"""Inference engine: exactly the notebook's pipeline (uniform JPEG -> frozen ResNet50
layer3+layer4 pooled to an 8x8 grid -> standardise -> patch scorer -> top-k mean)."""
import base64
import io
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model", "fishy_model.pt"))
BACKBONE_PATH = os.environ.get("BACKBONE_PATH", os.path.join(HERE, "weights", "resnet50_v2.pth"))

Image.MAX_IMAGE_PIXELS = 60_000_000          # reject decompression bombs
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


class PatchScorer(nn.Module):
    """Same layout as the notebook so the saved state_dict loads unchanged."""

    def __init__(self, in_dim, hidden, topk):
        super().__init__()
        self.topk = topk
        if hidden:
            self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(hidden, 1))
        else:
            self.f = nn.Linear(in_dim, 1)

    def forward(self, x):                        # x: B, P, D
        pl = self.f(x).squeeze(-1)               # B, P
        k = min(self.topk, pl.shape[1])
        return pl.topk(k, dim=1).values.mean(dim=1), pl


class Engine:
    def __init__(self):
        self.lock = threading.Lock()             # one image at a time keeps RAM predictable
        self.ready = False
        self.error = None
        self.info = {}
        try:
            self._load()
            self.ready = True
        except Exception as e:                   # the server still starts; the UI shows why
            self.error = str(e)
            print("[engine] not ready:", e, flush=True)

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model/fishy_model.pt is missing. Run the export cell at the end of the "
                "notebook and put the file in the model/ folder.")
        art = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

        self.size = int(art["uniform_size"])
        self.quality = int(art["jpeg_quality"])
        self.grid = int(art["grid"])
        self.threshold = float(art["threshold"])
        self.lo, self.hi = float(art["patch_lo"]), float(art["patch_hi"])
        self.mean, self.scale = art["mean"].float(), art["scale"].float()

        self.head = PatchScorer(int(art["in_dim"]), int(art["hidden"]), int(art["topk"]))
        self.head.load_state_dict(art["state_dict"])
        self.head.eval()

        backbone = models.resnet50(weights=None)
        if os.path.exists(BACKBONE_PATH):
            backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu", weights_only=True))
        else:
            print("[engine] weights/resnet50_v2.pth not found, downloading from torchvision...", flush=True)
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.info = {k: art.get(k) for k in (
            "model_name", "cv_auc", "cv_balanced_acc", "target_false_alarm",
            "real_cleared", "fakes_caught", "threshold")}

    # -- same treatment as the training images -------------------------------------------
    def _uniform(self, raw):
        try:
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img)   # phone photos display upright
            img = img.convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        except Exception as e:
            raise ValueError("Couldn't read that file as an image.") from e
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.quality)
        jpg = buf.getvalue()
        return Image.open(io.BytesIO(jpg)).convert("RGB"), jpg

    @torch.inference_mode()
    def _features(self, img):
        x = self.tf(img).unsqueeze(0)
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer2(b.layer1(x))
        l3 = b.layer3(x)
        l4 = b.layer4(l3)
        p = torch.cat([F.adaptive_avg_pool2d(l3, self.grid),
                       F.adaptive_avg_pool2d(l4, self.grid)], dim=1)   # 1, 3072, G, G
        return p.flatten(2).transpose(1, 2)[0]                        # G*G, 3072

    @torch.inference_mode()
    def check(self, raw):
        img, jpg = self._uniform(raw)
        with self.lock:
            feats = self._features(img)
            z = ((feats - self.mean) / self.scale).unsqueeze(0)
            logit, patch_logits = self.head(z)
        score = float(torch.sigmoid(logit)[0])
        pl = patch_logits[0]
        norm = ((pl - self.lo) / (self.hi - self.lo)).clamp(0, 1)
        top = pl.topk(3).indices.tolist()
        return {
            "flagged": score > self.threshold,
            "score": score,
            "threshold": self.threshold,
            "grid": self.grid,
            "patches": [round(v, 4) for v in norm.tolist()],
            "top": [[i // self.grid, i % self.grid] for i in top],
            "preview": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
        }

# ==== 15. Export for deployment (paste as the LAST cell, run after section 13) ====
# Saves the final "Patch-MIL NN + raw" model + scaler + threshold into one small file.
import copy, shutil
import numpy as np
import torch
from google.colab import files

NAME = "Patch-MIL NN + raw"
use_pca, hidden = MIL_VARIANTS[NAME]
assert not use_pca, "the web app expects the raw-feature model (no PCA)"

# threshold + heatmap scaling, computed for THIS model from the out-of-fold scores
real_scores = oof_prob[NAME][y_all == 0]
thr = float(np.quantile(real_scores, 1 - TARGET_FALSE_ALARM))
lo, hi = [float(v) for v in np.percentile(oof_patch[NAME], [5, 99.5])]
pred = (oof_prob[NAME] > thr).astype(int)
real_cleared = float(((pred == 0) & (y_all == 0)).sum() / (y_all == 0).sum())
fakes_caught = float(((pred == 1) & (y_all == 1)).sum() / (y_all == 1).sum())

model_cpu = copy.deepcopy(final_models[NAME]).cpu().eval()   # copy: keep the Colab one on GPU
artifact = {
    "model_name": NAME,
    "state_dict": {k: v.cpu() for k, v in model_cpu.state_dict().items()},
    "mean": torch.tensor(scaler_f.mean_, dtype=torch.float32),
    "scale": torch.tensor(scaler_f.scale_, dtype=torch.float32),
    "in_dim": int(P_all.shape[-1]), "hidden": int(hidden),
    "grid": int(GRID), "topk": int(TOPK),
    "uniform_size": int(UNIFORM_SIZE), "jpeg_quality": int(JPEG_QUALITY),
    "threshold": thr, "patch_lo": lo, "patch_hi": hi,
    "target_false_alarm": float(TARGET_FALSE_ALARM),
    "real_cleared": real_cleared, "fakes_caught": fakes_caught,
    "cv_auc": float(pooled_df.loc[NAME, "roc_auc"]),
    "cv_balanced_acc": float(pooled_df.loc[NAME, "balanced_acc"]),
}
torch.save(artifact, "fishy_model.pt")

# sanity check: rebuild scaling + head from the saved numbers and compare on 8 images
d = P_all.shape[-1]
chk = P_all[:8].astype(np.float32)
z_ref = scaler_f.transform(chk.reshape(-1, d)).astype(np.float32).reshape(8, -1, d)
ref, _ = predict_mil(final_models[NAME], z_ref)
with torch.no_grad():
    new, _ = model_cpu((torch.from_numpy(chk) - artifact["mean"]) / artifact["scale"])
print("max logit difference (should be ~0):", float(np.abs(ref - new.numpy()).max()))
print(f"threshold {thr:.3f} | real cleared {real_cleared:.0%} | fakes caught {fakes_caught:.0%}")

shutil.copy("fishy_model.pt", f"{GEN_DIR}/fishy_model.pt")   # backup on Drive
files.download("fishy_model.pt")                              # -> put it in the app's model/ folder

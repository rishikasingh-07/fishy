"""Run once at build time: saves the ImageNet ResNet50 weights next to the app,
so the server never has to download ~100 MB on every cold start."""
import os
import torch
from torchvision import models

os.makedirs("weights", exist_ok=True)
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
torch.save(m.state_dict(), os.path.join("weights", "resnet50_v2.pth"))
print("Saved weights/resnet50_v2.pth")

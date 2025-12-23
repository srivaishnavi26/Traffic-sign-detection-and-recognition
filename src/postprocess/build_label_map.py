import torch

ckpt = torch.load("traffic_sign_classifier.pth", map_location="cpu")
classes = ckpt["classes"]

for i, cls in enumerate(classes):
    print(i, "->", cls)

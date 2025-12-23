import torch

ckpt = torch.load("traffic_sign_classifier.pth", map_location="cpu")
classes = ckpt["classes"]

LABEL_MAP = {str(i): cls for i, cls in enumerate(classes)}

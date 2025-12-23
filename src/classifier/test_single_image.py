import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

ckpt = torch.load("traffic_sign_classifier.pth", map_location=DEVICE)
classes = ckpt["classes"]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

img_path = Path("data/classification/train/33/00033_00000_00027.png")
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(img)
    pred = logits.argmax(1).item()

print("Predicted index:", pred)
print("Mapped folder:", classes[pred])
print("Expected folder:", img_path.parent.name)

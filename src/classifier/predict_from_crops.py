import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path("traffic_sign_classifier.pth")
CROPS_DIR = Path("runs/detect/predict9/crops")
OUT_FILE = Path("runs/classifier/raw_predictions.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

THRESHOLD = 0.45
results = []

print("🔍 Predicting crops...\n")

with torch.no_grad():
    for img_path in CROPS_DIR.rglob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)

        conf = conf.item()
        if conf < THRESHOLD:
            label = "UNKNOWN"
        else:
            label = classes[pred.item()]

        results.append((img_path.stem, label, conf))
        print(f"{img_path.name} → {label} ({conf:.2f})")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    for name, label, conf in results:
        f.write(f"{name},{label},{conf:.3f}\n")

print(f"\n✅ Saved predictions to {OUT_FILE}")

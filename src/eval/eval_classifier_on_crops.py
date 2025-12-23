import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from postprocess.label_map import LABEL_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/classification/val")
CKPT_PATH = Path("traffic_sign_classifier.pth")
OUT_DIR = Path("runs/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
classes = ckpt["classes"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(preds)

print(classification_report(y_true, y_pred, target_names=classes))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png")
plt.close()

print("Saved confusion_matrix.png")

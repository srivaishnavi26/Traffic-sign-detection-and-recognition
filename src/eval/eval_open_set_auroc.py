import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/classification/val")
CKPT_PATH = Path("traffic_sign_classifier.pth")
OUT_DIR = Path("runs/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.75

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
y_score = []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)

        max_conf, _ = torch.max(probs, dim=1)

        for c in max_conf.cpu().tolist():
            y_score.append(c)
            y_true.append(1 if c >= CONF_THRESHOLD else 0)

auc = roc_auc_score(y_true, y_score)
print(f"Open-set AUROC: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Open-Set ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "open_set_roc.png")
plt.close()

print("Saved open_set_roc.png")

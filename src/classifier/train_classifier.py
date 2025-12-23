import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("data/classification")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

torch.save({
    "model": model.state_dict(),
    "classes": train_ds.classes
}, "traffic_sign_classifier.pth")

print("✅ Classifier trained & saved")

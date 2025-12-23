from pathlib import Path
import random
import shutil

RAW_ROOT = Path("data/raw/GTSRB/Train")
OUT_ROOT = Path("data/classification")

TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "val"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
val_ratio = 0.2

total_train = 0
total_val = 0

for class_dir in sorted(RAW_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    images = list(class_dir.glob("*.png"))
    if len(images) == 0:
        continue

    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    (TRAIN_DIR / class_name).mkdir(exist_ok=True)
    (VAL_DIR / class_name).mkdir(exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, TRAIN_DIR / class_name / img.name)
        total_train += 1

    for img in val_imgs:
        shutil.copy(img, VAL_DIR / class_name / img.name)
        total_val += 1

print("✅ GTSRB classification dataset prepared")
print(f"Train images: {total_train}")
print(f"Val images:   {total_val}")

from pathlib import Path
import shutil
import random

SRC = Path("data/raw/GTSRB/Train")
DST = Path("data/classification")

train_ratio = 0.8

for class_dir in SRC.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.ppm"))
    random.shuffle(images)

    split = int(len(images) * train_ratio)
    train_imgs = images[:split]
    val_imgs = images[split:]

    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = DST / split_name / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            shutil.copy(img, out_dir / img.name)

print("✅ GTSRB classification dataset prepared")

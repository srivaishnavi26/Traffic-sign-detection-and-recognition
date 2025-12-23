from pathlib import Path
import random
import cv2

RAW = Path("data/raw/GTSRB/Train")
OUT = Path("data/classification")

TRAIN = OUT / "train"
VAL = OUT / "val"

TRAIN.mkdir(parents=True, exist_ok=True)
VAL.mkdir(parents=True, exist_ok=True)

random.seed(42)

total_train = 0
total_val = 0

for class_dir in RAW.iterdir():
    if not class_dir.is_dir():
        continue

    cls = class_dir.name  # 00000, 00001, ...

    train_cls = TRAIN / cls
    val_cls = VAL / cls
    train_cls.mkdir(parents=True, exist_ok=True)
    val_cls.mkdir(parents=True, exist_ok=True)

    images = list(class_dir.glob("*.ppm"))
    random.shuffle(images)

    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]

    for img_path in train_imgs:
        img = cv2.imread(str(img_path))
        if img is not None:
            cv2.imwrite(str(train_cls / (img_path.stem + ".png")), img)
            total_train += 1

    for img_path in val_imgs:
        img = cv2.imread(str(img_path))
        if img is not None:
            cv2.imwrite(str(val_cls / (img_path.stem + ".png")), img)
            total_val += 1

print(f"✅ Done")
print(f"Train images: {total_train}")
print(f"Val images: {total_val}")

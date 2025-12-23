import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
RAW_DIR = Path("data/raw/GTSDB/TrainIJCNN2013/TrainIJCNN2013")
GT_FILE = RAW_DIR / "gt.txt"

IMG_OUT = Path("data/detection/images/train")
LBL_OUT = Path("data/detection/labels/train")

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

annotations = {}

# Read gt.txt
with open(GT_FILE, "r") as f:
    for line in f:
        img, x1, y1, x2, y2, cls = line.strip().split(";")
        x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])

        annotations.setdefault(img, []).append((x1, y1, x2, y2, cls))

# Convert
for img_name, boxes in tqdm(annotations.items(), desc="Converting GTSDB"):
    src_img = RAW_DIR / img_name
    if not src_img.exists():
        continue

    # Copy image
    shutil.copy(src_img, IMG_OUT / img_name)

    # Read image size
    with Image.open(src_img) as im:
        w, h = im.size

    label_file = (LBL_OUT / img_name).with_suffix(".txt")

    with open(label_file, "w") as lf:
        for x1, y1, x2, y2, cls in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lf.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

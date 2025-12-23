from pathlib import Path
import cv2

src_root = Path("data/raw/GTSDB/TrainIJCNN2013")
dst_root = Path("data/detection/images/train")
dst_root.mkdir(parents=True, exist_ok=True)

count = 0
for img_path in src_root.rglob("*.ppm"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    out_name = img_path.stem + ".png"
    cv2.imwrite(str(dst_root / out_name), img)
    count += 1

print(f"Copied & converted {count} images")

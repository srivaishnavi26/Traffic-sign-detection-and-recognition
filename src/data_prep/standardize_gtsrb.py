import os
from PIL import Image

INPUT_DIR = "data/raw/GTSRB"
OUTPUT_DIR = "data/classification/gtsrb"
SIZE = (64, 64)
MAX_PER_CLASS = 300  # limit for now

for class_id in os.listdir(INPUT_DIR):
    in_class = os.path.join(INPUT_DIR, class_id)
    out_class = os.path.join(OUTPUT_DIR, class_id)

    if not os.path.isdir(in_class):
        continue

    os.makedirs(out_class, exist_ok=True)

    count = 0
    for img_file in os.listdir(in_class):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(in_class, img_file))
                img = img.convert("RGB")
                img = img.resize(SIZE)
                img.save(os.path.join(out_class, img_file.replace('.png', '.jpg')), "JPEG")
                count += 1
            except:
                pass

        if count >= MAX_PER_CLASS:
            break

print("GTSRB standardization (sampled) completed")

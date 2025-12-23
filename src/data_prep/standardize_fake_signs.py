import os
from PIL import Image

INPUT_DIR = "data/raw/fake_signs"
OUTPUT_DIR = "data/classification/fake_signs"
SIZE = (64, 64)

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mpo')):
            input_path = os.path.join(root, file)

            try:
                img = Image.open(input_path)
                img = img.convert("RGB")
                img = img.resize(SIZE)

                out_name = f"fake_{count}.jpg"
                img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG")

                count += 1
            except:
                pass

print(f"Processed fake signs: {count}")

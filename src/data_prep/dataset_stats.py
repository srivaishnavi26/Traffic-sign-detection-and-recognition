import os
from PIL import Image
from collections import Counter

def analyze_folder(path, name, limit=500):
    sizes = []
    formats = []
    count = 0

    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, f)
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                        formats.append(img.format)
                        count += 1
                except:
                    pass

                if count >= limit:
                    break
        if count >= limit:
            break

    print(f"\nDataset: {name}")
    print(f"Sampled images: {count}")
    print(f"Top sizes: {Counter(sizes).most_common(3)}")
    print(f"Formats: {Counter(formats)}")


if __name__ == "__main__":
    analyze_folder("data/raw/GTSRB", "GTSRB")
    analyze_folder("data/raw/GTSDB", "GTSDB")
    analyze_folder("data/raw/fake_signs", "Fake Signs", limit=300)

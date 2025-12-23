from pathlib import Path

LABELS_DIR = Path("data/detection/labels")
OUT_DIR = Path("data/detection_binary/labels")

for split in ["train", "val"]:
    in_dir = LABELS_DIR / split
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_file in in_dir.glob("*.txt"):
        lines = label_file.read_text().strip().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, x, y, w, h = parts
            new_lines.append(f"0 {x} {y} {w} {h}")

        if new_lines:
            (out_dir / label_file.name).write_text("\n".join(new_lines))

from collections import defaultdict, Counter
from pathlib import Path

RAW_PRED_FILE = Path("runs/classifier/raw_predictions.txt")
OUT_FILE = Path("runs/classifier/temporal_predictions.txt")

WINDOW = 7
CONF_THRESH = 0.6

frames = defaultdict(list)

with open(RAW_PRED_FILE) as f:
    for line in f:
        if not line.strip():
            continue

        fname, cls, conf = line.strip().split(",")

        if cls == "UNKNOWN":
            continue

        _, frame_no = fname.split("_")
        frames[int(frame_no)].append((cls, float(conf)))

with open(OUT_FILE, "w") as out:
    for frame_no in sorted(frames):
        window_preds = []
        for i in range(frame_no - WINDOW, frame_no + WINDOW + 1):
            window_preds.extend(frames.get(i, []))

        labels = [cls for cls, conf in window_preds if conf >= CONF_THRESH]

        if not labels:
            continue

        stable = Counter(labels).most_common(1)[0][0]
        raw = frames[frame_no][0][0]
        conf = max(c for _, c in frames[frame_no])

        out.write(f"Video8_{frame_no},{raw},{stable},{conf:.3f}\n")

print("Temporal predictions regenerated (UNKNOWN removed)")

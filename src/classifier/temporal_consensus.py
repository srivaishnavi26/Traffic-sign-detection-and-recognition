from collections import deque, Counter
from pathlib import Path

WINDOW = 5
UNKNOWN = "UNKNOWN"

def majority_vote(labels):
    valid = [l for l in labels if l != UNKNOWN]
    if not valid:
        return UNKNOWN
    return Counter(valid).most_common(1)[0][0]

def parse_filename(name):
    # Video1_23 → ("Video1", 23)
    parts = name.split("_")
    return parts[0], int(parts[1])

pred_file = Path("runs/classifier/raw_predictions.txt")
out_file = Path("runs/classifier/temporal_predictions.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)

history = {}
results = []

with open(pred_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # CSV format: Video1_23,12,0.995
        fname, label, conf = line.split(",")
        conf = float(conf)

        video, frame = parse_filename(fname)

        if video not in history:
            history[video] = deque(maxlen=WINDOW)

        history[video].append(label)
        stable = majority_vote(history[video])

        results.append((fname, label, stable, conf))

with open(out_file, "w") as f:
    for fname, raw, stable, conf in results:
        f.write(f"{fname},{raw},{stable},{conf:.2f}\n")

print(f"✅ Temporal consensus applied: {len(results)} entries saved")

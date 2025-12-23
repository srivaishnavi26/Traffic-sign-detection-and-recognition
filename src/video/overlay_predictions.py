import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/road_videos/Video4.mp4")  # change if needed
YOLO_LABELS = Path("runs/detect/predict7/labels")
PRED_FILE = Path("runs/classifier/raw_predictions.txt")
OUT_VIDEO = Path("runs/final/output_labeled.mp4")

# Load classifier predictions
pred_map = {}
with open(PRED_FILE) as f:
    for line in f:
        name, label, conf = line.strip().split(",")
        pred_map[name] = (label, float(conf))

cap = cv2.VideoCapture(str(VIDEO_PATH))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(OUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_id = 0
print("🎬 Rendering labeled video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label_file = YOLO_LABELS / f"{Path(VIDEO_PATH).stem}_{frame_id}.txt"
    if label_file.exists():
        with open(label_file) as f:
            for i, line in enumerate(f):
                _, xc, yc, bw, bh = map(float, line.split())

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                crop_name = f"{Path(VIDEO_PATH).stem}_{frame_id}"
                label, conf = pred_map.get(crop_name, ("UNKNOWN", 0.0))

                text = f"{label} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    writer.write(frame)
    frame_id += 1

cap.release()
writer.release()
print(f"✅ Final video saved to {OUT_VIDEO}")

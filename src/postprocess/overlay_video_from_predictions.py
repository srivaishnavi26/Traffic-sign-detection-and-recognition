import cv2
import re
from pathlib import Path
from gtsrb_sign_names import SIGN_NAMES

VIDEO_NAME = "BKS_Road"
VIDEO_PATH = Path(f"data/raw/road_videos/{VIDEO_NAME}.mp4")

YOLO_DIR = Path("runs/detect/BKS_Road")
LABELS_DIR = YOLO_DIR / "labels"

PRED_FILE = Path("runs/classifier/temporal_predictions.txt")

OUT_PATH = Path("runs/final/BKS_temporal.mp4")

# ---------------- load predictions ----------------

frame_predictions = {}

with open(PRED_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        name, raw_label, stable_label, conf = line.split(",")

        # safely extract frame number
        match = re.search(r'_(\d+)', name)
        if not match:
            continue

        frame_id = int(match.group(1))

        frame_predictions[frame_id] = stable_label

print("Loaded predictions:", len(frame_predictions))


# ---------------- video processing ----------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

frame_id = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO label files start from frame 1
    frame_num = frame_id + 1

    label_file = LABELS_DIR / f"{VIDEO_NAME}_{frame_num}.txt"

    if label_file.exists():

        with open(label_file) as f:

            for line in f:

                _, xc, yc, w, h = map(float, line.split())

                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)

                label = frame_predictions.get(frame_num, "UNKNOWN")

                if label != "UNKNOWN":
                    label = SIGN_NAMES.get(label, label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    writer.write(frame)
    frame_id += 1

cap.release()
writer.release()

print("Temporal consensus video generated successfully")
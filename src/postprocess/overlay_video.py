import cv2
from pathlib import Path
from collections import Counter
from label_map import LABEL_MAP
from gtsrb_sign_names import SIGN_NAMES

VIDEO_PATH = Path("data/raw/road_videos/Video2.mp4")
PRED_FILE = Path("runs/classifier/temporal_predictions.txt")
OUT_PATH = Path("runs/final/video8_output.mp4")

labels = []

with open(PRED_FILE) as f:
    for line in f:
        if not line.strip():
            continue
        video_frame, _, stable, _ = line.strip().split(",")

        if not video_frame.startswith("Video8_"):
            continue

        if stable in LABEL_MAP:
            folder_id = LABEL_MAP[stable]
            sign_name = SIGN_NAMES.get(folder_id)
            if sign_name:
                labels.append(sign_name)

if not labels:
    raise RuntimeError("NO VALID LABELS FOUND")

final_label = Counter(labels).most_common(1)[0][0]
print("FINAL SIGN:", final_label)

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(OUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        final_label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    writer.write(frame)

cap.release()
writer.release()

print("FINAL VIDEO GENERATED")

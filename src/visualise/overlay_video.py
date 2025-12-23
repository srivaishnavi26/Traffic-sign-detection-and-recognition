print("OVERLAY FILE EXECUTED")
exit()

import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/road_videos/Video8.mp4")
OUT_PATH = Path("runs/final/debug_video8.mp4")

print("SCRIPT STARTED")

cap = cv2.VideoCapture(str(VIDEO_PATH))
print("VideoCapture created")

if not cap.isOpened():
    print("VIDEO NOT OPENED")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video props:", fps, w, h)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(OUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

if not writer.isOpened():
    print("WRITER NOT OPENED")
    exit(1)

frame_id = 0
max_frames = 300   # HARD STOP

while frame_id < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("FRAME READ FAILED AT", frame_id)
        break

    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    writer.write(frame)
    frame_id += 1

cap.release()
writer.release()

print("SCRIPT FINISHED")

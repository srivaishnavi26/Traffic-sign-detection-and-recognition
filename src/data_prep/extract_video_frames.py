import cv2
import os

VIDEO_DIR = "data/raw/road_videos"
OUTPUT_DIR = "data/detection/images/video_frames"
FRAME_INTERVAL = 30  # approx 1 frame per second (30 FPS videos)

os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_count = 0

for video in os.listdir(VIDEO_DIR):
    if not video.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(VIDEO_DIR, video)
    cap = cv2.VideoCapture(video_path)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_INTERVAL == 0:
            frame_name = f"{video.split('.')[0]}_frame_{saved}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), frame)
            saved += 1
            frame_count += 1

        count += 1

    cap.release()
    print(f"{video}: extracted {saved} frames")

print(f"\nTotal extracted frames: {frame_count}")

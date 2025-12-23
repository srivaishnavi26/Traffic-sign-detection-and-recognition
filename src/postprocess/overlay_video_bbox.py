import cv2
import torch
from pathlib import Path
from torchvision import transforms, models
from PIL import Image

from label_map import LABEL_MAP
from gtsrb_sign_names import SIGN_NAMES

VIDEO_NAME = "Video6"
VIDEO_PATH = Path(f"data/raw/road_videos/{VIDEO_NAME}.mp4")

YOLO_DIR = Path("runs/detect/predict9")
CROPS_DIR = YOLO_DIR / "crops" / "traffic_sign"
LABELS_DIR = YOLO_DIR / "labels"

OUT_PATH = Path("runs/final/video8_bbox_output.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- classifier -----------------
ckpt = torch.load("traffic_sign_classifier.pth", map_location=DEVICE)
classes = ckpt["classes"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_crop(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img).argmax(1).item()
    folder_id = LABEL_MAP[str(pred)]
    return SIGN_NAMES.get(folder_id, folder_id)

# ----------------- video -----------------
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

    label_file = LABELS_DIR / f"{VIDEO_NAME}_{frame_id}.txt"

    if label_file.exists():
        with open(label_file) as f:
            for det_idx, line in enumerate(f):
                _, xc, yc, w, h = map(float, line.split())

                x1 = int((xc - w / 2) * W)
                y1 = int((yc - h / 2) * H)
                x2 = int((xc + w / 2) * W)
                y2 = int((yc + h / 2) * H)

                # 🔥 CORRECT CROP NAME LOGIC
                if det_idx == 0:
                    crop_name = f"{VIDEO_NAME}_{frame_id}.jpg"
                else:
                    crop_name = f"{VIDEO_NAME}_{frame_id}{det_idx + 1}.jpg"

                crop_path = CROPS_DIR / crop_name

                if crop_path.exists():
                    label = classify_crop(crop_path)

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
print("✅ BBOX VIDEO GENERATED")

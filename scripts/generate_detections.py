import os
import json
from tqdm import tqdm
from ultralytics import YOLO
import cv2

# === CONFIG ===
IMG_DIR = "data/rgb"
OUT_DIR = "data/detections"
MODEL_WEIGHTS = "yolov8n.pt"  # or yolov8m.pt/yolov8l.pt
CONFIDENCE_THRESHOLD = 0.3
TARGET_CLASSES = {"car", "truck", "bus", "person"}

# === Setup ===
os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_WEIGHTS)

# === Run detection on each image ===
for fname in tqdm(sorted(os.listdir(IMG_DIR)), desc="Running detection"):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    image = cv2.imread(img_path)

    # Run YOLO inference
    results = model(image, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        if cls_name in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            detections.append({
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

    # Save detections as JSON
    out_file = os.path.join(OUT_DIR, fname.replace(".png", ".json"))
    with open(out_file, "w") as f:
        json.dump(detections, f, indent=2)

print("✅ Detection complete. Results saved to:", OUT_DIR)

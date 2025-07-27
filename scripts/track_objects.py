import os
import json
import cv2
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIG ===
IMG_DIR = "data/rgb"
DET_DIR = "data/detections"
OUT_DIR = "data/tracks"
os.makedirs(OUT_DIR, exist_ok=True)

# === Initialize DeepSORT tracker ===
tracker = DeepSort(max_age=30)

# === Process each frame ===
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])

for fname in tqdm(frames, desc="Tracking"):
    img_path = os.path.join(IMG_DIR, fname)
    det_path = os.path.join(DET_DIR, fname.replace(".png", ".json"))

    # Skip missing detection files
    if not os.path.exists(det_path):
        print(f"⚠️ Skipping missing detection: {det_path}")
        continue

    # Read image and detections
    image = cv2.imread(img_path)
    with open(det_path, 'r') as f:
        detections = json.load(f)

    # Prepare DeepSORT input
    inputs = []
    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        conf = det["conf"]
        cls = det["class"]
        inputs.append([bbox, conf, cls])  # [[x1,y1,x2,y2], conf, label]

    # Run tracking
    tracks = tracker.update_tracks(inputs, frame=image)

    # Collect per-frame tracking output
    frame_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = int(track.track_id)
        ltrb = list(map(float, track.to_ltrb()))  # left, top, right, bottom
        cls = track.det_class
        frame_tracks.append({
            "id": track_id,
            "class": cls,
            "bbox": ltrb
        })

    # Save tracking result
    out_file = os.path.join(OUT_DIR, fname.replace(".png", ".json"))
    with open(out_file, "w") as f:
        json.dump(frame_tracks, f, indent=2)

print("✅ Tracking complete. Results saved to:", OUT_DIR)

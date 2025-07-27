import os, json
import numpy as np
import cv2
from tqdm import tqdm

# === Config ===
RGB_DIR = "data/rgb"
DEPTH_DIR = "data/depth"
TRACK_DIR = "data/tracks"
POSE_FILE = "data/groundtruth.txt"
TIMESTAMPS_FILE = "data/rgb.txt"
OUT_DIR = "data/trajectories"
os.makedirs(OUT_DIR, exist_ok=True)

fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5

# === Load timestamps ===
timestamps = []
with open(TIMESTAMPS_FILE, 'r') as f:
    for line in f:
        if line.startswith("#"):
            continue
        timestamps.append(line.strip().split()[0])

# === Load poses (timestamp → [R|t]) ===
pose_map = {}
with open(POSE_FILE, 'r') as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) != 8: continue
        ts = parts[0]
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:])
        pose_map[ts] = {
            "translation": [tx, ty, tz],
            "quaternion": [qx, qy, qz, qw]
        }

# === Helper: quaternion to rotation matrix ===
from scipy.spatial.transform import Rotation as R
def get_pose_matrix(translation, quaternion):
    rot = R.from_quat(quaternion).as_matrix()  # (3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = translation
    return T

# === Gather object trajectories ===
trajectories = {}

frames = sorted([f for f in os.listdir(TRACK_DIR) if f.endswith(".json")])
for fname in tqdm(frames, desc="Processing frames"):
    frame_idx = int(fname.replace(".json", ""))
    if frame_idx >= len(timestamps):
        continue
    ts = timestamps[frame_idx]

    if ts not in pose_map:
        continue

    # Load image depth + track data
    depth_path = os.path.join(DEPTH_DIR, f"{frame_idx:06d}.png")
    if not os.path.exists(depth_path):
        continue
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0  # scale depth

    with open(os.path.join(TRACK_DIR, fname), 'r') as f:
        objects = json.load(f)

    T_wc = get_pose_matrix(pose_map[ts]["translation"], pose_map[ts]["quaternion"])

    for obj in objects:
        obj_id = f"{obj['class']}_{obj['id']}"
        x1, y1, x2, y2 = obj['bbox']
        cx_img = int((x1 + x2) / 2)
        cy_img = int((y1 + y2) / 2)

        if 0 <= cx_img < depth.shape[1] and 0 <= cy_img < depth.shape[0]:
            z = depth[cy_img, cx_img]
            if z == 0:
                continue

            # Backproject to 3D (camera frame)
            x = (cx_img - cx) * z / fx
            y = (cy_img - cy) * z / fy
            pt_cam = np.array([x, y, z, 1.0])  # homogenous

            # Transform to world frame
            pt_world = T_wc @ pt_cam
            position = pt_world[:3].tolist()

            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append({
                "frame": frame_idx,
                "timestamp": ts,
                "position": position
            })

# === Save individual object files ===
for obj_id, traj in trajectories.items():
    out_path = os.path.join(OUT_DIR, f"{obj_id}.json")
    with open(out_path, "w") as f:
        json.dump(traj, f, indent=2)

print(f"✅ Done. Saved {len(trajectories)} object trajectories to: {OUT_DIR}")

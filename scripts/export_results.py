import os
import json
import shutil
from tqdm import tqdm

# === CONFIG ===
PREDICTIONS_DIR = "outputs/predictions"
EXPLANATIONS_DIR = "outputs/explanations"
VIS_DIR = "outputs/visualizations"
EXPORT_DIR = "outputs/final_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# === Create export folder per object
objects = [f.replace(".json", "") for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".json")]

for obj_id in tqdm(objects, desc="Exporting results"):
    obj_folder = os.path.join(EXPORT_DIR, obj_id)
    os.makedirs(obj_folder, exist_ok=True)

    # === Copy prediction file
    pred_src = os.path.join(PREDICTIONS_DIR, f"{obj_id}.json")
    pred_dst = os.path.join(obj_folder, "trajectory_prediction.json")
    shutil.copyfile(pred_src, pred_dst)

    # === Copy explanation if exists
    expl_src = os.path.join(EXPLANATIONS_DIR, f"{obj_id}.txt")
    if os.path.exists(expl_src):
        expl_dst = os.path.join(obj_folder, "gpt4_explanation.txt")
        shutil.copyfile(expl_src, expl_dst)

    # === Copy visualization image (optional)
    vis_img = os.path.join(VIS_DIR, f"{obj_id}.png")
    if os.path.exists(vis_img):
        shutil.copyfile(vis_img, os.path.join(obj_folder, "trajectory_viz.png"))

    # === Copy video if available
    vis_vid = os.path.join(VIS_DIR, f"{obj_id}.mp4")
    if os.path.exists(vis_vid):
        shutil.copyfile(vis_vid, os.path.join(obj_folder, "trajectory_viz.mp4"))

print(f"✅ Export complete. Organized results saved to: {EXPORT_DIR}")

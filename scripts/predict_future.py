import os, json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# === Config ===
TRAJ_DIR = "data/trajectories"
OUT_DIR = "outputs/predictions"
os.makedirs(OUT_DIR, exist_ok=True)

OBS_FRAMES = 10
PRED_FRAMES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Simple GRU Predictor ===
class TrajectoryGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x, pred_len):
        output_seq = []
        h = None
        for _ in range(pred_len):
            out, h = self.gru(x.unsqueeze(0), h)  # (1, 1, hidden)
            next_pos = self.output(out[:, -1, :])  # (1, 3)
            output_seq.append(next_pos.squeeze(0))
            x = torch.cat([x[:, 1:], next_pos], dim=1)
        return torch.stack(output_seq, dim=1).squeeze(0)  # (pred_len, 3)

model = TrajectoryGRU().to(DEVICE)

# For demo/testing, use random weights — no training yet
model.eval()

# === Predict for each object ===
traj_files = [f for f in os.listdir(TRAJ_DIR) if f.endswith(".json")]
for fname in tqdm(traj_files, desc="Predicting"):
    with open(os.path.join(TRAJ_DIR, fname)) as f:
        traj = json.load(f)

    if len(traj) < OBS_FRAMES:
        continue

    obs_positions = [step["position"] for step in traj[-OBS_FRAMES:]]
    obs_tensor = torch.tensor(obs_positions, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, N, 3)

    with torch.no_grad():
        pred = model(obs_tensor, pred_len=PRED_FRAMES)  # (M, 3)

    out_data = {
        "observed": obs_positions,
        "predicted": pred.cpu().numpy().tolist()
    }

    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

print(f"✅ Prediction complete. Results saved in: {OUT_DIR}")

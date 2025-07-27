import json
import os
import openai

# === Config ===
PREDICTION_DIR = "outputs/predictions"
EXPLAIN_DIR = "outputs/explanations"
os.makedirs(EXPLAIN_DIR, exist_ok=True)
OBJ_ID = "person_5"
FILE = os.path.join(PREDICTION_DIR, f"{OBJ_ID}.json")

# === Your OpenAI API key (if using GPT-4)
openai.api_key = "YOUR_API_KEY"

# === Load Trajectory
with open(FILE, 'r') as f:
    data = json.load(f)

obs = data["observed"]
pred = data["predicted"]

# === Build Prompt
prompt = f"""
You are an AI assistant analyzing the motion of objects in a 3D driving scene.

Here is a sequence of 3D positions (x, y, z) over time:

Observed trajectory:
{obs}

Predicted future positions:
{pred}

Please describe in natural language what the object is likely doing, and why.
"""

# === Call GPT-4
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a smart motion analyst for autonomous driving."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

# === Save Result
explanation = response['choices'][0]['message']['content']
out_path = os.path.join(EXPLAIN_DIR, f"{OBJ_ID}.txt")
with open(out_path, "w") as f:
    f.write(explanation)

print("✅ Explanation saved:", out_path)

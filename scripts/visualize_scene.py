import json
import os
from panda3d.core import LineSegs, Vec3, Point3
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

# === Config ===
PREDICTION_DIR = "outputs/predictions"
OBJ_ID = "person_5"  # change per test
FILE = os.path.join(PREDICTION_DIR, f"{OBJ_ID}.json")

# === Visualization App ===
class TrajectoryViewer(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.camera.set_pos(0, -10, 3)
        self.camera.look_at(0, 0, 0)
        self.load_trajectory()

    def draw_path(self, points, color):
        segs = LineSegs()
        segs.set_thickness(3)
        segs.set_color(*color)
        for pt in points:
            segs.draw_to(Point3(*pt))
        node = segs.create()
        self.render.attach_new_node(node)

    def draw_point(self, pt, color, scale=0.1):
        marker = self.loader.load_model("models/smiley")
        marker.reparent_to(self.render)
        marker.set_scale(scale)
        marker.set_color(*color)
        marker.set_pos(*pt)

    def load_trajectory(self):
        with open(FILE, 'r') as f:
            data = json.load(f)

        obs = data["observed"]
        pred = data["predicted"]

        # Draw observed (blue)
        self.draw_path(obs, (0.2, 0.4, 1.0, 1))
        for pt in obs:
            self.draw_point(pt, (0.2, 0.4, 1.0, 1))

        # Draw predicted (red)
        self.draw_path(pred, (1.0, 0.1, 0.1, 1))
        for pt in pred:
            self.draw_point(pt, (1.0, 0.1, 0.1, 1))


# === Run Viewer ===
if __name__ == "__main__":
    app = TrajectoryViewer()
    app.run()

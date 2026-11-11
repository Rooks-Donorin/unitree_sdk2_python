"""Check the optical-flow mover tracker (motion_flow.py) on synthetic video where the truth is known.

    .venv_wifi/bin/python tests/flow_synthetic_test.py          # needs numpy + opencv

Video: a textured world seen by a camera that pans, drifts and zooms (like a walking dog's), with two textured
objects moving through the picture by themselves. Checks (exit code 1 if one fails):
  1. with the movers, both are tracked and each keeps one id;
  2. with camera motion only, nothing is tracked (no false alarms);
  3. the measured speed of a mover (what is left after the camera's own motion is removed) is within 25% of
     the truth on average.
This is synthetic video, not the dog's camera; see the README for what has and has not been tried on the dog.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import motion_flow as mf

W, H = 1280, 720
rng = np.random.default_rng(3)


def texture(h, w, blur):
    t = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8).astype(np.float32), (0, 0), blur)
    return ((t - t.min()) / (t.max() - t.min()) * 255).astype(np.uint8)


WORLD, OBJ_A, OBJ_B = texture(1500, 2600, 6), texture(160, 120, 5), texture(120, 160, 5)
OBJECTS = {"A": (OBJ_A, 150, 380, 200, 0), "B": (OBJ_B, 850, 80, -20, 150)}    # start x, y and image velocity px/s


def camera(t):
    """World -> image affine at time t: pan right 120 px/s, drift down 30 px/s, zoom +2 %/s."""
    z = 1 + 0.02 * t
    return np.float32([[z, 0, 200 + 120 * t - (z - 1) * W / 2], [0, z, 150 + 30 * t - (z - 1) * H / 2]])


def frame(t, objects=True):
    img = cv2.warpAffine(WORLD, camera(t), (W, H), borderMode=cv2.BORDER_REFLECT)
    where = {}
    if objects:
        for name, (tex, x0, y0, vx, vy) in OBJECTS.items():
            x, y = int(x0 + vx * t), int(y0 + vy * t)
            h, w = tex.shape[:2]
            if 0 <= x < W - w and 0 <= y < H - h:
                img[y:y + h, x:x + w] = tex
                where[name] = (x + w / 2, y + h / 2)
    return img, where


def background_velocity(t, p, dt=1e-3):
    """Image velocity (px/s) of the background at image point p, from the camera model."""
    world = cv2.invertAffineTransform(camera(t)) @ [p[0], p[1], 1]
    return (camera(t + dt) @ [*world, 1] - camera(t) @ [*world, 1]) / dt


def run(objects):
    engine, t, ids, ratios = mf.FlowEngine(), 0.0, set(), []
    jitter = np.random.default_rng(1)
    while t < 4.0:
        img, where = frame(t, objects)
        res = engine.process(img, 1000.0 + t + jitter.normal(0, 0.006))      # arrival-time jitter, as over Wi-Fi
        t += 1 / 15
        if res is None:
            continue
        for tr in res["tracks"]:
            if not tr["confirmed"]:
                continue
            ids.add(tr["id"])
            if objects and t > 2.0 and where:
                s = W / engine.width
                c = ((tr["box"][0] + tr["box"][2]) * s / 2, (tr["box"][1] + tr["box"][3]) * s / 2)
                name = min(where, key=lambda k: math.hypot(where[k][0] - c[0], where[k][1] - c[1]))
                bv = background_velocity(t, where[name])
                true = math.hypot(OBJECTS[name][3] - bv[0], OBJECTS[name][4] - bv[1])
                ratios.append(math.hypot(tr["vel"][0] * W, tr["vel"][1] * W) / true)
    return ids, ratios


ids_moving, ratios = run(True)
ids_static, _ = run(False)
ratio = float(np.median(ratios)) if ratios else float("nan")
checks = [
    ("both movers tracked (2 distinct ids)", len(ids_moving) == 2),
    ("no false tracks with camera motion only", len(ids_static) == 0),
    ("measured/true mover speed within 25%% (median %.2f)" % ratio, 0.75 <= ratio <= 1.25),
]
for label, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + label)
sys.exit(0 if all(ok for _, ok in checks) else 1)

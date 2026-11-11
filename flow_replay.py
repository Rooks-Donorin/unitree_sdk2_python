"""Replay a camera clip (recorded with key 7 in lidar_control.py) through the optical-flow code, offline.

    .venv_wifi/bin/python flow_replay.py captures/clip_20261110_153000            # defaults
    .venv_wifi/bin/python flow_replay.py CLIP --motion 2.5 --every 10 --view arrows

It feeds the frames to motion_flow.FlowEngine with the times they arrived at, prints a summary (how often the
camera fit was good, how many movers were found, how that compares with what the live flow showed) and writes
annotated pictures to CLIP/replay_<settings>/ so the result can be judged by eye. Change --motion, --width,
--preset to see how the settings change the result without going back to the dog.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

import motion_flow as mf


def annotate(rgb, result, view):
    """Draw the flow result on a copy of the frame (full size), in OpenCV (BGR)."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    fw, fh = result["size"]
    sx, sy = w / fw, h / fh
    if view == "colour" and result["viz"] is not None:
        viz = cv2.resize(cv2.cvtColor(result["viz"], cv2.COLOR_RGB2BGR), (w, h))
        out = cv2.addWeighted(out, 0.4, viz, 0.6, 0)
    elif view == "arrows" and result["arrows"] is not None:
        for x, y, fx, fy, moving in result["arrows"].tolist():
            p, q = (int(x * sx), int(y * sy)), (int((x + 3 * fx) * sx), int((y + 3 * fy) * sy))
            cv2.arrowedLine(out, p, q, (60, 60, 255) if moving else (200, 200, 200), 2, tipLength=0.3)
    elif result["mask"] is not None:
        tint = cv2.resize(result["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
        out[tint > 0] = (0.5 * out[tint > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for t in result["tracks"]:
        x0, y0, x1, y1 = [int(v) for v in (t["box"][0] * sx, t["box"][1] * sy, t["box"][2] * sx, t["box"][3] * sy)]
        if not t["confirmed"]:
            cv2.rectangle(out, (x0, y0), (x1, y1), (60, 140, 150), 1)
            continue
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 170, 255), 3)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.arrowedLine(out, (cx, cy), (int(cx + t["vel"][0] * w * 0.5), int(cy + t["vel"][1] * w * 0.5)), (255, 255, 255), 3)
        cv2.putText(out, f"#{t['id']} {t['speed'] * 100:.0f}%/s", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
    fit = result["inliers"] * 100
    note = result["note"] or "ok"
    cv2.putText(out, f"fit {fit:.0f}%  {note}  gap {result['gap'] * 1000:.0f} ms", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip folder (captures/clip_...)")
    ap.add_argument("--motion", type=float, default=mf.MOTION, help="sensitivity: leftover px that count as moving")
    ap.add_argument("--width", type=int, default=mf.FLOW_WIDTH, help="flow resolution (px wide)")
    ap.add_argument("--preset", default=mf.FLOW_PRESET, choices=["ultrafast", "fast", "medium"])
    ap.add_argument("--view", default="movers", choices=["movers", "colour", "arrows"])
    ap.add_argument("--every", type=int, default=15, help="save every Nth frame as a picture (0 = none)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.clip, "frames.csv"))))
    if not rows:
        sys.exit("frames.csv has no frames")
    view_id = {"movers": 0, "colour": 1, "arrows": 2}[a.view]
    engine = mf.FlowEngine(width=a.width, preset=a.preset, motion=a.motion, view=view_id)
    tag = f"replay_m{a.motion:g}_w{a.width}_{a.preset}_{a.view}"
    out_dir = os.path.join(a.clip, tag)
    os.makedirs(out_dir, exist_ok=True)

    notes, fits, movers, live_movers, live_fits, saved = {}, [], [], [], [], 0
    for i, row in enumerate(rows):
        bgr = cv2.imread(os.path.join(a.clip, f"frame_{int(row['frame']):05d}.jpg"))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = engine.process(rgb, float(row["time"]))
        if res is None:
            continue
        notes[res["note"] or "ok"] = notes.get(res["note"] or "ok", 0) + 1
        fits.append(res["inliers"])
        movers.append(sum(t["confirmed"] for t in res["tracks"]))
        if row.get("flow_movers", "") != "":
            live_movers.append(int(row["flow_movers"]))
            live_fits.append(float(row["flow_fit"]))
        if a.every and i % a.every == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), annotate(rgb, res, a.view))
            saved += 1

    span = float(rows[-1]["time"]) - float(rows[0]["time"])
    print(f"{len(rows)} frames over {span:.1f} s ({len(rows) / max(span, 1e-6):.1f} fps arriving); flow results for {len(fits)}")
    print(f"settings: motion {a.motion:g}, width {a.width}, preset {a.preset}")
    print("camera-fit quality: median %.0f%%, 10th percentile %.0f%%" % (100 * np.median(fits), 100 * np.percentile(fits, 10)))
    print("what the flow said about each frame:", ", ".join(f"{k}: {v}" for k, v in sorted(notes.items(), key=lambda kv: -kv[1])))
    print("confirmed movers per frame: median %d, max %d; frames with at least one: %.0f%%"
          % (np.median(movers), max(movers), 100 * np.mean(np.array(movers) > 0)))
    if live_movers:
        print("live (what you saw): movers median %d, frames with one %.0f%%; fit median %.0f%%"
              % (np.median(live_movers), 100 * np.mean(np.array(live_movers) > 0), 100 * np.median(live_fits)))
    print(f"{saved} annotated pictures in {out_dir}")


if __name__ == "__main__":
    main()

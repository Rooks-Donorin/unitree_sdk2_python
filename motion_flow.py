"""Optical flow on the dog's camera: finds what is moving and tracks it. Used by lidar_control.py (key 3).

How it works, per camera frame:
  1. The frame is shrunk (FLOW_WIDTH px wide) and compared with the newest earlier frame that is at least
     FLOW_GAP seconds old (a longer gap gives more motion to measure than back-to-back frames).
  2. DIS optical flow gives, for every pixel, how far it moved between the two frames.
  3. The camera's OWN motion is fitted to that flow (a similarity: shift + turn + zoom, found with RANSAC so
     things that move on their own don't spoil it). Walking, turning and leaning all show up here.
  4. What is left over once the camera's motion is taken away is motion of things themselves. Pixels whose
     leftover is bigger than `motion` are "moving"; blobs of them become detections.
  5. A small tracker links detections frame to frame, so each mover keeps an id and a speed, and a blob has to
     be seen CONFIRM times before it counts (stray flickers never do).

Limits worth knowing:
  * The fit is one shift/turn/zoom for the whole picture. Things at very different depths move differently when
    the dog walks forward, so close static objects can be flagged as moving at times; the fit is skipped (and
    the reason reported) when it explains too little of the picture.
  * Speeds are in picture widths per second, not metres: a flow has no scale without depth.
  * Frame times are when frames ARRIVED, so Wi-Fi jitter makes the speeds noisy.
This module only needs numpy; OpenCV (cv2) is imported when an engine is made, so importing it never fails.
"""
import math
from collections import deque

import numpy as np

FLOW_WIDTH = 320      # px: frames are shrunk to this width for the flow (smaller = faster, coarser)
FLOW_GAP = 0.10       # s: flow is measured against the newest frame at least this old
FLOW_PRESET = "medium"   # DIS preset: "ultrafast", "fast" or "medium" (~1, ~3, ~8 ms per frame at 320 px)
MOTION = 1.5          # px (at FLOW_WIDTH, per FLOW_GAP): leftover motion above this counts as moving
FIT_TOL = 1.0         # px: how far a pixel's flow may be from the camera-motion fit and still count as agreeing
MIN_INLIERS = 0.45    # the fit must explain at least this share of the picture, or no movers are reported
MAX_EGO = 0.10        # camera motion (picture widths per FLOW_GAP) above this is too fast to trust (turning, blur)
MIN_AREA = 0.0015     # smallest mover, as a share of the picture area
MAX_BLOBS = 8         # most movers looked at per frame (the biggest)
CONFIRM = 3           # a mover must be seen in this many frames before it is shown as a track
LOST = 0.6            # s a track survives without being seen
MAX_GAP = 0.6         # s: frames further apart than this (a stall) aren't compared
STEP = 8              # px between the points the camera-motion fit looks at


def iou_gate(a, b, width):
    """True when two (x0, y0, x1, y1) boxes are close enough to be the same thing."""
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1]) < 0.6 * size + 0.04 * width


class Track:
    def __init__(self, ident, box, vel, now):
        self.id = ident
        self.box = list(box)          # x0, y0, x1, y1 in flow pixels, smoothed
        self.vel = list(vel)          # picture widths per second (x, y), smoothed
        self.hits = 1
        self.first_seen = self.last_seen = now

    @property
    def confirmed(self):
        return self.hits >= CONFIRM

    def update(self, box, vel, now):
        k = 0.5
        self.box = [(1 - k) * o + k * n for o, n in zip(self.box, box)]
        self.vel = [(1 - k) * o + k * n for o, n in zip(self.vel, vel)]
        self.hits += 1
        self.last_seen = now

    @property
    def speed(self):
        return math.hypot(*self.vel)


class MotionTracker:
    """Links detections between frames by nearest centre; unmatched detections start new tracks."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def clear(self):
        self.tracks, self.next_id = [], 1

    def update(self, dets, now, width):
        """dets: list of (box, vel). Returns the tracks that are still alive."""
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, (box, _) in enumerate(dets):
                if iou_gate(tr.box, box, width):
                    ca = ((tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2)
                    cb = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    pairs.append((math.hypot(ca[0] - cb[0], ca[1] - cb[1]), ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ti].update(dets[di][0], dets[di][1], now)
        for di, (box, vel) in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, box, vel, now))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if now - t.last_seen <= LOST]
        return self.tracks


class FlowEngine:
    """Feed it frames (RGB uint8 arrays, with the time each arrived) and read back what moved."""

    def __init__(self, width=FLOW_WIDTH, preset=FLOW_PRESET, motion=MOTION, view=0):
        import cv2                       # raises ImportError if OpenCV isn't in this environment
        self.cv2 = cv2
        presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                   "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        self.dis = cv2.DISOpticalFlow_create(presets[preset])
        self.width = width
        self.motion = motion             # may be changed while running
        self.view = view                 # 0 tracks, 1 flow colour, 2 flow arrows: decides which picture is prepared
        self.fit_tol = FIT_TOL
        self.history = deque()           # (stamp, grey frame)
        self.tracker = MotionTracker()
        self._grid = None

    def reset(self):
        self.history.clear()
        self.tracker.clear()

    def _grid_for(self, h, w):
        if self._grid is None or self._grid[0] != (h, w):
            ys, xs = np.mgrid[STEP // 2:h:STEP, STEP // 2:w:STEP]
            X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid = ((h, w), ys.ravel(), xs.ravel(), X, Y)
        return self._grid

    def process(self, rgb, stamp):
        """One frame in. Returns a result dict, or None while there is nothing to compare it with yet."""
        cv2 = self.cv2
        h0, w0 = rgb.shape[:2]
        w = self.width
        h = max(8, round(w * h0 / w0))
        grey = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        if self.history and stamp - self.history[-1][0] > MAX_GAP:
            self.reset()                                  # a stall: the old frames no longer relate to this one
        ref = None
        for old_stamp, old in reversed(self.history):
            if stamp - old_stamp >= FLOW_GAP:
                ref = (old_stamp, old)
                break
        self.history.append((stamp, grey))
        while len(self.history) > 2 and stamp - self.history[1][0] >= FLOW_GAP:
            self.history.popleft()                        # keep just the reference frame and newer ones
        if ref is None:
            return None
        gap = stamp - ref[0]
        flow = self.dis.calc(ref[1], grey, None)
        res = {"stamp": stamp, "size": (w, h), "gap": gap, "flow": flow, "tracks": [], "mask": None,
               "ok": False, "note": "", "inliers": 0.0, "ego": (0.0, 0.0), "viz": None, "arrows": None}

        # 3. the camera's own motion: shift + turn + zoom, fitted robustly to a grid of flow vectors
        (_, ys, xs, X, Y) = self._grid_for(h, w)
        src = np.stack([xs, ys], axis=1).astype(np.float32)
        dst = src + flow[ys, xs]
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.fit_tol,
                                             maxIters=300, confidence=0.99)
        if A is None:
            res["note"] = "could not fit the camera's motion"
            self.tracker.update([], stamp, w)
            return self._finish(res, None)
        res["inliers"] = float(inl.mean())
        pred = np.dstack([A[0, 0] * X + A[0, 1] * Y + A[0, 2] - X, A[1, 0] * X + A[1, 1] * Y + A[1, 2] - Y])
        ego = (float(A[0, 2] + (A[0, 0] - 1) * w / 2 + A[0, 1] * h / 2),      # shift of the picture centre
               float(A[1, 2] + A[1, 0] * w / 2 + (A[1, 1] - 1) * h / 2))
        res["ego"] = ego
        if res["inliers"] < MIN_INLIERS:
            res["note"] = f"camera motion not clear ({res['inliers'] * 100:.0f}% fits)"
            self.tracker.update([], stamp, w)
            return self._finish(res, pred)
        if math.hypot(*ego) / w * (FLOW_GAP / gap) > MAX_EGO:
            res["note"] = "camera moving too fast to tell"
            self.tracker.update([], stamp, w)
            return self._finish(res, pred)

        # 4. what is left over is things moving by themselves
        left = flow - pred
        scale = FLOW_GAP / gap                            # express every flow as if the frames were FLOW_GAP apart
        mag = cv2.blur(np.hypot(left[..., 0], left[..., 1]) * scale, (5, 5))
        mask = (mag > self.motion).astype(np.uint8)
        mask[:4, :] = mask[-4:, :] = 0
        mask[:, :4] = mask[:, -4:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:MAX_BLOBS]
        dets = []
        for i in blobs:
            x, y, bw, bh, area = stats[i]
            if area < MIN_AREA * w * h:
                continue
            inside = labels[y:y + bh, x:x + bw] == i
            vec = left[y:y + bh, x:x + bw][inside]                   # leftover flow of the blob's pixels
            weight = np.hypot(vec[:, 0], vec[:, 1]) ** 2             # the strongly moving core counts most; the
            weight /= max(weight.sum(), 1e-9)                        # fringe (blur, the object's wake) hardly
            vx, vy = (vec * weight[:, None]).sum(axis=0) * scale / FLOW_GAP / w   # widths per second
            dets.append(((float(x), float(y), float(x + bw), float(y + bh)), (vx, vy)))
        res["ok"], res["mask"] = True, mask
        self.tracker.update(dets, stamp, w)
        return self._finish(res, pred, mag)

    def _finish(self, res, pred, mag=None):
        cv2 = self.cv2
        res["tracks"] = [{"id": t.id, "box": tuple(t.box), "vel": tuple(t.vel), "speed": t.speed,
                          "confirmed": t.confirmed, "age": t.last_seen - t.first_seen}
                         for t in self.tracker.tracks if t.last_seen == res["stamp"]]
        flow = res["flow"]
        w, h = res["size"]
        if self.view == 1:                                # flow as colour: direction = hue, speed = brightness
            ang = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi)
            speed = np.clip(np.hypot(flow[..., 0], flow[..., 1]) / 6.0, 0, 1)
            hsv = np.dstack([(ang * 179).astype(np.uint8), np.full((h, w), 255, np.uint8), (speed * 255).astype(np.uint8)])
            res["viz"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif self.view == 2 and pred is not None:         # arrows on a grid; red where the flow isn't the camera's
            ys, xs = np.mgrid[8:h - 4:16, 8:w - 4:16]
            f = flow[ys, xs]
            leftover = np.hypot(*(f - pred[ys, xs]).transpose(2, 0, 1)) * (FLOW_GAP / res["gap"])
            res["arrows"] = np.column_stack([xs.ravel(), ys.ravel(), f[..., 0].ravel(), f[..., 1].ravel(),
                                             (leftover > self.motion).ravel()]).astype(np.float32)
        res["flow"] = None                                 # not needed by the caller; saves copying
        return res

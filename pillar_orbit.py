"""Pillar picking, circle-around-it control and scan accumulation for lidar_control.py.

Everything works in the odom (world) frame that the deskewed lidar cloud and
rt/utlidar/robot_pose share. The orbit controller strafes sideways around the
pillar while turning to keep facing it, and every STEP_ANGLE of travel stops
for a nose-up tilt so the downward-facing lidar also sees the upper part.
"""
import math

import numpy as np

PILLAR_Z = (0.25, 1.8)        # m (odom z); points in this band count as pillar / obstacle
PILLAR_MAX_WIDTH = 1.2        # m; anything wider looks like a wall, not a pillar
CELL = 0.1                    # m grid for clustering points into objects
SEARCH_CONE = math.radians(30)  # C picks the nearest object within this angle of straight ahead
SEARCH_RANGE = (0.4, 4.0)     # m

ORBIT_SPEED = 0.2             # m/s sideways around the pillar
ORBIT_RADIUS = (0.9, 3.0)     # m from pillar centre; the starting distance is clamped to this
K_RADIAL = 0.8                # forward/back correction per metre of radius error
K_YAW = 1.5                   # turn rate per radian of facing error
MAX_VX, MAX_VYAW = 0.25, 0.8
FULL_TURN = math.radians(370)  # a little past one lap so the scan closes up
STEP_ANGLE = math.radians(30)  # tilt-scan this often (degrees of orbit)
TILT_HOLD = 1.5               # s held at full tilt for each tilt scan
CLEARANCE = 0.5               # m; stop if anything except the pillar is this close to the robot

SCAN_VOXEL = 0.03             # m; the accumulated scan keeps one point per voxel


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Pillar:
    def __init__(self, center, radius, n_points):
        self.center = np.asarray(center, float)
        self.radius = radius
        self.n_points = n_points


def _band(points):
    z = points[:, 2]
    return points[(z >= PILLAR_Z[0]) & (z <= PILLAR_Z[1]), :2]


def _fit_circle(xy):
    """Algebraic least-squares circle fit; returns (centre, radius) or None."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre = sol[:2]
    r2 = sol[2] + centre @ centre
    return (centre, math.sqrt(r2)) if r2 > 0 else None


def _object_at(xy, seed):
    """Points of the connected object (8-connected grid cells) containing the seed cell."""
    cells = np.floor(xy / CELL).astype(np.int64)
    occupied = {}
    for i, key in enumerate(map(tuple, cells)):
        occupied.setdefault(key, []).append(i)
    start = tuple(np.floor(np.asarray(seed) / CELL).astype(np.int64))
    if start not in occupied:  # snap to the nearest occupied cell within 0.3 m
        near = [k for k in occupied if abs(k[0] - start[0]) <= 3 and abs(k[1] - start[1]) <= 3]
        if not near:
            return None
        start = min(near, key=lambda k: (k[0] - start[0]) ** 2 + (k[1] - start[1]) ** 2)
    seen, todo = {start}, [start]
    limit = int(PILLAR_MAX_WIDTH / CELL) + 2  # stop growing far beyond pillar size
    while todo:
        cx, cy = todo.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (cx + dx, cy + dy)
                if k in occupied and k not in seen and max(abs(k[0] - start[0]), abs(k[1] - start[1])) <= limit:
                    seen.add(k)
                    todo.append(k)
    idx = [i for k in seen for i in occupied[k]]
    return xy[idx]


def find_pillar(points, robot_xy, yaw, click_xy=None):
    """Find a pillar at the clicked point, or the nearest object straight ahead.

    Returns (Pillar, None) or (None, reason).
    """
    if points is None or not len(points):
        return None, "no lidar points"
    xy = _band(points)
    if len(xy) < 10:
        return None, "no points at pillar height"
    robot_xy = np.asarray(robot_xy, float)

    if click_xy is None:
        rel = xy - robot_xy
        dist = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.abs((np.arctan2(rel[:, 1], rel[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        ahead = (bearing < SEARCH_CONE) & (dist > SEARCH_RANGE[0]) & (dist < SEARCH_RANGE[1])
        if ahead.sum() < 5:
            return None, "nothing ahead"
        # Nearest cell with a few points in it, so single stray points don't win
        cand = xy[ahead][np.argsort(dist[ahead])]
        cells = np.floor(cand / CELL).astype(np.int64)
        _, first, counts = np.unique(cells, axis=0, return_index=True, return_counts=True)
        good = first[counts >= 3]
        if not len(good):
            return None, "nothing solid ahead"
        seed = cand[good.min()]
    else:
        seed = np.asarray(click_xy, float)

    obj = _object_at(xy, seed)
    if obj is None or len(obj) < 8:
        return None, "no object there"
    extent = obj.max(0) - obj.min(0)
    if max(extent) > PILLAR_MAX_WIDTH:
        return None, f"object is {max(extent):.1f} m wide - looks like a wall"

    # Round pillar: circle fit works from one side. Otherwise (flat face, poor
    # fit) push the visible surface's centroid away from the robot by half its width.
    centroid = obj.mean(0)
    view_dir = (centroid - robot_xy) / max(np.linalg.norm(centroid - robot_xy), 1e-6)
    side = np.array([-view_dir[1], view_dir[0]])
    half_width = (np.ptp(obj @ side)) / 2
    fit = _fit_circle(obj)
    if fit is not None:
        c, r = fit
        if 0.03 < r < PILLAR_MAX_WIDTH / 2 and (c - centroid) @ view_dir > 0 \
                and np.median(np.abs(np.linalg.norm(obj - c, axis=1) - r)) < 0.03:
            return Pillar(c, r, len(obj)), None
    return Pillar(centroid + view_dir * half_width, max(half_width, 0.05), len(obj)), None


def clearance_ok(points, robot_xy, pillar):
    """False if anything other than the pillar is within CLEARANCE of the robot."""
    if points is None or not len(points):
        return True
    xy = _band(points)
    near = np.hypot(*(xy - robot_xy).T) < CLEARANCE
    not_pillar = np.hypot(*(xy - pillar.center).T) > pillar.radius + 0.3
    return not np.any(near & not_pillar)


class Orbit:
    """Circle the pillar facing it: sideways at ORBIT_SPEED, radius and heading held by feedback.

    step() returns (vx, vy, vyaw, lean_target): lean_target is a nose-up pitch
    to hold during a tilt scan, 0.0 to come back level, or None for no lean.
    """

    def __init__(self, pillar, robot_xy, direction, tilt_scans, scan_tilt):
        self.pillar = pillar
        self.direction = direction            # +1 strafe left (clockwise from above), -1 right
        self.tilt_scans = tilt_scans
        self.scan_tilt = scan_tilt
        rel = np.asarray(robot_xy, float) - pillar.center
        self.radius = min(max(float(np.hypot(*rel)), ORBIT_RADIUS[0]), ORBIT_RADIUS[1])
        self.last_angle = math.atan2(rel[1], rel[0])
        self.travelled = 0.0
        self.since_tilt = STEP_ANGLE if tilt_scans else 0.0  # start with a tilt scan
        self.phase = "move"
        self.phase_start = 0.0
        self.done = False

    def step(self, now, robot_xy, yaw, lean_pitch):
        rel = np.asarray(robot_xy, float) - self.pillar.center
        dist = float(np.hypot(*rel))
        angle = math.atan2(rel[1], rel[0])
        moved = abs(wrap(angle - self.last_angle))
        self.last_angle = angle

        if self.phase == "tilt":
            if now - self.phase_start < TILT_HOLD:
                return 0.0, 0.0, 0.0, -self.scan_tilt
            if lean_pitch != 0.0:
                return 0.0, 0.0, 0.0, 0.0          # ramping back level
            self.phase = "move"
            return 0.0, 0.0, 0.0, None             # level: leave pose mode, then keep moving

        self.travelled += moved
        self.since_tilt += moved
        if self.travelled >= FULL_TURN:
            self.done = True
            return 0.0, 0.0, 0.0, None
        if self.tilt_scans and self.since_tilt >= STEP_ANGLE:
            self.since_tilt = 0.0
            self.phase, self.phase_start = "tilt", now
            return 0.0, 0.0, 0.0, -self.scan_tilt

        face_err = wrap(math.atan2(-rel[1], -rel[0]) - yaw)
        # Sideways speed fades out if we're not facing the pillar, so we turn first
        vy = self.direction * ORBIT_SPEED * max(0.0, math.cos(face_err)) ** 2
        vx = max(-MAX_VX, min(MAX_VX, K_RADIAL * (dist - self.radius)))
        # Strafing left around a pillar ahead swings the bearing clockwise: feed forward -vy/R
        vyaw = max(-MAX_VYAW, min(MAX_VYAW, K_YAW * face_err - vy / self.radius))
        return vx, vy, vyaw, None

    @property
    def progress(self):
        return min(1.0, self.travelled / FULL_TURN)


class ScanCloud:
    """All points seen during an orbit, thinned to one per SCAN_VOXEL voxel."""

    def __init__(self):
        self.points = np.empty((0, 4), np.float32)
        self.pending = []

    def add(self, points):
        if points is not None and len(points):
            self.pending.append(points)
        if len(self.pending) >= 10:
            self.compact()

    def compact(self):
        if not self.pending:
            return
        pts = np.concatenate([self.points] + self.pending)
        self.pending = []
        k = np.floor(pts[:, :3] / SCAN_VOXEL).astype(np.int64) + (1 << 20)  # voxel index, made positive
        keys = k[:, 0] | (k[:, 1] << 21) | (k[:, 2] << 42)                  # one int per voxel (+-30 km)
        _, keep = np.unique(keys, return_index=True)
        self.points = pts[np.sort(keep)]

    def all(self):
        self.compact()
        return self.points

    def view(self):
        """Everything so far, without thinning the newest frames (cheap, for drawing)."""
        return np.concatenate([self.points] + self.pending) if self.pending else self.points

    def around(self, pillar, margin=0.5):
        pts = self.all()
        return pts[np.hypot(*(pts[:, :2] - pillar.center).T) < pillar.radius + margin]

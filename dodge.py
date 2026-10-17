"""Move out of the way of things moving towards the robot (for lidar_control.py).

Tracker  groups the last few lidar frames' body-height points near the robot
         into objects, follows them from frame to frame and fits each one's
         velocity over ~0.8 s (one frame is too noisy: the lidar samples a
         different part of an object every frame).
Dodger   flags an object as a threat if its predicted path brings it within
         DANGER of the robot soon, picks the escape direction (of 16) that
         keeps the most distance from that path without running into known
         obstacles, and returns body-frame velocities to get there while
         turning to keep the threat in the (forward-looking) lidar's view.
"""
import math

import numpy as np

import navigation as nav

TRACK_Z = (0.15, 1.9)        # m above the floor
TRACK_RANGE = 4.0            # m from the robot
CELL = 0.15                  # m clustering grid
MIN_POINTS = 8
MAX_WIDTH = 1.2              # m; bigger clusters (walls, furniture runs) aren't tracked
GATE = 0.4                   # m; a cluster this close to a track's predicted position continues it
HISTORY = 0.8                # s of positions used for the velocity fit
MIN_SAMPLES = 6
MIN_AGE = 0.7                # s a track must exist before it can count as moving
MAX_RESIDUAL = 0.08          # m RMS off the fitted straight line; jumpier tracks are noise
MIN_TRAVEL = 0.25            # m covered over the fit window
MAX_SPEED = 2.5              # m/s; faster is a tracking glitch
DROP_AFTER = 0.6             # s unseen -> track dropped

MIN_SPEED = 0.25             # m/s; slower things are treated as still
DANGER = 1.0                 # m; predicted closest approach below this is a threat
HORIZON = 3.0                # s to look ahead
CALM_TIME = 1.0              # s without a threat before the dodge ends
MAX_DODGE = 5.0              # s; a single dodge never lasts longer than this

SPEED = 0.35                 # m/s escape speed
MAX_SIDE, MAX_BACK = 0.3, 0.3
MAX_VYAW, K_YAW = 0.6, 1.2
CHECK_DIST = 0.8             # m along each escape direction that must be free


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    _next_id = 1

    def __init__(self, pos, now):
        self.id = Track._next_id
        Track._next_id += 1
        self.history = [(now, pos)]
        self.age = 0.0
        self.pos = pos
        self.vel = np.zeros(2)
        self.last_seen = now
        self.born = now
        self.steady = False   # last fit was a clean straight line

    def add(self, pos, now):
        self.history.append((now, pos))
        self.history = [(t, p) for t, p in self.history if now - t <= HISTORY]
        self.pos, self.last_seen = pos, now
        self.steady = False
        if len(self.history) >= MIN_SAMPLES:
            t = np.array([h[0] for h in self.history])
            p = np.array([h[1] for h in self.history], float)
            t = t - t.mean()
            self.vel = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()  # least-squares slope
            residual = p - p.mean(0) - t[:, None] * self.vel
            rms = math.sqrt((residual ** 2).sum(1).mean())
            travel = np.hypot(*self.vel) * (t.max() - t.min())
            self.steady = rms <= MAX_RESIDUAL and travel >= MIN_TRAVEL
        else:
            self.vel = np.zeros(2)
        self.age = now - self.born

    @property
    def moving(self):
        speed = np.hypot(*self.vel)
        return self.steady and self.age >= MIN_AGE and MIN_SPEED <= speed <= MAX_SPEED


def _clusters(xy):
    """Centroids of 8-connected grid clusters that are object-sized."""
    cells = np.floor(xy / CELL).astype(np.int64)
    groups = {}
    for i, key in enumerate(map(tuple, cells)):
        groups.setdefault(key, []).append(i)
    seen, out = set(), []
    for start in groups:
        if start in seen:
            continue
        seen.add(start)
        todo, members = [start], []
        while todo:
            c = todo.pop()
            members += groups[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in groups and n not in seen:
                        seen.add(n)
                        todo.append(n)
        pts = xy[members]
        if len(pts) >= MIN_POINTS and max(pts.max(0) - pts.min(0)) <= MAX_WIDTH:
            out.append(pts.mean(0))
    return out


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, points, robot_xy, floor_z, now):
        """points: the last ~0.25 s of lidar (odom frame). Call once per new lidar frame."""
        self.tracks = [t for t in self.tracks if now - t.last_seen <= DROP_AFTER]
        if points is None or not len(points):
            return
        z = points[:, 2] - floor_z
        xy = points[(z >= TRACK_Z[0]) & (z <= TRACK_Z[1]), :2]
        d = np.hypot(*(xy - robot_xy).T)
        xy = xy[(d > nav.SELF_RADIUS) & (d < TRACK_RANGE)]
        centroids = _clusters(xy) if len(xy) else []

        # Greedy nearest matching against each track's predicted position
        free = list(range(len(centroids)))
        for tr in sorted(self.tracks, key=lambda t: -len(t.history)):
            if not free:
                break
            pred = tr.pos + tr.vel * (now - tr.last_seen)
            k = min(free, key=lambda i: np.hypot(*(centroids[i] - pred)))
            if np.hypot(*(centroids[k] - pred)) <= GATE:
                tr.add(centroids[k], now)
                free.remove(k)
        self.tracks += [Track(centroids[i], now) for i in free]


def closest_approach(robot_xy, obj_pos, obj_vel):
    """(distance, time) of the object's closest approach to a still robot within HORIZON."""
    rel = np.asarray(robot_xy) - obj_pos
    v2 = obj_vel @ obj_vel
    t = 0.0 if v2 < 1e-9 else min(max((rel @ obj_vel) / v2, 0.0), HORIZON)
    return float(np.hypot(*(rel - obj_vel * t))), t


class Dodger:
    def __init__(self):
        self.tracker = Tracker()
        self.active = False
        self.started = 0.0
        self.calm_since = None
        self.threat = None   # the Track being avoided
        self.direction = None  # escape direction chosen last step (world unit vector)

    def threats(self, robot_xy):
        out = []
        for tr in self.tracker.tracks:
            if tr.moving:
                d, t = closest_approach(robot_xy, tr.pos, tr.vel)
                if d < DANGER and t > 0:  # t == 0: already past / moving away
                    out.append(tr)
        return out

    def step(self, now, robot_xy, yaw, occ, recent_points, floor_z):
        """Returns (vx, vy, vyaw) while dodging, or None when there's nothing to dodge."""
        robot_xy = np.asarray(robot_xy, float)
        threats = self.threats(robot_xy)
        if threats:
            self.threat = min(threats, key=lambda t: np.hypot(*(t.pos - robot_xy)))
            self.calm_since = None
            if not self.active:
                self.active, self.started, self.direction = True, now, None
        elif self.active:
            if self.calm_since is None:
                self.calm_since = now
            if now - self.calm_since >= CALM_TIME:
                self.active, self.threat = False, None
        if self.active and now - self.started > MAX_DODGE:
            self.active, self.threat = False, None
        if not self.active:
            return None
        if not threats:
            return 0.0, 0.0, 0.0  # threat gone: wait still for CALM_TIME before handing back

        tr = self.threat
        best, best_score = None, -math.inf
        for a in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            u = np.array([math.cos(a), math.sin(a)])
            if not self._free(occ, recent_points, floor_z, robot_xy, u):
                continue
            # Closest the object gets to us over the horizon if we walk this way (at most 1.2 m)
            ts = np.linspace(0, HORIZON, 16)
            ours = robot_xy + u * np.minimum(SPEED * ts, 1.2)[:, None]
            theirs = tr.pos + tr.vel * ts[:, None]
            score = np.hypot(*(ours - theirs).T).min()
            if abs(wrap(a - yaw)) > math.radians(120):
                score -= 0.2  # mild preference against walking backwards blind
            if self.direction is not None and u @ self.direction > 0.7:
                score += 0.15  # stick with the side already chosen instead of dithering
            if score > best_score:
                best, best_score = u, score
        bearing = math.atan2(*(tr.pos - robot_xy)[::-1])
        vyaw = max(-MAX_VYAW, min(MAX_VYAW, K_YAW * wrap(bearing - yaw)))
        self.direction = best
        if best is None:
            return 0.0, 0.0, vyaw  # boxed in: just face it
        fwd = best @ (math.cos(yaw), math.sin(yaw))
        left = best @ (-math.sin(yaw), math.cos(yaw))
        vx = max(-MAX_BACK, min(SPEED, SPEED * fwd))
        vy = max(-MAX_SIDE, min(MAX_SIDE, SPEED * left))
        return vx, vy, vyaw

    def _free(self, occ, recent_points, floor_z, robot_xy, u):
        """Is CHECK_DIST along u clear of mapped obstacles and of fresh lidar points?"""
        r = int(round(nav.ROBOT_RADIUS / nav.CELL))
        for s in (0.2, 0.4, 0.6, CHECK_DIST):
            i, j = nav.ObstacleMap.to_cell(robot_xy + u * s)
            if occ[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].any():
                return False
        if recent_points is not None and len(recent_points):
            z = recent_points[:, 2] - floor_z
            rel = recent_points[(z >= nav.OBST_Z[0]) & (z <= nav.OBST_Z[1]), :2] - robot_xy
            along = rel @ u
            across = np.abs(rel @ (-u[1], u[0]))
            if np.any((along > 0.1) & (along < CHECK_DIST) & (across < nav.ROBOT_RADIUS)):
                return False
        return True

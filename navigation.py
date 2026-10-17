"""Waypoint navigation with obstacle avoidance for lidar_control.py.

ObstacleMap  remembers where lidar points at body height have been seen
             repeatedly (odom frame, 10 cm grid). Heights are measured from
             the floor under the robot, so odom z drift doesn't matter.
             Obstacles are erased again when lidar beams pass through them
             (something that moved away, or a false obstacle).
plan_path    A* on that map, with obstacles padded by the robot's size and
             an extra cost for passing close to them; the result is shortened
             by cutting corners wherever there is a clear straight line.
Navigator    walks the robot along the plan facing forward (so the lidar,
             which looks forward, sees what's ahead), replanning every second
             and stopping for anything right in front of it.
"""
import heapq
import math

import numpy as np

CELL = 0.1                    # m
MAP_SIZE = 800                # cells per side (80 m square around the odom origin)
OBST_Z = (0.15, 0.7)          # m above the floor: points in this band block the robot
SELF_RADIUS = 0.25            # m; points this close to the robot centre aren't mapped (own body)
HIT_DECAY = 2.0               # s; each cell's hit score decays with this time constant...
MIN_SCORE = 3.0               # ...and the cell becomes an obstacle once the score reaches this,
                              # so stray noise points (single hits) never count
MAP_MEMORY = 120.0            # s; obstacles not confirmed again for this long are forgotten

# Seen-through clearing: each lidar point means the beam from the lidar to it
# crossed open space. Where that beam is at body height and crosses an
# obstacle cell, the cell gets a "miss"; CLEAR_MISSES misses (decaying like
# hits) erase it, unless the cell itself was hit within HIT_GUARD seconds
# (a thin pole can share a cell with beams passing beside it).
LIDAR_OFFSET = (0.28, 0.06)   # m: lidar position ahead of / above the robot's base (head front)
CLEAR_MISSES = 3.0
HIT_GUARD = 1.0               # s
CLEAR_RANGE = 6.0             # m; longer beams are too sparse to trust for clearing
END_GAP = 0.15                # m before each point that isn't counted (the point may be the obstacle)
RAY_STEP = 0.05               # m between samples along each beam
MAX_RAYS = 1000               # per frame; more are thinned out

ROBOT_RADIUS = 0.35           # m; the robot centre must stay this far from obstacles
SOFT_RADIUS = 0.7             # m; extra cost for being closer than this
SOFT_WEIGHT = 2.0
PLAN_MARGIN = 4.0             # m around start/goal that the planner searches

SPEED = 0.35                  # m/s forward
MAX_VYAW = 0.8                # rad/s
K_YAW = 1.5
LOOKAHEAD = 0.6               # m along the path to steer towards
TURN_IN_PLACE = math.radians(50)  # turn on the spot if the path is further off than this
REACHED = 0.3                 # m from a waypoint counts as reached
REPLAN_EVERY = 1.0            # s
STOP_BOX = (0.25, 0.6, 0.25)  # m: anything at x in [0.25, 0.6] ahead, |y| < 0.25 -> no forward motion
STUCK_AFTER = 10.0            # s without forward motion -> give up


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _disk(r_cells):
    return [(dx, dy) for dx in range(-r_cells, r_cells + 1) for dy in range(-r_cells, r_cells + 1)
            if dx * dx + dy * dy <= r_cells * r_cells]


def _dilate(mask, offsets):
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dx, dy in offsets:
        out[max(dx, 0):h + min(dx, 0), max(dy, 0):w + min(dy, 0)] |= \
            mask[max(-dx, 0):h + min(-dx, 0), max(-dy, 0):w + min(-dy, 0)]
    return out


class ObstacleMap:
    def __init__(self):
        self.clear()

    def clear(self):
        self.score = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)
        self.misses = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)
        self.confirmed = np.full((MAP_SIZE, MAP_SIZE), -1e9)
        self.last_hit = np.full((MAP_SIZE, MAP_SIZE), -1e9)
        self.last_add = None
        self.erased = 0

    @staticmethod
    def to_cell(xy):
        return np.floor(np.asarray(xy) / CELL).astype(np.int64) + MAP_SIZE // 2

    @staticmethod
    def to_world(ij):
        return (np.asarray(ij) - MAP_SIZE // 2 + 0.5) * CELL

    def add(self, points, robot_xy, yaw, robot_z, floor_z, now):
        """Add one lidar frame (odom-frame points) seen from the given robot pose."""
        if points is None or not len(points):
            return
        robot_xy = np.asarray(robot_xy, float)
        if self.last_add is not None:
            decay = math.exp(-max(now - self.last_add, 0.0) / HIT_DECAY)
            self.score *= decay
            self.misses *= decay
        self.last_add = now
        z = points[:, 2] - floor_z
        xy = points[(z >= OBST_Z[0]) & (z <= OBST_Z[1]), :2]
        xy = xy[np.hypot(*(xy - robot_xy).T) > SELF_RADIUS]
        ij = self.to_cell(xy)
        ok = np.all((ij >= 0) & (ij < MAP_SIZE), axis=1)
        ij = ij[ok]
        np.add.at(self.score, (ij[:, 0], ij[:, 1]), 1.0)
        self.last_hit[ij[:, 0], ij[:, 1]] = now
        hit = ij[self.score[ij[:, 0], ij[:, 1]] >= MIN_SCORE]
        self.confirmed[hit[:, 0], hit[:, 1]] = now

        origin = np.array([robot_xy[0] + LIDAR_OFFSET[0] * math.cos(yaw),
                           robot_xy[1] + LIDAR_OFFSET[0] * math.sin(yaw),
                           robot_z + LIDAR_OFFSET[1]])
        self._clear_seen_through(points[:, :3], origin, floor_z, now)

    def _clear_seen_through(self, points, origin, floor_z, now):
        rel = points - origin
        dist = np.linalg.norm(rel, axis=1)
        keep = (dist > END_GAP + RAY_STEP) & (dist < CLEAR_RANGE)
        rel, dist = rel[keep], dist[keep]
        if len(rel) > MAX_RAYS:
            pick = np.linspace(0, len(rel) - 1, MAX_RAYS).astype(int)
            rel, dist = rel[pick], dist[pick]
        if not len(rel):
            return
        along = np.arange(RAY_STEP, CLEAR_RANGE, RAY_STEP)           # S distances along a beam
        valid = along[None, :] < (dist - END_GAP)[:, None]           # R x S
        samples = origin + (along[None, :, None] / dist[:, None, None]) * rel[:, None, :]  # R x S x 3
        height = samples[..., 2] - floor_z
        valid &= (height >= OBST_Z[0]) & (height <= OBST_Z[1])
        cells = self.to_cell(samples[valid][:, :2])
        cells = cells[np.all((cells >= 0) & (cells < MAP_SIZE), axis=1)]
        keys = np.unique(cells[:, 0] * MAP_SIZE + cells[:, 1])       # each cell counts once per frame
        i, j = keys // MAP_SIZE, keys % MAP_SIZE
        mapped = now - self.confirmed[i, j] < MAP_MEMORY
        i, j = i[mapped], j[mapped]
        self.misses[i, j] += 1.0
        gone = (self.misses[i, j] >= CLEAR_MISSES) & (now - self.last_hit[i, j] > HIT_GUARD)
        self.confirmed[i[gone], j[gone]] = -1e9
        self.misses[i[gone], j[gone]] = 0.0
        self.erased += int(gone.sum())

    def occupied(self, now):
        return now - self.confirmed < MAP_MEMORY

    def obstacle_xy(self, now, center_xy, radius):
        """World centres of obstacle cells within `radius` of a point (for drawing)."""
        c = self.to_cell(center_xy)
        r = int(radius / CELL) + 1
        i0, j0 = np.maximum(c - r, 0)
        i1, j1 = np.minimum(c + r, MAP_SIZE)
        ii, jj = np.nonzero(self.occupied(now)[i0:i1, j0:j1])
        return self.to_world(np.column_stack([ii + i0, jj + j0]))


def plan_path(occ, start_xy, goal_xy):
    """A* from start to goal on the occupancy grid. Returns ([world xy, ...], None) or (None, reason)."""
    s, g = ObstacleMap.to_cell(start_xy), ObstacleMap.to_cell(goal_xy)
    if not (np.all((s >= 0) & (s < MAP_SIZE)) and np.all((g >= 0) & (g < MAP_SIZE))):
        return None, f"outside the {MAP_SIZE * CELL:.0f} m map"
    m = int(PLAN_MARGIN / CELL)
    lo = np.maximum(np.minimum(s, g) - m, 0)
    hi = np.minimum(np.maximum(s, g) + m + 1, MAP_SIZE)
    sub = occ[lo[0]:hi[0], lo[1]:hi[1]]
    hard = _dilate(sub, _disk(int(round(ROBOT_RADIUS / CELL))))
    cost = np.zeros(sub.shape)
    ring = hard
    for _ in range(int(round((SOFT_RADIUS - ROBOT_RADIUS) / CELL))):
        ring = _dilate(ring, _disk(1))
        cost += ring * SOFT_WEIGHT / 4
    s, g = tuple(s - lo), tuple(g - lo)

    # Let the robot leave an obstacle's padding it's already standing in
    for dx, dy in _disk(int(round(ROBOT_RADIUS / CELL))):
        i, j = s[0] + dx, s[1] + dy
        if 0 <= i < hard.shape[0] and 0 <= j < hard.shape[1] and not sub[i, j]:
            hard[i, j] = False
    if hard[g]:  # waypoint in an obstacle's padding: use the nearest free cell within 0.5 m
        free = [(g[0] + dx, g[1] + dy) for dx, dy in _disk(5)
                if 0 <= g[0] + dx < hard.shape[0] and 0 <= g[1] + dy < hard.shape[1]
                and not hard[g[0] + dx, g[1] + dy]]
        if not free:
            return None, "waypoint is inside an obstacle"
        g = min(free, key=lambda c: (c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2)

    def h(c):
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    steps = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    best = {s: 0.0}
    parent = {}
    todo = [(h(s), 0.0, s)]
    while todo:
        _, d, c = heapq.heappop(todo)
        if c == g:
            break
        if d > best.get(c, math.inf):
            continue
        for dx, dy, length in steps:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < hard.shape[0] and 0 <= n[1] < hard.shape[1]) or hard[n]:
                continue
            nd = d + length * (1 + cost[n])
            if nd < best.get(n, math.inf):
                best[n] = nd
                parent[n] = c
                heapq.heappush(todo, (nd + h(n), nd, n))
    else:
        return None, "no path - blocked, or the waypoint is walled in"

    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()

    # Shortcut: keep only the corners needed for clear straight lines
    def clear(a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        ii = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        jj = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        # no padding crossed, and no closer to obstacles than the two ends already are
        return not hard[ii, jj].any() and cost[ii, jj].max() <= max(cost[a], cost[b]) + 1e-9
    kept = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not clear(cells[i], cells[j]):
            j -= 1
        kept.append(cells[j])
        i = j
    return [tuple(ObstacleMap.to_world(np.array(c) + lo)) for c in kept], None


def blocked_ahead(points, robot_xy, yaw, floor_z=0.0):
    """True if lidar points at body height are right in front of the robot."""
    if points is None or not len(points):
        return False
    z = points[:, 2] - floor_z
    rel = points[(z >= OBST_Z[0]) & (z <= OBST_Z[1]), :2] - robot_xy
    fwd = rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw)
    left = -rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw)
    near, far, half = STOP_BOX
    return bool(np.any((fwd > near) & (fwd < far) & (np.abs(left) < half)))


class Navigator:
    """Follows waypoints in order. step() returns (vx, vyaw); check .done / .failed / .note."""

    def __init__(self, waypoints):
        self.waypoints = [np.asarray(w, float) for w in waypoints]
        self.index = 0
        self.path = None
        self.last_plan = -math.inf
        self.done = False
        self.failed = None
        self.note = ""
        self.last_forward = None

    def step(self, now, robot_xy, yaw, occ, recent_points, floor_z=0.0):
        robot_xy = np.asarray(robot_xy, float)
        if self.last_forward is None:
            self.last_forward = now
        goal = self.waypoints[self.index]
        # Reached: at the waypoint, or at the end of the plan (the planner moves
        # waypoints that are inside an obstacle's padding to the nearest free spot)
        end = self.path[-1] if self.path else goal
        if min(np.hypot(*(goal - robot_xy)), np.hypot(*(np.asarray(end) - robot_xy))) < REACHED:
            self.index += 1
            self.path = None
            if self.index == len(self.waypoints):
                self.done = True
                return 0.0, 0.0
            goal = self.waypoints[self.index]

        blocked = blocked_ahead(recent_points, robot_xy, yaw, floor_z)
        if blocked and now - self.last_plan >= 0.3:
            self.path = None  # replan with the new points; the new path should turn away

        if self.path is None or now - self.last_plan >= REPLAN_EVERY:
            self.path, why = plan_path(occ, robot_xy, goal)
            self.last_plan = now
            if self.path is None:
                self.failed = f"waypoint {self.index + 1}: {why}"
                return 0.0, 0.0
            self.note = f"to waypoint {self.index + 1}/{len(self.waypoints)}"

        # Steer at the point LOOKAHEAD metres along the path (pure pursuit)
        path = [robot_xy] + [np.asarray(p) for p in self.path[1:]]
        target, left = path[-1], LOOKAHEAD
        for a, b in zip(path, path[1:]):
            seg = np.hypot(*(b - a))
            if seg >= left:
                target = a + (b - a) * (left / seg)
                break
            left -= seg
        err = wrap(math.atan2(target[1] - robot_xy[1], target[0] - robot_xy[0]) - yaw)
        vyaw = max(-MAX_VYAW, min(MAX_VYAW, K_YAW * err))
        if blocked or abs(err) > TURN_IN_PLACE:
            # Turn on the spot only; give up if that goes on too long
            if blocked:
                self.note = "obstacle right ahead - turning"
            if now - self.last_forward > STUCK_AFTER:
                self.failed = f"waypoint {self.index + 1}: stuck (obstacle ahead)" if blocked else \
                    f"waypoint {self.index + 1}: stuck turning"
                return 0.0, 0.0
            return 0.0, vyaw
        self.last_forward = now
        to_goal = np.hypot(*(np.asarray(self.path[-1]) - robot_xy))
        vx = min(SPEED * math.cos(err) ** 2, 0.15 + to_goal)  # ease off near the waypoint
        return vx, vyaw

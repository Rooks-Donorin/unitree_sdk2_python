"""Drive the Go2 with the keyboard while viewing and capturing its lidar.

Usage:  .venv/bin/python lidar_control.py eth0

The left pane is a live view of the lidar point cloud around the robot,
coloured by height, either top-down or as an orbitable 3D view (V). Each lidar frame only has ~500 points, so the
view overlays the last few seconds of frames; since points are in the odom
(world) frame they stay put as the robot moves. Switching the lidar off (L)
freezes the points on screen; when it comes back on (~10 s spin-up) new
points line up with the old ones, because the robot pose keeps updating from
leg odometry while the lidar is off. The window must have focus for keys to
work.

Driving                          Lidar
  W / S   forward / backward       P       save snapshot (.ply)
  A / D   turn left / right        O       start / stop recording frames
  Q / E   strafe left / right      G       hide / show ground points
  Shift   hold for faster speed    V       top-down / 3D view
  Arrows  lean body (standing)     M       view follows robot heading / fixed
  Space   stop                     + / -   zoom (or mouse wheel)
  R       stand up                 L       lidar on / off
  F       lie down                 [ / ]   less / more persistence
  Esc     stop and quit            3D only: drag with the mouse to orbit

Pillar scan
  Click   pick the pillar under the mouse (top-down view)
  C       pick the nearest object straight ahead
  Z / X   circle the pillar to the left / right, facing it, for one lap
  T       tilt scans on / off (stop every 30 deg and tilt nose-up)
  Bksp    clear the pillar and its scan
  Any drive key, arrow, Space, R or F cancels the lap.

Waypoints
  R-click add a waypoint (top-down view)
  U       remove the last waypoint (Shift+U: remove all)
  Enter   walk the waypoints in order / stop
  K       forget the obstacle map
  The dog plans a path around obstacles (red squares: lidar points 0.15-0.7 m
  above the floor seen repeatedly, remembered for 2 min, erased when the
  lidar sees through them), walks it facing forward, replans every
  second, and turns away from anything right in front of it. Any drive key,
  arrow, Space, R or F stops it.

Moving obstacles
  B       dodge things moving towards the dog on / off (off at start)
  Things the lidar sees moving (orange; red = on a collision course) are
  tracked. While dodging is on and the dog is standing, anything predicted to
  pass within 1 m in the next 3 s makes the dog step out of its way (avoiding
  mapped obstacles) while turning to keep it in view. Waypoints and pillar
  laps pause meanwhile. Holding a drive key overrides the dodge. The lidar
  only looks forward, so things coming from behind aren't seen.

Captures go to ./captures/. Snapshots are PLY files of the points currently
on screen (open in CloudCompare, MeshLab, Open3D). Recordings are a folder of frame_NNNNN.npy arrays
(N x 4 float32: x, y, z, intensity in the odom frame) plus poses.csv with
the robot pose for each frame. A finished pillar lap saves scan_*.ply (every
point seen during the lap) and pillar_*.ply (just the pillar).

Leaning (Up = nose down, Down = nose up, Left/Right = roll) only works standing
still. The dog ignores lean (Euler) commands unless its "pose mode" is on, so
the first arrow press turns pose mode on, and walking turns it off again
(leaning while walking makes the dog jitter and lock up). The lidar faces
mostly downward and only sees up to ~18 degrees above horizontal (straight
ahead), so a nose-up tilt raises the top of its view by about the tilt angle;
the pillar lap's tilt scans use this.
The dog only moves while keys are held. Releasing all keys, or the window
losing focus, sends StopMove.
"""
import math
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import pygame

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

import dodge
import navigation as nav
import pillar_orbit as po

# Speeds (m/s, m/s, rad/s) for normal and Shift-held driving
NORMAL = (0.4, 0.3, 0.8)
FAST = (0.8, 0.5, 1.5)
SEND_HZ = 20
LEAN = 0.5             # rad of pitch/roll while an arrow is held (SDK limit 0.75)
SCAN_TILT = 0.4        # rad nose-up for the pillar lap's tilt scans (standing still)
LEAN_RATE = 1.0        # rad/s, how fast the body tilts toward the target
LEAN_HZ = 10           # the dog drifts back unless Euler is re-sent continuously
POSE_EXIT_DELAY = 1.0  # s level, stopped, with arrows released before leaving pose mode
STANDING_HEIGHT = 0.2  # m; body_height above this counts as standing

CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"  # odom frame, ~12 Hz
POSE_TOPIC = "rt/utlidar/robot_pose"       # robot pose in the same odom frame

VIEW = 720            # lidar pane is VIEW x VIEW pixels
PANEL = 380           # width of the text panel on the right
Z_MIN, Z_MAX = 0.0, 2.0   # height range for the colour map (m, odom frame)
GROUND_Z = 0.12           # points below this are "ground" when hidden
PERSIST_STEPS = [0.0, 1.0, 3.0, 5.0, 10.0, 30.0]  # seconds of lidar-on time to overlay (0 = latest only)
MAX_GAP = 0.5             # gaps longer than this (lidar off) don't count toward persistence
LIDAR_SPINUP = 20.0       # seconds to show "starting up" after switching the lidar on
CAPTURE_DIR = "captures"
FOCAL = (VIEW / 2) / math.tan(math.radians(60) / 2)  # 60 degree field of view
NEAR = 0.2                # points closer to the 3D camera than this are skipped

# Height colour map: blue -> cyan -> green -> yellow -> red
_anchors = np.array([[40, 60, 230], [0, 200, 230], [40, 220, 60], [240, 220, 30], [230, 40, 30]], float)
_t = np.linspace(0, 1, 256) * (len(_anchors) - 1)
_i = np.minimum(_t.astype(int), len(_anchors) - 2)
_f = (_t - _i)[:, None]
COLORMAP = ((1 - _f) * _anchors[_i] + _f * _anchors[_i + 1]).astype(np.uint8)


class LidarFeed:
    """Subscribes to the point cloud and robot pose, keeps recent frames."""

    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque()  # (lidar time, N x 4 float32 x/y/z/intensity in odom frame)
        self.lidar_time = 0.0   # advances only while frames arrive, so time with the lidar off doesn't age points
        self.pose = None        # (x, y, z, qx, qy, qz, qw)
        self.frames = 0
        self.last_frame_time = 0.0
        self.record_queue = None  # set to a Queue while recording

        self.cloud_sub = ChannelSubscriber(CLOUD_TOPIC, PointCloud2_)
        self.cloud_sub.Init(self._on_cloud, 10)
        self.pose_sub = ChannelSubscriber(POSE_TOPIC, PoseStamped_)
        self.pose_sub.Init(self._on_pose, 10)

    def _on_cloud(self, msg: PointCloud2_):
        raw = np.array(msg.data, dtype=np.uint8)
        stride = msg.point_step // 4
        flat = raw.view(np.float32).reshape(-1, stride)
        offsets = {f.name: f.offset // 4 for f in msg.fields}
        cols = [offsets["x"], offsets["y"], offsets["z"], offsets.get("intensity", offsets["z"])]
        points = np.ascontiguousarray(flat[:, cols])
        points = points[np.any(points[:, :3] != 0, axis=1)]  # drop zero padding (most of each message)
        now = time.time()
        with self.lock:
            if self.last_frame_time:
                self.lidar_time += min(now - self.last_frame_time, MAX_GAP)
            self.history.append((self.lidar_time, points))
            while self.lidar_time - self.history[0][0] > PERSIST_STEPS[-1]:
                self.history.popleft()
            self.frames += 1
            self.last_frame_time = now
            pose = self.pose
            rec = self.record_queue
        if rec is not None:
            rec.put((time.time(), points, pose))

    def _on_pose(self, msg: PoseStamped_):
        p, q = msg.pose.position, msg.pose.orientation
        with self.lock:
            self.pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def latest(self, persist):
        """Points from the last `persist` seconds of lidar-on time (at least the newest frame)."""
        with self.lock:
            if not self.history:
                return None, self.pose, self.frames, self.last_frame_time
            newest = self.history[-1][0]
            recent = [p for t, p in self.history if newest - t <= persist] or [self.history[-1][1]]
            return np.concatenate(recent), self.pose, self.frames, self.last_frame_time

    def frames_since(self, count):
        """Frames that arrived after the feed's frame counter was `count`."""
        with self.lock:
            n = min(self.frames - count, len(self.history))
            return [p for _, p in list(self.history)[len(self.history) - n:]] if n > 0 else []


class Recorder:
    """Writes queued lidar frames to disk from the main thread."""

    def __init__(self, feed: LidarFeed):
        self.feed = feed
        self.dir = None
        self.count = 0
        self.poses_file = None

    @property
    def active(self):
        return self.dir is not None

    def start(self):
        self.dir = os.path.join(CAPTURE_DIR, time.strftime("rec_%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.poses_file = open(os.path.join(self.dir, "poses.csv"), "w")
        self.poses_file.write("frame,time,x,y,z,qx,qy,qz,qw\n")
        self.count = 0
        with self.feed.lock:
            self.feed.record_queue = queue.Queue()

    def stop(self):
        with self.feed.lock:
            q, self.feed.record_queue = self.feed.record_queue, None
        self._drain(q)
        self.poses_file.close()
        saved = self.dir
        self.dir = None
        return saved

    def write_pending(self):
        if self.active:
            self._drain(self.feed.record_queue)

    def _drain(self, q):
        while q is not None and not q.empty():
            stamp, points, pose = q.get_nowait()
            np.save(os.path.join(self.dir, f"frame_{self.count:05d}.npy"), points)
            pose_str = ",".join(f"{v:.6f}" for v in pose) if pose else ",,,,,,"
            self.poses_file.write(f"{self.count},{stamp:.6f},{pose_str}\n")
            self.count += 1


class Lean:
    """Body pitch/roll from the arrow keys.

    The dog acknowledges Euler commands but ignores them unless pose mode is
    on (Pose(True)). It can walk in pose mode, but leaning while walking makes
    it jitter and lock up, so the caller only leans while standing still and
    calls leave() before walking.
    """

    def __init__(self, sport: SportClient):
        self.sport = sport
        self.pitch = self.roll = 0.0
        self.active = False       # pose mode on
        self.last_send = 0.0
        self.level_since = None
        self.body_height = 0.0
        self.state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.state_sub.Init(self._on_state, 10)

    def _on_state(self, msg: SportModeState_):
        self.body_height = msg.body_height

    @property
    def standing(self):
        return self.body_height > STANDING_HEIGHT

    def update(self, target_pitch, target_roll, dt, now, moving):
        wants_lean = bool(target_pitch or target_roll)
        if wants_lean and not self.active and self.standing:
            self.sport.Pose(True)
            self.active = True
        if not self.active:
            return

        step = LEAN_RATE * dt
        self.pitch += max(-step, min(step, target_pitch - self.pitch))
        self.roll += max(-step, min(step, target_roll - self.roll))
        if now - self.last_send >= 1.0 / LEAN_HZ:
            self.sport.Euler(self.roll, self.pitch, 0.0)
            self.last_send = now

        # Back level and stopped for a moment: hand back to balance stand
        if not wants_lean and not moving and self.pitch == 0.0 and self.roll == 0.0:
            if self.level_since is None:
                self.level_since = now
            elif now - self.level_since > POSE_EXIT_DELAY:
                self.leave()
        else:
            self.level_since = None

    def leave(self):
        """Level the body and leave pose mode so Move / StandDown work again."""
        if self.active:
            self.sport.Euler(0.0, 0.0, 0.0)
            self.sport.Pose(False)
            if self.standing:
                self.sport.BalanceStand()
        self.active = False
        self.pitch = self.roll = 0.0
        self.level_since = None


def save_ply(points, name="snapshot"):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, time.strftime(f"{name}_%Y%m%d_%H%M%S.ply"))
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nproperty float intensity\n"
        "end_header"
    )
    np.savetxt(path, points, fmt="%.4f", header=header, comments="")
    return path


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def screen_to_world(u, v, pose, scale, robot_up):
    """Inverse of the top-down view mapping: pixel -> odom (x, y)."""
    rx, ry, yaw = (pose[0], pose[1], yaw_from_quat(*pose[3:])) if pose else (0.0, 0.0, 0.0)
    view_yaw = yaw if robot_up else 0.0
    c = VIEW // 2
    fwd, left = (c - v) / scale, (c - u) / scale
    cs, sn = math.cos(view_yaw), math.sin(view_yaw)
    return rx + cs * fwd - sn * left, ry + sn * fwd + cs * left


def draw_cloud(surface, small, points, pose, scale, robot_up, hide_ground, pillar=None, orbit_radius=None,
               nav_overlay=None):
    """Top-down view: robot at centre; in robot-up mode its heading points up.

    nav_overlay: dict with "obstacles" (N x 2 cell centres), "waypoints", "path" (lists of xy).
    """
    surface.fill((12, 12, 16))
    c = VIEW // 2

    # Range rings every metre, labelled every 2 m
    for r in range(1, int(VIEW / 2 / scale) + 2):
        pygame.draw.circle(surface, (45, 45, 55), (c, c), int(r * scale), 1)
        if r % 2 == 0:
            surface.blit(small.render(f"{r}m", True, (90, 90, 100)), (c + 3, c - int(r * scale) - 14))

    rx, ry, rz, yaw = 0.0, 0.0, 0.0, 0.0
    if pose is not None:
        rx, ry, rz = pose[:3]
        yaw = yaw_from_quat(*pose[3:])
    view_yaw = yaw if robot_up else 0.0
    cs, sn = math.cos(view_yaw), math.sin(view_yaw)

    def to_screen(xy):
        dx, dy = float(xy[0]) - rx, float(xy[1]) - ry
        return c - (-sn * dx + cs * dy) * scale, c - (cs * dx + sn * dy) * scale

    if points is not None and len(points):
        pts = points[points[:, 2] >= GROUND_Z] if hide_ground else points
        pts = pts[np.argsort(pts[:, 2])]  # draw tall points last so they stay visible
        dx, dy = pts[:, 0] - rx, pts[:, 1] - ry
        cs, sn = math.cos(view_yaw), math.sin(view_yaw)
        fwd = cs * dx + sn * dy     # screen up
        left = -sn * dx + cs * dy   # screen left
        u = (c - left * scale).astype(np.int32)
        v = (c - fwd * scale).astype(np.int32)
        ok = (u >= 0) & (u < VIEW - 1) & (v >= 0) & (v < VIEW - 1)
        u, v = u[ok], v[ok]
        level = np.clip((pts[ok, 2] - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
        colors = COLORMAP[(level * 255).astype(np.int32)]
        px = pygame.surfarray.pixels3d(surface)
        for du in (0, 1):           # 2x2 pixel dots
            for dv in (0, 1):
                px[u + du, v + dv] = colors
        del px

    # Obstacle map cells (red outlines, on top of the points so they stay visible)
    if nav_overlay is not None and len(nav_overlay["obstacles"]):
        size = max(3, int(nav.CELL * scale))
        for x, y in nav_overlay["obstacles"]:
            u, v = to_screen((x, y))
            pygame.draw.rect(surface, (255, 50, 60), (u - size / 2, v - size / 2, size, size), 1)

    # Picked pillar (magenta) and the lap's path
    if pillar is not None:
        cs, sn = math.cos(view_yaw), math.sin(view_yaw)
        dx, dy = pillar.center[0] - rx, pillar.center[1] - ry
        pu = c - (-sn * dx + cs * dy) * scale
        pv = c - (cs * dx + sn * dy) * scale
        pygame.draw.circle(surface, (255, 60, 255), (pu, pv), max(3, int(pillar.radius * scale)), 2)
        pygame.draw.line(surface, (255, 60, 255), (pu - 5, pv), (pu + 5, pv))
        pygame.draw.line(surface, (255, 60, 255), (pu, pv - 5), (pu, pv + 5))
        if orbit_radius:
            pygame.draw.circle(surface, (130, 50, 130), (pu, pv), int(orbit_radius * scale), 1)

    # Moving things: orange circle + arrow to where it'll be in 1 s (red if it's a threat)
    if nav_overlay is not None:
        for pos, vel, threat in nav_overlay.get("tracks", []):
            color = (255, 50, 50) if threat else (255, 150, 30)
            u, v = to_screen(pos)
            pygame.draw.circle(surface, color, (u, v), max(4, int(0.25 * scale)), 2)
            pygame.draw.line(surface, color, (u, v), to_screen(pos + vel), 2)

    # Planned path (green) and numbered waypoints (yellow)
    if nav_overlay is not None:
        path = nav_overlay["path"]
        if path and len(path) > 1:
            pygame.draw.lines(surface, (60, 230, 90), False, [to_screen(p) for p in path], 2)
        wps = [to_screen(w) for w in nav_overlay["waypoints"]]
        if len(wps) > 1:
            pygame.draw.lines(surface, (120, 110, 40), False, wps, 1)
        for i, (u, v) in enumerate(wps):
            pygame.draw.circle(surface, (255, 220, 40), (u, v), 6, 2)
            surface.blit(small.render(str(i + 1), True, (255, 220, 40)), (u + 7, v - 7))

    # Robot marker (triangle pointing along its heading)
    rel = yaw - view_yaw
    shape = [(0.35, 0.0), (-0.25, 0.17), (-0.25, -0.17)]
    tri = [
        (c - (math.sin(rel) * fx + math.cos(rel) * fy) * scale,
         c - (math.cos(rel) * fx - math.sin(rel) * fy) * scale)
        for fx, fy in shape
    ]
    pygame.draw.polygon(surface, (255, 255, 255), tri)


def draw_3d(surface, points, pose, cam_az, cam_el, cam_dist, follow, hide_ground, pillar=None, orbit_radius=None,
            nav_overlay=None):
    """Perspective view from a camera orbiting the robot.

    cam_az is the camera's direction from the robot (pi = behind it when
    following the robot's heading), cam_el its elevation above the ground.
    """
    surface.fill((12, 12, 16))
    c = VIEW // 2

    rx, ry, rz, yaw = 0.0, 0.0, 0.0, 0.0
    if pose is not None:
        rx, ry, rz = pose[:3]
        yaw = yaw_from_quat(*pose[3:])

    az = cam_az + (yaw if follow else 0.0)
    target = np.array([rx, ry, 0.3])
    cam = target + cam_dist * np.array([
        math.cos(cam_el) * math.cos(az), math.cos(cam_el) * math.sin(az), math.sin(cam_el)])
    fwd = (target - cam) / np.linalg.norm(target - cam)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    to_cam = np.stack([right, up, fwd])  # rows: camera x (right), y (up), z (depth)

    def project(cam_pts):
        return c + FOCAL * cam_pts[..., 0] / cam_pts[..., 2], c - FOCAL * cam_pts[..., 1] / cam_pts[..., 2]

    def line(a, b, color, width=1):
        """Draw a world-space segment, clipped against the near plane."""
        a, b = (np.array(a) - cam) @ to_cam.T, (np.array(b) - cam) @ to_cam.T
        if a[2] < NEAR and b[2] < NEAR:
            return
        if a[2] < NEAR or b[2] < NEAR:
            t = (NEAR - a[2]) / (b[2] - a[2])
            clipped = a + t * (b - a)
            a, b = (clipped, b) if a[2] < NEAR else (a, clipped)
        (ax, ay), (bx, by) = project(a), project(b)
        pygame.draw.line(surface, color, (ax, ay), (bx, by), width)

    # Ground grid (1 m squares, 20 m across) under the robot
    gx, gy = math.floor(rx), math.floor(ry)
    for k in range(-10, 11):
        line((gx + k, gy - 10, 0), (gx + k, gy + 10, 0), (45, 45, 55))
        line((gx - 10, gy + k, 0), (gx + 10, gy + k, 0), (45, 45, 55))

    if points is not None and len(points):
        pts = points[points[:, 2] >= GROUND_Z] if hide_ground else points
        cp = (pts[:, :3] - cam) @ to_cam.T
        ok = cp[:, 2] > NEAR
        pts, cp = pts[ok], cp[ok]
        order = np.argsort(-cp[:, 2])  # far to near, so near points end up on top
        pts, cp = pts[order], cp[order]
        u, v = project(cp)
        u, v = u.astype(np.int32), v.astype(np.int32)
        ok = (u >= 0) & (u < VIEW - 1) & (v >= 0) & (v < VIEW - 1)
        u, v = u[ok], v[ok]
        level = np.clip((pts[ok, 2] - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
        colors = COLORMAP[(level * 255).astype(np.int32)]
        px = pygame.surfarray.pixels3d(surface)
        for du in (0, 1):           # 2x2 pixel dots
            for dv in (0, 1):
                px[u + du, v + dv] = colors
        del px

    # Picked pillar as a magenta cylinder outline, and the lap's path on the ground
    if pillar is not None:
        px_, py_ = pillar.center
        ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)]
        for (ax, ay), (bx, by) in zip(ring, ring[1:]):
            for z in (0.0, po.PILLAR_Z[1]):
                line((px_ + pillar.radius * ax, py_ + pillar.radius * ay, z),
                     (px_ + pillar.radius * bx, py_ + pillar.radius * by, z), (255, 60, 255))
            if orbit_radius:
                line((px_ + orbit_radius * ax, py_ + orbit_radius * ay, 0.0),
                     (px_ + orbit_radius * bx, py_ + orbit_radius * by, 0.0), (130, 50, 130))
        for ax, ay in ring[:-1:6]:
            line((px_ + pillar.radius * ax, py_ + pillar.radius * ay, 0.0),
                 (px_ + pillar.radius * ax, py_ + pillar.radius * ay, po.PILLAR_Z[1]), (255, 60, 255))

    # Planned path on the ground (green), waypoints as yellow posts
    if nav_overlay is not None:
        path = nav_overlay["path"] or []
        for a, b in zip(path, path[1:]):
            line((a[0], a[1], 0.02), (b[0], b[1], 0.02), (60, 230, 90), 2)
        for w in nav_overlay["waypoints"]:
            line((w[0], w[1], 0.0), (w[0], w[1], 0.6), (255, 220, 40), 2)
        for pos, vel, threat in nav_overlay.get("tracks", []):
            color = (255, 50, 50) if threat else (255, 150, 30)
            line((pos[0], pos[1], 0.0), (pos[0], pos[1], 1.5), color, 3)
            line((pos[0], pos[1], 0.05), (pos[0] + vel[0], pos[1] + vel[1], 0.05), color, 2)

    # Robot marker: triangle at body height pointing along its heading, plus a post to the ground
    cs, sn = math.cos(yaw), math.sin(yaw)
    tri = [(rx + cs * fx - sn * fy, ry + sn * fx + cs * fy, rz) for fx, fy in
           [(0.35, 0.0), (-0.25, 0.17), (-0.25, -0.17)]]
    for i in range(3):
        line(tri[i], tri[(i + 1) % 3], (255, 255, 255), 2)
    line((rx, ry, rz), (rx, ry, 0.0), (160, 160, 160))


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <network_interface>   e.g. eth0")
        sys.exit(1)

    print("WARNING: make sure the area around the robot is clear.")
    ChannelFactoryInitialize(0, sys.argv[1])
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    feed = LidarFeed()
    lean = Lean(sport)
    recorder = Recorder(feed)
    lidar_switch = ChannelPublisher("rt/utlidar/switch", String_)
    lidar_switch.Init()

    pygame.init()
    screen = pygame.display.set_mode((VIEW + PANEL, VIEW))
    pygame.display.set_caption("Go2 lidar + keyboard control")
    font = pygame.font.SysFont("monospace", 15)
    small = pygame.font.SysFont("monospace", 12)
    view = pygame.Surface((VIEW, VIEW))
    clock = pygame.time.Clock()

    status = "Press R to stand up"
    moving = False
    last_send = 0.0
    scale = 40.0          # pixels per metre
    robot_up = True       # top-down: heading up; 3D: camera follows heading
    view_3d = False
    cam_az, cam_el, cam_dist = math.pi, math.radians(35), 6.0  # 3D camera: behind and above
    hide_ground = False
    persist_idx = 2
    lidar_on = None       # unknown until frames arrive or L is pressed
    lidar_on_since = 0.0
    pillar = None         # picked pillar (po.Pillar)
    orbit = None          # active lap (po.Orbit)
    scan = None           # points gathered during the lap (po.ScanCloud)
    scan_frame = 0        # feed frame counter already added to the scan
    tilt_scans = True
    obstacle_map = nav.ObstacleMap()
    map_frame = 0         # feed frame counter already added to the obstacle map
    waypoints = []
    navigator = None      # active nav.Navigator
    dodger = dodge.Dodger()
    dodge_on = False
    track_frame = 0       # feed frame counter the tracker has seen
    was_dodging = False

    def end_nav(reason):
        nonlocal navigator, moving, status
        if navigator is None:
            return
        sport.StopMove()
        moving = False
        status = ("Reached the last waypoint" if navigator.done else
                  f"Stopped: {navigator.failed}" if navigator.failed else f"Waypoints stopped: {reason}")
        navigator = None

    def end_orbit(reason):
        """Stop the lap; save the scan if it finished."""
        nonlocal orbit, moving, status
        if orbit is None:
            return
        sport.StopMove()
        moving = False
        lean.leave()
        if orbit.done:
            full = save_ply(scan.all(), "scan")
            save_ply(scan.around(orbit.pillar), "pillar")
            status = f"Lap done - saved {full} and pillar_*.ply"
        else:
            status = f"Lap stopped: {reason}"
        orbit = None
    hz, hz_frames, hz_time = 0.0, 0, time.time()
    running = True

    while running:
        points, pose, frames, last_frame = feed.latest(PERSIST_STEPS[persist_idx])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWFOCUSLOST and (moving or orbit or navigator):
                end_orbit("window lost focus")
                end_nav("window lost focus")
                sport.StopMove()
                moving = False
                status = "Window lost focus - stopped"
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not view_3d
                  and event.pos[0] < VIEW and orbit is None):
                click = screen_to_world(*event.pos, pose, scale, robot_up)
                pillar, why = po.find_pillar(points, pose[:2] if pose else (0, 0),
                                             yaw_from_quat(*pose[3:]) if pose else 0.0, click)
                status = (f"Pillar picked: {pillar.radius * 2:.2f} m wide" if pillar
                          else f"No pillar: {why}")
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not view_3d
                  and event.pos[0] < VIEW and navigator is None):
                waypoints.append(screen_to_world(*event.pos, pose, scale, robot_up))
                status = f"Waypoint {len(waypoints)} added - Enter to go"
            elif event.type == pygame.MOUSEWHEEL:
                if view_3d:
                    cam_dist = min(40.0, max(1.0, cam_dist / (1.15 ** event.y)))
                else:
                    scale = min(200.0, max(10.0, scale * (1.15 ** event.y)))
            elif event.type == pygame.MOUSEMOTION and event.buttons[0] and view_3d:
                cam_az -= event.rel[0] * 0.01
                cam_el = min(math.radians(89), max(math.radians(3), cam_el + event.rel[1] * 0.01))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    end_orbit("Space")
                    end_nav("Space")
                    sport.StopMove()
                    moving = False
                    status = "Stopped"
                elif event.key == pygame.K_t:
                    tilt_scans = not tilt_scans
                    status = f"Tilt scans {'on' if tilt_scans else 'off'} (next lap)"
                elif event.key == pygame.K_c and orbit is None and pose:
                    pillar, why = po.find_pillar(points, pose[:2], yaw_from_quat(*pose[3:]))
                    status = (f"Pillar picked: {pillar.radius * 2:.2f} m wide" if pillar
                              else f"No pillar: {why}")
                elif event.key in (pygame.K_z, pygame.K_x) and orbit is None:
                    if pillar is None:
                        status = "Pick a pillar first (click it or press C)"
                    elif not lean.standing:
                        status = "Stand up first (R)"
                    elif not pose:
                        status = "No robot pose"
                    else:
                        lean.leave()
                        orbit = po.Orbit(pillar, pose[:2], +1 if event.key == pygame.K_z else -1,
                                         tilt_scans, SCAN_TILT)
                        scan, scan_frame = po.ScanCloud(), frames
                        status = f"Circling pillar at {orbit.radius:.1f} m - any key stops"
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if navigator is not None:
                        end_nav("Enter")
                    elif not waypoints:
                        status = "Right-click to add waypoints first"
                    elif not lean.standing:
                        status = "Stand up first (R)"
                    elif orbit is not None or not pose:
                        status = "Busy (lap running) or no robot pose"
                    else:
                        lean.leave()
                        navigator = nav.Navigator(waypoints)
                        status = f"Walking {len(waypoints)} waypoint(s) - any key stops"
                elif event.key == pygame.K_u and navigator is None:
                    if event.mod & pygame.KMOD_SHIFT:
                        waypoints.clear()
                    elif waypoints:
                        waypoints.pop()
                    status = f"{len(waypoints)} waypoint(s)"
                elif event.key == pygame.K_b:
                    dodge_on = not dodge_on
                    status = f"Dodging moving obstacles {'ON' if dodge_on else 'off'}"
                elif event.key == pygame.K_k:
                    obstacle_map.clear()
                    status = "Obstacle map cleared"
                elif event.key == pygame.K_BACKSPACE:
                    end_orbit("cleared")
                    pillar = scan = None
                    status = "Pillar cleared"
                elif event.key == pygame.K_r:
                    end_orbit("R")
                    end_nav("R")
                    lean.leave()
                    sport.StandUp()
                    time.sleep(1.0)
                    sport.BalanceStand()
                    status = "Standing - ready to drive"
                elif event.key == pygame.K_f:
                    sport.StopMove()
                    moving = False
                    end_orbit("F")
                    end_nav("F")
                    lean.leave()
                    sport.StandDown()
                    status = "Lying down"
                elif event.key == pygame.K_p:
                    if points is not None:
                        status = f"Saved {save_ply(points)}"
                    else:
                        status = "No lidar data to save"
                elif event.key == pygame.K_o:
                    if recorder.active:
                        n = recorder.count
                        status = f"Saved {n} frames to {recorder.stop()}"
                    else:
                        recorder.start()
                        status = f"Recording to {recorder.dir}"
                elif event.key == pygame.K_g:
                    hide_ground = not hide_ground
                elif event.key == pygame.K_LEFTBRACKET:
                    persist_idx = max(0, persist_idx - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    persist_idx = min(len(PERSIST_STEPS) - 1, persist_idx + 1)
                elif event.key == pygame.K_m:
                    robot_up = not robot_up
                elif event.key == pygame.K_v:
                    view_3d = not view_3d
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    if view_3d:
                        cam_dist = max(1.0, cam_dist / 1.25)
                    else:
                        scale = min(200.0, scale * 1.25)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if view_3d:
                        cam_dist = min(40.0, cam_dist * 1.25)
                    else:
                        scale = max(10.0, scale / 1.25)
                elif event.key == pygame.K_l:
                    lidar_on = not lidar_on
                    msg = std_msgs_msg_dds__String_()
                    msg.data = "ON" if lidar_on else "OFF"
                    lidar_switch.Write(msg)
                    if lidar_on:
                        lidar_on_since = time.time()
                        status = "Lidar ON - spinning up (~10 s)"
                    else:
                        status = "Lidar OFF - points frozen"

        # Driving
        keys = pygame.key.get_pressed()
        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        vx_max, vy_max, vyaw_max = FAST if fast else NORMAL
        vx = vx_max * (keys[pygame.K_w] - keys[pygame.K_s])
        vy = vy_max * (keys[pygame.K_q] - keys[pygame.K_e])
        vyaw = vyaw_max * (keys[pygame.K_a] - keys[pygame.K_d])

        arrow_pitch = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        arrow_roll = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        now = time.time()
        dt = clock.get_time() / 1000.0

        # Obstacle map from every new lidar frame
        # Floor height in odom z: robot height minus its reported body height
        floor_z = pose[2] - lean.body_height if pose is not None and lean.body_height > 0 else 0.0
        if pose is not None:
            for p in feed.frames_since(map_frame):
                obstacle_map.add(p, pose[:2], yaw_from_quat(*pose[3:]), pose[2], floor_z, now)
        map_frame = frames
        recent_points = feed.latest(1.0)[0]  # stop checks use only the last second, not the persistence window

        # Track moving things (once per new lidar frame) and dodge them if enabled
        robot_yaw = yaw_from_quat(*pose[3:]) if pose is not None else 0.0
        if pose is not None and frames != track_frame:
            dodger.tracker.update(feed.latest(0.25)[0], np.array(pose[:2]), floor_z, now)
            track_frame = frames
        manual = bool(vx or vy or vyaw or arrow_pitch or arrow_roll)
        dodge_cmd = None
        if dodge_on and not manual and pose is not None and lean.standing and now - last_frame < 2.0:
            dodge_cmd = dodger.step(now, pose[:2], robot_yaw, obstacle_map.occupied(now), recent_points, floor_z)
        if dodge_cmd is not None:
            vx, vy, vyaw = dodge_cmd
            status = "DODGING a moving obstacle"
            was_dodging = True
        elif was_dodging:
            was_dodging = False  # resume: replan the waypoints / re-anchor the lap from here
            if navigator is not None:
                navigator.path = None
            if orbit is not None:
                rel = np.array(pose[:2]) - orbit.pillar.center
                orbit.last_angle = math.atan2(rel[1], rel[0])
            status = "Dodge over"

        # Waypoints: any manual key takes over; otherwise the navigator drives
        if navigator is not None and dodge_cmd is None:
            if vx or vy or vyaw or arrow_pitch or arrow_roll:
                end_nav("manual key")
            elif pose is None or now - last_frame > 2.0:
                end_nav("lost lidar / pose")
            else:
                vx, vyaw = navigator.step(now, pose[:2], yaw_from_quat(*pose[3:]),
                                          obstacle_map.occupied(now), recent_points, floor_z)
                vy = 0.0
                if navigator.done or navigator.failed:
                    end_nav("done")
                    vx = vyaw = 0.0
                else:
                    status = navigator.note

        # Pillar lap: any manual key takes over; otherwise the orbit drives
        lap_lean = None
        if orbit is not None and dodge_cmd is None:
            if vx or vy or vyaw or arrow_pitch or arrow_roll:
                end_orbit("manual key")
            elif pose is None or now - last_frame > 2.0:
                end_orbit("lost lidar / pose")
            elif orbit.phase == "move" and not po.clearance_ok(recent_points, np.array(pose[:2]), orbit.pillar):
                end_orbit(f"something within {po.CLEARANCE} m")
            else:
                for p in feed.frames_since(scan_frame):
                    scan.add(p)
                scan_frame = frames
                vx, vy, vyaw, lap_lean = orbit.step(now, pose[:2], yaw_from_quat(*pose[3:]), lean.pitch)
                if orbit.done:
                    end_orbit("done")
                    vx = vy = vyaw = 0.0
                    lap_lean = None

        if (vx or vy or vyaw) and lean.active:
            lean.leave()  # leaning while walking makes the dog jitter, so level out first
        if vx or vy or vyaw:
            if now - last_send >= 1.0 / SEND_HZ:
                sport.Move(vx, vy, vyaw)
                last_send = now
                moving = True
        elif moving:
            sport.StopMove()
            moving = False

        # Leaning, standing still only: the lap's tilt scans, or the arrow keys
        if dodge_cmd is not None:
            pass  # walking; leaning was ended above
        elif orbit is not None:
            if lap_lean is None:
                lean.leave()
            else:
                lean.update(lap_lean, 0.0, dt, now, moving)
        elif not moving:
            lean.update(LEAN * arrow_pitch, LEAN * arrow_roll, dt, now, moving)

        recorder.write_pending()

        if now - hz_time >= 1.0:
            hz = (frames - hz_frames) / (now - hz_time)
            hz_frames, hz_time = frames, now

        # Draw
        shown = points
        if scan is not None and len(scan.view()):
            shown = scan.view() if points is None else np.concatenate([points, scan.view()])
        orbit_radius = orbit.radius if orbit else None
        nav_overlay = {
            "obstacles": obstacle_map.obstacle_xy(now, pose[:2], VIEW / scale) if pose and not view_3d else [],
            "waypoints": waypoints,
            "path": navigator.path if navigator else None,
            "tracks": [(t.pos, t.vel, t in dodger.threats(np.array(pose[:2])))
                       for t in dodger.tracker.tracks if t.moving] if pose else [],
        }
        if view_3d:
            draw_3d(view, shown, pose, cam_az, cam_el, cam_dist, robot_up, hide_ground, pillar, orbit_radius,
                    nav_overlay)
        else:
            draw_cloud(view, small, shown, pose, scale, robot_up, hide_ground, pillar, orbit_radius, nav_overlay)
        screen.blit(view, (0, 0))
        pygame.draw.rect(screen, (25, 25, 30), (VIEW, 0, PANEL, VIEW))

        stale = points is None or now - last_frame > 2.0
        if lidar_on is None and not stale:
            lidar_on = True
        if lidar_on is False:
            lidar_line = ("lidar OFF - points frozen", (255, 200, 90))
        elif stale and now - lidar_on_since < LIDAR_SPINUP:
            lidar_line = ("lidar spinning up...", (255, 200, 90))
        elif stale:
            lidar_line = ("NO LIDAR DATA - press L", (255, 120, 100))
        else:
            lidar_line = (f"{hz:4.1f} Hz", None)
        lines = [
            ("DRIVE", (150, 200, 255)),
            ("W/S fwd/back  A/D turn  Q/E strafe", None),
            ("Shift fast  Space stop  Esc quit", None),
            ("R stand up  F lie down  Arrows lean", None),
            ("", None),
            (f"vx   {vx:+.2f} m/s", None),
            (f"vy   {vy:+.2f} m/s", None),
            (f"vyaw {vyaw:+.2f} rad/s", None),
            (f"speed: {'FAST' if fast else 'normal'}", None),
            (f"lean p{math.degrees(lean.pitch):+.0f}° r{math.degrees(lean.roll):+.0f}°"
             f"{'  POSE' if lean.active else ''}", None),
            ("", None),
            ("LIDAR", (150, 200, 255)),
            ("P snapshot  O record  G ground", None),
            ("V 2D/3D  M follow  +/- zoom  drag", None),
            ("[ ] persistence  L lidar on/off", None),
            ("", None),
            (f"{lidar_line[0]}  {0 if points is None else len(points)} points", lidar_line[1]),
            (f"view: 3D {'follow' if robot_up else 'fixed'}  {cam_dist:.1f} m away" if view_3d else
             f"view: top {'robot-up' if robot_up else 'world-up'}  {scale:.0f} px/m", None),
            (f"ground: {'hidden' if hide_ground else 'shown'}   "
             f"persist: {PERSIST_STEPS[persist_idx]:g} s", None),
            (f"REC {recorder.count} frames" if recorder.active else "not recording",
             (255, 80, 80) if recorder.active else None),
            ("", None),
            ("PILLAR", (150, 200, 255)),
            ("Click/C pick  Z/X lap  T tilt  Bksp", None),
            ((f"LAP {orbit.progress * 100:3.0f}%  {orbit.phase}  r={orbit.radius:.1f} m" if orbit else
              f"pillar {pillar.radius * 2:.2f} m wide  tilt scans {'on' if tilt_scans else 'off'}" if pillar else
              f"no pillar  tilt scans {'on' if tilt_scans else 'off'}"),
             (255, 60, 255) if orbit else None),
            ("", None),
            ("WAYPOINTS", (150, 200, 255)),
            ("R-click add  U undo  Enter go  K map", None),
            (f"WALKING to {navigator.index + 1}/{len(waypoints)}" if navigator else
             f"{len(waypoints)} waypoint(s)", (255, 220, 40) if navigator else None),
            ("DODGING!" if dodge_cmd is not None else
             f"B dodge movers: {'ON' if dodge_on else 'off'}",
             (255, 50, 50) if dodge_cmd is not None else (255, 150, 30) if dodge_on else None),
        ]
        if pose is not None:
            lines.append((f"pos ({pose[0]:+.2f}, {pose[1]:+.2f}) yaw {math.degrees(yaw_from_quat(*pose[3:])):+.0f}°", None))
        y = 14
        for text, color in lines:
            screen.blit(font.render(text, True, color or (220, 220, 220)), (VIEW + 16, y))
            y += 19

        # Status may be a long path; wrap it at the bottom of the panel
        for i in range(0, len(status), 38):
            screen.blit(font.render(status[i:i + 38], True, (255, 220, 120)), (VIEW + 16, y + 10))
            y += 20

        # Height colour legend
        legend_y = VIEW - 40
        for i in range(PANEL - 32):
            color = COLORMAP[int(i / (PANEL - 33) * 255)]
            pygame.draw.line(screen, color, (VIEW + 16 + i, legend_y), (VIEW + 16 + i, legend_y + 10))
        screen.blit(small.render(f"{Z_MIN:.1f} m", True, (180, 180, 180)), (VIEW + 16, legend_y + 14))
        screen.blit(small.render(f"{Z_MAX:.1f} m", True, (180, 180, 180)), (VIEW + PANEL - 52, legend_y + 14))

        pygame.display.flip()
        clock.tick(30)

    end_orbit("quit")
    end_nav("quit")
    sport.StopMove()
    lean.leave()
    if lidar_on is False:
        msg = std_msgs_msg_dds__String_()
        msg.data = "ON"
        lidar_switch.Write(msg)
    if recorder.active:
        print(f"Saved {recorder.count} frames to {recorder.stop()}")
    pygame.quit()


if __name__ == "__main__":
    main()

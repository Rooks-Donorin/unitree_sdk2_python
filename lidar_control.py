"""Drive the Go2 with the keyboard while viewing and capturing its lidar.

Usage:  .venv/bin/python lidar_control.py eth0              (Ethernet cable, DDS)
        .venv_wifi/bin/python lidar_control.py wifi ap      (on the dog's own Wi-Fi hotspot)
        .venv_wifi/bin/python lidar_control.py wifi 192.168.1.23   (dog and PC on one router)
        add --slam to the wifi forms to use the dog's SLAM point cloud instead of the voxel map (it starts the
        dog's mapping, which OVERWRITES its saved map, and stops it again on exit); --raw asks for the raw lidar,
        which the dog doesn't send over Wi-Fi

Over Wi-Fi the dog talks WebRTC like the phone app (see transport.py): close the app first, the dog
takes one client, firmware 1.1.15+ needs its AES key in GO2_AES_KEY or ./.go2_aes_key, and the lidar
is the dog's voxel map rather than the raw deskewed cloud, so it looks sparser than over the cable.

The window can be resized or maximized, and F11 toggles fullscreen: the lidar pane grows to fit and the
panel on the right gets the rest. The panel lists every command below its live values (the bigger the window,
the more of the list it shows), and F1 or Tab shows the whole list over the lidar pane.

The left pane is a live view of the lidar point cloud around the robot,
coloured by height: top-down, an orbitable 3D view, or a first-person view
from the dog's body looking along its heading (V cycles through them). Each
lidar frame only has ~500 points, so the view builds a map: points stay where they were seen (walls and
obstacles stay) and are removed only when lidar beams keep passing through where they were, i.e. the thing
moved. [ ] shortens that memory to the last few seconds instead. Points are in the odom
(world) frame, so they stay put as the robot moves. Switching the lidar off (L)
freezes the points on screen; when it comes back on (~10 s spin-up) new
points line up with the old ones, because the robot pose keeps updating from
leg odometry while the lidar is off. The window must have focus for keys to
work.

Driving                          Lidar
  W / S   forward / backward       P       save snapshot (.ply)
  A / D   turn left / right        O       start / stop recording frames
  Q / E   strafe left / right      G       hide / show ground points
  Shift   hold for faster speed    V       top-down / 3D / first-person view
  Arrows  lean body (standing)     M       view follows robot heading / fixed
  Space   stop                     + / -   zoom (or mouse wheel)
  R       stand up                 L       lidar on / off
  F       lie down                 [ / ]   less / more persistence
  Esc     stop and quit            H       colour: height / distance (floor light blue, closest red) / camera
                                   I       camera picture + lidar on / off (first-person)
                                   1       depth map (Depth Anything) in place of the camera picture
                                   2       with it on, drop lidar points the depth map disagrees with
                                   3       optical flow on the camera picture: tracks what moves (see motion_flow.py)
                                   4       flow view: movers / flow colour / flow arrows
                                   5 / 6   flow less / more sensitive
                                   7       record a camera clip on / off (for tuning flow offline; flow_replay.py)
                                   Y       camera picture size
                                   F2      keep / drop the lidar points behind the dog
                                   F3      on-screen buttons (right edge) show / hide: every key as a mouse button
                                   N       block mesh view on / off (3D, first-person)
                                   J       save the block mesh as a coloured .ply
                                   /       pick a camera-mount setting (fov, fwd, left, up, yaw, pitch)
                                   , .     change it (saved to camera_mount.json)    \\  reset
                                   3D: drag to orbit  First-person: drag to look around

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
  K       forget the maps (points and obstacles)
  The dog plans a path around obstacles (red squares: lidar points 0.15-0.7 m
  above the floor seen repeatedly, remembered until the lidar sees through them, erased when the
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

Terminal commands
  Type in the terminal you started this from (the prompt reads go2>): jump, pounce, shake, stretch, happy,
  heart, scrape, dance1, dance2, sit, rise, recover, stand, down, stop; help lists them all. They need the dog
  standing, cancel any waypoint walk or pillar lap first, and jump / pounce ask for a second typing once per run.

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
import contextlib
import io
import json
import math
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pygame

import dodge
import motion_flow
import navigation as nav
import onscreen_buttons as ob
import pillar_orbit as po
import transport

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

VIEW = 720            # lidar pane is VIEW x VIEW pixels (follows the window size, see set_view_size)
PANEL = 380           # width of one column of the text panel on the right (the panel gets more columns in a wide window)
MAX_VIEW = 1200       # largest lidar pane; a bigger window gives the extra width to the panel
MIN_VIEW = 400
Z_MIN, Z_MAX = 0.0, 2.0   # height range for the colour map (m, odom frame)
DIST_MAX = 8.0            # distance from the lidar (m) that maps to the red end when colouring by distance
GROUND_Z = 0.12           # points below this are "ground" when hidden
PERSIST_STEPS = [0.0, 1.0, 3.0, 5.0, 10.0, 30.0, math.inf]  # seconds of lidar-on time to overlay (0 = latest only, inf = the whole map)
MAX_GAP = 0.5             # gaps longer than this (lidar off) don't count toward persistence
LIDAR_SPINUP = 20.0       # seconds to show "starting up" after switching the lidar on
LIDAR_NUDGE = 20.0        # while no lidar data arrives and the lidar should be on, send it ON again this often (s)
CAPTURE_DIR = "captures"
CLIP_MAX_SECONDS = 60       # a camera clip (key 7) stops by itself after this long, so it can't fill the disk
FOCAL = (VIEW / 2) / math.tan(math.radians(60) / 2)  # 60 degree field of view
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"   # Hugging Face model id; -Base-hf / -Large-hf are slower and better
CAMERA_SIZES = [260, 480, VIEW]  # widths of the floating camera picture (Y cycles); the last fills the pane
MOUNT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_mount.json")
NEAR = 0.2                # points closer to the 3D camera than this are skipped

# Height colour map: blue -> cyan -> green -> yellow -> red
_anchors = np.array([[40, 60, 230], [0, 200, 230], [40, 220, 60], [240, 220, 30], [230, 40, 30]], float)
_t = np.linspace(0, 1, 256) * (len(_anchors) - 1)
_i = np.minimum(_t.astype(int), len(_anchors) - 2)
_f = (_t - _i)[:, None]
COLORMAP = ((1 - _f) * _anchors[_i] + _f * _anchors[_i + 1]).astype(np.uint8)

# Distance colour map: bright red (closest) -> green -> blue (farthest)
_d_anchors = np.array([[255, 20, 20], [40, 230, 60], [40, 80, 255]], float)
_d_t = np.linspace(0, 1, 256) * (len(_d_anchors) - 1)
_d_i = np.minimum(_d_t.astype(int), len(_d_anchors) - 2)
_d_f = (_d_t - _d_i)[:, None]
FLOOR_BLEND = (0.07, 0.22)  # m above the floor: pure floor colour up to the first, pure distance colour from the second
FLOOR_KEEP = 6              # in distance colouring only one floor dot in this many is drawn, so the floor doesn't swamp the scene
NEAR_DOTS = (0.15, 2.5)     # m from the dog: dots start growing at the second distance and keep growing, ~1/distance, down to the first
NEAR_GROW = 5.0             # how fast: extra dot width in px is NEAR_GROW * (1 / distance - 1 / 2.5 m)
MAX_STORE = 1_500_000       # points in the map; beyond this the ones seen longest ago go first
MAX_DRAW = 60000            # most points drawn at once (the nearest to the dog), so a big map doesn't slow the window
CARVE_RAYS = 600            # beams per frame used to clear points the lidar sees through
CARVE_STEP = 0.04           # m between samples along a beam (each beam starts at a random offset, see _carve)
MESH_VOXEL = 0.08           # m; blob size of the block mesh
MESH_MIN_POINTS = 2         # a blob needs this many map points, so stray single points don't become blocks
MESH_POINTS = 150000        # most map points (nearest the dog) the block mesh is built from
MESH_DRAW = 3500            # most block faces drawn per frame (the nearest ones)
MESH_REBUILD = 1.5          # s between rebuilds of the block mesh while its view is on
UNPAINTED = (70, 70, 80)    # colour for a point the camera hasn't seen (such points aren't drawn in camera-colour mode)
DEPTH_RANGE = (0.3, 8.0)    # m: lidar points this far from the camera are checked against the depth map
DEPTH_MIN_FIT = 30          # lidar points in view needed to line the depth map up with the lidar
DEPTH_K = 3.0               # a point is rejected when it is more than this many spreads off the fitted depth map
PAINT_WEIGHT = 8            # a point's camera colour is the average of up to this many sightings
MERGE_DIST = 0.03           # m; a new point this close to one already kept just refreshes it instead of being stored
FLOOR_COLOR = (130, 225, 255)  # light blue, so the floor reads differently from far (deep blue) objects
DIST_COLORMAP = ((1 - _d_f) * _d_anchors[_d_i] + _d_f * _d_anchors[_d_i + 1]).astype(np.uint8)


@dataclass
class CameraMount:
    """Where the dog's camera sits and where it points, so lidar points land where the picture shows them.

    One mount drives both the lidar drawn onto the camera picture and the first-person 3D view, so lining
    one up lines up the other. The defaults are Go2Py's calibration (about 80 degrees across at 640 px; the
    stream is 16:9, so it is an estimate); the , . / keys adjust it live and it is saved to MOUNT_FILE.
    """
    fov: float = 80.0    # degrees, across the picture
    fwd: float = 0.28    # camera position in the body frame, m (forward / left / up of the body centre)
    left: float = 0.0
    up: float = 0.0
    yaw: float = 0.0     # aim trim, degrees (left / up positive)
    pitch: float = 0.0

    # name, step, min, max
    FIELDS = (("fov", 2.5, 30.0, 140.0), ("fwd", 0.02, -0.5, 0.8), ("left", 0.02, -0.5, 0.5),
              ("up", 0.02, -0.5, 0.5), ("yaw", 0.5, -30.0, 30.0), ("pitch", 0.5, -30.0, 30.0))

    @classmethod
    def load(cls):
        try:
            with open(MOUNT_FILE) as f:
                return cls(**{k: float(v) for k, v in json.load(f).items() if k in asdict(cls())})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self):
        try:
            with open(MOUNT_FILE, "w") as f:
                json.dump(asdict(self), f, indent=1)
        except OSError:
            pass

    def adjust(self, index, direction):
        name, step, low, high = self.FIELDS[index]
        setattr(self, name, round(min(high, max(low, getattr(self, name) + direction * step)), 3))

    def axes(self, extra_yaw=0.0, extra_pitch=0.0):
        """Unit vectors of the camera's forward, left and up directions in the body frame."""
        yaw, pitch = math.radians(self.yaw) + extra_yaw, math.radians(self.pitch) + extra_pitch
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        return (np.array([cp * cy, cp * sy, sp]), np.array([-sy, cy, 0.0]), np.array([-sp * cy, -sp * sy, cp]))

    def position(self):
        return np.array([self.fwd, self.left, self.up])


def body_rotation(pose):
    """3x3 matrix turning body-frame vectors (forward, left, up) into odom-frame vectors."""
    qx, qy, qz, qw = pose[3:]
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


class LidarFeed:
    """Takes the point cloud and robot pose from the link, keeps a map of the points and recent frames.

    Two views of the same data: `history` holds each frame whole (for the obstacle map and the pillar scan,
    which count how often things are seen), and the point store is the map the view shows: one point per
    MERGE_DIST cell with the last time it was seen. Points stay for good; one goes only when lidar beams keep
    passing through where it was (it moved away), by the same rule the obstacle map uses (navigation.py).
    """

    def __init__(self, link):
        self.link = link
        self.rng = np.random.default_rng()
        self.lock = threading.Lock()
        self.history = deque()  # (lidar time, N x 4 float32 x/y/z/intensity in odom frame)
        self.lidar_time = 0.0   # advances only while frames arrive, so time with the lidar off doesn't age points
        self.keys = np.empty(0, np.int64)       # point store, sorted by cell key
        self.store = np.empty((0, 4), np.float32)
        self.seen = np.empty(0, np.float64)     # lidar time each stored point was last seen
        self.misses = np.empty(0, np.float32)   # decaying count of beams that passed through each point
        self.paint = np.empty((0, 4), np.uint8)  # camera colour r, g, b of each point and how many sightings made it
        self.painter = None     # painter(points, pose) -> (rgb, painted) from the camera, or None; set by main
        self.floor_z = None     # floor height measured from the points, set by main; None until known
        self.keep_behind = False  # True: keep the points behind the dog too (set by main)
        self.gate = None        # gate(points, pose) -> keep mask (True = keep) from the depth map, or None; set by main
        self.pose = None        # (x, y, z, qx, qy, qz, qw)
        self.frames = 0
        self.last_frame_time = 0.0
        self.record_queue = None  # set to a Queue while recording

        link.start(self._on_cloud, self._on_pose)

    def _on_cloud(self, points):
        now = time.time()
        with self.lock:
            pose = self.pose
            rec = self.record_queue
            if self.last_frame_time:
                self.lidar_time += min(now - self.last_frame_time, MAX_GAP)
            # Points behind the dog (where it can't be looked at or steered toward) are dropped as the
            # frame comes in, so they never reach the view or the maps. Recordings keep the whole frame.
            ahead = points if self.keep_behind else drop_behind(points, pose)
            keep = self.gate(ahead, pose) if self.gate and pose is not None and len(ahead) else None
            if keep is not None and not keep.all():
                # Points the depth map says are wrong never get in, and any already stored in those cells go
                at, found = self._find(np.unique(self._cell_keys(ahead[~keep][:, :3])))
                if found.any():
                    gone = np.ones(len(self.keys), bool)
                    gone[at[found]] = False
                    self._select(gone)
                ahead = ahead[keep]
            self.history.append((self.lidar_time, ahead))
            while self.lidar_time - self.history[0][0] > PERSIST_STEPS[-2]:
                self.history.popleft()
            dt = min(now - self.last_frame_time, MAX_GAP) if self.last_frame_time else 0.0
            colored = self.painter(ahead, pose) if self.painter and pose is not None and len(ahead) else None
            self._merge(ahead, *(colored or (None, None)))
            if pose is not None:
                height = getattr(self.link, "body_height", 0.0)
                floor = self.floor_z if self.floor_z is not None else pose[2] - height if height > 0 else 0.0
                self._carve(ahead, pose, floor, dt)
            self.frames += 1
            self.last_frame_time = now
        if rec is not None:
            rec.put((time.time(), points, pose))

    @staticmethod
    def _cell_keys(xyz):
        cells = np.floor(xyz / MERGE_DIST).astype(np.int64) + (1 << 20)
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def _find(self, keys):
        """Index of each key in the sorted store, and whether it is there."""
        at = np.searchsorted(self.keys, keys)
        if not len(self.keys):
            return at, np.zeros(len(keys), bool)
        return at, (at < len(self.keys)) & (self.keys[np.minimum(at, len(self.keys) - 1)] == keys)

    def _select(self, idx):
        """Keep only the stored points at `idx` (an index array or a boolean mask)."""
        self.keys, self.store, self.seen, self.misses, self.paint = (
            self.keys[idx], self.store[idx], self.seen[idx], self.misses[idx], self.paint[idx])

    def _merge(self, points, rgb=None, painted=None):
        """Add a frame to the point store: points in a cell already held only refresh it (lock held).

        rgb / painted: the camera colour for each point and whether the camera saw it; those colours are averaged
        into the stored point.
        """
        keys, first = np.unique(self._cell_keys(points[:, :3]), return_index=True)  # also drops repeats in the frame
        points = points[first]
        fresh = np.zeros((len(points), 4), np.uint8)
        if rgb is not None:
            rgb, painted = rgb[first], painted[first]
            fresh[painted, :3], fresh[painted, 3] = rgb[painted], 1
        at, known = self._find(keys)
        self.seen[at[known]] = self.lidar_time
        self.misses[at[known]] = 0.0     # seen again: whatever beams passed through it were wrong or it came back
        if rgb is not None:
            hit = at[known][painted[known]]
            weight = self.paint[hit, 3:4].astype(np.float32)
            mixed = (self.paint[hit, :3] * weight + rgb[known][painted[known]]) / (weight + 1)
            self.paint[hit, :3] = mixed.astype(np.uint8)
            self.paint[hit, 3] = np.minimum(weight[:, 0] + 1, PAINT_WEIGHT)
        new = ~known
        self.keys = np.insert(self.keys, at[new], keys[new])
        self.store = np.insert(self.store, at[new], points[new], axis=0)
        self.seen = np.insert(self.seen, at[new], self.lidar_time)
        self.misses = np.insert(self.misses, at[new], 0.0)
        self.paint = np.insert(self.paint, at[new], fresh[new], axis=0)
        if len(self.keys) > MAX_STORE:
            self._select(np.sort(np.argpartition(self.seen, len(self.keys) - MAX_STORE)[len(self.keys) - MAX_STORE:]))

    def _carve(self, points, pose, floor_z, dt):
        """Remove stored points the lidar can now see through (something moved away). Lock held.

        Every point in the frame means the beam from the lidar to it crossed open space, so any stored point
        on that beam (stopping short of the end, and above the floor) gets a miss; enough misses, with no
        sighting in the last HIT_GUARD seconds, delete it. Same numbers as the obstacle map's clearing.
        """
        if not len(self.keys):
            return
        self.misses *= math.exp(-dt / nav.HIT_DECAY)
        origin = np.array(pose[:3]) + body_rotation(pose) @ np.array([nav.LIDAR_OFFSET[0], 0.0, nav.LIDAR_OFFSET[1]])
        rel = points[:, :3] - origin
        dist = np.linalg.norm(rel, axis=1)
        keep = (dist > nav.END_GAP + nav.RAY_STEP) & (dist < nav.CLEAR_RANGE)
        rel, dist = rel[keep], dist[keep]
        if len(rel) > CARVE_RAYS:
            pick = np.linspace(0, len(rel) - 1, CARVE_RAYS).astype(int)
            rel, dist = rel[pick], dist[pick]
        if not len(rel):
            return
        # Each beam is sampled from a random starting offset: with fixed distances from a stationary lidar, a
        # cell lying between two sample distances would never be checked and could never be cleared.
        along = CARVE_STEP * (np.arange(1, int(nav.CLEAR_RANGE / CARVE_STEP))[None, :] + self.rng.random((len(rel), 1)))
        valid = along < (dist - nav.END_GAP)[:, None]
        samples = origin + (along[..., None] / dist[:, None, None]) * rel[:, None, :]
        valid &= samples[..., 2] >= floor_z + GROUND_Z   # the floor can't move away
        at, present = self._find(np.unique(self._cell_keys(samples[valid])))
        hit = at[present]
        self.misses[hit] += 1.0
        gone = hit[(self.misses[hit] >= nav.CLEAR_MISSES) & (self.lidar_time - self.seen[hit] > nav.HIT_GUARD)]
        if len(gone):
            alive = np.ones(len(self.keys), bool)
            alive[gone] = False
            self._select(alive)

    def clear(self):
        """Forget the point map."""
        with self.lock:
            self.keys = np.empty(0, np.int64)
            self.store = np.empty((0, 4), np.float32)
            self.seen = np.empty(0, np.float64)
            self.misses = np.empty(0, np.float32)
            self.paint = np.empty((0, 4), np.uint8)

    def _on_pose(self, pose):
        with self.lock:
            self.pose = pose

    def latest(self, persist, colors=False):
        """Points seen in the last `persist` seconds of lidar-on time, one per MERGE_DIST cell.

        colors=True adds three columns, the camera colour r, g, b (0 where the camera never saw the point).
        """
        with self.lock:
            if not self.history:
                return None, self.pose, self.frames, self.last_frame_time
            recent = self.seen >= self.lidar_time - persist
            points = self.store[recent]
            if colors:
                points = np.column_stack([points, self.paint[recent, :3].astype(np.float32)])
            return points, self.pose, self.frames, self.last_frame_time

    def frames_since(self, count):
        """Frames that arrived after the feed's frame counter was `count`."""
        with self.lock:
            n = min(self.frames - count, len(self.history))
            return [p for _, p in list(self.history)[len(self.history) - n:]] if n > 0 else []


class CameraFeed:
    """The dog's front camera, scaled for the first-person overlay. Only streams while switched on."""

    def __init__(self, link):
        self.link = link
        self.on = False
        self.lock = threading.Lock()
        self.rgb = None       # latest H x W x 3 frame, not yet turned into a surface
        self.latest_rgb = None  # latest full-size frame, kept for painting lidar points
        self.surface = None
        self.last_frame = 0.0
        self.tap = None       # tap(rgb, time): called for every new frame (the clip recorder uses it)

    def toggle(self):
        self.on = not self.on
        if self.on:
            self.link.start_camera(self._on_frame)
        else:
            self.link.stop_camera()
        return self.on

    def _on_frame(self, rgb):
        with self.lock:
            self.rgb = self.latest_rgb = rgb
            self.last_frame = now = time.time()
        tap = self.tap
        if tap is not None:
            tap(rgb, now)

    def scaled(self, width):
        """The newest frame as a surface `width` px wide (None before the first frame)."""
        with self.lock:
            rgb, self.rgb = self.rgb, None
        if rgb is not None:
            full = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.surface = pygame.transform.smoothscale(full, (width, round(width * rgb.shape[0] / rgb.shape[1])))
        return self.surface


class DepthFeed:
    """Depth Anything V2 on the camera picture, run on a worker thread.

    The model gives relative depth (how much nearer one pixel is than another, not metres), so each frame is
    stretched to its own near-to-far range and coloured like the distance view: red nearest, green, blue farthest.
    It loads on first use (downloading the model once) and only runs while switched on.
    """

    def __init__(self, camera):
        self.camera = camera
        self.on = False
        self.status = ""          # "loading the model...", an error, or "" when running
        self.fps = 0.0
        self.lock = threading.Lock()
        self.colored = None       # latest H x W x 3 uint8, not yet turned into a surface
        self.surface = None
        self.last_frame = 0.0
        self.depth = None         # latest H x W float32 relative inverse depth (bigger = nearer)
        self._thread = None
        self._quit = False

    def toggle(self):
        self.on = not self.on
        if self.on and (self._thread is None or not self._thread.is_alive()):
            self.status = "loading the model..."
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self.on

    def _run(self):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as err:
            self.status, self.on = f"needs torch + transformers ({err})", False
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
            model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
            if device == "cuda":
                model.half()
        except Exception as err:   # no internet for the first download, out of memory, ...
            self.status, self.on = f"could not load {DEPTH_MODEL}: {err}", False
            return
        self.status = "" if device == "cuda" else "running on the CPU (slow)"
        done = 0.0
        while not self._quit:
            frame, stamp = self.camera.latest_rgb, self.camera.last_frame
            if not self.on or frame is None or stamp == done:
                time.sleep(0.03)
                continue
            done, started = stamp, time.time()
            try:
                inputs = processor(images=frame, return_tensors="pt")
                pixels = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    out = model(pixel_values=pixels.half() if device == "cuda" else pixels).predicted_depth
                depth = torch.nn.functional.interpolate(out[None].float(), size=frame.shape[:2], mode="bilinear")
                depth = depth[0, 0].cpu().numpy()
            except Exception as err:
                self.status, self.on = f"depth failed: {err}", False
                return
            low, high = np.percentile(depth, [2, 98])
            near = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)   # 1 = nearest
            colored = DIST_COLORMAP[((1 - near) * 255).astype(np.int32)]
            with self.lock:
                self.depth, self.colored, self.last_frame = depth.astype(np.float32), colored, time.time()
            self.fps = 0.8 * self.fps + 0.2 / max(time.time() - started, 1e-3)

    def close(self):
        """Stop the worker, waiting briefly. False if it is still busy (the model is mid-load)."""
        self._quit = True
        if self._thread is not None:
            self._thread.join(3.0)
            return not self._thread.is_alive()
        return True

    def scaled(self, width):
        """The newest depth picture as a surface `width` px wide (None before the first one)."""
        with self.lock:
            colored, self.colored = self.colored, None
        if colored is not None:
            full = pygame.surfarray.make_surface(colored.swapaxes(0, 1))
            self.surface = pygame.transform.smoothscale(
                full, (width, round(width * colored.shape[0] / colored.shape[1])))
        return self.surface


class FlowFeed:
    """Optical flow on the camera picture, run on a worker thread (the work itself is in motion_flow.py).

    Takes out the camera's own motion and tracks what is left moving: people, other robots, a rolling ball.
    Needs OpenCV (pip install opencv-python-headless, in this venv). It only runs while switched on, and the
    camera has to be on too (switching flow on turns it on).
    """
    VIEWS = ("movers", "colour", "arrows")

    def __init__(self, camera):
        self.camera = camera
        self.on = False
        self.status = ""          # an error, or "" when running
        self.fps = 0.0
        self.view = 0             # index into VIEWS
        self.motion = motion_flow.MOTION    # px (at motion_flow.FLOW_WIDTH, per FLOW_GAP) of leftover motion that counts
        self.lock = threading.Lock()
        self.result = None        # newest motion_flow result dict
        self.last_result = 0.0    # when it arrived
        self._thread = None
        self._quit = False

    def toggle(self):
        self.on = not self.on
        with self.lock:
            self.result = None
        if self.on:
            self.status = ""
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        return self.on

    def cycle_view(self):
        self.view = (self.view + 1) % len(self.VIEWS)
        return self.VIEWS[self.view]

    def adjust(self, direction):
        """+1 = more sensitive (smaller leftover motion counts as moving), -1 = less. Returns a status line."""
        self.motion = min(8.0, max(0.4, self.motion / 1.25 ** direction))
        return f"Flow sensitivity: leftover motion must exceed {self.motion:.1f} px"

    def snapshot(self):
        """The newest result, or None when there is none or it has gone stale."""
        with self.lock:
            return self.result if time.time() - self.last_result < 1.0 else None

    def _run(self):
        try:
            engine = motion_flow.FlowEngine(motion=self.motion, view=self.view)
        except ImportError:
            self.status, self.on = "needs OpenCV: pip install opencv-python-headless", False
            return
        except Exception as err:
            self.status, self.on = f"flow could not start: {err}", False
            return
        done = 0.0
        was_on = True
        while not self._quit:
            frame, stamp = self.camera.latest_rgb, self.camera.last_frame
            if not self.on:
                if was_on:
                    engine.reset()                 # frames from before a pause don't relate to the ones after
                    was_on = False
                time.sleep(0.05)
                continue
            was_on = True
            if frame is None or stamp == done or time.time() - stamp > 1.0:
                time.sleep(0.01)
                continue
            done = stamp
            engine.motion, engine.view = self.motion, self.view
            try:
                res = engine.process(frame, stamp)
            except Exception as err:
                self.status, self.on = f"flow failed: {err}", False
                return
            if res is not None and self.on:          # a frame in flight when flow was switched off is dropped
                now = time.time()
                with self.lock:
                    previous, self.result, self.last_result = self.last_result, res, now
                if now - previous < 1.0:           # results per second, as the picture sees them
                    self.fps = 0.8 * self.fps + 0.2 / max(now - previous, 1e-3)

    def close(self):
        self._quit = True
        if self._thread is not None:
            self._thread.join(2.0)


class ClipRecorder:
    """Saves the camera stream so it can be replayed offline (flow_replay.py), e.g. to tune optical flow.

    A clip is a folder captures/clip_<date>/ with frame_NNNNN.jpg, frames.csv (arrival time, the dog's pose, the
    drive command, and what the live optical flow saw at that moment) and notes.txt (settings). The JPEGs are
    written by a worker thread; if the disk can't keep up, frames are dropped (and counted) instead of slowing
    the display. It stops by itself after CLIP_MAX_SECONDS.
    """

    HEADER = "frame,time,x,y,z,qx,qy,qz,qw,vx,vy,vyaw,flow_fit,flow_movers\n"

    def __init__(self):
        self.dir = None
        self.accepted = self.dropped = self.written = 0
        self.context = (None, (0.0, 0.0, 0.0))   # latest (pose, (vx, vy, vyaw)); the main loop keeps it current
        self.started = 0.0
        self._queue = None
        self._thread = None

    @property
    def active(self):
        return self.dir is not None

    def over_time(self):
        return self.active and time.time() - self.started > CLIP_MAX_SECONDS

    def start(self, notes=""):
        self.dir = os.path.join(CAPTURE_DIR, time.strftime("clip_%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "notes.txt"), "w") as f:
            f.write(notes + "\nReplay with:  python flow_replay.py " + self.dir + "\n")
        self.accepted = self.dropped = self.written = 0
        self.started = time.time()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._write_loop, args=(self.dir, self._queue), daemon=True)
        self._thread.start()

    def add(self, rgb, stamp, pose, cmd, flow_result):
        """Queue one frame (called from the camera thread)."""
        q = self._queue
        if not self.active or q is None:
            return
        if q.qsize() > 90:                       # ~3 s behind: drop rather than pile up in memory
            self.dropped += 1
            return
        fit, movers = ("", "") if flow_result is None else (
            f"{flow_result['inliers']:.3f}", sum(t["confirmed"] for t in flow_result["tracks"]))
        q.put((self.accepted, rgb, stamp, pose, cmd, fit, movers))
        self.accepted += 1

    def stop(self):
        """Finish the clip; returns (folder, frames saved, frames dropped)."""
        folder, q, thread = self.dir, self._queue, self._thread
        self.dir = None
        if q is not None:
            q.put(None)
        if thread is not None:
            thread.join(10.0)
        return folder, self.written, self.dropped

    def _write_loop(self, folder, q):
        try:
            import cv2
        except ImportError:
            cv2 = None
        with open(os.path.join(folder, "frames.csv"), "w") as csv_file:
            csv_file.write(self.HEADER)
            while True:
                item = q.get()
                if item is None:
                    break
                n, rgb, stamp, pose, cmd, fit, movers = item
                path = os.path.join(folder, f"frame_{n:05d}.jpg")
                try:
                    if cv2 is not None:
                        cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    else:
                        pygame.image.save(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), path)
                except Exception as err:
                    print(f"camera clip: could not save frame {n}: {err}", flush=True)
                    continue
                pose_str = ",".join(f"{v:.6f}" for v in pose) if pose else ",,,,,,"
                csv_file.write(f"{n},{stamp:.6f},{pose_str},{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f},{fit},{movers}\n")
                self.written += 1
                if self.written % 20 == 0:
                    csv_file.flush()


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

    def __init__(self, link):
        self.link = link
        self.sport = link.sport
        self.pitch = self.roll = 0.0
        self.active = False       # pose mode on
        self.last_send = 0.0
        self.level_since = None

    @property
    def body_height(self):
        return self.link.body_height

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


def nearest_points(points, pose, limit):
    """At most `limit` points, the ones nearest the dog, so drawing stays fast with a big map."""
    if points is None or pose is None or len(points) <= limit:
        return points
    d2 = np.sum((points[:, :2] - np.array(pose[:2])) ** 2, axis=1)
    return points[np.argpartition(d2, limit)[:limit]]


def drop_behind(points, pose):
    """Only the points in front of the robot (or all of them while its pose is unknown)."""
    if pose is None or not len(points):
        return points
    yaw = yaw_from_quat(*pose[3:])
    ahead = math.cos(yaw) * (points[:, 0] - pose[0]) + math.sin(yaw) * (points[:, 1] - pose[1])
    return points[ahead >= 0.0]


def screen_to_world(u, v, pose, scale, robot_up):
    """Inverse of the top-down view mapping: pixel -> odom (x, y)."""
    rx, ry, yaw = (pose[0], pose[1], yaw_from_quat(*pose[3:])) if pose else (0.0, 0.0, 0.0)
    view_yaw = yaw if robot_up else 0.0
    c = VIEW // 2
    fwd, left = (c - v) / scale, (c - u) / scale
    cs, sn = math.cos(view_yaw), math.sin(view_yaw)
    return rx + cs * fwd - sn * left, ry + sn * fwd + cs * left


def point_colors(pts, pose, by_distance, floor_z=0.0):
    """N x 3 colours: by height, or by distance from the lidar (floor blue, everything above it red -> blue)."""
    if by_distance == 2 and pts.shape[1] >= 7:   # camera colour; grey where the camera hasn't seen the point
        painted = pts[:, 4:7].sum(axis=1) > 0
        return np.where(painted[:, None], pts[:, 4:7], np.array(UNPAINTED)).astype(np.uint8)
    by_distance = 1 if by_distance == 2 else by_distance
    if not by_distance:
        level = np.clip((pts[:, 2] - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
        return COLORMAP[(level * 255).astype(np.int32)]
    origin = np.array(pose[:3]) if pose else np.zeros(3)
    level = np.clip(np.linalg.norm(pts[:, :3] - origin, axis=1) / DIST_MAX, 0, 1)
    colors = DIST_COLORMAP[(level * 255).astype(np.int32)].astype(np.float32)
    # Fade from the floor colour into the distance colour over the first few cm above the floor
    above = np.clip((pts[:, 2] - floor_z - FLOOR_BLEND[0]) / (FLOOR_BLEND[1] - FLOOR_BLEND[0]), 0, 1)[:, None]
    return ((1 - above) * np.array(FLOOR_COLOR, np.float32) + above * colors).astype(np.uint8)


def dot_sizes(pts, pose, base, max_size):
    """Dot width in pixels per point: `base`, growing steeply as a point gets close to the dog (up to `max_size`)."""
    origin = np.array(pose[:3]) if pose else np.zeros(3)
    dist = np.maximum(np.linalg.norm(pts[:, :3] - origin, axis=1), NEAR_DOTS[0])
    grow = np.maximum(NEAR_GROW * (1.0 / dist - 1.0 / NEAR_DOTS[1]), 0.0)
    return np.minimum(base + np.round(grow).astype(np.int32), max_size)


def stamp_dots(surface, u, v, colors, sizes):
    """Draw square dots centred on (u, v), smallest first so the big near ones end up on top.

    Small dots are written into the pixel array in bulk; the few big ones (points right next to the dog) are
    filled as rectangles, which is much cheaper than writing hundreds of pixels each.
    """
    big = sizes > 4
    px = pygame.surfarray.pixels3d(surface)
    width, height = px.shape[:2]
    for size in np.unique(sizes[~big]):
        group = sizes == size
        gu, gv, gc = u[group], v[group], colors[group]
        for du in range(size):
            for dv in range(size):
                x, y = gu + du - size // 2, gv + dv - size // 2
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                px[x[inside], y[inside]] = gc[inside]
    del px
    idx = np.flatnonzero(big)
    idx = idx[np.argsort(sizes[idx])][-4000:]    # the biggest few thousand, smallest first
    for size, x, y, color in zip(sizes[idx].tolist(), u[idx].tolist(), v[idx].tolist(), colors[idx].tolist()):
        surface.fill(color, (x - size // 2, y - size // 2, size, size))


def view_points(points, hide_ground, by_distance, floor_z):
    """The points to draw: ground hidden if asked; in distance colouring the floor is thinned to a regular dot pattern;
    in camera colouring only the points the camera has painted."""
    if by_distance == 2 and points.shape[1] >= 7:
        points = points[points[:, 4:7].sum(axis=1) > 0]
    if hide_ground:
        return points[points[:, 2] >= floor_z + GROUND_Z]
    if by_distance == 1 and len(points):
        floor = points[:, 2] < floor_z + FLOOR_BLEND[0]
        cell = (np.floor(points[:, 0] * 20) + np.floor(points[:, 1] * 20)).astype(np.int64)  # fixed to the world, so it doesn't flicker
        return points[~floor | (cell % FLOOR_KEEP == 0)]
    return points


FACE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def build_mesh(points):
    """Block mesh from map points: N x 7 (x, y, z, intensity, r, g, b) -> dict of faces, or None.

    The points are grouped into MESH_VOXEL blobs, each blob takes the average camera colour of its points (the
    height colour where the camera never saw it), and every blob face with no blob next to it becomes a quad.
    Joined together the quads form one closed-up surface.
    """
    if points is None or len(points) == 0:
        return None
    cells = np.floor(points[:, :3] / MESH_VOXEL).astype(np.int64) + (1 << 20)
    keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    painted = points[:, 4:7].sum(axis=1) > 0
    seen = np.bincount(inverse[painted], minlength=len(uniq))
    rgb = np.stack([np.bincount(inverse[painted], points[painted, 4 + c], minlength=len(uniq)) for c in range(3)], 1)
    height = np.bincount(inverse, points[:, 2], minlength=len(uniq)) / counts
    by_height = COLORMAP[(np.clip((height - Z_MIN) / (Z_MAX - Z_MIN), 0, 1) * 255).astype(np.int32)]
    color = np.where(seen[:, None] > 0, rgb / np.maximum(seen, 1)[:, None], by_height)
    keep = counts >= MESH_MIN_POINTS
    uniq, color = uniq[keep], color[keep].astype(np.uint8)
    if not len(uniq):
        return None
    cell = np.stack([uniq >> 42, (uniq >> 21) & 0x1FFFFF, uniq & 0x1FFFFF], 1) - (1 << 20)
    center = (cell + 0.5) * MESH_VOXEL
    half = MESH_VOXEL / 2
    out = {"centers": [], "normals": [], "corners": [], "colors": []}
    for direction in FACE_DIRS:
        d = np.array(direction)
        neighbour = uniq + int(d[0]) * (1 << 42) + int(d[1]) * (1 << 21) + int(d[2])
        at = np.minimum(np.searchsorted(uniq, neighbour), len(uniq) - 1)
        exposed = uniq[at] != neighbour
        if not exposed.any():
            continue
        axis = int(np.argmax(np.abs(d)))
        a, b = [i for i in range(3) if i != axis]
        offsets = np.zeros((4, 3))
        offsets[:, axis] = half * d[axis]
        offsets[:, a] = half * np.array([-1, 1, 1, -1])
        offsets[:, b] = half * np.array([-1, -1, 1, 1])
        c = center[exposed]
        out["centers"].append(c + half * d)
        out["normals"].append(np.tile(d, (len(c), 1)))
        out["corners"].append((c[:, None, :] + offsets[None]).astype(np.float32))
        out["colors"].append(color[exposed])
    mesh = {k: np.concatenate(v) for k, v in out.items()}
    mesh["centers"] = mesh["centers"].astype(np.float32)
    return mesh


class MeshBuilder:
    """Rebuilds the block mesh on a worker thread so the window doesn't stall while it works."""

    def __init__(self):
        self.mesh = None
        self.busy = False
        self.built = 0.0

    def request(self, points):
        if self.busy or time.time() - self.built < MESH_REBUILD:
            return
        self.busy = True
        threading.Thread(target=self._work, args=(points,), daemon=True).start()

    def _work(self, points):
        try:
            self.mesh = build_mesh(points)
        finally:
            self.built = time.time()
            self.busy = False


def save_mesh(mesh, name="mesh"):
    """Write the block mesh as an ASCII PLY: shared vertices, one coloured quad per face."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, time.strftime(f"{name}_%Y%m%d_%H%M%S.ply"))
    corners = np.round(mesh["corners"].reshape(-1, 3) / 0.001).astype(np.int64)
    verts, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = np.column_stack([np.full(len(mesh["colors"]), 4), inverse.reshape(-1, 4), mesh["colors"]])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, verts * 0.001, fmt="%.3f")
        np.savetxt(f, faces, fmt="%d")
    return path


def draw_mesh(surface, mesh, cam, to_cam, project):
    """Paint the nearest block faces that face the camera, far to near, shaded by how squarely they face it."""
    centers = mesh["centers"]
    away = np.sum(mesh["normals"] * (cam - centers), axis=1) <= 0
    dist2 = np.sum((centers - cam) ** 2, axis=1)
    dist2[away] = np.inf                    # faces turned away from the camera can't be seen
    idx = np.flatnonzero(np.isfinite(dist2))
    if len(idx) > MESH_DRAW:
        idx = idx[np.argpartition(dist2[idx], MESH_DRAW)[:MESH_DRAW]]
    idx = idx[np.argsort(-dist2[idx])]
    if not len(idx):
        return
    cp = (mesh["corners"][idx].reshape(-1, 3) - cam) @ to_cam.T
    u, v = project(cp)
    in_front = (cp[:, 2] > NEAR).reshape(-1, 4).all(axis=1)
    u, v = u.reshape(-1, 4), v.reshape(-1, 4)
    # Lit from the viewer's side (a face looking straight at the camera is brightest) plus a little from above
    toward = (cam - centers[idx]) / np.sqrt(dist2[idx])[:, None]
    shade = np.clip(0.40 + 0.40 * np.sum(mesh["normals"][idx] * toward, axis=1)
                    + 0.20 * np.maximum(mesh["normals"][idx][:, 2], 0), 0.3, 1.0)
    colors = (mesh["colors"][idx] * shade[:, None]).astype(np.uint8)
    visible = in_front & (u.max(axis=1) >= 0) & (u.min(axis=1) < VIEW) & (v.max(axis=1) >= 0) & (v.min(axis=1) < VIEW)
    for k in np.flatnonzero(visible).tolist():
        pygame.draw.polygon(surface, colors[k].tolist(), list(zip(u[k].tolist(), v[k].tolist())))


def draw_cloud(surface, small, points, pose, scale, robot_up, hide_ground, pillar=None, orbit_radius=None,
               nav_overlay=None, by_distance=False, floor_z=0.0):
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
        pts = view_points(points, hide_ground, by_distance, floor_z)
        pts = pts[np.argsort(pts[:, 2])]  # draw tall points last so they stay visible
        dx, dy = pts[:, 0] - rx, pts[:, 1] - ry
        cs, sn = math.cos(view_yaw), math.sin(view_yaw)
        fwd = cs * dx + sn * dy     # screen up
        left = -sn * dx + cs * dy   # screen left
        u = (c - left * scale).astype(np.int32)
        v = (c - fwd * scale).astype(np.int32)
        ok = (u >= 0) & (u < VIEW - 1) & (v >= 0) & (v < VIEW - 1)
        u, v = u[ok], v[ok]
        colors = point_colors(pts[ok], pose, by_distance, floor_z)
        stamp_dots(surface, u, v, colors, np.full(len(u), 2))

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
            nav_overlay=None, look=None, by_distance=False, floor_z=0.0, mount=None, mesh=None):
    """Perspective view from a camera orbiting the robot.

    cam_az is the camera's direction from the robot (pi = behind it when
    following the robot's heading), cam_el its elevation above the ground.

    look=(yaw, pitch) switches to first person: the view is from the dog's camera mount (position, aim and
    field of view of `mount`), turned by `look` (radians, left / up positive).
    """
    surface.fill((12, 12, 16))
    c = VIEW // 2

    rx, ry, rz, yaw = 0.0, 0.0, 0.0, 0.0
    if pose is not None:
        rx, ry, rz = pose[:3]
        yaw = yaw_from_quat(*pose[3:])

    if look is not None:
        mount = mount or CameraMount()
        rot = body_rotation(pose) if pose is not None else np.eye(3)
        cam = np.array([rx, ry, rz]) + rot @ mount.position()
        fwd = rot @ mount.axes(*look)[0]
    else:
        az = cam_az + (yaw if follow else 0.0)
        target = np.array([rx, ry, floor_z + 0.3])
        cam = target + cam_dist * np.array([
            math.cos(cam_el) * math.cos(az), math.cos(cam_el) * math.sin(az), math.sin(cam_el)])
        fwd = (target - cam) / np.linalg.norm(target - cam)
    focal = (VIEW / 2) / math.tan(math.radians(mount.fov) / 2) if look is not None else FOCAL
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    to_cam = np.stack([right, up, fwd])  # rows: camera x (right), y (up), z (depth)

    def project(cam_pts):
        return c + focal * cam_pts[..., 0] / cam_pts[..., 2], c - focal * cam_pts[..., 1] / cam_pts[..., 2]

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
        line((gx + k, gy - 10, floor_z), (gx + k, gy + 10, floor_z), (45, 45, 55))
        line((gx - 10, gy + k, floor_z), (gx + 10, gy + k, floor_z), (45, 45, 55))

    if mesh is not None:
        draw_mesh(surface, mesh, cam, to_cam, project)
    elif points is not None and len(points):
        pts = view_points(points, hide_ground, by_distance, floor_z)
        cp = (pts[:, :3] - cam) @ to_cam.T
        ok = cp[:, 2] > NEAR
        pts, cp = pts[ok], cp[ok]
        order = np.argsort(-cp[:, 2])  # far to near, so near points end up on top
        pts, cp = pts[order], cp[order]
        u, v = project(cp)
        u, v = u.astype(np.int32), v.astype(np.int32)
        ok = (u >= 0) & (u < VIEW - 1) & (v >= 0) & (v < VIEW - 1)
        u, v = u[ok], v[ok]
        colors = point_colors(pts[ok], pose, by_distance, floor_z)
        # Only the first-person view makes points that are right next to the dog big
        stamp_dots(surface, u, v, colors,
                   dot_sizes(pts[ok], pose, 2, 30) if look is not None else np.full(len(u), 2))

    # Picked pillar as a magenta cylinder outline, and the lap's path on the ground
    if pillar is not None:
        px_, py_ = pillar.center
        ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)]
        for (ax, ay), (bx, by) in zip(ring, ring[1:]):
            for z in (floor_z, floor_z + po.PILLAR_Z[1]):
                line((px_ + pillar.radius * ax, py_ + pillar.radius * ay, z),
                     (px_ + pillar.radius * bx, py_ + pillar.radius * by, z), (255, 60, 255))
            if orbit_radius:
                line((px_ + orbit_radius * ax, py_ + orbit_radius * ay, floor_z),
                     (px_ + orbit_radius * bx, py_ + orbit_radius * by, floor_z), (130, 50, 130))
        for ax, ay in ring[:-1:6]:
            line((px_ + pillar.radius * ax, py_ + pillar.radius * ay, floor_z),
                 (px_ + pillar.radius * ax, py_ + pillar.radius * ay, floor_z + po.PILLAR_Z[1]), (255, 60, 255))

    # Planned path on the ground (green), waypoints as yellow posts
    if nav_overlay is not None:
        path = nav_overlay["path"] or []
        for a, b in zip(path, path[1:]):
            line((a[0], a[1], floor_z + 0.02), (b[0], b[1], floor_z + 0.02), (60, 230, 90), 2)
        for w in nav_overlay["waypoints"]:
            line((w[0], w[1], floor_z), (w[0], w[1], floor_z + 0.6), (255, 220, 40), 2)
        for pos, vel, threat in nav_overlay.get("tracks", []):
            color = (255, 50, 50) if threat else (255, 150, 30)
            line((pos[0], pos[1], floor_z), (pos[0], pos[1], floor_z + 1.5), color, 3)
            line((pos[0], pos[1], floor_z + 0.05), (pos[0] + vel[0], pos[1] + vel[1], floor_z + 0.05), color, 2)

    if look is not None:
        return  # the camera is the robot, so there's no marker to draw

    # Robot marker: triangle at body height pointing along its heading, plus a post to the ground
    cs, sn = math.cos(yaw), math.sin(yaw)
    tri = [(rx + cs * fx - sn * fy, ry + sn * fx + cs * fy, rz) for fx, fy in
           [(0.35, 0.0), (-0.25, 0.17), (-0.25, -0.17)]]
    for i in range(3):
        line(tri[i], tri[(i + 1) % 3], (255, 255, 255), 2)
    line((rx, ry, rz), (rx, ry, floor_z), (160, 160, 160))


def project_to_camera(points, pose, width, height, mount):
    """Odom-frame points -> pixel columns/rows on the camera picture, and which points are in front of it."""
    body = (points[:, :3] - np.array(pose[:3])) @ body_rotation(pose) - mount.position()
    forward, left, up = mount.axes()
    depth = body @ forward
    ok = depth > 0.25
    focal = (width / 2) / math.tan(math.radians(mount.fov) / 2)
    safe = np.where(ok, depth, 1.0)
    u = width / 2 - focal * (body @ left) / safe
    v = height / 2 - focal * (body @ up) / safe
    return u, v, ok, depth


# Typed commands: name -> (SportClient method, what it does, kind). kind "acrobatic" launches the dog and needs
# room; "stand" and "down" are the R / F key sequences; the rest need the dog standing. Aliases map to these names.
COMMANDS = {
    "jump":    ("FrontJump", "jump forward", "acrobatic"),
    "pounce":  ("FrontPounce", "pounce forward", "acrobatic"),
    "shake":   ("Hello", "shake hands / wave a paw (Unitree's 'Hello')", "trick"),
    "stretch": ("Stretch", "stretch", "trick"),
    "happy":   ("Content", "happy wiggle (Unitree's 'Content')", "trick"),
    "heart":   ("Heart", "make a heart shape", "trick"),
    "scrape":  ("Scrape", "scrape the floor", "trick"),
    "dance1":  ("Dance1", "dance routine 1", "trick"),
    "dance2":  ("Dance2", "dance routine 2", "trick"),
    "sit":     ("Sit", "sit down like a dog", "posture"),
    "rise":    ("RiseSit", "get up from sitting", "posture"),
    "recover": ("RecoveryStand", "get up after a fall", "posture"),
    "stand":   (None, "stand up and balance (same as R)", "stand"),
    "down":    (None, "lie down (same as F)", "down"),
    "stop":    (None, "stop moving and cancel waypoints / laps", "stop"),
    "depth":   (None, "depth map of what the camera sees on / off (Depth Anything)", "ui"),
    "filter":  (None, "drop lidar points the depth map disagrees with, on / off", "ui"),
    "flow":    (None, "optical flow on the camera picture (tracks what moves) on / off", "ui"),
    "clip":    (None, "record a camera clip (for tuning flow offline) on / off (same as 7)", "ui"),
    "behind":  (None, "keep the lidar points behind the dog too, on / off (same as F2)", "ui"),
}
ALIASES = {"hello": "shake", "wave": "shake", "paw": "shake", "hop": "jump", "leap": "jump", "content": "happy",
           "wiggle": "happy", "lie": "down", "liedown": "down", "sitdown": "sit", "up": "stand", "standup": "stand",
           "s": "stop", "halt": "stop"}

# Every key and mouse command, for the side panel and the F1 overlay: (section, [lines of at most 38 characters]).
# Keep it in step with the key handling in main() and the docstring at the top.
HELP = [
    ("DRIVE", ["W / S      forward / backward", "A / D      turn left / right", "Q / E      strafe left / right",
               "Shift      hold for faster speed", "Arrows     lean body (standing)", "Space      stop (and cancel walks)",
               "R          stand up", "F          lie down", "Esc        stop and quit"]),
    ("VIEW", ["V          top-down / 3D / 1st-person", "M          follow heading / fixed",
              "+ / -      zoom (or mouse wheel)", "drag       3D orbit / 1st look around",
              "G          hide / show ground", "H          colour: height/dist/camera",
              "[ / ]      less / more persistence", "N          block mesh view on / off",
              "F2         keep points behind the dog", "F3         on-screen buttons on / off",
              "F11        fullscreen on / off", "F1 / Tab   all commands on / off"]),
    ("LIDAR + CAPTURE", ["L          lidar on / off", "P          save snapshot (.ply)", "O          record frames on / off",
                         "J          save block mesh (.ply)", "K          forget the maps"]),
    ("CAMERA", ["I          camera picture on / off", "Y          camera picture size",
                "1          depth map on / off", "2          depth filter on / off",
                "3          optical flow on / off", "4          flow view: next",
                "5 / 6      flow less / more sensitive",
                "7          record camera clip on / off",
                "/          pick a mount setting", ", / .      change that setting", "\\          reset the mount"]),
    ("PILLAR SCAN", ["Click      pick pillar (top-down)", "C          pick object straight ahead",
                     "Z / X      circle it left / right", "T          tilt scans on / off", "Bksp       clear the pillar"]),
    ("WAYPOINTS", ["R-click    add waypoint (top-down)", "U          remove the last one",
                   "Shift+U    remove all", "Enter      walk them / stop", "B          dodge movers on / off"]),
    ("TERMINAL (type in the shell)", None),
]


def estimate_floor(points, pose):
    """Height of the floor in the points' own frame: the lowest level where a lot of points sit at once, or None.

    The dog's pose height minus its body height says where the floor should be, but over Wi-Fi the cloud and
    the pose don't always share a height reference, which makes the floor look like it floats (and everything
    above it an obstacle). The points themselves know where the floor is: it is the lowest dense layer.
    """
    if points is None or len(points) < 100:
        return None
    near = points[:, :3]
    if pose is not None:
        near = near[np.hypot(near[:, 0] - pose[0], near[:, 1] - pose[1]) < 4.0]
    if len(near) < 100:
        return None
    z = near[:, 2]
    low = np.percentile(z, 1)
    z = z[z < low + 1.5]
    counts, edges = np.histogram(z, bins=np.arange(low, low + 1.5 + 0.03, 0.03))
    first = int(np.argmax(counts >= max(15, 0.25 * counts.max())))    # the lowest bin that is dense enough
    layer = z[(z >= edges[first]) & (z < edges[first] + 0.09)]
    return float(np.median(layer)) if len(layer) else None


def depth_consistency(points, pose, depth_map, mount):
    """Which lidar points agree with the depth map: (keep mask, how many were checked, how many were rejected).

    The depth model gives relative inverse depth (bigger = nearer, scale and offset unknown), so for the points the
    camera can see a line  depth_map = a / range + b  is fitted, dropping outliers as it goes; a point far off that
    line is one the depth map disagrees with. Points the camera can't see are kept, and so is everything when
    the line doesn't hold up (too few points, or it fits less than half of them).
    """
    keep = np.ones(len(points), bool)
    height, width = depth_map.shape
    u, v, ok, forward = project_to_camera(points, pose, width, height, mount)
    judged = np.flatnonzero(ok & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
                            & (forward > DEPTH_RANGE[0]) & (forward < DEPTH_RANGE[1]))
    if len(judged) < DEPTH_MIN_FIT:
        return keep, 0, 0
    inverse = 1.0 / forward[judged]
    seen = depth_map[v[judged].astype(int), u[judged].astype(int)].astype(np.float64)
    use = np.ones(len(judged), bool)
    for _ in range(5):
        a, b = np.polyfit(inverse[use], seen[use], 1)
        resid = seen - (a * inverse + b)
        spread = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use]))) + 1e-6
        use = np.abs(resid) < 2.5 * spread
        if use.sum() < DEPTH_MIN_FIT // 2:
            return keep, 0, 0
    if a <= 0 or use.mean() < 0.5:     # nearer must read as bigger; and the fit has to explain most points
        return keep, 0, 0
    bad = np.abs(seen - (a * inverse + b)) > DEPTH_K * spread
    keep[judged[bad]] = False
    return keep, len(judged), int(bad.sum())


def set_view_size(size):
    """Resize the lidar pane. The drawing code reads VIEW (and what is derived from it) when it draws."""
    global VIEW, FOCAL
    VIEW = size
    FOCAL = (VIEW / 2) / math.tan(math.radians(60) / 2)
    CAMERA_SIZES[-1] = VIEW


def help_lines():
    """HELP flattened into (text, colour) lines, with the terminal commands taken from COMMANDS."""
    out = []
    for section, rows in HELP:
        if out:
            out.append(("", None))
        out.append((section, (150, 200, 255)))
        if rows is None:
            names = [f"{name}*" if kind == "acrobatic" else name for name, (_, _, kind) in COMMANDS.items()]
            rows, row = [], ""
            for name in names + ["help", "quit"]:
                if row and len(row) + len(name) + 1 > 38:
                    rows.append(row)
                    row = ""
                row = f"{row} {name}" if row else name
            rows += [row, "* throws the dog: clear 2 m first"]
        out.extend((row, None) for row in rows)
    return out


def columns_fit(surface, count, x0, top, bottom, col_width, line_height=19):
    """Whether draw_columns can draw `count` lines in that space."""
    rows = max(1, (bottom - top) // line_height)
    return x0 + (-(-count // rows)) * col_width - 16 <= surface.get_width()


def draw_columns(surface, font, lines, x0, top, bottom, col_width, line_height=19):
    """Draw (text, colour) lines top to bottom, moving to the next column of the surface when one is full."""
    x, y = x0, top
    for text, color in lines:
        if y + line_height > bottom:
            x, y = x + col_width, top
            if x + col_width - 16 > surface.get_width():
                return False         # no room left: some lines were not drawn
        if text:
            surface.blit(font.render(text, True, color or (220, 220, 220)), (x, y))
        y += line_height
    return True


class Terminal:
    """Lines typed in the terminal the program was started from, read without blocking.

    The main loop looks every frame. (A thread sitting in input() can't be stopped, and Python aborts at exit if
    one is still blocked on stdin.)
    """

    def __init__(self):
        self.pending = ""
        self.closed = False   # stdin ended (or isn't there), so stop looking

    def lines(self):
        out = []
        try:
            while not self.closed and select.select([0], [], [], 0)[0]:
                data = os.read(0, 4096)
                if not data:
                    self.closed = True
                    break
                self.pending += data.decode(errors="replace")
        except (OSError, ValueError):
            self.closed = True
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            out.append(line)
        return out


def draw_flow(img, small, result):
    """Optical-flow results (motion_flow.py) drawn onto the camera picture `img` (a copy; its own pixels).

    movers view: red tint where something moves by itself, boxes round the tracked movers (orange once confirmed,
    thin yellow while a flicker is still unproven) with id and speed, and a line showing where each is heading.
    flow colour: every pixel's flow, hue = direction, brightness = speed. flow arrows: flow on a grid, grey where
    it matches the camera's own motion and red where it doesn't.
    """
    w, h = result["size"]
    sx, sy = img.get_width() / w, img.get_height() / h
    if result["viz"] is not None:
        viz = pygame.transform.smoothscale(pygame.surfarray.make_surface(result["viz"].swapaxes(0, 1)), img.get_size())
        viz.set_alpha(150)
        img.blit(viz, (0, 0))
    elif result["arrows"] is not None:
        for x, y, fx, fy, moving in result["arrows"].tolist():   # tolist: pygame wants plain Python numbers
            a = (x * sx, y * sy)
            color = (255, 70, 70) if moving else (190, 190, 205)
            pygame.draw.line(img, color, a, (a[0] + 3 * fx * sx, a[1] + 3 * fy * sy), 1)
            pygame.draw.circle(img, color, (int(a[0]), int(a[1])), 2)
    elif result["mask"] is not None:
        tint = np.zeros((h, w, 3), np.uint8)
        tint[..., 0] = result["mask"] * 90
        img.blit(pygame.transform.smoothscale(pygame.surfarray.make_surface(tint.swapaxes(0, 1)), img.get_size()),
                 (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    for tr in result["tracks"]:
        x0, y0, x1, y1 = tr["box"]
        rect = pygame.Rect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
        if not tr["confirmed"]:
            pygame.draw.rect(img, (150, 140, 60), rect, 1)
            continue
        pygame.draw.rect(img, (255, 170, 40), rect, 2)
        centre = rect.center
        ahead = (centre[0] + tr["vel"][0] * img.get_width() * 0.5, centre[1] + tr["vel"][1] * img.get_width() * 0.5)
        pygame.draw.line(img, (255, 255, 255), centre, ahead, 2)          # where it will be in half a second
        pygame.draw.circle(img, (255, 255, 255), (int(ahead[0]), int(ahead[1])), 3)
        img.blit(small.render(f"#{tr['id']}  {tr['speed'] * 100:.0f}%/s", True, (255, 200, 80)),
                 (rect.x + 2, max(0, rect.y - 16)))


def draw_camera(surface, small, camera, points, pose, size_idx, mount, by_distance, hide_ground,
                floor_z=0.0, depth_feed=None, flow_feed=None):
    """The camera picture as a heads-up display with the lidar points drawn on it where they hit.

    It is fixed on screen, so it doesn't move as the first-person view looks around.
    """
    by_distance = 1 if by_distance == 2 else by_distance   # camera colours would vanish on the camera picture
    width = CAMERA_SIZES[size_idx]
    img = camera.scaled(width)
    live = img is not None and time.time() - camera.last_frame < 2.0
    flow_result = flow_feed.snapshot() if flow_feed is not None and flow_feed.on else None
    showing_depth = False
    if depth_feed is not None and depth_feed.on:
        depth_img = depth_feed.scaled(width)      # keeps showing the camera picture until the first depth map is ready
        if depth_img is not None:
            img, showing_depth = depth_img, True
    box = pygame.Rect(0, 0, width, img.get_height() if img else width * 9 // 16)
    if width >= VIEW:
        box.midleft = (0, VIEW // 2)
    else:
        box.topright = (VIEW - 12, 12)

    if img is None:
        pygame.draw.rect(surface, (20, 20, 26), box)
    else:
        img = img.copy()
        if live and pose is not None and points is not None and len(points):
            pts = view_points(points, hide_ground, by_distance, floor_z)
            u, v, ok, depth = project_to_camera(pts, pose, img.get_width(), img.get_height(), mount)
            order = np.argsort(-depth)           # far to near, so near points end up on top
            order = order[ok[order]]
            u, v = u[order].astype(np.int32), v[order].astype(np.int32)
            inside = (u >= 1) & (u < img.get_width() - 2) & (v >= 1) & (v < img.get_height() - 2)
            order, u, v = order[inside], u[inside], v[inside]
            colors = point_colors(pts[order], pose, by_distance, floor_z)
            stamp_dots(img, u, v, colors, dot_sizes(pts[order], pose, 3, max(8, img.get_width() // 20)))
        if flow_result is not None and not showing_depth:     # flow is measured on the camera picture, not the depth one
            draw_flow(img, small, flow_result)
        surface.blit(img, box)
    if not live:
        surface.blit(small.render("waiting for camera..." if img is None else "camera stalled", True,
                                  (255, 200, 90)), (box.x + 8, box.y + 8))
    pygame.draw.rect(surface, (200, 200, 210), box.inflate(2, 2), 1)
    label = (f"depth (Depth Anything V2, near = red)  {depth_feed.fps:.0f}/s" if showing_depth
             else f"camera + lidar  fov {mount.fov:.0f}")
    if depth_feed is not None and depth_feed.status:
        label += f"  [{depth_feed.status[:60]}]"
    surface.blit(small.render(label, True, (200, 200, 210)), (box.x, box.bottom + 4))
    if flow_feed is not None and flow_feed.on:
        if flow_result is None:
            text = f"flow: {flow_feed.VIEWS[flow_feed.view]}  " + (flow_feed.status or "waiting for frames...")
        else:
            movers = sum(t["confirmed"] for t in flow_result["tracks"])
            text = (f"flow: {flow_feed.VIEWS[flow_feed.view]}  {flow_feed.fps:.0f}/s  {movers} moving  "
                    f"fit {flow_result['inliers'] * 100:.0f}%  sens {flow_feed.motion:.1f}")
            if flow_result["note"]:
                text += f"  [{flow_result['note']}]"
        text_img = small.render(text[:100], True, (255, 190, 90))
        surface.blit(text_img, (min(box.x, surface.get_width() - text_img.get_width() - 4), box.bottom + 20))


def main():
    if not [a for a in sys.argv[1:] if a not in ("--raw", "--slam")]:
        print(f"Usage: {sys.argv[0]} <network_interface>   e.g. eth0 (cable)\n"
              f"       {sys.argv[0]} wifi [ap|<dog ip>] [--raw]   (Wi-Fi; ap = on the dog's own hotspot;\n"
              f"                      --raw: try the raw lidar (the dog doesn't send it)  --slam: use the dog's SLAM cloud)")
        sys.exit(1)

    print("WARNING: make sure the area around the robot is clear.")
    args = [a for a in sys.argv[1:] if a not in ("--raw", "--slam")]
    if "--slam" in sys.argv:
        print("--slam starts the dog's own mapping. Every mapping/start OVERWRITES the one map saved on the dog.")
        input("Press Enter to continue, Ctrl+C to cancel ... ")
    try:
        link = transport.WifiLink(args[1] if len(args) > 1 else "ap", raw="--raw" in sys.argv,
                                  slam="--slam" in sys.argv) \
            if args[0] == "wifi" else transport.DdsLink(args[0])
    except transport.LinkError as err:
        sys.exit(str(err))
    sport = link.sport

    feed = LidarFeed(link)
    lean = Lean(link)
    recorder = Recorder(feed)
    clip = ClipRecorder()

    def camera_tap(rgb, now):
        """Every camera frame: save it if a clip is being recorded."""
        if clip.active:
            pose_now, cmd_now = clip.context
            clip.add(rgb, now, pose_now, cmd_now, flow.snapshot() if flow.on else None)

    def toggle_clip():
        """Camera clip on / off (turns the camera on with it); returns a status line."""
        if clip.active:
            folder, saved, dropped = clip.stop()
            print(f"Camera clip saved: {saved} frames in {folder}" + (f" ({dropped} dropped)" if dropped else ""), flush=True)
            return f"Clip saved: {saved} frames" + (f", {dropped} dropped" if dropped else "")
        if not camera.on:
            camera.toggle()
        clip.start("\n".join([
            time.strftime("Camera clip recorded %Y-%m-%d %H:%M:%S"),
            f"link: {getattr(link, 'name', '?')}",
            f"camera picture width on screen: {CAMERA_SIZES[cam_size]}",
            f"optical flow: {'on' if flow.on else 'off'}, view {flow.VIEWS[flow.view]}, sensitivity {flow.motion:.2f}",
            f"motion_flow: width {motion_flow.FLOW_WIDTH}, gap {motion_flow.FLOW_GAP}, preset {motion_flow.FLOW_PRESET}",
            "frames.csv: arrival time of each frame, the dog's pose, the drive command, the live flow's camera fit"
            " and number of confirmed movers (blank when flow was off)."]))
        print(f"Recording camera clip to {clip.dir}", flush=True)
        return f"Clip REC: {os.path.basename(clip.dir)}  (7 stops)"
    camera = CameraFeed(link)
    depth = DepthFeed(camera)
    flow = FlowFeed(camera)
    camera.tap = lambda rgb, now: camera_tap(rgb, now)   # camera_tap is defined below, once the recorder exists
    cam_size = 0
    mount, mount_sel = CameraMount.load(), 0   # camera mount being tuned, and which setting , . change
    view_floor = 0.0      # smoothed floor height measured from the points; used for colouring, the obstacle map...
    floor_known = False   # until the first measurement the pose-based estimate is used for the floor
    floor_frame = -1      # lidar frame the floor was last measured on

    def paint_points(pts, pose):
        """Camera colours for lidar points, from the newest camera picture (none while the camera is off)."""
        img = camera.latest_rgb
        if not camera.on or img is None or time.time() - camera.last_frame > 0.5:
            return None
        height, width = img.shape[:2]
        u, v, ok, _ = project_to_camera(pts, pose, width, height, mount)
        inside = ok & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        rgb = np.zeros((len(pts), 3), np.uint8)
        rgb[inside] = img[v[inside].astype(int), u[inside].astype(int)]
        return rgb, inside

    feed.painter = paint_points

    pygame.init()
    window_size = (VIEW + PANEL + ob.WIDTH, VIEW)    # the lidar pane, the text panel and the button column
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)   # resizable, so the maximize button works
    fullscreen = False
    show_help = False     # F1 / Tab: every command over the lidar pane
    pygame.display.set_caption("Go2 lidar + keyboard control")
    font = pygame.font.SysFont("monospace", 15)
    small = pygame.font.SysFont("monospace", 12)
    view = pygame.Surface((VIEW, VIEW))
    clock = pygame.time.Clock()

    status = "Press R to stand up"
    terminal = Terminal()

    # On-screen buttons (onscreen_buttons.py): every key as a mouse button
    buttons_on = True            # F3 / the "hide" button
    buttons_visible = True       # on AND the window is wide enough for them
    button_groups = ob.default_groups()
    button_items = []            # (Rect, Button) from the last draw, for clicks
    button_scroll = 0
    button_drag = False          # a click began on the buttons: mouse movement must not turn the view
    held_keys = set()            # keys the buttons are holding down (driving, leaning, FAST)
    held_button = None           # the hold button under the mouse right now
    held_repeat_at = 0.0
    armed_id, armed_at = None, 0.0   # QUIT waiting for its confirming second click
    tab_rect = pygame.Rect(0, 0, 0, 0)   # the small "buttons" tab shown while they are hidden
    acrobatics_ok = False         # set once the jump / pounce warning has been confirmed
    print("Type commands here (try: help). The window must have focus for the driving keys, not for these.")
    print("go2> ", end="", flush=True)
    moving = False
    last_send = 0.0
    scale = 40.0          # pixels per metre
    robot_up = True       # top-down: heading up; 3D: camera follows heading
    view_3d = False
    first_person = False  # with view_3d: camera at the dog's body instead of orbiting it
    look_yaw = look_pitch = 0.0  # first-person head turn relative to the dog's heading
    cam_az, cam_el, cam_dist = math.pi, math.radians(35), 6.0  # 3D camera: behind and above
    hide_ground = False
    color_mode = 0        # point colours: 0 height, 1 distance from the lidar, 2 camera colour (H cycles)
    mesh_view = False     # draw the block mesh instead of dots (3D and first-person views)
    mesher = MeshBuilder()
    persist_idx = len(PERSIST_STEPS) - 1   # the whole map
    lidar_on = None       # unknown until frames arrive or L is pressed
    lidar_on_since = 0.0
    last_nudge = 0.0      # when ON was last re-sent because the lidar had gone quiet
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

    depth_filter = True               # with the depth map on, drop lidar points that disagree with it (2 toggles)
    depth_stats = {"checked": 0, "rejected": 0}

    def depth_gate(pts, pose):
        """Keep mask for a lidar frame from the depth map, or None when it can't be used (off, none yet, stale)."""
        if not (depth.on and depth_filter and depth.depth is not None and time.time() - depth.last_frame < 1.0):
            return None
        keep, checked, rejected = depth_consistency(pts, pose, depth.depth, mount)
        depth_stats["checked"] += checked
        depth_stats["rejected"] += rejected
        return keep

    feed.gate = depth_gate

    def toggle_depth_filter():
        nonlocal depth_filter
        depth_filter = not depth_filter
        depth_stats.update(checked=0, rejected=0)
        return f"Depth filter {'ON' if depth_filter else 'off'} (only acts while the depth map is on)"

    def toggle_behind():
        feed.keep_behind = not feed.keep_behind
        return ("Keeping lidar points behind the dog too" if feed.keep_behind
                else "Dropping lidar points behind the dog")

    def toggle_depth():
        """Depth map on / off (turns the camera on with it); returns a status line."""
        if not depth.on and not camera.on:
            camera.toggle()
        on = depth.toggle()
        return ("Depth map ON (first-person view; first use downloads the model)" if on else "Depth map off")

    def toggle_flow():
        """Optical flow on / off (turns the camera on with it); returns a status line."""
        if not flow.on and not camera.on:
            camera.toggle()
        on = flow.toggle()
        return ("Flow ON: see the camera picture (V)"
                if on else "Optical flow off")

    def run_command(line):
        """Do one command typed in the terminal. Runs on the main loop, so it can cancel laps and waypoints first."""
        nonlocal status, moving, running, acrobatics_ok
        words = line.strip().lower().split()
        if not words:
            return
        name = ALIASES.get(words[0], words[0])
        if name in ("help", "?", "commands"):
            for key, (_, what, kind) in COMMANDS.items():
                print(f"  {key:8} {what}" + ("   [needs room!]" if kind == "acrobatic" else ""))
            print("  quit     close the window\n  (aliases: hello/wave -> shake, hop -> jump, ...)")
            return
        if name in ("quit", "exit"):
            running = False
            return
        if name not in COMMANDS:
            print(f"Unknown command '{words[0]}'. Type help for the list.")
            return
        method, what, kind = COMMANDS[name]
        if kind == "ui":
            status = (toggle_depth_filter() if name == "filter" else toggle_behind() if name == "behind"
                      else toggle_flow() if name == "flow" else toggle_clip() if name == "clip" else toggle_depth())
            print(status)
            return
        if kind in ("acrobatic", "trick", "posture") and not lean.standing:
            print(f"The dog isn't standing: type stand first. ({name} not sent)")
            return
        if kind == "acrobatic" and not acrobatics_ok:
            acrobatics_ok = True
            print(f"'{name}' throws the dog forward. Clear about 2 m ahead, keep the remote in hand "
                  f"(L2 + B = damp), then type {name} again to go. (Asked once per run.)")
            return
        end_orbit("command")
        end_nav("command")
        lean.leave()
        sport.StopMove()
        moving = False
        if kind == "stand":
            sport.StandUp()
            time.sleep(1.0)
            sport.BalanceStand()
        elif kind == "down":
            sport.StandDown()
        elif kind == "trick" or kind == "acrobatic" or kind == "posture":
            code = getattr(sport, method)()
            if code not in (None, 0):
                print(f"{name}: the dog answered with error code {code}")
        status = f"Command: {name} - {what}"
        print(f"{name}: sent")

    def button_active(button_id):
        """Whether a button should be lit (its feature is on)."""
        lit = {
            "view": lambda: view_3d, "follow": lambda: robot_up, "help": lambda: show_help,
            "fullscreen": lambda: fullscreen, "ground": lambda: hide_ground, "colour": lambda: color_mode != 0,
            "mesh": lambda: mesh_view, "behind": lambda: feed.keep_behind, "lidar": lambda: lidar_on is not False,
            "record": lambda: recorder.active, "camera": lambda: camera.on, "clip": lambda: clip.active,
            "depth": lambda: depth.on, "depth_filter": lambda: depth_filter, "flow": lambda: flow.on,
            "tilt": lambda: tilt_scans, "go": lambda: navigator is not None, "dodge": lambda: dodge_on,
            "lap_l": lambda: orbit is not None, "lap_r": lambda: orbit is not None,
            "fast": lambda: pygame.K_LSHIFT in held_keys,
        }.get(button_id)
        try:
            return bool(lit()) if lit else False
        except Exception:
            return False

    def post_key(btn):
        """Hand the program the key press this button stands for; the normal key handler does the rest."""
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=btn.key, mod=btn.mod, unicode="", scancode=0))

    def release_buttons():
        nonlocal held_button
        if held_button is not None and not held_button.repeat:
            held_keys.discard(held_button.key)
        held_button = None

    def press_button(btn):
        """A click on an on-screen button."""
        nonlocal status, running, buttons_on, held_button, held_repeat_at, armed_id, armed_at
        clicked = time.time()
        if btn.id == "hide":
            buttons_on = False
            status = "Buttons hidden. F3 or the tab shows them"
            return
        if btn.id == "quit":                              # the one button that ends the program: ask twice
            if armed_id == btn.id and clicked - armed_at < 2.5:
                running = False
            else:
                armed_id, armed_at = btn.id, clicked
                status = "Click QUIT again within 2 s to quit"
            return
        armed_id = None
        if btn.command:                                   # a trick: what the terminal would say goes in the status line
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                run_command(btn.command)
            lines_said = [line.strip() for line in said.getvalue().splitlines() if line.strip()]
            if lines_said:
                status = lines_said[-1]
        elif btn.latch:
            held_keys.symmetric_difference_update({btn.key})
        elif btn.hold:
            held_button = btn
            if btn.repeat:                                # zoom: steps repeat while held
                post_key(btn)
                held_repeat_at = clicked + 0.35
            else:
                held_keys.add(btn.key)
        else:
            post_key(btn)

    while running:
        points, pose, frames, last_frame = feed.latest(PERSIST_STEPS[persist_idx], colors=color_mode == 2 or mesh_view)
        drawn = nearest_points(points, pose, MAX_DRAW)  # what gets drawn, coloured and projected
        if mesh_view and points is not None:
            mesher.request(nearest_points(points, pose, MESH_POINTS))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWFOCUSLOST and (moving or orbit or navigator):
                end_orbit("window lost focus")
                end_nav("window lost focus")
                sport.StopMove()
                moving = False
                status = "Window lost focus - stopped"
                release_buttons()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (
                    buttons_visible and event.pos[0] >= pygame.display.get_surface().get_width() - ob.WIDTH
                    or not buttons_visible and tab_rect.collidepoint(event.pos)):
                if not buttons_visible:                   # the "buttons" tab at the top right
                    buttons_on = True
                    status = ("Buttons shown" if pygame.display.get_surface().get_width() >= MIN_VIEW + PANEL + ob.WIDTH
                              else "Window too narrow: widen it")
                else:
                    button_drag = True
                    clicked_on = ob.hit_test(button_items, event.pos)
                    if clicked_on is not None and (clicked_on.id == "hide" or ob.TOP_H <= event.pos[1]
                                                   < pygame.display.get_surface().get_height() - ob.FOOT_H):
                        press_button(clicked_on)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                release_buttons()
                button_drag = False
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not view_3d
                  and event.pos[0] < VIEW and event.pos[1] < VIEW and orbit is None):
                click = screen_to_world(*event.pos, pose, scale, robot_up)
                pillar, why = po.find_pillar(points, pose[:2] if pose else (0, 0),
                                             yaw_from_quat(*pose[3:]) if pose else 0.0, click)
                status = (f"Pillar picked: {pillar.radius * 2:.2f} m wide" if pillar
                          else f"No pillar: {why}")
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not view_3d
                  and event.pos[0] < VIEW and event.pos[1] < VIEW and navigator is None):
                waypoints.append(screen_to_world(*event.pos, pose, scale, robot_up))
                status = f"Waypoint {len(waypoints)} added - Enter to go"
            elif (event.type == pygame.MOUSEWHEEL and buttons_visible
                  and pygame.mouse.get_pos()[0] >= pygame.display.get_surface().get_width() - ob.WIDTH):
                column = pygame.Rect(pygame.display.get_surface().get_width() - ob.WIDTH, 0, ob.WIDTH,
                                     pygame.display.get_surface().get_height())
                button_scroll = min(ob.max_scroll(button_groups, column), max(0, button_scroll - event.y * 36))
            elif event.type == pygame.MOUSEWHEEL:
                if first_person:
                    pass
                elif view_3d:
                    cam_dist = min(40.0, max(1.0, cam_dist / (1.15 ** event.y)))
                else:
                    scale = min(200.0, max(10.0, scale * (1.15 ** event.y)))
            elif event.type == pygame.MOUSEMOTION and event.buttons[0] and first_person and not button_drag:
                look_yaw = (look_yaw - event.rel[0] * 0.005 + math.pi) % (2 * math.pi) - math.pi
                look_pitch = min(math.radians(80), max(math.radians(-80), look_pitch - event.rel[1] * 0.005))
            elif event.type == pygame.MOUSEMOTION and event.buttons[0] and view_3d and not button_drag:
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
                elif event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        window_size = pygame.display.get_surface().get_size()
                        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        pygame.display.set_mode(window_size, pygame.RESIZABLE)
                    status = f"Fullscreen {'ON' if fullscreen else 'off'} (F11)"
                elif event.key == pygame.K_F2:
                    status = toggle_behind()
                elif event.key == pygame.K_F3:
                    buttons_on = not buttons_on
                    status = ("Buttons " + ("shown" if buttons_on else "hidden") + " (F3)"
                              + ("" if pygame.display.get_surface().get_width() >= MIN_VIEW + PANEL + ob.WIDTH or not buttons_on
                                 else ": window too narrow"))
                elif event.key in (pygame.K_F1, pygame.K_TAB):
                    show_help = not show_help
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
                    feed.clear()
                    status = "Maps cleared"
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
                        status = f"Saved {save_ply(points[:, :4])}"
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
                elif event.key == pygame.K_h:
                    color_mode = (color_mode + 1) % 3
                    status = f"Colour: {['height', 'distance', 'camera'][color_mode]}" + (
                        " (turn the camera on with I to paint)" if color_mode == 2 and not camera.on else "")
                elif event.key == pygame.K_n:
                    mesh_view = not mesh_view
                    status = f"Block mesh view {'ON' if mesh_view else 'off'}" + (
                        " (shown in the 3D and first-person views)" if mesh_view and not view_3d else "")
                elif event.key == pygame.K_j:
                    mesh = build_mesh(nearest_points(feed.latest(math.inf, colors=True)[0], pose, MESH_POINTS))
                    status = f"Saved {save_mesh(mesh)}" if mesh is not None else "Nothing to mesh yet"
                elif event.key == pygame.K_y:
                    cam_size = (cam_size + 1) % len(CAMERA_SIZES)
                elif event.key == pygame.K_SLASH:
                    mount_sel = (mount_sel + 1) % len(CameraMount.FIELDS)
                    status = f"Camera mount: , . change {CameraMount.FIELDS[mount_sel][0]}"
                elif event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                    mount.adjust(mount_sel, 1 if event.key == pygame.K_PERIOD else -1)
                    mount.save()
                    name = CameraMount.FIELDS[mount_sel][0]
                    status = f"Camera mount {name} = {getattr(mount, name):g}"
                elif event.key == pygame.K_BACKSLASH:
                    mount = CameraMount()
                    mount.save()
                    status = "Camera mount reset to defaults"
                elif event.key == pygame.K_1:
                    status = toggle_depth()
                elif event.key == pygame.K_2:
                    status = toggle_depth_filter()
                elif event.key == pygame.K_3:
                    status = toggle_flow()
                elif event.key == pygame.K_4:
                    status = f"Flow view: {flow.cycle_view()}"
                elif event.key in (pygame.K_5, pygame.K_6):
                    status = flow.adjust(1 if event.key == pygame.K_6 else -1)
                elif event.key == pygame.K_7:
                    status = toggle_clip()
                elif event.key == pygame.K_i:
                    status = f"Camera picture {'ON' if camera.toggle() else 'off'} (shown in first-person view)"
                elif event.key == pygame.K_LEFTBRACKET:
                    persist_idx = max(0, persist_idx - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    persist_idx = min(len(PERSIST_STEPS) - 1, persist_idx + 1)
                elif event.key == pygame.K_m:
                    robot_up = not robot_up
                elif event.key == pygame.K_v:
                    if not view_3d:
                        view_3d = True
                    elif not first_person:
                        first_person = True
                        look_yaw = look_pitch = 0.0
                    else:
                        view_3d = first_person = False
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    if first_person:
                        pass
                    elif view_3d:
                        cam_dist = max(1.0, cam_dist / 1.25)
                    else:
                        scale = min(200.0, scale * 1.25)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if first_person:
                        pass
                    elif view_3d:
                        cam_dist = min(40.0, cam_dist * 1.25)
                    else:
                        scale = max(10.0, scale / 1.25)
                elif event.key == pygame.K_l:
                    lidar_on = not lidar_on
                    link.set_lidar(lidar_on)
                    if lidar_on:
                        lidar_on_since = time.time()
                        status = "Lidar ON - spinning up (~10 s)"
                    else:
                        status = "Lidar OFF - points frozen"

        for line in terminal.lines():
            run_command(line)
            print("go2> ", end="", flush=True)

        # Driving
        if held_button is not None:                       # let go if the pointer slid off the button or the mouse came up
            mouse_now = pygame.mouse.get_pos()
            if not pygame.mouse.get_pressed()[0] or not any(r.collidepoint(mouse_now) and b is held_button
                                                            for r, b in button_items):
                release_buttons()
            elif held_button.repeat and time.time() >= held_repeat_at:
                post_key(held_button)
                held_repeat_at = time.time() + 0.12
        keys = ob.VirtualKeys(pygame.key.get_pressed(), held_keys)
        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        vx_max, vy_max, vyaw_max = FAST if fast else NORMAL
        vx = vx_max * (keys[pygame.K_w] - keys[pygame.K_s])
        vy = vy_max * (keys[pygame.K_q] - keys[pygame.K_e])
        vyaw = vyaw_max * (keys[pygame.K_a] - keys[pygame.K_d])

        arrow_pitch = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        arrow_roll = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        now = time.time()
        dt = clock.get_time() / 1000.0

        # Floor height in odom z. Measured from the points (smoothed), because the pose height minus the body
        # height can be off from where the cloud puts the floor, and then the floor floats and reads as obstacles.
        pose_floor = pose[2] - lean.body_height if pose is not None and lean.body_height > 0 else 0.0
        if frames != floor_frame and pose is not None:
            floor_frame = frames
            seen = estimate_floor(drawn, pose)
            if seen is not None and seen < pose[2] - 0.1:      # the floor is below the dog's body
                view_floor = seen if not floor_known else view_floor + 0.1 * (seen - view_floor)
                floor_known = True
        floor_z = view_floor if floor_known else pose_floor
        feed.floor_z = view_floor if floor_known else None
        if pose is not None and floor_known:
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
        clip.context = (pose, (vx, vy, vyaw))
        if clip.over_time():
            status = f"Time limit ({CLIP_MAX_SECONDS} s). " + toggle_clip()

        if now - hz_time >= 1.0:
            hz = (frames - hz_frames) / (now - hz_time)
            hz_frames, hz_time = frames, now

        # Draw. The window may have been resized, maximized or made fullscreen: the lidar pane is square and
        # as big as fits, and the panel gets the rest.
        screen = pygame.display.get_surface()
        win_w, win_h = screen.get_size()
        buttons_visible = buttons_on and win_w >= MIN_VIEW + PANEL + ob.WIDTH      # too narrow: they step aside
        button_w = ob.WIDTH if buttons_visible else 0
        want = max(MIN_VIEW, min(win_h, win_w - PANEL - button_w, MAX_VIEW))
        if want != VIEW:
            set_view_size(want)
            view = pygame.Surface((VIEW, VIEW))
        shown = drawn
        if scan is not None and len(scan.view()):
            shown = scan.view() if drawn is None else np.concatenate(
                [drawn, np.pad(scan.view(), ((0, 0), (0, drawn.shape[1] - scan.view().shape[1])))])
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
                    nav_overlay, (look_yaw, look_pitch) if first_person else None, color_mode, view_floor, mount,
                    mesher.mesh if mesh_view else None)
        else:
            draw_cloud(view, small, shown, pose, scale, robot_up, hide_ground, pillar, orbit_radius, nav_overlay,
                       by_distance=color_mode, floor_z=view_floor)
        if camera.on and first_person:
            draw_camera(view, small, camera, drawn, pose, cam_size, mount, color_mode, hide_ground,
                        view_floor, depth, flow)
        screen.fill((25, 25, 30))
        screen.blit(view, (0, 0))
        if show_help:
            veil = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
            veil.fill((10, 10, 14, 225))
            screen.blit(veil, (0, 0))
            screen.blit(font.render("ALL COMMANDS  (F1 or Tab closes this)", True, (255, 220, 120)), (16, 8))
            draw_columns(screen, font, help_lines(), 16, 36, VIEW, 360)

        stale = points is None or now - last_frame > 2.0
        if lidar_on is None and not stale:
            lidar_on = True
        if stale and lidar_on is not False and now - max(last_nudge, lidar_on_since) > LIDAR_NUDGE:
            link.set_lidar(True)      # the dog may have switched it off during a move or pose
            last_nudge = now
            status = "Lidar went quiet - sent ON again (it takes ~10 s to spin up)"
        if lidar_on is False:
            lidar_line = ("lidar OFF - points frozen", (255, 200, 90))
        elif stale and now - lidar_on_since < LIDAR_SPINUP:
            lidar_line = ("lidar spinning up...", (255, 200, 90))
        elif stale:
            lidar_line = ("NO LIDAR DATA - press L", (255, 120, 100))
        else:
            lidar_line = (f"{hz:4.1f} Hz {getattr(link, 'cloud_source', '')}".rstrip(), None)
        lines = [
            ("F1 all commands   F11 fullscreen", (255, 220, 120)),
            ("", None),
            ("DRIVE", (150, 200, 255)),
            (f"vx   {vx:+.2f} m/s", None),
            (f"vy   {vy:+.2f} m/s", None),
            (f"vyaw {vyaw:+.2f} rad/s", None),
            (f"speed: {'FAST' if fast else 'normal'}", None),
            (f"lean p{math.degrees(lean.pitch):+.0f}° r{math.degrees(lean.roll):+.0f}°"
             f"{'  POSE' if lean.active else ''}", None),
            ("", None),
            ("LIDAR", (150, 200, 255)),
            (f"{lidar_line[0]}  {0 if points is None else len(points)} points", lidar_line[1]),
            (f"view: first-person  look {math.degrees(look_yaw):+.0f}° {math.degrees(look_pitch):+.0f}°"
             if first_person else
             f"view: 3D {'follow' if robot_up else 'fixed'}  {cam_dist:.1f} m away" if view_3d else
             f"view: top {'robot-up' if robot_up else 'world-up'}  {scale:.0f} px/m", None),
            (f"behind the dog: {'kept (F2)' if feed.keep_behind else 'dropped (F2)'}", None),
            (f"ground: {'hidden' if hide_ground else 'shown'}   "
             f"persist: {'map' if PERSIST_STEPS[persist_idx] == math.inf else f'{PERSIST_STEPS[persist_idx]:g} s'}", None),
            (f"colour: {['height', 'distance from lidar', 'camera'][color_mode]}   I camera: "
             f"{'ON' if camera.on else 'off'}", None),
            ("mount: " + " ".join(f"{'>' if i == mount_sel else ''}{n} {getattr(mount, n):.{1 if n in ('fov', 'yaw', 'pitch') else 2}f}"
                                  for i, (n, *_) in enumerate(CameraMount.FIELDS[:3])), None),
            ("       " + " ".join(f"{'>' if i + 3 == mount_sel else ''}{n} {getattr(mount, n):.{1 if n in ('fov', 'yaw', 'pitch') else 2}f}"
                                  for i, (n, *_) in enumerate(CameraMount.FIELDS[3:])), None),
            (f"REC {recorder.count} frames" if recorder.active else "not recording",
             (255, 80, 80) if recorder.active else None),
            *([(f"7 CLIP REC {clip.written}/{clip.accepted} frames" + (f"  {clip.dropped} dropped" if clip.dropped else ""),
                (255, 80, 80))] if clip.active else []),      # only while recording: the panel is short of room
            ("", None),
            ("PILLAR", (150, 200, 255)),
            ((f"LAP {orbit.progress * 100:3.0f}%  {orbit.phase}  r={orbit.radius:.1f} m" if orbit else
              f"pillar {pillar.radius * 2:.2f} m wide  tilt scans {'on' if tilt_scans else 'off'}" if pillar else
              f"no pillar  tilt scans {'on' if tilt_scans else 'off'}"),
             (255, 60, 255) if orbit else None),
            ("", None),
            ("WAYPOINTS", (150, 200, 255)),
            (f"WALKING to {navigator.index + 1}/{len(waypoints)}" if navigator else
             f"{len(waypoints)} waypoint(s)", (255, 220, 40) if navigator else None),
            ("", None),
            (f"1 depth map: {'ON' if depth.on else 'off'}" + (f"  {depth.status[:22]}" if depth.status else ""), None),
            (f"2 depth filter: {'ON' if depth_filter else 'off'}" + (
                f"  dropped {100 * depth_stats['rejected'] / depth_stats['checked']:.0f}% of "
                f"{depth_stats['checked']} checked" if depth.on and depth_filter and depth_stats["checked"] else ""),
             None),
            (f"3 optical flow: {'ON' if flow.on else 'off'}" + (f"  {flow.status[:22]}" if flow.status else "") + (
                f"  {flow.VIEWS[flow.view]}  sens {flow.motion:.1f}" if flow.on and not flow.status else ""), None),
            ("DODGING!" if dodge_cmd is not None else
             f"B dodge movers: {'ON' if dodge_on else 'off'}",
             (255, 50, 50) if dodge_cmd is not None else (255, 150, 30) if dodge_on else None),
        ]
        if hasattr(link, "diagnosis"):      # Wi-Fi: which stage (map message, points, pose) has gone quiet
            lines.append((link.diagnosis(), None))
        if pose is not None:
            lines.append((f"pos ({pose[0]:+.2f}, {pose[1]:+.2f}) yaw {math.degrees(yaw_from_quat(*pose[3:])):+.0f}°", None))
            lines.append((f"z {pose[2]:+.2f}  body {lean.body_height:.2f}  floor {floor_z:+.2f}"
                          f"{' (measured)' if floor_known else ' (pose)'}", None))
        # Status may be a long path; wrap it under the live values
        lines.append(("", None))
        lines.extend((status[i:i + 38], (255, 220, 120)) for i in range(0, len(status), 38))
        # The full command list follows when the window is big enough to show all of it (F1 shows it otherwise)
        legend_y = win_h - 40
        everything = lines + [("", None)] + help_lines()
        text_area = screen.subsurface((0, 0, win_w - button_w, win_h))    # the text panel keeps clear of the buttons
        if columns_fit(text_area, len(everything), VIEW + 16, 14, legend_y - 8, PANEL):
            lines = everything
        draw_columns(text_area, font, lines, VIEW + 16, 14, legend_y - 8, PANEL)

        # Height colour legend
        for i in range(0 if color_mode == 2 else PANEL - 32):
            color = (DIST_COLORMAP if color_mode == 1 else COLORMAP)[int(i / (PANEL - 33) * 255)]
            pygame.draw.line(screen, color, (VIEW + 16 + i, legend_y), (VIEW + 16 + i, legend_y + 10))
        lo, hi = (0.0, DIST_MAX) if color_mode == 1 else (Z_MIN, Z_MAX)
        if color_mode != 2:
            screen.blit(small.render(f"{lo:.1f} m", True, (180, 180, 180)), (VIEW + 16, legend_y + 14))
            screen.blit(small.render(f"{hi:.1f} m", True, (180, 180, 180)), (VIEW + PANEL - 52, legend_y + 14))
        screen.blit(small.render(["height", "distance (floor light blue)", "camera colour (unseen points hidden)"][color_mode],
                                 True, (180, 180, 180)), (VIEW + 16 if color_mode == 2 else VIEW + PANEL // 2 - 50, legend_y + 14))

        if buttons_visible:
            column = pygame.Rect(win_w - ob.WIDTH, 0, ob.WIDTH, win_h)
            button_scroll = min(button_scroll, ob.max_scroll(button_groups, column))
            button_items = ob.draw(screen, small, column, button_groups, button_active, pygame.mouse.get_pos(),
                                   held_button, armed_id if time.time() - armed_at < 2.5 else None, button_scroll)
            tab_rect = pygame.Rect(0, 0, 0, 0)
        else:
            button_items = []
            tab_rect = pygame.Rect(win_w - 84, 0, 84, 22)               # click to bring the buttons back
            pygame.draw.rect(screen, (48, 50, 64), tab_rect, border_radius=4)
            pygame.draw.rect(screen, (96, 98, 118), tab_rect, 1, border_radius=4)
            screen.blit(small.render("buttons F3", True, (228, 228, 235)), (tab_rect.x + 8, tab_rect.y + 5))

        pygame.display.flip()
        clock.tick(30)

    end_orbit("quit")
    end_nav("quit")
    sport.StopMove()
    lean.leave()
    if lidar_on is False:
        link.set_lidar(True)
    if recorder.active:
        print(f"Saved {recorder.count} frames to {recorder.stop()}")
    if clip.active:
        folder, saved, dropped = clip.stop()
        print(f"Saved camera clip: {saved} frames to {folder}" + (f" ({dropped} dropped)" if dropped else ""))
    depth_stopped = depth.close()
    flow.close()
    link.close()
    pygame.quit()
    if not depth_stopped:
        # Quit while the depth model was still loading: letting Python shut down under a half-started CUDA
        # thread aborts the process, so leave directly (everything above has already been cleaned up)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

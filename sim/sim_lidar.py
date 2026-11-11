"""Simulated Go2 lidar for unitree_mujoco.

Casts rays from where the Go2's utlidar sits on the body and publishes what they hit on the topics the real
dog uses, so lidar_control.py / navigation.py / dodge.py / pillar_orbit.py can run against the sim:

  rt/utlidar/cloud_deskewed   sensor_msgs PointCloud2_   x y z intensity (float32), odom (= sim world) frame, ~12 Hz
  rt/utlidar/robot_pose       geometry_msgs PoseStamped_ base_link pose in the same frame, ~20 Hz
  rt/utlidar/switch           std_msgs String_            "ON" / "OFF" (listened to, like the dog; starts ON)

Geometry follows the Go2 URDF: lidar at (0.28945, 0, -0.046825) in the body, pitched 2.8782 rad, so its
hemisphere points ~15 deg forward of straight down and sees only up to ~15 deg above horizontal ahead.
Only world geoms (MuJoCo geom group 0) are hit, so the robot never scans its own body.

This is an approximation tuned to what was measured on the real dog (about 1300-1700 returns per frame), not a
model of the L1's scan pattern or noise: each frame jitters the ray grid so coverage fills in over time, and
range noise is plain Gaussian.
"""
import threading
import time

import mujoco
import numpy as np

CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"
POSE_TOPIC = "rt/utlidar/robot_pose"
SWITCH_TOPIC = "rt/utlidar/switch"

MOUNT_POS = np.array([0.28945, 0.0, -0.046825])   # lidar in base_link, m
MOUNT_PITCH = 2.8782                               # rad about y (URDF radar_joint)


def cap_dds_message_size(limit="1400B"):
    """Make CycloneDDS send small UDP datagrams.  Call BEFORE ChannelFactoryInitialize.

    On this WSL2 setup (loopback interface) datagrams of roughly 11 KB and up never arrive, so a lidar frame
    (~27 KB) is silently lost while small topics like robot_pose work.  Capping the sender's MaxMessageSize
    makes DDS fragment the sample itself; receivers can keep their default settings.
    """
    import unitree_sdk2py.core.channel as channel
    import unitree_sdk2py.core.channel_config as channel_config

    cfg = channel_config.ChannelConfigHasInterface
    if "MaxMessageSize" not in cfg:
        cfg = cfg.replace("</Interfaces>", "</Interfaces>\n<MaxMessageSize>%s</MaxMessageSize>" % limit, 1)
    channel_config.ChannelConfigHasInterface = channel.ChannelConfigHasInterface = cfg


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class SimLidar:
    def __init__(self, mj_model, mj_data, locker, base_body="base_link", cloud_hz=12.0, pose_hz=20.0,
                 n_azimuth=72, n_polar=24, max_range=15.0, min_range=0.1, noise=0.01, frame_id="odom"):
        self.m, self.d, self.locker = mj_model, mj_data, locker
        self.base_id = mj_model.body(base_body).id
        self.cloud_dt, self.pose_dt = 1.0 / cloud_hz, 1.0 / pose_hz
        self.n_az, self.n_pol = n_azimuth, n_polar
        self.max_range, self.min_range, self.noise = max_range, min_range, noise
        self.frame_id = frame_id
        self.on = True
        self.R_mount = _rot_y(MOUNT_PITCH)
        self.rng = np.random.default_rng(0)
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # world only
        self.snapshot = mujoco.MjData(mj_model)
        self.stats = dict(frames=0, last_points=0)

    # -- ray pattern: hemisphere about the lidar's +z axis, jittered per frame --------------------------------
    def _local_dirs(self):
        az = (np.arange(self.n_az)[None, :] + self.rng.random((self.n_pol, self.n_az))) / self.n_az * 2 * np.pi
        # polar angle from the axis: 0 (straight out of the dome) .. 90 deg (horizon of the dome)
        pol = (np.arange(self.n_pol)[:, None] + self.rng.random((self.n_pol, self.n_az))) / self.n_pol * (np.pi / 2)
        az, pol = az.ravel(), pol.ravel()
        return np.stack([np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)], axis=1)

    def scan(self):
        """One lidar frame from the current sim state -> (N x 4 float32 x y z intensity in the odom frame)."""
        with self.locker:
            mujoco.mj_copyData(self.snapshot, self.m, self.d)
        d = self.snapshot
        R_base = d.xmat[self.base_id].reshape(3, 3)
        origin = d.xpos[self.base_id] + R_base @ MOUNT_POS
        dirs_w = (self._local_dirs() @ self.R_mount.T) @ R_base.T          # world-frame unit vectors
        n = len(dirs_w)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        mujoco.mj_multiRay(self.m, d, origin, dirs_w.ravel(), self.geomgroup, 1, -1, geomid, dist, None, n,
                           self.max_range)
        hit = (geomid >= 0) & (dist > self.min_range) & (dist < self.max_range)
        r = dist[hit] + self.rng.normal(0.0, self.noise, hit.sum())
        pts = origin + dirs_w[hit] * r[:, None]
        out = np.empty((len(pts), 4), np.float32)
        out[:, :3] = pts
        out[:, 3] = 100.0
        return out

    # -- DDS ------------------------------------------------------------------------------------------------------
    def start(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self.cloud_pub = ChannelPublisher(CLOUD_TOPIC, PointCloud2_)
        self.cloud_pub.Init()
        self.pose_pub = ChannelPublisher(POSE_TOPIC, PoseStamped_)
        self.pose_pub.Init()

        def on_switch(msg):
            self.on = str(msg.data).strip().upper() != "OFF"
            print("[lidar]", "ON" if self.on else "OFF", flush=True)

        self.switch_sub = ChannelSubscriber(SWITCH_TOPIC, String_)
        self.switch_sub.Init(on_switch, 10)
        threading.Thread(target=self._run, daemon=True).start()
        print("[lidar] publishing %s and %s" % (CLOUD_TOPIC, POSE_TOPIC), flush=True)

    def _stamp(self, header):
        now = time.time()
        header.stamp.sec, header.stamp.nanosec = int(now), int((now % 1) * 1e9)
        header.frame_id = self.frame_id

    def _publish_cloud(self, pts):
        from unitree_sdk2py.idl.default import std_msgs_msg_dds__Header_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_

        header = std_msgs_msg_dds__Header_()
        self._stamp(header)
        fields = [PointField_(name, 4 * i, 7, 1) for i, name in enumerate(("x", "y", "z", "intensity"))]  # 7 = FLOAT32
        data = list(pts.astype(np.float32).tobytes())
        self.cloud_pub.Write(PointCloud2_(header, 1, len(pts), fields, False, 16, 16 * len(pts), data, True))

    def _publish_pose(self):
        from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__PoseStamped_

        with self.locker:
            pos, quat = self.d.qpos[0:3].copy(), self.d.qpos[3:7].copy()    # free joint: x y z, w x y z
        msg = geometry_msgs_msg_dds__PoseStamped_()
        self._stamp(msg.header)
        p, q = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = map(float, pos)
        q.w, q.x, q.y, q.z = map(float, quat)
        self.pose_pub.Write(msg)

    def _run(self):
        next_cloud = next_pose = next_log = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now >= next_log:
                next_log = now + 5.0
                print("[lidar] %s, %d frames sent, last frame %d points" % (
                    "on" if self.on else "off", self.stats["frames"], self.stats["last_points"]), flush=True)
            if now >= next_pose:
                next_pose = now + self.pose_dt
                self._publish_pose()
            if now >= next_cloud:
                next_cloud = now + self.cloud_dt
                if self.on:
                    pts = self.scan()
                    self.stats["frames"] += 1
                    self.stats["last_points"] = len(pts)
                    self._publish_cloud(pts)
            time.sleep(0.003)

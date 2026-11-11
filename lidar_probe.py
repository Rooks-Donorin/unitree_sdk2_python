"""Print what the dog's lidar topics are doing, once a second. Read-only: nothing is sent to the dog.

Usage:  .venv/bin/python lidar_probe.py eth0

Run it with the dog in the pose you are asking about (e.g. sitting), and close lidar_control.py first or run
it in a second terminal. Per second it shows, for each topic, how many messages arrived and how many real
(non-zero) points they held, plus the dog's pose (z, and its pitch from the quaternion) and the lidar switch.
"""
import math
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

CLOUDS = ("rt/utlidar/cloud_deskewed", "rt/utlidar/cloud")
lock = threading.Lock()
stats = {t: [0, 0] for t in CLOUDS}      # messages, real points since the last print
pose = [None]
switch = ["(none seen)"]


def cloud_handler(topic):
    def handler(msg):
        raw = np.array(msg.data, dtype=np.uint8)
        flat = raw.view(np.float32).reshape(-1, msg.point_step // 4)
        real = int(np.count_nonzero(np.any(flat[:, :3] != 0, axis=1)))
        with lock:
            stats[topic][0] += 1
            stats[topic][1] += real
    return handler


def on_pose(msg):
    p, q = msg.pose.position, msg.pose.orientation
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))
    pose[0] = (p.z, pitch)


def on_switch(msg):
    switch[0] = msg.data


def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <network_interface>   e.g. eth0")
    ChannelFactoryInitialize(0, sys.argv[1])
    subs = []
    for topic in CLOUDS:
        sub = ChannelSubscriber(topic, PointCloud2_)
        sub.Init(cloud_handler(topic), 10)
        subs.append(sub)
    for topic, kind, handler in (("rt/utlidar/robot_pose", PoseStamped_, on_pose),
                                 ("rt/utlidar/switch", String_, on_switch)):
        sub = ChannelSubscriber(topic, kind)
        sub.Init(handler, 10)
        subs.append(sub)
    print("Ctrl+C to stop. Columns: messages/s and real points/s per topic.")
    while True:
        time.sleep(1.0)
        with lock:
            row = {t: tuple(stats[t]) for t in CLOUDS}
            for t in CLOUDS:
                stats[t][:] = [0, 0]
        where = f"z {pose[0][0]:.2f} m  pitch {pose[0][1]:+.0f} deg" if pose[0] else "no pose"
        print("  ".join(f"{t.rsplit('/', 1)[1]}: {m} msg {n} pts" for t, (m, n) in row.items())
              + f"  | {where}  | switch {switch[0]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

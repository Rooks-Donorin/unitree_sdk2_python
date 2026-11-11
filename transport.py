"""How lidar_control.py talks to the dog: DDS over the Ethernet cable, or WebRTC over Wi-Fi.

Both links give lidar_control the same things:
  link.sport           StandUp / StandDown / BalanceStand / StopMove / Move / Euler / Pose
  link.start(on_points, on_pose)
                       on_points(N x 4 float32 x/y/z/intensity, odom frame), on_pose((x, y, z, qx, qy, qz, qw))
  link.body_height     latest body height from the sport mode state (m)
  link.set_lidar(on)   switch the lidar on / off
  link.start_camera(on_frame) / link.stop_camera()
                       on_frame(H x W x 3 uint8 RGB array) for each front-camera picture
  link.close()

DdsLink is what lidar_control always did (CycloneDDS, needs the cable, EDU / X models).
WifiLink is the way the phone app reaches the dog (WebRTC, library unitree_webrtc_connect), so it needs
no cable. The dog accepts ONE Wi-Fi client at a time: close the Unitree app first. Over Wi-Fi the lidar
comes as the voxel-map topic instead of rt/utlidar/cloud_deskewed, and there is no full-rate lowstate.
The Wi-Fi library and the DDS stack don't share an environment (see go2-robot-dog/README.md), so each
link imports its own libraries only when it is used.

Firmware 1.1.15 and newer needs the dog's own AES key for the Wi-Fi handshake: put it in the environment
variable GO2_AES_KEY or in the file .go2_aes_key next to this script (it is git-ignored).
"""
import asyncio
import json
import math
import os
import random
import threading
import time

import numpy as np

CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"  # odom frame, ~12 Hz
POSE_TOPIC = "rt/utlidar/robot_pose"       # robot pose in the same odom frame

CAMERA_POLL = 0.1      # s between camera frame requests over the cable (each JPEG is ~170 KB)
CAMERA_TIMEOUT = 0.2   # s to wait for a camera reply before skipping that frame (a good reply takes 10-45 ms)

WIFI_CLOUD_TOPIC = "rt/utlidar/voxel_map_compressed"
USLAM_CMD = "rt/uslam/client_command"           # takes a plain string, e.g. "mapping/start"
USLAM_LOG = "rt/uslam/server_log"
USLAM_CLOUD = "rt/uslam/frontend/cloud_world_ds"
USLAM_ODOM = "rt/uslam/frontend/odom"
SLAM_DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2-robot-dog", "lab", "slam_decoder.py")
SLAM_WAIT = 12.0  # s after starting the dog's mapping before saying no SLAM cloud has arrived
RAW_WAIT = 8.0  # s of Wi-Fi without a raw frame before saying the dog isn't sending it
WIFI_STATE_TOPIC = "rt/lf/sportmodestate"
SPORT_TOPIC = "rt/api/sport/request"
SWITCH_TOPIC = "rt/utlidar/switch"
DOG_AP_IP = "192.168.12.1"  # the dog's address on its own hotspot
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".go2_aes_key")

API = {"StandUp": 1004, "StandDown": 1005, "BalanceStand": 1002, "StopMove": 1003, "Euler": 1007,
       "Move": 1008, "Pose": 1028, "RecoveryStand": 1006, "Sit": 1009, "RiseSit": 1010, "Hello": 1016,
       "Stretch": 1017, "Content": 1020, "Dance1": 1022, "Dance2": 1023, "Scrape": 1029, "FrontJump": 1031,
       "FrontPounce": 1032, "Heart": 1036}  # same ids as the cable SDK (unitree_sdk2py/go2/sport/sport_api.py)


class LinkError(Exception):
    """The dog could not be reached; the message says what to do about it."""


class DdsLink:
    name = "cable"

    def __init__(self, interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        ChannelFactoryInitialize(0, interface)
        self.sport = SportClient()
        self.sport.SetTimeout(10.0)
        self.sport.Init()
        self.body_height = 0.0
        self._switch = ChannelPublisher("rt/utlidar/switch", String_)
        self._switch.Init()
        self._subs = []
        self._camera = None  # (stop event) while the camera thread runs

    def start_camera(self, on_frame):
        import io

        import pygame
        from unitree_sdk2py.go2.video.video_client import VideoClient

        if self._camera is not None:
            return
        client = VideoClient()
        # A good call answers in ~15 ms (measured over the cable), but now and then a reply is lost and the
        # call then sits out its whole timeout. 3 s of that froze the picture, and two in a row made a 6 s
        # stall, so wait only 0.5 s for a reply and carry on with the next frame.
        client.SetTimeout(CAMERA_TIMEOUT)
        client.Init()
        stop = self._camera = threading.Event()

        def poll():
            while not stop.is_set():
                started = time.time()
                try:
                    code, data = client.GetImageSample()
                    if code == 0 and data:
                        surf = pygame.image.load(io.BytesIO(bytes(data)), "frame.jpg")
                        on_frame(pygame.surfarray.array3d(surf).swapaxes(0, 1))
                except Exception as err:  # one bad frame must not end the camera thread for good
                    print(f"camera frame skipped: {err!r}", flush=True)
                stop.wait(max(0.0, CAMERA_POLL - (time.time() - started)))

        threading.Thread(target=poll, daemon=True).start()

    def stop_camera(self):
        if self._camera is not None:
            self._camera.set()
            self._camera = None

    def start(self, on_points, on_pose):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        def cloud(msg):
            raw = np.array(msg.data, dtype=np.uint8)
            stride = msg.point_step // 4
            flat = raw.view(np.float32).reshape(-1, stride)
            offsets = {f.name: f.offset // 4 for f in msg.fields}
            cols = [offsets["x"], offsets["y"], offsets["z"], offsets.get("intensity", offsets["z"])]
            points = np.ascontiguousarray(flat[:, cols])
            on_points(points[np.any(points[:, :3] != 0, axis=1)])  # drop zero padding (most of each message)

        def pose(msg):
            p, q = msg.pose.position, msg.pose.orientation
            on_pose((p.x, p.y, p.z, q.x, q.y, q.z, q.w))

        def state(msg):
            self.body_height = msg.body_height

        for topic, kind, handler in ((CLOUD_TOPIC, PointCloud2_, cloud), (POSE_TOPIC, PoseStamped_, pose),
                                     ("rt/sportmodestate", SportModeState_, state)):
            sub = ChannelSubscriber(topic, kind)
            sub.Init(handler, 10)
            self._subs.append(sub)

    def set_lidar(self, on):
        from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

        msg = std_msgs_msg_dds__String_()
        msg.data = "ON" if on else "OFF"
        self._switch.Write(msg)

    def close(self):
        self.stop_camera()


def aes_key():
    key = os.environ.get("GO2_AES_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        key = open(KEY_FILE).read().strip()
    return key or None


class WifiSport:
    """The SportClient calls lidar_control uses, sent over the Wi-Fi link.

    Nothing here waits for the dog: the UI loop must not stall on Wi-Fi latency. Move is
    fire-and-forget like the cable SDK's; the rest are sent as requests whose answers are
    only checked for a failure code.
    """

    def __init__(self, link):
        self.link = link

    def Init(self):
        pass

    def SetTimeout(self, seconds):
        pass

    def Move(self, vx, vy, vyaw):
        self.link.post(self.link.send_move, vx, vy, vyaw)

    def StopMove(self):
        self._call("StopMove")

    def StandUp(self):
        self._call("StandUp")

    def StandDown(self):
        self._call("StandDown")

    def BalanceStand(self):
        self._call("BalanceStand")

    def Euler(self, roll, pitch, yaw):
        self._call("Euler", {"x": roll, "y": pitch, "z": yaw})

    def Pose(self, flag):
        self._call("Pose", {"data": flag})

    def _call(self, name, parameter=None):
        self.link.post(self.link.request, name, parameter)


# The tricks and posture changes lidar_control's command line uses; each takes no arguments, like the SportClient ones
for _name in ("RecoveryStand", "Sit", "RiseSit", "Hello", "Stretch", "Content", "Dance1", "Dance2", "Scrape",
              "FrontJump", "FrontPounce", "Heart"):
    setattr(WifiSport, _name, lambda self, _name=_name: self._call(_name))


class WifiLink:
    name = "Wi-Fi"

    def __init__(self, target, timeout=25.0, raw=False, slam=False):
        """target: "ap" when this computer is on the dog's own hotspot, or the dog's IP on a shared router.

        raw: also ask the dog for the raw lidar cloud (the cable's rt/utlidar/cloud_deskewed) over Wi-Fi. Nobody
        has confirmed the dog sends it; if it does, it is used instead of the voxel map, and if nothing comes
        the voxel map is used as before.

        slam: start the dog's own SLAM mapping and show its world point cloud (rt/uslam/frontend/cloud_world_ds)
        instead of the voxel map, with the robot pose taken from the same SLAM so the two agree. This OVERWRITES
        the dog's one saved map. The cloud needs libslam.wasm's decoder from go2-robot-dog/lab. Mapping is stopped
        again on close().
        """
        try:
            from unitree_webrtc_connect import unitree_auth as auth
            from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
        except OSError as err:  # sounddevice (imported by the library) is missing the system PortAudio
            raise LinkError(f"{err}. Install it with:  sudo apt install libportaudio2")
        except ImportError as err:
            raise LinkError(f"Wi-Fi needs the unitree_webrtc_connect library ({err}). Make a separate venv: "
                            "python3.10 -m venv .venv_wifi && .venv_wifi/bin/pip install 'aiortc>=1.9.0' "
                            "pycryptodome requests curl_cffi wasmtime lz4 numpy packaging sounddevice pydub "
                            "pygame && .venv_wifi/bin/pip install --no-deps unitree_webrtc_connect==2.2.0 "
                            "(sounddevice also needs the system PortAudio: sudo apt install libportaudio2).")
        self.body_height = 0.0
        self.sport = WifiSport(self)
        self.raw = raw
        self.slam = slam
        self._slam_dec = load_slam_decoder() if slam else None
        self._slam_last = 0.0             # time of the last SLAM cloud frame
        self._slam_odom_last = 0.0        # ...and of the last SLAM pose
        self._slam_shown = False
        self.cloud_source = "voxel map"   # what the view is showing: "voxel map" or "raw lidar"
        self._raw_last = 0.0              # time of the last usable raw frame
        self._raw_started = 0.0
        self._raw_told = False
        self._pose = None
        self._seen = {}                   # when the voxel map, its points and the pose last arrived (diagnosis)
        self._frame = None  # "odom" or "body", once worked out
        self._on_points = self._on_pose = None
        self._on_frame = None
        self._video_hooked = False
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

        key = aes_key()
        if target == "ap":
            self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=key)
            where = f"the dog's own Wi-Fi ({DOG_AP_IP})"
        else:
            self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=target, aes_128_key=key)
            where = f"{target} on the shared router"
        print(f"connecting over Wi-Fi to {where}" + (" with the AES key" if key else "") + " ...", flush=True)
        try:
            self._run(self._conn.connect(), timeout)
        except auth.RobotBusyError:
            raise LinkError("The dog already has a Wi-Fi client: close the Unitree phone app (or any other "
                            "program) and try again.")
        except auth.AesKeyRequiredError:
            raise LinkError("This dog's firmware needs its own AES key over Wi-Fi: set GO2_AES_KEY or put it "
                            f"in {KEY_FILE}.")
        except auth.AesKeyRejectedError:
            raise LinkError(f"The dog rejected the AES key (GO2_AES_KEY or {KEY_FILE}): it belongs to another "
                            "dog, or it is mistyped.")
        except (auth.LocalSignalingPortError, OSError, ValueError):
            raise LinkError(f"Cannot reach the dog at {where}. Is this computer on the dog's network, "
                            "and is the dog on?")
        except (asyncio.TimeoutError, TimeoutError, auth.NoSdpAnswerError, auth.DataChannelTimeoutError):
            raise LinkError(f"The dog did not finish the Wi-Fi handshake within {timeout:.0f} s. Move closer, "
                            "check that the phone app is closed, then try again.")
        print("connected", flush=True)

    # Everything that touches the connection runs on its own thread's event loop
    def _run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def post(self, fn, *args):
        self._loop.call_soon_threadsafe(fn, *args)

    @property
    def _pubsub(self):
        return self._conn.datachannel.pub_sub

    def send_move(self, vx, vy, vyaw):
        self._pubsub.publish_without_callback(SPORT_TOPIC, {
            "header": {"identity": {"id": int(time.time() * 1000) % 2147483648 + random.randint(0, 1000),
                                    "api_id": API["Move"]},
                       "policy": {"priority": 0, "noreply": True}},
            "parameter": json.dumps({"x": vx, "y": vy, "z": vyaw}),
            "binary": [],
        })

    def request(self, name, parameter=None):
        payload = {"api_id": API[name]}
        if parameter is not None:
            payload["parameter"] = parameter
        asyncio.ensure_future(self._request(name, payload))

    async def _request(self, name, payload):
        try:
            answer = await asyncio.wait_for(self._pubsub.publish_request_new(SPORT_TOPIC, payload), 5.0)
            code = answer.get("data", {}).get("header", {}).get("status", {}).get("code", -1)
        except Exception:
            code = -1
        if code != 0:
            print(f"Wi-Fi {name}: {'no answer' if code == -1 else f'code {code}'}", flush=True)

    def start(self, on_points, on_pose):
        self._on_points, self._on_pose = on_points, on_pose
        self._run(self._subscribe(), 10.0)

    async def _subscribe(self):
        ps = self._pubsub
        ps.subscribe(POSE_TOPIC, self._pose_msg)
        ps.subscribe(WIFI_STATE_TOPIC, self._state_msg)
        # native returns points in metres (the default returns voxel indices); and without turning
        # traffic saving off the dog throttles the cloud away
        self._conn.datachannel.set_decoder("native")
        await self._conn.datachannel.disableTrafficSaving(True)
        ps.subscribe(WIFI_CLOUD_TOPIC, self._cloud_msg)
        if self.slam:
            ps.subscribe(USLAM_CLOUD, self._slam_msg)
            ps.subscribe(USLAM_ODOM, self._slam_odom_msg)
            ps.subscribe(USLAM_LOG, self._slam_log)
        if self.raw:
            self._raw_started = time.time()
            ps.subscribe(CLOUD_TOPIC, self._raw_msg)
            print(f"asking the dog for the raw lidar ({CLOUD_TOPIC}) over Wi-Fi; the voxel map is the fallback", flush=True)
        ps.publish_without_callback(SWITCH_TOPIC, "ON")
        if self.slam:
            # The mapping frontend does nothing useful with the lidar off, so give it a moment to spin up
            await asyncio.sleep(1.0)
            ps.publish_without_callback(USLAM_CMD, "mapping/start")
            print("→ mapping/start (this overwrites the dog's saved map)", flush=True)
            await asyncio.sleep(0.4)
            ps.publish_without_callback(USLAM_CMD, "mapping/get_status")
            self._loop.call_later(SLAM_WAIT, self._slam_check)

    @staticmethod
    def _parse_pose(d):
        """PoseStamped ({pose: {position, orientation}}) or Odometry ({pose: {pose: {...}}}) -> pose tuple, or None."""
        try:
            pose = d["pose"]
            pose = pose["pose"] if "pose" in pose else pose
            p, q = pose["position"], pose["orientation"]
            return (float(p["x"]), float(p["y"]), float(p["z"]),
                    float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _pose_msg(self, msg):
        pose = self._parse_pose(msg.get("data", {}))
        if pose is None:
            return
        self._seen["pose"] = time.time()
        if self.slam and time.time() - self._slam_odom_last < 2.0:
            return   # the SLAM cloud is in the SLAM's frame, so its own pose is the one that matches
        self._pose = pose
        self._on_pose(pose)

    def _slam_odom_msg(self, msg):
        pose = self._parse_pose(msg.get("data", {}))
        if pose is None:
            if not getattr(self, "_odom_unreadable", False):
                self._odom_unreadable = True
                print(f"SLAM pose message not understood: {str(msg)[:200]}", flush=True)
            return
        self._slam_odom_last = time.time()
        self._pose = pose
        self._on_pose(pose)

    def _slam_msg(self, msg):
        d = msg.get("data", {})
        if not isinstance(d, dict):
            return
        _, frame = self._slam_dec.decode(d.get("data", b""), d)
        if frame is None or not len(frame):
            return
        frame = frame[np.isfinite(frame).all(axis=1)]
        if not len(frame):
            return
        if not self._slam_shown:
            self._slam_shown = True
            print(f"SLAM cloud is arriving ({len(frame)} points in the first frame)", flush=True)
        self._slam_last = time.time()
        self.cloud_source = "slam cloud"
        self._on_points(np.column_stack([frame, np.zeros(len(frame), np.float32)]).astype(np.float32))

    def _slam_log(self, msg):
        try:
            data = msg["data"]
            print("dog SLAM:", str(data["data"] if isinstance(data, dict) else data)[:200], flush=True)
        except (KeyError, TypeError):
            pass

    def _slam_check(self):
        if not self._slam_last:
            print("no SLAM cloud yet - is the dog's mapping running? (see the 'dog SLAM:' lines above; "
                  "showing the voxel map meanwhile)", flush=True)

    def _state_msg(self, msg):
        try:
            self.body_height = float(msg["data"]["body_height"])
        except (KeyError, TypeError, ValueError):
            pass

    def _raw_msg(self, msg):
        if not self._raw_last and not getattr(self, "_raw_shown", False):
            self._raw_shown = True   # what the first raw message looks like, since the format is unconfirmed
            body = msg.get("data") if isinstance(msg, dict) else None
            print(f"raw lidar message arrived: {type(body).__name__} "
                  f"{sorted(body)[:8] if isinstance(body, dict) else ''}", flush=True)
        points = parse_pointcloud2(msg.get("data", {}) if isinstance(msg, dict) else {})
        if points is None or not len(points):
            return
        self._raw_last = time.time()
        self.cloud_source = "raw lidar"
        self._on_points(points)

    def diagnosis(self):
        """Short text: how long ago a voxel-map message, a decoded point set and a pose last arrived."""
        now = time.time()
        age = lambda key: f"{now - self._seen[key]:.0f}s" if key in self._seen else "never"
        return f"last data: map {age('map')}  pts {age('pts')}  pose {age('pose')}"

    def _cloud_msg(self, msg):
        self._seen["map"] = time.time()
        if self.slam and time.time() - self._slam_last < 2.0:
            return   # SLAM frames are coming in (in a different frame from this map), so don't mix them
        if self.raw:
            if time.time() - self._raw_last < 2.0:
                return   # raw frames are coming in, so the voxel map would only mix in coarser copies
            self.cloud_source = "voxel map"
            if not self._raw_last and not self._raw_told and time.time() - self._raw_started > RAW_WAIT:
                self._raw_told = True
                print("no usable raw lidar over Wi-Fi - using the voxel map (the dog doesn't send it this way)",
                      flush=True)
        xyz = decode_voxel_cloud(msg.get("data", {}).get("data", {}))
        if xyz is None or not len(xyz):
            return
        self._seen["pts"] = time.time()
        if self._pose is None:
            return
        px, py, pz, qx, qy, qz, qw = self._pose
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        if self._frame is None and math.hypot(px, py) > 1.5:
            # Once the robot is away from the origin the two frames can be told apart: a sweep in the odom
            # frame is centred near the robot, one in the body frame near (0, 0). Until then they are close
            # enough to treat as odom.
            mid = xyz[:, :2].mean(axis=0)
            self._frame = "odom" if math.dist(mid, (px, py)) < math.hypot(*mid) * 0.6 + 1.5 else "body"
        if self._frame == "body":
            c, s = math.cos(yaw), math.sin(yaw)
            xyz = np.stack([px + c * xyz[:, 0] - s * xyz[:, 1], py + s * xyz[:, 0] + c * xyz[:, 1],
                            pz + xyz[:, 2]], axis=1)
        self._on_points(np.column_stack([xyz, np.zeros(len(xyz), np.float32)]).astype(np.float32))

    def set_lidar(self, on):
        self.post(self._pubsub.publish_without_callback, SWITCH_TOPIC, "ON" if on else "OFF")

    def start_camera(self, on_frame):
        self._on_frame = on_frame
        self.post(self._video, True)

    def stop_camera(self):
        self._on_frame = None
        self.post(self._video, False)

    def _video(self, on):
        video = self._conn.video
        if on and not self._video_hooked:
            video.add_track_callback(self._track)
            self._video_hooked = True
        video.switchVideoChannel(on)

    async def _track(self, track):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            on_frame = self._on_frame
            if on_frame is not None:
                on_frame(frame.to_ndarray(format="rgb24"))

    def close(self):
        if self.slam:
            try:   # leave the dog's mapping the way it was found: not running
                self.post(self._pubsub.publish_without_callback, USLAM_CMD, "mapping/stop")
                time.sleep(0.5)
            except Exception:
                pass
        try:
            self._run(self._conn.disconnect(), 5.0)  # a dangling connection blocks the next client
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def load_slam_decoder():
    """The dog's SLAM-cloud decoder (libslam.wasm, from the go2-robot-dog repo), or a LinkError saying what's missing."""
    if not os.path.exists(SLAM_DECODER):
        raise LinkError(f"--slam needs the SLAM-cloud decoder from the go2-robot-dog repo, expected at {SLAM_DECODER}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("slam_decoder", SLAM_DECODER)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.SlamDecoder()
    except ImportError as err:
        raise LinkError(f"--slam needs the wasmtime package in this environment ({err}): "
                        ".venv_wifi/bin/pip install wasmtime")


def parse_pointcloud2(d):
    """A sensor_msgs/PointCloud2 that arrived as a plain dict -> (N, 4) float32 x/y/z/intensity, or None."""
    try:
        fields = {f["name"]: int(f["offset"]) for f in d["fields"]}
        step = int(d["point_step"])
        data = d["data"]
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        raw = np.frombuffer(bytes(data), np.uint8)
        n = len(raw) // step
        if n == 0 or not all(k in fields for k in ("x", "y", "z")):
            return None
        raw = raw[:n * step].reshape(n, step)
        out = np.zeros((n, 4), np.float32)
        for i, name in enumerate(("x", "y", "z", "intensity")):
            if name in fields:
                out[:, i] = raw[:, fields[name]:fields[name] + 4].copy().view(np.float32).ravel()
        out = out[np.isfinite(out).all(axis=1)]
        return out[np.any(out[:, :3] != 0, axis=1)]   # drop zero padding
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def decode_voxel_cloud(data):
    """The Wi-Fi library's decoded voxel map -> (N, 3) float32 metres, or None.

    The native decoder returns {"points": (N, 3)} in metres; the default one returns voxel indices
    ("positions"), which are scaled here in case it is ever the one in use.
    """
    try:
        pts = data.get("points")
        if pts is None:
            pos, n = data.get("positions"), int(data.get("face_count", 0))
            if pos is None or n <= 0:
                return None
            pts = np.frombuffer(bytes(pos), dtype=np.float32)[:n * 3].reshape(-1, 3)
            pts = pts * float(data.get("resolution", 0.05)) + np.asarray(data.get("origin", (0, 0, 0)), np.float32)
        xyz = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        return np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1)])
    except Exception:
        return None

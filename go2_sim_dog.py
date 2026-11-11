"""Stand-in for the Go2's onboard sport-mode service, for the MuJoCo sim.

unitree_mujoco only exposes raw joint topics (rt/lowstate, rt/lowcmd).  This
program closes the gap: it answers the same `sport` and `motion_switcher` RPC
services the real dog does, so scripts that use SportClient (keyboard_control.py,
navigation.py, ...) can drive the simulated dog unchanged, and it turns their
velocity commands into joint targets with either

  --backend gait   the hand-written IK trot in go2_trot.py (no training needed)
  --backend rl     a policy trained with legged_gym / rsl_rl (see train_go2.sh)

SIM ONLY.  It publishes rt/lowcmd, so never point it at a real robot's network.

Typical use (three terminals):
    ~/eng6010/run_sim.sh                                   # the MuJoCo sim (DDS domain 0, iface lo)
    ~/eng6010/.venv_sim/bin/python go2_sim_dog.py --backend gait
    ~/eng6010/.venv_sim/bin/python keyboard_control.py lo  # or any SportClient script, interface "lo"

Supported sport calls: Move, StopMove, StandUp, BalanceStand, RecoveryStand,
StandDown, Damp, Sit, RiseSit.  Euler/Pose/SpeedLevel/tricks are acknowledged but do nothing.
"""
import argparse
import glob
import json
import math
import os
import threading
import time

import numpy as np

from go2_trot import TrotGait, require_loopback

# --- sport api ids (unitree_sdk2py/go2/sport/sport_api.py) -------------------
API = dict(Damp=1001, BalanceStand=1002, StopMove=1003, StandUp=1004, StandDown=1005,
           RecoveryStand=1006, Euler=1007, Move=1008, Sit=1009, RiseSit=1010)
IGNORED = [1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
           1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038,
           1039, 1040, 1041, 1042, 1043, 1044, 1045]   # SpeedLevel, Pose, tricks ...: ack only

# --- joint conventions --------------------------------------------------------
# SDK / MuJoCo motor order: FR, FL, RR, RL (hip, thigh, calf each).
# legged_gym order (URDF): FL, FR, RL, RR.  POLICY_LEG[j] = SDK leg index of policy leg j.
POLICY_LEG_DEFAULT = "1,0,3,2"
POLICY_DEFAULT_ANGLES = np.array([0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,  0.1, 1.0, -1.5,  -0.1, 1.0, -1.5])  # FL,FR,RL,RR
Q_LYING = np.array([0.0, 1.36, -2.65] * 4)       # same pose Unitree's stand_go2 example lies down in


def newest_checkpoint(root):
    files = glob.glob(os.path.join(root, "*", "model_*.pt"))
    files.sort(key=lambda f: (os.path.getmtime(f)))
    return files[-1] if files else None


class Policy:
    """Actor MLP from an rsl_rl checkpoint, evaluated with plain torch (no isaacgym needed)."""

    def __init__(self, path, leg_map):
        import torch
        self.torch = torch
        ck = torch.load(path, map_location="cpu")
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
        layers, idx = [], 0
        while f"{idx}.weight" in sd:
            w, b = sd[f"{idx}.weight"], sd[f"{idx}.bias"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w, b
            layers.append(lin)
            idx += 2
            if f"{idx}.weight" in sd:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers).eval()
        self.n_obs = layers[0].in_features
        leg = [int(c) for c in leg_map.split(",")]
        self.perm = np.array([leg[j] * 3 + k for j in range(4) for k in range(3)])  # policy idx -> SDK idx
        self.default_sdk = np.zeros(12)
        self.default_sdk[self.perm] = POLICY_DEFAULT_ANGLES
        self.action = np.zeros(12)
        self.kp, self.kd = 20.0, 0.5

    def reset(self):
        self.action[:] = 0.0

    def act(self, q, dq, quat, gyro, v_body, cmd):
        """One 50 Hz policy step; returns joint targets in SDK order."""
        w, x, y, z = quat
        g = np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])  # projected gravity
        p = self.perm
        obs = np.concatenate([v_body * 2.0, gyro * 0.25, g, cmd * np.array([2.0, 2.0, 0.25]),
                              (q - self.default_sdk)[p], dq[p] * 0.05, self.action]).astype(np.float32)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(obs)[None])[0].numpy()
        self.action = np.clip(a, -100, 100)
        tgt = np.zeros(12)
        tgt[p] = self.action * 0.25 + POLICY_DEFAULT_ANGLES
        return tgt

    def stand(self):
        return self.default_sdk.copy()


class Brain:
    """Shared state between the RPC threads (commands) and the 200 Hz control loop."""

    def __init__(self, move_timeout=0.5):
        self.lock = threading.Lock()
        self.mode = "stand"           # "stand" (walking allowed) or "down"
        self.mode_changed = True
        self.cmd = np.zeros(3)
        self.cmd_time = 0.0
        self.move_timeout = move_timeout

    def set_mode(self, mode):
        with self.lock:
            if mode != self.mode:
                self.mode, self.mode_changed = mode, True
            self.cmd[:] = 0.0

    def move(self, vx, vy, vyaw):
        with self.lock:
            self.cmd[:] = (vx, vy, vyaw)
            self.cmd_time = time.perf_counter()

    def stop(self):
        with self.lock:
            self.cmd[:] = 0.0

    def get_cmd(self):
        with self.lock:
            if time.perf_counter() - self.cmd_time > self.move_timeout:
                return np.zeros(3)
            return self.cmd.copy()


def start_services(brain):
    from unitree_sdk2py.rpc.server import Server

    class SportServer(Server):
        def __init__(self):
            super().__init__("sport")

        def Init(self):
            self._SetApiVersion("1.0.0.1")
            for name, aid in API.items():
                self._RegistHandler(aid, self._make(name), False)
            for aid in IGNORED:
                self._RegistHandler(aid, lambda p: (0, ""), False)

        def _make(self, name):
            def h(param):
                if name == "Move":
                    d = json.loads(param)
                    brain.move(float(d["x"]), float(d["y"]), float(d["z"]))
                elif name == "StopMove":
                    brain.stop()
                elif name in ("StandUp", "BalanceStand", "RecoveryStand", "RiseSit"):
                    brain.set_mode("stand")
                elif name in ("StandDown", "Damp", "Sit"):
                    brain.set_mode("down")
                return 0, ""        # Euler etc.: accepted, no effect
            return h

    class SwitcherServer(Server):
        def __init__(self):
            super().__init__("motion_switcher")

        def Init(self):
            self._SetApiVersion("1.0.0.1")
            self._RegistHandler(1001, lambda p: (0, json.dumps({"form": "0", "name": "mcf"})), False)
            for aid in (1002, 1003, 1004):
                self._RegistHandler(aid, lambda p: (0, ""), False)
            self._RegistHandler(1005, lambda p: (0, json.dumps({"silent": False})), False)

    servers = []
    for cls in (SportServer, SwitcherServer):
        s = cls()
        s.Init()
        s.Start(False)
        servers.append(s)
    return servers


def main():
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
    from unitree_sdk2py.utils.crc import CRC

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gait", "rl"], default="gait")
    ap.add_argument("--checkpoint", default="latest",
                    help="rl backend: path to model_N.pt, or 'latest' (newest in unitree_rl_gym/logs/rough_go2)")
    ap.add_argument("--leg-map", default=POLICY_LEG_DEFAULT,
                    help="rl backend: SDK leg index (FR=0,FL=1,RR=2,RL=3) for each policy leg (FL,FR,RL,RR)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--no-autostand", action="store_true", help="start lying down until StandUp is called")
    ap.add_argument("--print-state", action="store_true", help="print position once a second")
    a = ap.parse_args()

    require_loopback(a.iface)
    ChannelFactoryInitialize(a.domain, a.iface)
    st = {"q": None, "dq": None, "quat": None, "gyro": None, "vel": np.zeros(3), "pos": np.zeros(3)}

    def on_low(m):
        st["q"] = np.array([m.motor_state[i].q for i in range(12)])
        st["dq"] = np.array([m.motor_state[i].dq for i in range(12)])
        st["quat"] = np.array(m.imu_state.quaternion)
        st["gyro"] = np.array(m.imu_state.gyroscope)

    def on_high(m):
        st["vel"] = np.array(m.velocity)
        st["pos"] = np.array(m.position)

    low_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_sub.Init(on_low, 10)
    high_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    high_sub.Init(on_high, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0], cmd.head[1], cmd.level_flag, cmd.gpio = 0xFE, 0xEF, 0xFF, 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 2.146e9
        cmd.motor_cmd[i].dq = 16000.0
        cmd.motor_cmd[i].kp = cmd.motor_cmd[i].kd = cmd.motor_cmd[i].tau = 0.0

    def send(q, kp, kd):
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q[i]), 0.0, kp, kd, 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    # --- backend ---
    if a.backend == "rl":
        ck = a.checkpoint
        if ck == "latest":
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unitree_rl_gym", "logs", "rough_go2")
            ck = newest_checkpoint(os.path.normpath(root))
            if ck is None:
                raise SystemExit("no checkpoints found; train first (~/eng6010/train_go2.sh)")
        print("rl backend, checkpoint:", ck)
        policy = Policy(ck, a.leg_map)
        q_stand, walk_kp, walk_kd = policy.stand(), policy.kp, policy.kd
    else:
        print("gait backend (go2_trot.TrotGait)")
        gait = TrotGait()
        q_stand, walk_kp, walk_kd = gait.stand(), 100.0, 4.0
    TRANS_KP, TRANS_KD = 60.0, 3.0

    brain = Brain()
    if a.no_autostand:
        brain.mode = "down"
    start_services(brain)

    while st["q"] is None:
        print("waiting for rt/lowstate (is the sim running on domain %d / %s?)" % (a.domain, a.iface))
        time.sleep(1.0)
    time.sleep(1.5)                                         # sim spawns the dog in the air; let it land
    print("ready: serving sport + motion_switcher; mode =", brain.mode)

    dt, ctrl_n = 0.005, 0
    TRANS_T = 2.0
    q_from, t_trans, q_cmd = st["q"].copy(), time.perf_counter(), st["q"].copy()
    walking_t0, target, was_walking = 0.0, q_stand.copy(), False
    yaw_ref = None                                           # heading hold for the open-loop gait

    def yaw_of(quat):
        w, x, y, z = quat
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    last, next_print = time.perf_counter(), 0.0
    while True:
        now = time.perf_counter()
        with brain.lock:
            mode, changed = brain.mode, brain.mode_changed
            brain.mode_changed = False
        if changed:
            q_from, t_trans = q_cmd.copy(), now
            if a.backend == "rl":
                policy.reset()
        goal = q_stand if mode == "stand" else Q_LYING
        cmd_v = brain.get_cmd() if mode == "stand" else np.zeros(3)
        s = min(1.0, (now - t_trans) / TRANS_T)
        if s < 1.0:                                          # smooth posture change
            k = 0.5 - 0.5 * math.cos(math.pi * s)
            q_cmd = (1 - k) * q_from + k * goal
            send(q_cmd, TRANS_KP, TRANS_KD)
            was_walking = False
        elif mode == "down":
            q_cmd = Q_LYING.copy()
            send(q_cmd, TRANS_KP, TRANS_KD)
        else:
            if a.backend == "rl":
                if ctrl_n % 4 == 0:                          # 50 Hz policy, 200 Hz command stream
                    quat = st["quat"]
                    w, x, y, z = quat
                    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                    v_body = R.T @ st["vel"]
                    target = policy.act(st["q"], st["dq"], quat, st["gyro"], v_body, cmd_v)
                q_cmd = target
            else:
                if not was_walking:
                    walking_t0 = now
                tw = now - walking_t0
                vyaw = float(np.clip(cmd_v[2], -1.0, 1.0))
                if abs(vyaw) > 0.05 or np.linalg.norm(cmd_v) < 1e-3:
                    yaw_ref = None                           # turning or standing: no hold
                else:                                        # walking straight: hold the heading
                    yaw = yaw_of(st["quat"])
                    if yaw_ref is None:
                        yaw_ref = yaw
                    err = math.atan2(math.sin(yaw - yaw_ref), math.cos(yaw - yaw_ref))
                    vyaw = float(np.clip(-2.0 * err, -0.5, 0.5))
                q_cmd = gait.targets(tw, float(np.clip(cmd_v[0], -0.8, 0.8)), vyaw)
            was_walking = True
            send(q_cmd, walk_kp, walk_kd)
        ctrl_n += 1
        if a.print_state and now >= next_print:
            next_print = now + 1.0
            print("mode=%s cmd=%s pos=%s" % (mode, np.round(cmd_v, 2), np.round(st["pos"], 2)), flush=True)
        last += dt
        if last < now - 0.1:                                 # fell behind; don't try to catch up
            last = now
        time.sleep(max(0.0, last - time.perf_counter()))


if __name__ == "__main__":
    main()

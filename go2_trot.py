"""Model-based trot gait for the Go2 (no learning, no sport-mode service needed).

Pure function `TrotGait.targets(t, vx, vyaw)` -> 12 joint targets in Unitree SDK
motor order (FR, FL, RR, RL; hip/thigh/calf each).  The same function drives
both the MuJoCo sim (via DDS, see `main()`) and could drive a real low-level
loop, but it has ONLY been exercised in simulation.

Run (sim must be running, see ~/eng6010/run_sim.sh):
    ~/eng6010/.venv_sim/bin/python go2_trot.py --vx 0.3 --seconds 10
Keys are not handled here; use --vx / --vyaw.  Ctrl-C ends it (the sim dog slumps once commands stop).
"""
import argparse
import math
import time

import numpy as np

L1 = L2 = 0.213          # thigh / calf length [m]
HALF_TRACK = 0.14        # lateral distance hip-to-body-centre line used for yaw [m]

# SDK motor order: FR, FL, RR, RL  (3 joints each: hip, thigh, calf)
LEGS = ["FR", "FL", "RR", "RL"]
# Trot: diagonal pairs move together.  phase offset per leg, in cycles.
PHASE = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
LEFT = {"FL": True, "RL": True, "FR": False, "RR": False}


def leg_ik(x, z):
    """Foot position (x forward, z up; z<0) in the thigh frame -> (thigh, calf).

    Convention checked against the MuJoCo model:
    x = -L1 sin(t1) - L2 sin(t1+t2),  z = -L1 cos(t1) - L2 cos(t1+t2).
    """
    r = min(math.hypot(x, z), 0.99 * (L1 + L2))
    gamma = math.acos((L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2))
    calf = -(math.pi - gamma)
    phi = math.atan2(-x, -z)
    thigh = phi - calf / 2.0
    return thigh, calf


def require_loopback(iface):
    """These programs publish raw joint commands (rt/lowcmd).  On a real Go2 that fights the dog's own
    controller and can make it fall or thrash, so they only run on the loopback interface (the sim)."""
    if iface != "lo":
        raise SystemExit("REFUSING to run on interface %r: this program sends low-level joint commands and is "
                         "for the MuJoCo simulation only (interface 'lo'). Never run it against the real dog." % iface)


class TrotGait:
    def __init__(self, period=0.35, duty=0.6, stand_z=-0.28, swing_h=0.07, x0=-0.08):
        self.period, self.duty = period, duty
        self.stand_z, self.swing_h, self.x0 = stand_z, swing_h, x0

    def stand(self):
        th, ca = leg_ik(self.x0, self.stand_z)
        return np.array([0.0, th, ca] * 4)

    def targets(self, t, vx=0.0, vyaw=0.0):
        """Joint targets at time t [s] since walking started."""
        out = np.zeros(12)
        moving = abs(vx) > 1e-3 or abs(vyaw) > 1e-3
        for i, leg in enumerate(LEGS):
            # forward speed of this side's feet relative to the ground
            v = vx - vyaw * HALF_TRACK if LEFT[leg] else vx + vyaw * HALF_TRACK
            if not moving:
                x, z = self.x0, self.stand_z
            else:
                stride = v * self.duty * self.period
                ph = (t / self.period + PHASE[leg]) % 1.0
                if ph < self.duty:                       # stance: foot slides back
                    s = ph / self.duty
                    x = self.x0 + stride * (0.5 - s)
                    z = self.stand_z
                else:                                    # swing: foot goes forward
                    s = (ph - self.duty) / (1.0 - self.duty)
                    x = self.x0 - stride / 2 + stride * (s - math.sin(2 * math.pi * s) / (2 * math.pi))
                    z = self.stand_z + self.swing_h * (1 - math.cos(2 * math.pi * s)) / 2
            th, ca = leg_ik(x, z)
            out[3 * i:3 * i + 3] = (0.0, th, ca)
        return out


def main():
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.3, help="forward speed [m/s]")
    ap.add_argument("--vyaw", type=float, default=0.0, help="yaw rate [rad/s], +ccw")
    ap.add_argument("--seconds", type=float, default=10.0, help="walking time")
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--kd", type=float, default=4.0)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default="lo")
    a = ap.parse_args()

    require_loopback(a.iface)
    ChannelFactoryInitialize(a.domain, a.iface)
    state = {"q": None}

    def on_state(msg):
        state["q"] = np.array([msg.motor_state[i].q for i in range(12)])

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)
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

    while state["q"] is None:
        print("waiting for rt/lowstate (is the sim running on domain %d / %s?)" % (a.domain, a.iface))
        time.sleep(1.0)

    gait = TrotGait()
    SETTLE_T, STAND_T, RAMP_T, dt = 1.5, 2.0, 1.5, 0.005
    time.sleep(SETTLE_T)                                 # sim spawns the dog in the air; let it land
    q_start = state["q"].copy()
    q_stand = gait.stand()
    t0 = time.perf_counter()
    last = t0

    def send(q):
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q[i]), 0.0, a.kp, a.kd, 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    total = STAND_T + a.seconds
    try:
        while True:
            t = time.perf_counter() - t0
            if t > total:
                break
            if t < STAND_T:                              # smooth stand-up
                s = 0.5 - 0.5 * math.cos(math.pi * t / STAND_T)
                q = (1 - s) * q_start + s * q_stand
            else:
                tw = t - STAND_T
                k = min(1.0, tw / RAMP_T)                # ramp speed up gently
                q = gait.targets(tw, a.vx * k, a.vyaw * k)
            send(q)
            last += dt
            time.sleep(max(0.0, last - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    # The sim bridge only applies a command when one arrives, so if we stopped
    # sending the dog would slump.  Keep holding the stand pose until Ctrl-C.
    print("walk finished, holding stand pose (Ctrl-C to exit)")
    try:
        while True:
            send(q_stand)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

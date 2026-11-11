"""Evaluate a trained walking policy in the MuJoCo Go2 model (sim-to-sim check), with no DDS in the loop.

    .venv_sim/bin/python rl/eval_in_mujoco.py rl/checkpoints/go2_walk_model_1500.pt --commands
    .venv_sim/bin/python rl/eval_in_mujoco.py rl/checkpoints/go2_walk_model_1500.pt --scan-legs

--commands   walk a list of velocity commands for 10 s each on flat ground and print the distance covered (and so
             the speed achieved) and whether the dog stayed up.
--scan-legs  the policy was trained with legs ordered FL, FR, RL, RR but the SDK / MuJoCo use FR, FL, RR, RL. Try
             all 24 ways of assigning legs and print which one walks furthest (the right one is far ahead).

The policy wrapper is go2_sim_dog.Policy, the same code the DDS controller uses. Needs the unitree_mujoco clone
with our scene files (sim/install_into_unitree_mujoco.sh); set UNITREE_MUJOCO if it is not ~/eng6010/unitree_mujoco.
"""
import argparse
import itertools
import math
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCENE = os.path.join(os.environ.get("UNITREE_MUJOCO", os.path.expanduser("~/eng6010/unitree_mujoco")),
                     "unitree_robots", "go2", "scene_flat.xml")
SDK_ORDER = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
             "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]


def run(job):
    """One rollout: (checkpoint, leg map, command (vx, vy, vyaw), seconds) -> (leg map, command, outcome, x, y, yaw)."""
    import torch
    torch.set_num_threads(1)
    from go2_sim_dog import Policy
    ckpt, legmap, cmd, secs = job
    pol = Policy(ckpt, legmap)
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    m.opt.timestep = 0.005                               # same as training: 200 Hz physics, policy every 4th step
    qa = [m.joint(n + "_joint").qposadr[0] for n in SDK_ORDER]
    va = [m.joint(n + "_joint").dofadr[0] for n in SDK_ORDER]
    d.qpos[2], d.qpos[3] = 0.35, 1.0
    for a, v in zip(qa, pol.default_sdk):
        d.qpos[a] = v
    target, cmd = pol.default_sdk.copy(), np.array(cmd, float)
    yaw_prev, yaw_total = 0.0, 0.0                        # heading unwrapped, so several turns are counted
    for i in range(int(secs / 0.005)):
        if i % 4 == 0:
            quat = d.qpos[3:7]
            w, x, y, z = quat
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            target = pol.act(d.qpos[qa], d.qvel[va], quat, d.qvel[3:6], R.T @ d.qvel[0:3], cmd)
        d.ctrl[:] = pol.kp * (target - d.qpos[qa]) + pol.kd * (0 - d.qvel[va])   # MuJoCo applies the torque limits
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.12:
            return legmap, cmd, "FELL at %.1f s" % (i * 0.005), d.qpos[0], d.qpos[1], 0.0
        w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw_total += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw
    return legmap, cmd, "ok", d.qpos[0], d.qpos[1], math.degrees(yaw_total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--commands", action="store_true")
    ap.add_argument("--scan-legs", action="store_true")
    ap.add_argument("--leg-map", default="1,0,3,2")
    a = ap.parse_args()
    if a.scan_legs:
        maps = [",".join(map(str, p)) for p in itertools.permutations(range(4))]
        with Pool(min(12, os.cpu_count())) as pool:
            res = pool.map(run, [(a.checkpoint, mp, (0.5, 0, 0), 8.0) for mp in maps])
        print("leg map (policy FL,FR,RL,RR -> SDK index)   outcome        distance in 8 s at 0.5 m/s")
        for mp, _, outcome, x, _, _ in sorted(res, key=lambda r: -r[3])[:6]:
            print("  %-10s %-14s %6.2f m" % (mp, outcome, x))
    if a.commands:
        cmds = [(0, 0, 0), (0.3, 0, 0), (0.8, 0, 0), (1.0, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, 0, 0.8), (0.4, 0, 0.6)]
        with Pool(min(8, os.cpu_count())) as pool:
            res = pool.map(run, [(a.checkpoint, a.leg_map, c, 10.0) for c in cmds])
        print("command (vx, vy, vyaw)   outcome   x [m]   y [m]   turned [deg]   speed x / y [m/s]   yaw rate [rad/s]")
        for _, c, outcome, x, y, yaw in res:
            print("  %-22s %-8s %6.2f  %6.2f  %9.0f   %6.2f / %5.2f   %9.2f"
                  % (tuple(c.tolist()), outcome, x, y, yaw, x / 10.0, y / 10.0, math.radians(yaw) / 10.0))


if __name__ == "__main__":
    main()

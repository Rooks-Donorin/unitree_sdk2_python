"""Demo for the MuJoCo sim: stand the dog up and walk a small loop (four forward legs, a left turn after each)
through the same SportClient API the real-dog scripts use. Needs sim/run_sim.sh and go2_sim_dog.py running.

    .venv_sim/bin/python sim/demo_loop.py
"""
import time, math
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
ChannelFactoryInitialize(0, "lo"); pos = [0, 0, 0]
ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(lambda m: pos.__setitem__(slice(0, 3), list(m.position)), 10)
sp = SportClient(); sp.SetTimeout(5.0); sp.Init()
def go(label, vx, vyaw, secs):
    t0 = time.time()
    while time.time() - t0 < secs: sp.Move(vx, 0, vyaw); time.sleep(0.1)
    sp.StopMove(); time.sleep(0.5); print("%-22s at (%5.2f, %5.2f)  z=%.2f" % (label, pos[0], pos[1], pos[2]), flush=True)
sp.StandUp(); time.sleep(4); print("%-22s at (%5.2f, %5.2f)  z=%.2f" % ("standing", pos[0], pos[1], pos[2]), flush=True)
for leg in range(1, 5):
    go("leg %d: forward" % leg, 0.5, 0.0, 3.0)
    go("      turn left 90 deg", 0.0, 0.8, 2.0)
print("loop done")

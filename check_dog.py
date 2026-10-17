import sys, time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

count = 0
def handler(msg: LowState_):
    global count
    count += 1
    if count % 100 == 0:  # the dog sends ~500 msgs/sec; print a few per second
        print(f"msgs: {count} | IMU roll/pitch/yaw: {[round(x, 3) for x in msg.imu_state.rpy]}"
              f" | battery: {msg.power_v:.1f} V | FR hip angle: {msg.motor_state[0].q:.3f}")

ChannelFactoryInitialize(0, sys.argv[1])
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)
print("Listening to the dog... (Ctrl+C to stop)")
while True:
    time.sleep(1)
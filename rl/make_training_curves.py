"""Draw the RL training curves figure (docs/figures/rl_training_curves.svg) from rl/training_curves.csv.

    python rl/make_training_curves.py            # needs matplotlib
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "training_curves.csv"))))
col = lambda name: [float(r[name]) if r[name] else float("nan") for r in rows]
it = col("iteration")

fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
ax[0][0].plot(it, col("Train/mean_reward"), color="#1f77b4")
ax[0][0].set_title("Mean episode reward")
ax[0][1].plot(it, col("Train/mean_episode_length"), color="#2ca02c")
ax[0][1].set_title("Mean episode length (steps; 1000 = full 20 s, no fall)")
ax[1][0].plot(it, col("Episode/rew_tracking_lin_vel"), label="lin. velocity tracking (max 1.0)")
ax[1][0].plot(it, col("Episode/rew_tracking_ang_vel"), label="yaw-rate tracking (max 0.5)")
ax[1][0].set_title("Velocity-tracking rewards")
ax[1][0].legend(fontsize=8)
for name, label in (("Episode/rew_torques", "torques"), ("Episode/rew_action_rate", "action rate"),
                    ("Episode/rew_feet_air_time", "feet air time")):
    ax[1][1].plot(it, col(name), label=label)
ax[1][1].set_title("Penalty / shaping terms (per episode)")
ax[1][1].legend(fontsize=8)
for a in ax.ravel():
    a.set_xlabel("PPO iteration")
    a.grid(alpha=0.3)
fig.suptitle("Go2 walking policy: PPO, 4096 parallel envs, 1500 iterations (147 M steps, ~85 min on an RTX 500 Ada laptop GPU)")
fig.tight_layout()
out = os.path.join(here, "..", "docs", "figures", "rl_training_curves.svg")
fig.savefig(out)
print("wrote", os.path.normpath(out))

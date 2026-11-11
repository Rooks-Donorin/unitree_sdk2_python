"""Export the scalar curves of a legged_gym / rsl_rl training run (TensorBoard events) to a CSV.

    python rl/export_tensorboard.py <run folder> rl/training_curves.csv

Needs `tensorboard` (it is in the unitree-rl conda env). The CSV has one row per training iteration.
"""
import csv
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run, out = sys.argv[1], sys.argv[2]
events = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
if not events:
    sys.exit(f"no TensorBoard events file in {run}")
acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = [t for t in acc.Tags()["scalars"] if "/time" not in t]
series = {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags}
steps = sorted(set().union(*[set(s) for s in series.values()]))
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iteration"] + tags)
    for s in steps:
        w.writerow([s] + [f"{series[t][s]:.6g}" if s in series[t] else "" for t in tags])
print(f"{len(steps)} iterations x {len(tags)} curves -> {out}")

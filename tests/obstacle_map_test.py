"""Check the "three separate lidar frames" rule of the obstacle map (navigation.ObstacleMap).

    .venv/bin/python tests/obstacle_map_test.py          # needs numpy only

A 10 cm cell becomes an obstacle only once 3 separate lidar frames have hit it; several points of one frame in the
same cell count once; the count starts again after HIT_WINDOW seconds without a hit. Exit code 1 if a case fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import navigation as nav

CELL = (2.0, 0.0)


def run(hits):
    """hits: [(time, points in that frame)] -> was the cell an obstacle after each frame?"""
    m, out = nav.ObstacleMap(), []
    for t, n in hits:
        pts = np.array([[CELL[0] + 0.001 * k, CELL[1], 0.4, 100.0] for k in range(n)], np.float32)
        m.add(pts, (0, 0), 3.14, 0.3, 0.0, t)      # the robot is far from the cell, looking away from it
        out.append(bool(m.occupied(t)[tuple(m.to_cell(CELL))]))
    return out


cases = [
    ("one frame with 5 points in the cell", [(0.0, 5)], [False]),
    ("two frames", [(0.0, 3), (0.1, 3)], [False, False]),
    ("three separate frames, one point each", [(0.0, 1), (0.1, 1), (0.2, 1)], [False, False, True]),
    ("three frames spread over 4 s (gaps < 5 s)", [(0.0, 1), (2.0, 1), (4.0, 1)], [False, False, True]),
    ("hits 6.5 s apart start the count again", [(0.0, 1), (6.5, 1), (13.0, 1)], [False, False, False]),
    ("stays an obstacle once confirmed", [(0.0, 1), (0.1, 1), (0.2, 1), (9.0, 1)], [False, False, True, True]),
]
ok_all = True
for label, hits, expect in cases:
    got = run(hits)
    ok = got == expect
    ok_all &= ok
    print(("PASS  " if ok else "FAIL  ") + f"{label}: {got}")
sys.exit(0 if ok_all else 1)

"""Synthetic stand-in for a hand tracker's output (MediaPipe 21-landmark format).

Generates what a monocular hand tracker would emit for a person doing a
pick-and-place in front of a phone camera, in *camera* coordinates:
x right, y down, z depth (meters). Real videos go through
extract_keypoints.py instead and produce the same (T, 21, 3) array.

Noise model (the variable studied in the experiments):
  * per-frame jitter, larger along depth (monocular depth is the weak axis)
  * random tracking dropouts (frames set to NaN)
  * human timing variation and curved approach paths
"""
from __future__ import annotations

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

# Rough open-hand landmark offsets from the wrist (meters, camera frame).
_OPEN_HAND = np.array([
    [0.00, 0.00, 0.00],
    [0.02, -0.02, 0.0], [0.035, -0.04, 0.0], [0.045, -0.06, 0.0], [0.05, -0.08, 0.0],
    [0.02, -0.09, 0.0], [0.02, -0.12, 0.0], [0.02, -0.14, 0.0], [0.02, -0.16, 0.0],
    [0.00, -0.09, 0.0], [0.00, -0.13, 0.0], [0.00, -0.15, 0.0], [0.00, -0.17, 0.0],
    [-0.02, -0.085, 0.0], [-0.02, -0.12, 0.0], [-0.02, -0.14, 0.0], [-0.02, -0.155, 0.0],
    [-0.035, -0.075, 0.0], [-0.04, -0.10, 0.0], [-0.04, -0.115, 0.0], [-0.04, -0.13, 0.0],
])


def _min_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _segment(a, b, n, rng, curve=0.03):
    """Min-jerk path from a to b with a random lateral bow (humans don't move straight)."""
    s = _min_jerk(np.linspace(0, 1, n))[:, None]
    bow = rng.normal(0, curve, 3)
    bow[1] = -abs(bow[1])  # arc upward (y is down in camera frame)
    return a + (b - a) * s + np.sin(np.pi * s) * bow


def generate_demo(obj_cam, goal_cam, rng, fps=30, jitter=0.004, depth_jitter_mult=3.0,
                  dropout=0.03):
    """Return (T, 21, 3) landmarks for one pick-and-place demo plus ground truth."""
    obj_cam, goal_cam = np.asarray(obj_cam, float), np.asarray(goal_cam, float)
    start = obj_cam + np.array([rng.uniform(-0.15, 0.15), -0.18, rng.uniform(-0.05, 0.05)])
    pre = obj_cam + np.array([0, -0.08, 0])
    lift = obj_cam + np.array([0, -0.12, 0])
    above_goal = goal_cam + np.array([0, -0.05, 0])

    speed = rng.uniform(0.8, 1.25)
    n = lambda base: max(4, int(base * fps / 30 / speed))
    k_close, k_open = n(8), n(8)
    parts = [
        (_segment(start, pre, n(18), rng), 1.0),
        (_segment(pre, obj_cam, n(10), rng, 0.005), 1.0),
        (np.repeat(obj_cam[None], k_close, 0), np.linspace(1, 0, k_close)),
        (_segment(obj_cam, lift, n(10), rng, 0.005), 0.0),
        (_segment(lift, above_goal, n(22), rng), 0.0),
        (_segment(above_goal, goal_cam, n(8), rng, 0.005), 0.0),
        (np.repeat(goal_cam[None], k_open, 0), np.linspace(0, 1, k_open)),
        (_segment(goal_cam, goal_cam + np.array([0, -0.1, 0]), n(10), rng), 1.0),
    ]
    pinch = np.concatenate([p for p, _ in parts])
    aperture = np.concatenate([np.broadcast_to(a, (len(p),)) for p, a in parts])

    T = len(pinch)
    lm = np.empty((T, 21, 3))
    for t in range(T):
        hand = _OPEN_HAND.copy()
        mid = (hand[THUMB_TIP] + hand[INDEX_TIP]) / 2
        for j in (THUMB_TIP, INDEX_TIP):
            hand[j] = mid + (hand[j] - mid) * (0.15 + 0.85 * aperture[t])
        lm[t] = hand - mid + pinch[t]  # pinch midpoint follows the planned path

    noise = rng.normal(0, jitter, lm.shape)
    noise[..., 2] *= depth_jitter_mult
    lm += noise
    drop = rng.random(T) < dropout
    drop[:2] = drop[-2:] = False
    lm[drop] = np.nan
    return lm, {"obj": obj_cam, "goal": goal_cam, "fps": fps}

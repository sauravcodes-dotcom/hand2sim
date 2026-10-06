"""Retarget hand landmarks (T, 21, 3) into a robot end-effector plan.

Pipeline (each step is a separate function so it can be tested and ablated):
  1. fill tracking dropouts by linear interpolation
  2. smooth with a centered moving average
  3. pinch point = midpoint of thumb tip and index tip; aperture = their
     distance normalized by hand size (wrist -> middle MCP), so it doesn't
     depend on how far the hand is from the camera
  4. detect grasp / release with hysteresis (avoids chattering on noisy frames)
  5. map camera axes to robot axes
  6. object-centric re-anchoring: the pre-grasp segment is expressed relative
     to the grasp point and re-anchored on the robot's object; the carry
     segment is warped so it ends on the robot's goal. This is what lets one
     human demo transfer to a scene with different object/goal positions,
     and it means we never need to detect the object in the video:
     the grasp moment tells us where it was.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

# camera (x right, y down, z away) -> robot (x forward, y left, z up)
DEFAULT_AXES = np.array([[0, 0, 1],
                         [-1, 0, 0],
                         [0, -1, 0]], float)


class RetargetError(RuntimeError):
    """Raised when a demo can't be segmented (e.g. no grasp detected)."""


@dataclass
class Plan:
    waypoints: np.ndarray   # (T, 3) robot-frame pinch positions
    gripper: np.ndarray     # (T,) +1 open, -1 closed
    grasp_idx: int
    release_idx: int


def fill_dropouts(x):
    x = np.array(x, float)
    T = x.shape[0]
    flat = x.reshape(T, -1)
    good = ~np.isnan(flat).any(1)
    if good.sum() < 2:
        raise RetargetError("hand tracked in fewer than 2 frames")
    idx = np.arange(T)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(idx, idx[good], flat[good, c])
    return flat.reshape(x.shape)


def smooth(x, k=5):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, ker, "valid"), 0, xp)


def pinch_and_aperture(lm):
    pinch = (lm[:, THUMB_TIP] + lm[:, INDEX_TIP]) / 2
    hand = np.linalg.norm(lm[:, MIDDLE_MCP] - lm[:, WRIST], axis=1)
    ap = np.linalg.norm(lm[:, THUMB_TIP] - lm[:, INDEX_TIP], axis=1) / np.maximum(hand, 1e-6)
    return pinch, ap


def detect_grasp(ap, close_thr=0.35, open_thr=0.5, min_hold=3):
    """Hysteresis threshold on normalized aperture. Returns (closed_mask, grasp_idx, release_idx)."""
    closed = np.zeros(len(ap), bool)
    state = False
    for t, a in enumerate(ap):
        if not state and a < close_thr:
            state = True
        elif state and a > open_thr:
            state = False
        closed[t] = state
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    if len(edges) < 2:
        raise RetargetError("no complete grasp/release found")
    # take the longest closed interval as the carry phase
    starts = [e + 1 for e in edges if not closed[e]]
    ends = [e + 1 for e in edges if closed[e]]
    pairs = [(s, e) for s in starts for e in ends if e > s]
    s, e = max(pairs, key=lambda p: p[1] - p[0])
    if e - s < min_hold:
        raise RetargetError("grasp too short to be real")
    mask = np.zeros(len(ap), bool)
    mask[s:e] = True
    return mask, s, e


def retarget(lm, robot_obj, robot_goal, axes=DEFAULT_AXES, scale=1.0, smooth_k=5,
             grasp_offset=np.zeros(3)):
    lm = smooth(fill_dropouts(lm), smooth_k)
    pinch_cam, ap = pinch_and_aperture(lm)
    closed, g, r = detect_grasp(ap)
    p = (pinch_cam @ axes.T) * scale

    robot_obj = np.asarray(robot_obj, float) + grasp_offset
    robot_goal = np.asarray(robot_goal, float)
    out = np.empty_like(p)
    # approach: relative to human grasp point, re-anchored on the robot's object
    out[:g + 1] = p[:g + 1] - p[g] + robot_obj
    # carry: linear warp so the segment starts at the object and ends at the goal
    seg = p[g:r + 1] - p[g]
    human_disp = p[r] - p[g]
    robot_disp = robot_goal - robot_obj
    w = np.linspace(0, 1, len(seg))[:, None]
    out[g:r + 1] = robot_obj + seg + w * (robot_disp - human_disp)
    # retreat: relative to release point
    out[r:] = p[r:] - p[r] + robot_goal
    return Plan(out, np.where(closed, -1.0, 1.0), g, r)

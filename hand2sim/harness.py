"""Evaluation harness: seeded episodes, failure taxonomy, Wilson intervals, rollout videos.

Every result row carries the config hash and seed, so any number in the report
can be regenerated exactly with `scripts/run_experiments.py --only <name>`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from .retarget import RetargetError
from .sim import make_env, unpack

FAILURE_MODES = ["success", "never_grasped", "dropped_in_transit", "missed_goal", "retarget_failed"]


@dataclass
class Episode:
    seed: int
    outcome: str
    final_dist: float
    max_lift: float
    steps: int


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def classify(success, obj_z, slipped):
    """slipped: object separated from the gripper while the gripper was commanded closed."""
    lift = float(np.max(obj_z) - obj_z[0])
    if success:
        return "success", lift
    if lift < 0.01:
        return "never_grasped", lift
    if slipped:
        return "dropped_in_transit", lift
    return "missed_goal", lift


def render_without_side_effects(env):
    """Grab a frame straight from the renderer.

    Fetch's env.render() runs a callback that calls mj_forward, which refreshes
    body positions that the next step's mocap action reads (mj_step leaves them
    one step stale). Calling it mid-episode nudges the controller, so a filmed
    episode can diverge from the same episode run unfilmed. We place the goal
    marker once at reset instead (sim.TablePlace) and read pixels directly.
    The gallery asserts filmed outcome == evaluated outcome.
    """
    return env.unwrapped.mujoco_renderer.render("rgb_array")


def run_episode(env, factory, seed, record=False, frame_every=2, collect=False):
    obs, _ = env.reset(seed=seed)
    frames, traj = [], []
    try:
        act = factory(obs, seed)
    except RetargetError:
        return Episode(seed, "retarget_failed", float("nan"), 0.0, 0), frames, traj
    obj_z = [unpack(obs)["obj"][2]]
    info = {"is_success": 0.0}
    holding, slipped = False, False
    t = 0
    for t in range(env.spec.max_episode_steps):
        a = act(obs, t)
        if collect:
            traj.append((obs, a))
        if record and t % frame_every == 0:
            frames.append(render_without_side_effects(env))
        obs, _, term, trunc, info = env.step(a)
        s = unpack(obs)
        obj_z.append(s["obj"][2])
        near = np.linalg.norm(s["obj"] - s["grip"]) < 0.03
        lifted = s["obj"][2] - obj_z[0] > 0.01
        if a[3] < 0 and near and lifted:
            holding = True
        elif holding and a[3] < 0 and not near:
            slipped = True
        if a[3] > 0:
            holding = False
        if term or trunc:
            break
    s = unpack(obs)
    dist = float(np.linalg.norm(s["obj"] - s["goal"]))
    outcome, lift = classify(bool(info["is_success"]), np.array(obj_z), slipped)
    return Episode(seed, outcome, dist, lift, t + 1), frames, traj


def evaluate(factory, seeds, max_steps=100, record_seeds=()):
    env = make_env(max_steps, render=bool(record_seeds))
    eps, videos = [], {}
    for sd in seeds:
        ep, frames, _ = run_episode(env, factory, sd, record=sd in record_seeds)
        eps.append(ep)
        if frames:
            videos[sd] = (ep.outcome, frames)
    env.close()
    return eps, videos


def summarize(eps):
    n = len(eps)
    k = sum(e.outcome == "success" for e in eps)
    p, lo, hi = wilson(k, n)
    counts = {m: sum(e.outcome == m for e in eps) for m in FAILURE_MODES}
    return {"n": n, "success": k, "rate": p, "ci95": [lo, hi], "outcomes": counts}


def episodes_to_jsonl(eps, path, extra):
    with open(path, "w") as f:
        for e in eps:
            f.write(json.dumps({**extra, **asdict(e)}) + "\n")

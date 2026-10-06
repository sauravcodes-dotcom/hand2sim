"""Retarget YOUR hand video onto the robot and film the result.

    python -m hand2sim.extract_keypoints my_demo.mp4 --model hand_landmarker.task
    python scripts/replay_video.py my_demo.npy --seeds 0 1 2 3 4

Each seed is a different robot scene (object/goal placement). The same human
demo is re-anchored onto each one, which is the point of object-centric
retargeting.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import numpy as np

from hand2sim.harness import run_episode
from hand2sim.policies import PlanFollower
from hand2sim.retarget import retarget
from hand2sim.sim import make_env, unpack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("landmarks", help=".npy from extract_keypoints")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--out", default="results/my_video")
    a = ap.parse_args()
    lm = np.load(a.landmarks)
    os.makedirs(a.out, exist_ok=True)
    env = make_env(100, render=True)

    def factory(obs0, seed):
        s = unpack(obs0)
        return PlanFollower(retarget(lm, s["obj"], s["goal"], smooth_k=a.smooth))

    for sd in a.seeds:
        ep, frames, _ = run_episode(env, factory, sd, record=True)
        path = f"{a.out}/seed{sd}_{ep.outcome}.gif"
        if frames:
            imageio.mimsave(path, frames, duration=0.08, loop=0)
        print(f"seed {sd}: {ep.outcome} (final distance {ep.final_dist:.3f} m) -> {path}")


if __name__ == "__main__":
    main()

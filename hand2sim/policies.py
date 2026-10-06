"""Policies evaluated by the harness. Each is a factory returning
`act(obs, t) -> action`, built fresh per episode from that episode's start obs."""
from __future__ import annotations

import numpy as np

from .retarget import DEFAULT_AXES, Plan, retarget
from .sim import servo, unpack
from .synthetic_hand import generate_demo


def random_human_scene(rng):
    """Object/goal placement in the *human's* scene (camera frame), independent of the robot's."""
    obj = np.array([rng.uniform(-0.15, 0.15), 0.0, rng.uniform(0.45, 0.65)])
    goal = np.array([rng.uniform(-0.15, 0.15), 0.0, rng.uniform(0.45, 0.65)])
    return obj, goal


class PlanFollower:
    """Executes a retargeted plan by time-indexing it over `exec_steps` sim steps."""

    def __init__(self, plan: Plan, exec_steps=85):
        self.plan, self.exec_steps = plan, exec_steps

    def __call__(self, obs, t):
        T = len(self.plan.waypoints)
        i = min(T - 1, int(t * T / self.exec_steps))
        return servo(unpack(obs)["grip"], self.plan.waypoints[i], self.plan.gripper[i])


HUMAN_CENTER = np.array([0.0, 0.0, 0.55])       # middle of the human's table, camera frame
ROBOT_CENTER = np.array([1.342, 0.749, 0.425])   # middle of Fetch's object spawn area


def fixed_calibration(p_cam):
    """Map a camera-frame point to the robot frame with one fixed table-to-table calibration."""
    return (np.asarray(p_cam) - HUMAN_CENTER) @ DEFAULT_AXES.T + ROBOT_CENTER


def replay_policy(noise, base_seed=0, reanchor=True, smooth_k=5):
    """Record a (synthetic) human demo in a *different* scene, retarget it to this episode.

    reanchor=True : object-centric retargeting onto the robot's actual object/goal.
    reanchor=False: ablation. Fixed calibration only, so the robot goes where the
                    human's object was, which is how naive "replay the hand path" fails.
    """
    def factory(obs0, episode_seed):
        # per-episode RNG: any single episode can be re-run (and filmed) on its own
        rng = np.random.default_rng([base_seed, episode_seed])
        h_obj, h_goal = random_human_scene(rng)
        lm, _ = generate_demo(h_obj, h_goal, rng, **noise)
        if reanchor:
            s = unpack(obs0)
            obj, goal = s["obj"], s["goal"]
        else:
            obj, goal = fixed_calibration(h_obj), fixed_calibration(h_goal)
        return PlanFollower(retarget(lm, obj, goal, smooth_k=smooth_k))
    return factory


def oracle_policy():
    """Scripted policy with privileged state. Upper bound, not a learning result."""
    def factory(obs0, episode_seed):
        def act(obs, t):
            s = unpack(obs)
            grip, obj, goal = s["grip"], s["obj"], s["goal"]
            held = s["fingers"].sum() < 0.06 and np.linalg.norm(obj - grip) < 0.02
            if not held:
                above = obj + [0, 0, 0.05]
                if np.linalg.norm(grip[:2] - obj[:2]) > 0.01 and grip[2] < above[2] - 0.01 and t < 5:
                    return servo(grip, above, 1.0)
                if np.linalg.norm(grip - obj) > 0.01:
                    return servo(grip, obj if np.linalg.norm(grip[:2] - obj[:2]) < 0.01 else above, 1.0)
                return servo(grip, obj, -1.0)
            return servo(grip, goal, -1.0)
        return act
    return factory

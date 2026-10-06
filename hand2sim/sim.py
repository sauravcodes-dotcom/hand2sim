"""Thin layer over Gymnasium-Robotics FetchPickAndPlace (MuJoCo).

Keeps every sim detail (obs layout, action scaling, episode length) in one
place so swapping in another simulator means editing one file.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPickAndPlace-v4"
ACTION_SCALE = 0.05  # Fetch moves the gripper 5 cm per unit action per step


class TablePlace(gym.Wrapper):
    """Put every goal on the table surface.

    Stock FetchPickAndPlace puts half its goals in mid-air and counts success
    only if the robot is still *holding* the object there at the final step.
    A human demo of "put it there" ends with a release, so the in-air variant
    measures task-definition mismatch, not transfer quality. Placing on the
    table makes the robot task match what the human actually demonstrates.
    """

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        u = self.env.unwrapped
        u.goal[2] = obs["observation"][5]  # object resting height
        u._render_callback()  # move the goal marker now (runs mj_forward), never mid-episode
        return u._get_obs(), info


def make_env(max_steps=100, render=False, size=240):
    kw = dict(max_episode_steps=max_steps)
    if render:
        kw.update(render_mode="rgb_array", width=size, height=size)
    return TablePlace(gym.make(ENV_ID, **kw))


def unpack(obs):
    o = obs["observation"]
    return {
        "grip": o[0:3],
        "obj": o[3:6],
        "fingers": o[9:11],
        "goal": obs["desired_goal"],
    }


def policy_features(obs):
    """State vector for learned policies: gripper, object, goal and the relative vectors."""
    s = unpack(obs)
    return np.concatenate([s["grip"], s["obj"], s["goal"], s["obj"] - s["grip"],
                           s["goal"] - s["obj"], s["fingers"]])


def servo(grip, target, gripper_cmd, gain=1.0):
    """P-controller from a Cartesian target to a Fetch action."""
    a = np.clip(gain * (target - grip) / ACTION_SCALE, -1, 1)
    return np.append(a, gripper_cmd).astype(np.float32)

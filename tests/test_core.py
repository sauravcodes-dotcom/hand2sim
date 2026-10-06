import numpy as np
import pytest

from hand2sim.extract_keypoints import backproject
from hand2sim.harness import wilson
from hand2sim.retarget import RetargetError, detect_grasp, fill_dropouts, retarget
from hand2sim.synthetic_hand import generate_demo


def test_fill_dropouts_interpolates():
    x = np.array([[0.0], [np.nan], [2.0]])
    assert fill_dropouts(x)[1, 0] == pytest.approx(1.0)


def test_hysteresis_ignores_single_noisy_frame():
    ap = np.array([1.0] * 5 + [0.1] * 10 + [0.45] + [0.1] * 10 + [1.0] * 5)
    _, g, r = detect_grasp(ap)
    assert (g, r) == (5, 26)  # one frame at 0.45 must not count as a release


def test_no_grasp_raises():
    with pytest.raises(RetargetError):
        detect_grasp(np.ones(30))


def test_retarget_lands_on_robot_object_and_goal():
    rng = np.random.default_rng(0)
    lm, _ = generate_demo([0.05, 0, 0.5], [-0.1, 0, 0.6], rng, jitter=0.0, dropout=0.0)
    obj, goal = np.array([1.3, 0.7, 0.42]), np.array([1.4, 0.9, 0.42])
    plan = retarget(lm, obj, goal, smooth_k=1)
    assert np.allclose(plan.waypoints[plan.grasp_idx], obj, atol=1e-9)
    assert np.allclose(plan.waypoints[plan.release_idx], goal, atol=1e-9)
    assert (plan.gripper[plan.grasp_idx:plan.release_idx] == -1).all()


def test_backproject_recovers_known_point():
    W, H = 1920, 1080
    f = (W / 2) / np.tan(np.radians(65) / 2)
    P = np.zeros((21, 3)); P[9] = [0, -0.09, 0]; P += [0.1, 0.05, 0.5]
    norm = np.c_[(P[:, 0] * f / P[:, 2] + W / 2) / W, (P[:, 1] * f / P[:, 2] + H / 2) / H, np.zeros(21)]
    assert np.allclose(backproject(norm, W, H)[[0, 9]], P[[0, 9]], atol=1e-6)


def test_wilson_bounds():
    p, lo, hi = wilson(97, 100)
    assert lo < p < hi and hi <= 1.0
    assert wilson(0, 0) == (0.0, 0.0, 0.0)

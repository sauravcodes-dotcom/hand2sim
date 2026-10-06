# hand2sim

**Human hand video → robot pick-and-place → reproducible simulation evaluation.**

A small, end-to-end pipeline for learning manipulation from human videos: extract hand
landmarks from a phone video, retarget them into a robot end-effector plan, execute it on a
simulated Fetch arm (MuJoCo), and evaluate with seeded episodes, confidence intervals, and a
labeled failure gallery. Runs on a laptop CPU.

![success](results/gallery/replay_phone_k5__success__seed10000.gif)
![dropped](results/gallery/replay_very_bad_k5__dropped_in_transit__seed10012.gif)
![never grasped](results/gallery/replay_very_bad_k5__never_grasped__seed10005.gif)

## Pipeline

```
phone video ──MediaPipe──▶ (T,21,3) landmarks ──▶ fill dropouts ▶ smooth ▶ pinch point + aperture
                                                     ▶ grasp/release (hysteresis)
                                                     ▶ object-centric re-anchoring ──▶ EE plan ──▶ Fetch (MuJoCo)
```

* **Depth from one camera** (`extract_keypoints.py`): MediaPipe gives image coordinates and only
  relative depth. Absolute depth comes from apparent hand size, `Z = f·L / l_px`, with L ≈ 9 cm
  (wrist to middle knuckle).
* **Grasp detection** (`retarget.py`): thumb–index distance normalized by hand size (so it does not
  depend on distance to the camera), thresholded with hysteresis so one noisy frame cannot fake a
  release.
* **Object-centric re-anchoring**: the human demo is recorded in a *different* scene from the
  robot's. The approach is expressed relative to the grasp point and re-anchored on the robot's
  object; the carry is warped to end on the robot's goal. No object detection is needed: the
  grasp moment tells us where the object was.
* **Evaluation harness** (`harness.py`): 100 held-out seeds per run, Wilson 95% intervals, every
  failure classified (`never_grasped`, `dropped_in_transit`, `missed_goal`, `retarget_failed`),
  config hash + seed on every row, and any single episode can be re-run and filmed on its own.

## Results

Hand-tracking noise is swept with a synthetic tracker (`synthetic_hand.py`) that emits the same
21-landmark format as MediaPipe, with per-landmark jitter (3× larger along depth) and random
dropped frames. Success rate over 100 episodes, 95% CI in brackets:

| tracking noise (jitter, dropout) | no smoothing | k=5 | k=9 |
|---|---|---|---|
| clean (0.2 cm, 0%) | 0.95 [0.89, 0.98] | 0.98 [0.93, 0.99] | 1.00 [0.96, 1.00] |
| phone (1 cm, 10%) | 0.78 [0.69, 0.85] | 0.97 [0.92, 0.99] | 1.00 [0.96, 1.00] |
| poor (2 cm, 15%) | 0.41 [0.32, 0.51] | 0.96 [0.90, 0.98] | 0.97 [0.92, 0.99] |
| bad (3 cm, 20%) | 0.14 [0.09, 0.22] | 0.78 [0.69, 0.85] | 0.86 [0.78, 0.91] |
| very bad (4.5 cm, 30%) | 0.07 [0.03, 0.14] | 0.48 [0.38, 0.58] | 0.68 [0.58, 0.76] |

| baseline (phone noise) | success |
|---|---|
| scripted policy with privileged state (upper bound) | 1.00 [0.96, 1.00] |
| **object-centric retargeting (ours, k=5)** | **0.97 [0.92, 0.99]** |
| fixed calibration, no re-anchoring | 0.04 [0.02, 0.10] |

Takeaways:
1. **Re-anchoring is what makes transfer work at all** (4% → 97%). Without it the robot reaches
   for where the human's object was; 93 of 100 failures are `never_grasped`.
2. **Smoothing buys a lot of noise tolerance**, but not for free: as noise grows, failures shift
   from `never_grasped` (jittery approach misses the object) to `missed_goal` (smoothing lags the
   placement).
3. Full per-run failure counts and the gallery are in [`results/REPORT.md`](results/REPORT.md).

## Engineering notes (things that would have silently corrupted results)

* **Filming changed the physics.** Fetch's `env.render()` runs a callback that calls
  `mj_forward`, which refreshes body positions the next mocap action reads. A filmed episode could
  end differently from the same episode run unfilmed (seed 10045 went from `dropped_in_transit`
  to `missed_goal`). Fix: place the goal marker once at reset and read pixels straight from the
  renderer. The gallery script now asserts filmed outcome == evaluated outcome for every clip.
* **Task definition mismatch.** Stock FetchPickAndPlace puts half its goals in mid-air and only
  counts success if the robot is still *holding* the object at the last step. A human "put it
  there" demo ends with a release, so the in-air variant measures the task definition, not transfer.
  Goals are placed on the table (`sim.TablePlace`).
* **Version pinning.** Newer MuJoCo (tested: 3.15) breaks Fetch's joint setup in gymnasium-robotics 1.4.2;
  `mujoco==3.3.0` is pinned.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # unit tests
python scripts/run_experiments.py           # full grid (~6 min on one CPU core) + gallery + REPORT.md
python scripts/run_experiments.py --only replay_phone_k5   # one run
```

Or with Docker: `docker build -t hand2sim . && docker run --rm -v $PWD/results:/app/results hand2sim`.

### With your own hand video

Record a ~3 s pinch-pick-and-place from a fixed phone camera (hand side-on, fingers visible), then:

```bash
curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python -m hand2sim.extract_keypoints my_demo.mp4 --model hand_landmarker.task
python scripts/replay_video.py my_demo.npy --seeds 0 1 2 3 4   # same demo, 5 different robot scenes
```

## Limitations and next steps

* Noise results use a synthetic tracker so noise can be controlled; real-video numbers are next.
* Single task, single arm, open-loop plan following. The robot does not react if the object moves.
* A plain MLP behavior-cloning policy trained on the retargeted demos did poorly in early tests
  (5–17%). Human pauses make actions ambiguous given state alone, which points to action chunking
  (ACT / Diffusion Policy via LeRobot) as the next step.
* Port the evaluation harness to TRI's LBM Eval (Drake) for multi-task, bimanual evaluation.

## Layout

```
hand2sim/   synthetic_hand.py  extract_keypoints.py  retarget.py  policies.py  sim.py  harness.py
scripts/    run_experiments.py  replay_video.py
tests/      test_core.py
results/    summary.json  REPORT.md  runs/*.jsonl  gallery/*.gif
```

MIT License.

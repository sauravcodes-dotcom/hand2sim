"""Extract 3D hand landmarks from a phone video with MediaPipe HandLandmarker.

    python -m hand2sim.extract_keypoints my_demo.mp4 --model hand_landmarker.task -o my_demo.npy

Output: (T, 21, 3) float array in camera frame (x right, y down, z depth, meters),
NaN rows where the hand wasn't detected. Same format as synthetic_hand.py, so
everything downstream is shared.

Depth from a single camera: MediaPipe gives normalized image coordinates plus
only *relative* depth. We recover absolute depth from apparent hand size,
Z = f * L / l_px, with L = wrist-to-middle-knuckle length (~9 cm for an adult
hand) and f approximated from the field of view. The error this introduces is
mostly along depth, which is exactly the noise axis the experiments stress.

Model file (one-time download):
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
from __future__ import annotations

import argparse

import numpy as np

HAND_LEN_M = 0.09
WRIST, MIDDLE_MCP = 0, 9


def backproject(norm_lm, width, height, hfov_deg=65.0, hand_len=HAND_LEN_M):
    """(21, 3) MediaPipe normalized landmarks -> (21, 3) camera-frame meters."""
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    px = norm_lm[:, :2] * [width, height]
    l_px = np.linalg.norm(px[MIDDLE_MCP] - px[WRIST])
    Z = f * hand_len / max(l_px, 1e-6)
    z = Z + norm_lm[:, 2] * width * Z / f  # MediaPipe z is in roughly image-width units
    x = (px[:, 0] - width / 2) * z / f
    y = (px[:, 1] - height / 2) * z / f
    return np.stack([x, y, z], 1)


def extract(video_path, model_path, hfov_deg=65.0):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    with vision.HandLandmarker.create_from_options(opts) as lmk:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int(i * 1000 / fps))
            if res.hand_landmarks:
                norm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
                out.append(backproject(norm, w, h, hfov_deg))
            else:
                out.append(np.full((21, 3), np.nan))
            i += 1
    cap.release()
    return np.array(out), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--hfov", type=float, default=65.0, help="camera horizontal FOV (deg)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    lm, fps = extract(a.video, a.model, a.hfov)
    out = a.out or a.video.rsplit(".", 1)[0] + ".npy"
    np.save(out, lm)
    tracked = (~np.isnan(lm).any((1, 2))).mean()
    print(f"{len(lm)} frames @ {fps:.0f} fps, hand tracked in {tracked:.0%} -> {out}")


if __name__ == "__main__":
    main()

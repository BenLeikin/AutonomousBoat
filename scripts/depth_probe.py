"""Monocular-depth probe for AutoBoat: does a pretrained depth net see the pool wall?

Runs MiDaS-small (ONNX) over a recorded session's frames and writes:
  - proximity.csv : per-frame center-band proximity metric (higher = closer ahead)
  - depth_XXXXXX.png : colorized depth maps for a sample of frames (visual check)

Run on the Pi or any PC with internet (downloads the 64MB model on first use):
    pip install onnxruntime opencv-python-headless numpy --break-system-packages
    python3 depth_probe.py ~/autoboat-data/sessions/session_YYYYMMDD_HHMMSS

Then zip the probe_out folder it creates and upload it for evaluation. The metric
to beat: proximity must RISE in the seconds before known stalls (telemetry says
when those were) and stay flat during free cruising. If it does, this becomes a
~1 Hz advisory input to the controller; if reflections fool the net, we'll see
that immediately in the colorized maps.
"""
import csv
import os
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midas_small.onnx")
NET_SIZE = 256                    # midas-small input resolution
SAMPLE_EVERY = 1                  # process every Nth frame (1 = all; raise if slow)
SAVE_DEPTH_EVERY = 10             # save a colorized depth map every Nth processed frame


def get_model():
    if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000_000:
        print("downloading midas-small (~64MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def infer_inverse_depth(sess, bgr):
    # MiDaS outputs relative INVERSE depth: larger = closer. Unscaled, so only
    # within-frame comparisons are meaningful; the proximity metric below is a
    # ratio against the near-water reference for that reason.
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    blob = img.transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    return cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def proximity(inv_depth):
    """Center-band closeness relative to the near-water reference.

    center band = rows 35-75%, cols 25-75% (where a wall ahead sits in frame)
    reference   = bottom strip rows 90-100% (water right at the bow: a known-near
                  surface, so the ratio self-calibrates each frame)
    Open water ahead -> center reads much farther than the bow strip -> ratio well
    below 1. Wall filling the view -> center nearly as close as the bow -> ratio
    approaches/exceeds ~0.8-1. Those are the hypotheses this probe tests.
    """
    h, w = inv_depth.shape
    center = inv_depth[int(h * 0.35):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
    ref = inv_depth[int(h * 0.90):, :]
    ref_med = float(np.median(ref))
    if ref_med <= 1e-6:
        return None
    return float(np.median(center) / ref_med)


def colorize(inv_depth):
    d = inv_depth - inv_depth.min()
    d = (255 * d / max(d.max(), 1e-6)).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 depth_probe.py <session_dir_or_frames_dir>")
        sys.exit(1)
    root = sys.argv[1]
    frames_dir = os.path.join(root, "frames") if os.path.isdir(os.path.join(root, "frames")) else root
    names = sorted(n for n in os.listdir(frames_dir)
                   if n.startswith("frame_") and n.endswith((".png", ".jpg")))
    if not names:
        print("no frames found in", frames_dir)
        sys.exit(1)
    out_dir = os.path.join(root, "probe_out")
    os.makedirs(out_dir, exist_ok=True)
    sess = get_model()
    print("processing %d frames from %s" % (len(names), frames_dir))
    import time as _t
    rows, t0 = [], _t.time()
    for i, name in enumerate(names):
        if i % SAMPLE_EVERY:
            continue
        img = cv2.imread(os.path.join(frames_dir, name))
        if img is None:
            continue
        inv = infer_inverse_depth(sess, img)
        p = proximity(inv)
        rows.append((name, "%.4f" % p if p is not None else ""))
        if (len(rows) % SAVE_DEPTH_EVERY) == 1:
            side = np.hstack([img, colorize(inv)])
            cv2.imwrite(os.path.join(out_dir, "depth_" + name), side)
        if len(rows) % 25 == 0:
            rate = len(rows) / (_t.time() - t0)
            print("  %d frames, %.2f fps" % (len(rows), rate))
    with open(os.path.join(out_dir, "proximity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "proximity"])
        w.writerows(rows)
    print("done: %d rows -> %s" % (len(rows), out_dir))
    print("zip the probe_out folder (plus the session's telemetry.csv) and upload it")


if __name__ == "__main__":
    main()

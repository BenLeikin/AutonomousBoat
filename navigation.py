#!/usr/bin/env python3
"""
Navigation logic — VFH gap-finding + IMU stuck detection + pool-wall avoidance.
Compatible with VisionProcessor outputs from BG (contours) or DNN (bboxes).
"""
import os, sys, signal, time, logging, yaml, math
import numpy as np
import cv2

from camera import CameraReader
from vision import VisionProcessor
from imu import IMU
import motor_control as mc

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f)
except Exception as e:
    print(f"[Error] Failed to load config: {e}")
    sys.exit(1)

nav_cfg = _full_cfg.get('navigation', {}) or {}
log_cfg = _full_cfg.get('logging', {}) or {}

NUM_SECTORS      = int(nav_cfg.get('num_sectors', 36))
MIN_GAP          = int(nav_cfg.get('min_gap_width', 3))
WALL_MARGIN_FRAC = float(nav_cfg.get('wall_margin_frac', 0.05))
ANGLE_TOL_RAD    = float(nav_cfg.get('angle_threshold_rad', math.radians(10)))
LOOP_DELAY       = float(nav_cfg.get('loop_delay', 0.1))

stuck_cfg    = nav_cfg.get('stuck', {})
STUCK_WINDOW = float(stuck_cfg.get('window_sec', 2.0))
MOTION_THRESH= float(stuck_cfg.get('motion_thresh', 0.01))
REVERSE_DUR  = float(stuck_cfg.get('reverse_dur', 1.0))

# --- Logging ---
level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

motion_history = []  # (timestamp, |ax|)

def shutdown(signum, frame):
    logger.info("Shutdown signal received, stopping motors")
    mc.stop_all()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# --- Helpers ---

def _obj_center(det):
    """Return (cx, cy) for either a contour or a bbox detection dict."""
    if isinstance(det, dict):
        if 'contour' in det:
            c = det['contour']
            M = cv2.moments(c)
            if M['m00'] == 0:
                return None
            return int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        if 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            return (x1 + x2) // 2, (y1 + y2) // 2
        return None
    # Back-compat: raw contour array
    if hasattr(det, 'shape'):
        M = cv2.moments(det)
        if M['m00'] == 0:
            return None
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    return None

def _touches_wall(det, w, h):
    """True if detection touches image edges within WALL_MARGIN_FRAC."""
    m = WALL_MARGIN_FRAC
    if isinstance(det, dict) and 'bbox' in det:
        x1, y1, x2, y2 = det['bbox']
        return (x1 <= w*m) or (x2 >= w*(1-m)) or (y1 <= h*m) or (y2 >= h*(1-m))
    if isinstance(det, dict) and 'contour' in det:
        x, y, cw, ch = cv2.boundingRect(det['contour'])
        return (x <= w*m) or (x+cw >= w*(1-m)) or (y <= h*m) or (y+ch >= h*(1-m))
    if hasattr(det, 'shape'):
        x, y, cw, ch = cv2.boundingRect(det)
        return (x <= w*m) or (x+cw >= w*(1-m)) or (y <= h*m) or (y+ch >= h*(1-m))
    return False

def compute_histogram(dets, frame_shape):
    """Build Boolean occupancy histogram from detections (contours or bboxes)."""
    h, w = frame_shape
    hist = np.zeros(NUM_SECTORS, dtype=bool)
    for det in dets:
        ctr = _obj_center(det)
        if ctr is None:
            continue
        cx, cy = ctr
        angle = math.atan2((h/2) - cy, cx - (w/2))
        idx = int((angle + math.pi) / (2*math.pi) * NUM_SECTORS) % NUM_SECTORS
        hist[idx] = True
        if _touches_wall(det, w, h):
            hist[idx] = True
    return hist

def find_best_gap(hist: np.ndarray) -> float:
    """Return steering angle (rad) for the largest free sector gap."""
    best_len = curr_len = 0
    best_start = curr_start = 0
    doubled = np.concatenate([hist, hist])
    for i, occ in enumerate(doubled):
        if not occ:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
        else:
            curr_len = 0
    if best_len < MIN_GAP:
        return 0.0
    mid = (best_start + best_len // 2) % NUM_SECTORS
    return (mid / NUM_SECTORS) * 2*math.pi - math.pi

def detect_stuck(imu_data: dict) -> bool:
    now = time.time()
    motion = abs(imu_data.get('ax', 0.0))
    motion_history.append((now, motion))
    cutoff = now - STUCK_WINDOW
    while motion_history and motion_history[0][0] < cutoff:
        motion_history.pop(0)
    return all(m <= MOTION_THRESH for _, m in motion_history)

def apply_motion(angle: float) -> None:
    if abs(angle) <= ANGLE_TOL_RAD:
        mc.both_forward()
    elif angle > 0:
        mc.pivot_right()
    else:
        mc.pivot_left()

# --- Main ---

def main() -> None:
    with CameraReader() as cam, IMU() as imu:
        vision = VisionProcessor()
        logger.info("Navigation loop started")
        while True:
            frame = cam.read()
            if frame is None:
                continue

            detections = vision.detect(frame)  # may be contours or bboxes
            imu_data = imu.read()

            hist = compute_histogram(detections, frame.shape[:2])
            best_angle = find_best_gap(hist)

            if detect_stuck(imu_data):
                logger.warning("Stuck detected, reversing for %.2fs", REVERSE_DUR)
                mc.both_reverse()
                time.sleep(REVERSE_DUR)
            else:
                apply_motion(best_angle)

            time.sleep(LOOP_DELAY)

    mc.stop_all()

if __name__ == '__main__':
    main()

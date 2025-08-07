#!/usr/bin/env python3
"""
Navigation logic — merges VFH gap-finding, vision contours,
IMU-based stuck detection, and pool-wall avoidance.
"""
import os
import sys
import signal
import time
import logging
import yaml
import math
import numpy as np
from camera import CameraReader
from vision import VisionProcessor
from imu import IMU
import motor_control as mc

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f)
except Exception as e:
    print(f"[Error] Failed to load config: {e}")
    sys.exit(1)
nav_cfg = _full_cfg.get('navigation', {}) or {}
log_cfg = _full_cfg.get('logging', {}) or {}

# Navigation parameters
NUM_SECTORS      = int(nav_cfg.get('num_sectors', 36))
MIN_GAP          = int(nav_cfg.get('min_gap_width', 3))
WALL_MARGIN_FRAC = float(nav_cfg.get('wall_margin_frac', 0.05))
ANGLE_TOL_RAD    = float(nav_cfg.get('angle_threshold_rad', math.radians(10)))
LOOP_DELAY       = float(nav_cfg.get('loop_delay', 0.1))

# Stuck detection parameters
stuck_cfg    = nav_cfg.get('stuck', {})
STUCK_WINDOW = float(stuck_cfg.get('window_sec', 2.0))
MOTION_THRESH= float(stuck_cfg.get('motion_thresh', 0.01))
REVERSE_DUR  = float(stuck_cfg.get('reverse_dur', 1.0))

# Set up logging
level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
logging.basicConfig(level=level,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Motion history for stuck detection: list of (timestamp, motion_value)
motion_history = []

def shutdown(signum, frame):
    logger.info("Shutdown signal received, stopping motors")
    mc.stop_all()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def compute_histogram(contours: list, frame_shape: tuple) -> np.ndarray:
    """Build a Boolean occupancy histogram from vision contours."""
    h, w = frame_shape
    hist = np.zeros(NUM_SECTORS, dtype=bool)
    sector_angle = 2 * math.pi / NUM_SECTORS

    for c in contours:
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # angle from center to contour point
        angle = math.atan2((h/2) - cy, cx - (w/2))
        idx = int((angle + math.pi) / (2*math.pi) * NUM_SECTORS) % NUM_SECTORS
        hist[idx] = True
        # mark walls (contours touching frame edges)
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= w*WALL_MARGIN_FRAC or (x+cw) >= w*(1-WALL_MARGIN_FRAC) \
           or y <= h*WALL_MARGIN_FRAC or (y+ch) >= h*(1-WALL_MARGIN_FRAC):
            hist[idx] = True
    return hist


def find_best_gap(hist: np.ndarray) -> float:
    """Select the mid-angle of the largest free (False) gap in the histogram."""
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
    mid = (best_start + best_len//2) % NUM_SECTORS
    return (mid / NUM_SECTORS) * 2*math.pi - math.pi


def detect_stuck(imu_data: dict) -> bool:
    """Return True if no significant motion detected within window."""
    now = time.time()
    motion = abs(imu_data.get('ax', 0.0))
    motion_history.append((now, motion))
    # purge old entries
    cutoff = now - STUCK_WINDOW
    while motion_history and motion_history[0][0] < cutoff:
        motion_history.pop(0)
    # if all motions below threshold, we're stuck
    return all(m <= MOTION_THRESH for _, m in motion_history)


def apply_motion(angle: float) -> None:
    """Map a steering angle to motor commands."""
    if abs(angle) <= ANGLE_TOL_RAD:
        mc.both_forward()
    elif angle > 0:
        mc.pivot_right()
    else:
        mc.pivot_left()


def main() -> None:
    with CameraReader() as cam, IMU() as imu:
        vision = VisionProcessor()
        logger.info("Navigation loop started")
        while True:
            frame = cam.read()
            if frame is None:
                continue

            contours = vision.detect(frame)
            imu_data = imu.read()

            hist = compute_histogram(contours, frame.shape[:2])
            best_angle = find_best_gap(hist)

            if detect_stuck(imu_data):
                logger.warning("Stuck detected, reversing for %.2f seconds", REVERSE_DUR)
                mc.both_reverse()
                time.sleep(REVERSE_DUR)
            else:
                apply_motion(best_angle)

            time.sleep(LOOP_DELAY)

    mc.stop_all()


if __name__ == '__main__':
    main()

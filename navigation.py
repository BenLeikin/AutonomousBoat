#!/usr/bin/env python3
"""
Navigation — Reactive 90° with pre-roll, edge/Hough wall fallback,
auto-correct recording FPS, ROI overlay, and target trace.

- Records at loop FPS when recording.fps == 0 (prevents time-lapse playback)
- Draws forward ROI and a trace of the forward target over time
- ObstacleAhead via: DNN boxes, contours, edge density, and Hough horizontal line
"""

import os, sys, signal, time, logging, yaml, math
import numpy as np
import cv2
from collections import deque

from camera import CameraReader
from vision import VisionProcessor
from imu import IMU
from annotator import Annotator
import motor_control as mc

# ─────────── Config ───────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        CFG = yaml.safe_load(f) or {}
except Exception as e:
    print(f"[Error] Failed to load config: {e}")
    sys.exit(1)

cam_cfg = CFG.get('camera', {}) or {}
nav_cfg = CFG.get('navigation', {}) or {}
log_cfg = CFG.get('logging', {}) or {}
rec_cfg = CFG.get('recording', {}) or {}
ann_cfg = CFG.get('annotator', {}) or {}

MODE               = str(nav_cfg.get('mode', 'reactive_90'))
LOOP_DELAY         = float(nav_cfg.get('loop_delay', 0.1))             # controls loop FPS
FWD_CORRIDOR_FRAC  = float(nav_cfg.get('forward_corridor_frac', 0.60))
FWD_BAND_FRAC      = float(nav_cfg.get('forward_band_frac', 0.45))
MIN_BOX_H_FRAC     = float(nav_cfg.get('min_bbox_height_frac', 0.25))
DEFAULT_TURN       = str(nav_cfg.get('default_turn', 'right'))
TURN_90_DURATION_S = float(nav_cfg.get('turn_90_duration_s', 0.8))
POST_TURN_COAST_S  = float(nav_cfg.get('post_turn_coast_s', 1.0))

PREROLL_SEC            = float(nav_cfg.get('preroll_sec', 5.0))
PREROLL_BLOCKED_RATIO  = float(nav_cfg.get('preroll_blocked_ratio', 0.4))

edge_cfg           = nav_cfg.get('edge_fallback', {}) or {}
EDGE_ENABLED       = bool(edge_cfg.get('enabled', True))
CANNY1             = int(edge_cfg.get('canny1', 60))
CANNY2             = int(edge_cfg.get('canny2', 180))
OPEN_K             = int(edge_cfg.get('open_kernel', 3))
EDGE_ENTER         = float(edge_cfg.get('enter_density', 0.08))
EDGE_EXIT          = float(edge_cfg.get('exit_density', 0.04))
CLAHE_CLIP         = float(edge_cfg.get('clahe_clip', 0.0))
CLAHE_GRID         = int(edge_cfg.get('clahe_grid', 8))
USE_HOUGH          = bool(edge_cfg.get('use_hough', True))
HOUGH_MIN_LINE_FR  = float(edge_cfg.get('hough_min_line_frac', 0.5))
HOUGH_THETA_TOL    = float(edge_cfg.get('hough_theta_tol_deg', 15.0))
HOUGH_THRESH       = int(edge_cfg.get('hough_threshold', 30))
HOUGH_MAX_GAP      = int(edge_cfg.get('hough_max_gap', 10))

stuck_cfg          = nav_cfg.get('stuck', {}) or {}
STUCK_ENABLED      = bool(stuck_cfg.get('enabled', True))
STUCK_WINDOW       = float(stuck_cfg.get('window_sec', 3.0))
YAW_THRESH_DPS     = float(stuck_cfg.get('yaw_rate_thresh_dps', 1.5))
USE_ACCEL          = bool(stuck_cfg.get('use_accel', False))
ACCEL_THRESH       = float(stuck_cfg.get('accel_thresh', 0.03))
RECOVER_REVERSE_DUR= float(stuck_cfg.get('reverse_dur', 0.5))

REC_ENABLED  = bool(rec_cfg.get('enabled', False))
REC_DIR      = rec_cfg.get('dir', '/mnt/usb')
REC_PREFIX   = rec_cfg.get('filename_prefix', 'annotated')
REC_FPS      = int(rec_cfg.get('fps', 0))  # 0 => auto = loop fps
REC_FOURCC   = rec_cfg.get('fourcc', 'mp4v')
ANN_FLAGS    = rec_cfg.get('annotate', {}) or {}
_det_style   = ann_cfg.get('det', {}) or {}
BOX_COLOR    = tuple(_det_style.get('box_color', [0, 255, 255]))
BOX_TH       = int(_det_style.get('box_thickness', 2))
CNT_COLOR    = tuple(_det_style.get('contour_color', [0, 255, 255]))
TXT_COLOR    = tuple(_det_style.get('text_color', [0, 255, 255]))

# Logging
level = getattr(logging, str(log_cfg.get('level', 'INFO')).upper(), logging.INFO)
logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────── State ───────────
CRUISE, TURN_90 = 0, 1
state = CRUISE
turn_dir = DEFAULT_TURN
turn_end = 0.0
cooldown_until = 0.0
last_turn_dir = DEFAULT_TURN

yaw_hist: list[tuple[float, float]] = []
writer = None
video_path = None
last_edge_blocked = False  # hysteresis

# Trace of forward target center (on-video)
TRACE = deque(maxlen=120)   # ~12s at 10 fps
TRACE_COLOR = (0, 255, 0)

# concise status logs
_last_action = None
_last_blocked = None
_last_causes = None

# ─────────── Signals ───────────
def shutdown(*_):
    logger.info("Shutdown signal received, stopping motors")
    try: mc.stop_all()
    except Exception: pass
    global writer
    if writer is not None:
        try: writer.release(); logger.info("Video writer released (%s)", video_path)
        except Exception: pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ─────────── Helpers ───────────
def forward_roi_rect(shape):
    h, w = shape
    x0 = int((1.0 - FWD_CORRIDOR_FRAC) * 0.5 * w)
    x1 = w - x0
    y0 = int((1.0 - FWD_BAND_FRAC) * h)
    return x0, y0, x1, h

def obstacle_ahead_dnn(dets, shape) -> bool:
    h, w = shape
    x0, y0, x1, _ = forward_roi_rect(shape)
    for det in dets:
        if not (isinstance(det, dict) and 'bbox' in det):
            continue
        xa, ya, xb, yb = det['bbox']
        cx = (xa + xb) // 2
        h_frac = (yb - ya) / max(1, h)
        if (x0 <= cx <= x1) and (yb >= y0) and (h_frac >= MIN_BOX_H_FRAC):
            return True
    return False

def obstacle_ahead_contours(dets, shape) -> bool:
    h, w = shape
    x0, y0, x1, _ = forward_roi_rect(shape)
    for det in dets:
        if not (isinstance(det, dict) and 'contour' in det):
            continue
        x, y, bw, bh = cv2.boundingRect(det['contour'])
        cx = x + bw // 2
        if (x0 <= cx <= x1) and (y + bh >= y0) and (bh / max(1, h) >= MIN_BOX_H_FRAC):
            return True
    return False

def edge_blocked(frame) -> tuple[bool, float, bool]:
    """Edge density + optional Hough horizontal line in forward ROI."""
    global last_edge_blocked
    if not EDGE_ENABLED:
        return False, 0.0, False
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = forward_roi_rect((h, w))
    roi = frame[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if CLAHE_CLIP > 0.0:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    edges = cv2.Canny(gray, CANNY1, CANNY2)
    if OPEN_K > 1:
        k = np.ones((OPEN_K, OPEN_K), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k)

    density = float(np.count_nonzero(edges)) / max(1, edges.size)
    enter = density >= EDGE_ENTER
    exit_  = density >= EDGE_EXIT
    blocked_density = enter if not last_edge_blocked else exit_

    hough_hit = False
    if USE_HOUGH:
        min_len = int(HOUGH_MIN_LINE_FR * max(1, edges.shape[1]))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESH,
                                minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP)
        if lines is not None:
            for (x1l, y1l, x2l, y2l) in lines[:,0,:]:
                angle = abs(math.degrees(math.atan2((y2l - y1l), (x2l - x1l))))
                if angle <= HOUGH_THETA_TOL or abs(angle-180) <= HOUGH_THETA_TOL:
                    hough_hit = True
                    break

    blocked = blocked_density or hough_hit
    last_edge_blocked = blocked
    return blocked, density, hough_hit

def choose_turn_direction(dets, w):
    left = right = 0
    for det in dets:
        if not (isinstance(det, dict) and ('bbox' in det or 'contour' in det)):
            continue
        if 'bbox' in det:
            x1b, y1b, x2b, y2b = det['bbox']; cx = (x1b + x2b) // 2
        else:
            x, y, bw, bh = cv2.boundingRect(det['contour']); cx = x + bw // 2
        if cx < w // 2: left += 1
        else: right += 1
    if left == right: return DEFAULT_TURN
    return 'left' if right > left else 'right'

def update_yaw(gz_abs):
    now = time.time()
    yaw_hist.append((now, gz_abs))
    cutoff = now - STUCK_WINDOW
    while yaw_hist and yaw_hist[0][0] < cutoff:
        yaw_hist.pop(0)

def is_stuck(cruising, blocked, imu_data) -> bool:
    if not (STUCK_ENABLED and cruising and blocked and yaw_hist):
        return False
    yaw_ok = all(v <= YAW_THRESH_DPS for _, v in yaw_hist)
    if not USE_ACCEL:
        return yaw_ok
    ax = float(imu_data.get('ax', 0.0)); ay = float(imu_data.get('ay', 0.0))
    accel_mag = math.hypot(ax, ay)
    return yaw_ok and (accel_mag <= ACCEL_THRESH)

def pivot(direction: str):
    if direction == 'left': mc.pivot_left()
    else: mc.pivot_right()

def init_writer_if_needed(frame, loop_fps):
    """Auto set recording fps to loop fps when config fps==0 to avoid time-lapse."""
    global writer, video_path
    if not REC_ENABLED or writer is not None: return
    try:
        os.makedirs(REC_DIR, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(REC_DIR, f"{REC_PREFIX}_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*REC_FOURCC)
        out_fps = REC_FPS if REC_FPS > 0 else max(1, int(round(loop_fps)))
        h, w = frame.shape[:2]
        wr = cv2.VideoWriter(video_path, fourcc, out_fps, (w, h))
        if not wr.isOpened():
            logger.warning("Failed to open VideoWriter at %s; disabling recording", video_path)
            return
        writer = wr
        logger.info("Recording annotated video to %s (fps=%d)", video_path, out_fps)
    except Exception as e:
        logger.warning("Could not initialize VideoWriter: %s", e)

def draw_roi_and_trace(img):
    x0, y0, x1, y1 = forward_roi_rect(img.shape[:2])
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)  # ROI box
    # trace polyline
    if len(TRACE) >= 2:
        pts = np.array(TRACE, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=TRACE_COLOR, thickness=2)
    return img

def annotate_and_record(frame, detections, action_str, thoughts="", loop_fps=10.0):
    if not REC_ENABLED or writer is None: return
    overlay = frame.copy()
    # detections
    for det in detections:
        if isinstance(det, dict) and 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, BOX_TH)
        elif isinstance(det, dict) and 'contour' in det:
            cv2.drawContours(overlay, [det['contour']], -1, CNT_COLOR, BOX_TH)
    # action & thoughts
    annot = Annotator()
    overlay = annot.draw_action(overlay, action_str)
    if thoughts:
        for i, line in enumerate(thoughts.split("\n")[:4]):
            cv2.putText(overlay, line, (10, 20 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    overlay = draw_roi_and_trace(overlay)
    try:
        writer.write(overlay)
    except Exception as e:
        logger.warning("VideoWriter write failed: %s", e)

def log_status_if_changed(action: str, blocked: bool, causes: str):
    global _last_action, _last_blocked, _last_causes
    if action != _last_action or blocked != _last_blocked or causes != _last_causes:
        if blocked:
            logger.info("Motors: %-18s | ObstacleAhead: TRUE  via %s", action, causes or "unknown")
        else:
            logger.info("Motors: %-18s | ObstacleAhead: FALSE", action)
        _last_action, _last_blocked, _last_causes = action, blocked, causes

def pick_forward_target(dets, shape):
    """Return (cx,cy) of the closest detection inside forward ROI for trace."""
    h, w = shape
    x0, y0, x1, _ = forward_roi_rect(shape)
    best = None; best_y2 = -1
    for det in dets:
        if isinstance(det, dict) and 'bbox' in det:
            x1b, y1b, x2b, y2b = det['bbox']
            cx = (x1b + x2b) // 2
            if x0 <= cx <= x1 and y2b >= y0:
                if y2b > best_y2:
                    best_y2 = y2b
                    best = (cx, (y1b + y2b)//2)
        elif isinstance(det, dict) and 'contour' in det:
            x, y, bw, bh = cv2.boundingRect(det['contour'])
            cx = x + bw // 2
            y2 = y + bh
            if x0 <= cx <= x1 and y2 >= y0:
                if y2 > best_y2:
                    best_y2 = y2
                    best = (cx, y + bh//2)
    return best

# ─────────── Pre-roll analysis ───────────
def preroll_phase(cam, vision, seconds, loop_fps):
    logger.info("Pre-roll analysis for %.1fs (no motion)", seconds)
    start = time.time()
    samples = blocked_samples = left_ct = right_ct = 0
    avg_edge = 0.0

    while time.time() - start < seconds:
        frame = cam.read()
        if frame is None:
            time.sleep(LOOP_DELAY); continue

        init_writer_if_needed(frame, loop_fps)

        dets = vision.detect(frame)
        h, w = frame.shape[:2]
        b_dnn = obstacle_ahead_dnn(dets, (h, w))
        b_cnt = obstacle_ahead_contours(dets, (h, w))
        b_edge, edge_density, b_hough = edge_blocked(frame)
        blocked = b_dnn or b_cnt or b_edge or b_hough

        samples += 1
        if blocked: blocked_samples += 1
        avg_edge += edge_density

        # side preference for first turn
        for det in dets:
            if isinstance(det, dict) and 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']; cx = (x1 + x2) // 2
            elif isinstance(det, dict) and 'contour' in det:
                x, y, bw, bh = cv2.boundingRect(det['contour']); cx = x + bw // 2
            else:
                continue
            if cx < w // 2: left_ct += 1
            else: right_ct += 1

        tgt = pick_forward_target(dets, (h, w))
        if tgt: TRACE.append(tgt)

        thoughts = f"PREROLL blocked={blocked} dnn={b_dnn} cnt={b_cnt} edge={edge_density:.2f} hough={b_hough}"
        annotate_and_record(frame, dets, "Pre-roll", thoughts, loop_fps)
        time.sleep(LOOP_DELAY)

    blocked_ratio = (blocked_samples / samples) if samples else 0.0
    logger.info("Pre-roll result: samples=%d blocked_ratio=%.2f avg_edge=%.3f (L:%d R:%d)",
                samples, blocked_ratio, (avg_edge/samples if samples else 0.0), left_ct, right_ct)

    start_with_turn = blocked_ratio >= PREROLL_BLOCKED_RATIO
    chosen = DEFAULT_TURN if left_ct == right_ct else ('left' if right_ct > left_ct else 'right')
    return start_with_turn, chosen

# ─────────── Main loop ───────────
def main():
    global state, turn_dir, turn_end, cooldown_until, last_turn_dir

    loop_fps = 1.0 / max(LOOP_DELAY, 1e-6)

    with CameraReader() as cam, IMU() as imu:
        vision = VisionProcessor()
        logger.info("Navigation loop starting…")
        mc.stop_all()
        logger.info("Motors: Stopped | ObstacleAhead: (pre-roll)")

        # Pre-roll (no motion)
        start_with_turn, first_dir = preroll_phase(cam, vision, PREROLL_SEC, loop_fps)

        if start_with_turn:
            turn_dir = first_dir; last_turn_dir = turn_dir
            turn_end = time.time() + TURN_90_DURATION_S
            pivot(turn_dir); state = TURN_90
            action = f"Turning 90° {turn_dir}"
        else:
            mc.both_forward(); state = CRUISE
            action = "Forward"

        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(LOOP_DELAY); continue

            init_writer_if_needed(frame, loop_fps)

            dets = vision.detect(frame)
            imu_data = imu.read()
            gz_abs = abs(float(imu_data.get('gz', 0.0)))
            update_yaw(gz_abs)

            h, w = frame.shape[:2]
            b_dnn   = obstacle_ahead_dnn(dets, (h, w))
            b_cnt   = obstacle_ahead_contours(dets, (h, w))
            b_edge, edge_density, b_hough = edge_blocked(frame)
            blocked = b_dnn or b_cnt or b_edge or b_hough

            causes_list = []
            if b_dnn:  causes_list.append("dnn")
            if b_cnt:  causes_list.append("contour")
            if b_edge: causes_list.append(f"edge:{edge_density:.2f}")
            if b_hough: causes_list.append("hough")
            causes = ",".join(causes_list)

            # update trace with closest forward target
            tgt = pick_forward_target(dets, (h, w))
            if tgt: TRACE.append(tgt)

            now = time.time()
            if state == CRUISE:
                if now < cooldown_until:
                    mc.both_forward(); action = 'Forward (Cooldown)'
                else:
                    if blocked:
                        turn_dir = choose_turn_direction(dets, w)
                        last_turn_dir = turn_dir
                        turn_end = now + TURN_90_DURATION_S
                        state = TURN_90
                        pivot(turn_dir)
                        action = f"Turning 90° {turn_dir}"
                        logger.info("Blocked → TURN_90 %s (via %s)", turn_dir, causes or "unknown")
                    else:
                        mc.both_forward(); action = 'Forward'

                if is_stuck(cruising=True, blocked=blocked, imu_data=imu_data):
                    logger.warning("Stuck: reverse %.1fs then 90° turn", RECOVER_REVERSE_DUR)
                    mc.both_reverse(); time.sleep(RECOVER_REVERSE_DUR)
                    turn_dir = 'left' if last_turn_dir == 'right' else 'right'
                    last_turn_dir = turn_dir
                    turn_end = time.time() + TURN_90_DURATION_S
                    state = TURN_90
                    pivot(turn_dir)
                    action = f"Recover Turn 90° {turn_dir}"
                    causes = (causes + ",stuck").strip(",")

            elif state == TURN_90:
                if now >= turn_end:
                    mc.both_forward(); action = 'Forward (Post-Turn)'
                    state = CRUISE; cooldown_until = now + POST_TURN_COAST_S
                else:
                    pivot(turn_dir); action = f"Turning 90° {turn_dir}"

            # Log concise status on change
            log_status_if_changed(action, blocked, causes)

            # Thoughts line for overlay
            t_turn_left = max(0.0, turn_end - now) if state == TURN_90 else 0.0
            t_cool_left = max(0.0, cooldown_until - now) if state == CRUISE else 0.0
            thoughts = (f"state={'CRUISE' if state==CRUISE else 'TURN_90'} action={action}\n"
                        f"blocked={blocked} via={causes or 'none'} edge={edge_density:.2f} hough={b_hough}\n"
                        f"yaw_dps={gz_abs:.2f} t_turn={t_turn_left:.2f}s t_cool={t_cool_left:.2f}s")

            annotate_and_record(frame, dets, action, thoughts, loop_fps)
            time.sleep(LOOP_DELAY)

    mc.stop_all()
    if writer is not None:
        try: writer.release(); logger.info("Video writer released (%s)", video_path)
        except Exception: pass

if __name__ == '__main__':
    main()

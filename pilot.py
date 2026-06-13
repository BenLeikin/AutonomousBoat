"""The pilot: owns the camera, runs vision -> controller -> actuation each frame,
maintains the live `proc`/frame buffers the dashboard serves, computes the
frame-motion stuck signal, and orchestrates Start/Stop (controller reset + arm +
auto-record). Also hosts offline re-analysis for the Analyze tab, since this is
the module that owns the vision stack."""
import os
import threading
import time

import numpy as np

from control import controller as autoboat_control
import hardware.imu as imu
import hardware.motors as motors
import hardware.power as hwpower
import hardware.tof as hwtof
import recording

# Vision stack is optional: if it's missing (e.g. running on a box without
# picamera2/opencv), the camera section just shows "--" and everything else works.
try:
    import cv2
    from picamera2 import Picamera2
    from vision.pipeline import analyze, annotate
    CAMERA_AVAILABLE = True
except Exception as e:
    print(f"[camera] vision stack unavailable, camera section disabled: {e}")
    CAMERA_AVAILABLE = False

recording.set_camera_available(lambda: CAMERA_AVAILABLE)

# The avoidance brain. One instance; reset on Start so each run begins in assess.
CONTROLLER = autoboat_control.Controller()

# ---------- Camera + vision shared state ----------
proc = {"fps": 0.0, "latency_ms": 0.0, "best_zone": 0,
        "center_pct": 0.0, "zones": [0, 0, 0, 0, 0], "cam_ok": False,
        "cmd": {"mode": "run", "turn": 0.0, "throttle": 0.0, "left": 0.0,
                "right": 0.0, "reason": "idle", "armed": False}}
plock = threading.Lock()
frame_buf = {"jpeg": None}
flock = threading.Lock()
raw_buf = {"frame": None}         # latest raw BGR frame, for on-demand snapshots
rlock = threading.Lock()

def camera_loop():
    # NOTE: this opens the camera directly for monitoring. Once a real control
    # loop owns the camera, only one process can hold the CSI device, so at that
    # point the dashboard should read frames from the control loop instead of
    # opening the camera itself.
    try:
        picam = Picamera2()
        cfg = picam.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"})
        picam.configure(cfg)
        picam.start()
        time.sleep(0.5)
    except Exception as e:
        print(f"[camera] init failed: {e}")
        return

    count = 0
    win = time.monotonic()
    fps = 0.0
    # Frame-motion stuck signal: mean |gray difference| vs a reference frame ~0.5s
    # old (consecutive frames at 16-30fps differ too little to be meaningful). A
    # pinned boat reads ~1 on this scale; a moving one ~15+ (measured on the
    # 20260609 wall sessions). Cheap: 80x60 grayscale.
    mref = {"gray": None, "t": 0.0}
    motion_val = None
    while True:
        try:
            frame = picam.capture_array()         # "RGB888" but actually BGR order
        except Exception:
            time.sleep(0.05)
            continue

        t0 = time.monotonic()
        result = analyze(frame, is_rgb=False)
        latency = (time.monotonic() - t0) * 1000.0

        try:
            small = cv2.cvtColor(cv2.resize(frame, (80, 60)),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
            now_m = time.monotonic()
            if mref["gray"] is not None:
                motion_val = float(np.abs(small - mref["gray"]).mean())
            if now_m - mref["t"] >= 0.4:
                mref["gray"] = small
                mref["t"] = now_m
        except Exception:
            motion_val = None

        with imu.alock:
            yaw_rate = imu.YAW_SIGN * imu.att["gz"]          # rad/s, for the stuck check
        with hwpower.pwlock:
            cur_ma = hwpower.power["current_ma"] if hwpower.power["connected"] else None
        # stale-data guard lives in hardware.tof: a wedged reader returns None
        rng_m = hwtof.fresh_range(max_age_s=1.0)
        cmd = CONTROLLER.step(result.zones, result.center_depth_pct,
                              yaw_rate, time.monotonic(), armed=motors.ARMED,
                              depths=result.depths, current_ma=cur_ma,
                              motion=motion_val, range_m=rng_m)
        # ACTUATION POINT. Drives the DRV8833 from cmd["left"]/cmd["right"] when
        # armed; a no-op with the driver asleep while disarmed. See the motor
        # actuation block (arm/disarm, throttle cap, watchdog) defined below.
        motors.actuate(cmd, time.monotonic())

        vis = annotate(frame, result, is_rgb=False)  # frame is already BGR
        with rlock:
            raw_buf["frame"] = frame
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with flock:
                frame_buf["jpeg"] = buf.tobytes()

        count += 1
        now = time.monotonic()
        if now - win >= 1.0:
            fps = count / (now - win)
            count = 0
            win = now

        with plock:
            proc["fps"] = fps
            proc["latency_ms"] = latency
            proc["best_zone"] = int(result.best_zone)
            proc["center_pct"] = float(result.center_depth_pct)
            proc["zones"] = list(result.zones)
            proc["cam_ok"] = True
            proc["cmd"] = cmd
            proc["tof_m"] = rng_m

        recording.maybe_capture(frame, result, fps, latency, cmd, motion_val, rng_m)



def start_run():
    # Single Start: reset the controller to assess, arm the motors, and auto-record
    # the run. One action so the operator doesn't manage arm + capture separately.
    if not motors.MOTORS_AVAILABLE:
        return {"ok": False, "message": "motor driver not available; cannot start"}
    CONTROLLER.reset(time.monotonic())
    motors.arm()
    cap_res = recording.capture_start()
    sess = cap_res.get("session") if isinstance(cap_res, dict) else None
    msg = "running at %d%% cap" % int(round(motors.MOTOR_THROTTLE_CAP * 100))
    if sess:
        msg += " | recording " + sess
    elif isinstance(cap_res, dict) and cap_res.get("error"):
        msg += " | capture: " + cap_res["error"]
    return {"ok": True, "running": True, "armed": True, "message": msg}


def stop_run():
    # Single Stop: disarm (motors coast) and finalize the run recording.
    motors.disarm()
    cap_res = recording.capture_stop(reason="run_stopped")
    msg = "stopped; motors coasting"
    if isinstance(cap_res, dict) and cap_res.get("frames"):
        msg += " | saved %d frames" % cap_res["frames"]
    return {"ok": True, "running": False, "armed": False, "message": msg}


def snapshot_still():
    with rlock:
        frame = raw_buf["frame"]
    if frame is None:
        return {"ok": False, "error": "no camera frame yet"}
    cap_dir = os.path.join(recording.DATA_ROOT, "captures")
    try:
        os.makedirs(cap_dir, exist_ok=True)
        fname = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
        # frame is BGR (picamera2 order), which is what imwrite expects.
        cv2.imwrite(os.path.join(cap_dir, fname), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    except Exception as e:
        return {"ok": False, "error": "save failed: %s" % e}
    return {"ok": True, "message": "saved " + fname}



def analyze_frame_file(fp, method):
    """Re-run the pipeline on one saved frame; returns annotated JPEG bytes."""
    if not CAMERA_AVAILABLE:
        raise RuntimeError("vision stack unavailable")
    img = cv2.imread(fp)            # BGR, matches how frames were saved
    res = analyze(img, is_rgb=False, method=method)
    vis = annotate(img, res, is_rgb=False)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()


def reanalyze_session(frames_dir, method, frame_file, limit=20000):
    """Re-run the pipeline over a whole session. `frame_file` is the path helper
    from recording (idx -> path or None). Returns the summary dict served by the
    Analyze tab."""
    if not CAMERA_AVAILABLE:
        raise RuntimeError("vision stack unavailable")
    centers, zones, i = [], [], 0
    while i < limit:
        fp = frame_file(frames_dir, i)
        if not fp:
            break
        img = cv2.imread(fp)
        if img is None:
            break
        try:
            res = analyze(img, is_rgb=False, method=method)
            centers.append(round(float(res.center_depth_pct), 1))
            zones.append(int(res.best_zone))
        except Exception:
            centers.append(None); zones.append(None)
        i += 1
    valid = [c for c in centers if c is not None]
    hist = [0, 0, 0, 0, 0]
    for z in zones:
        if z is not None and 0 <= z < 5:
            hist[z] += 1
    return {"method": method, "frames": len(centers), "center": centers,
            "best_zone_hist": hist,
            "avg_center": round(sum(valid) / len(valid), 1) if valid else None}

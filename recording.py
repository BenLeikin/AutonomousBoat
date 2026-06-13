"""Session recording: telemetry schema, frame capture, manifests, water-HSV
calibration, and read helpers for the Analyze tab. Pulls sensor snapshots from
the hardware modules; the camera-availability flag is injected by the pilot
(recording must not import the pilot: the pilot calls maybe_capture)."""
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time

try:
    import cv2
except Exception:
    cv2 = None

# Module handle to the pipeline so a session manifest can record exactly which
# segmentation method and parameters produced its data. Optional.
try:
    import vision.pipeline as _vp
except Exception:
    _vp = None

from control import controller as autoboat_control
import hardware.motors as motors
from hardware.imu import att, alock, IMU_AVAILABLE, ALPHA, ROLL_SIGN, PITCH_SIGN, YAW_SIGN
from hardware.gps import gps, glock, GPS_AVAILABLE, GPS_BAUD
from hardware.power import power, pwlock
from hardware.sysmon import sysm, slock
from hardware.tof import TOF_LIB_AVAILABLE, TOF_BUS, TOF_BOW_OFFSET_M

# Camera availability is injected by the pilot at import time.
camera_available_fn = lambda: False


def set_camera_available(fn):
    global camera_available_fn
    camera_available_fn = fn

# ---------- Capture / data-collection state ----------
# Records raw frames + synced telemetry to a session folder for offline tuning
# and labeling, and accumulates a water-HSV sample to suggest segmentation
# thresholds for the vision pipeline. Started and stopped from the dashboard.
# NOTE: writes to the SD card. Keep the read-only overlay OFF during capture, or
# the frames land in RAM and vanish on reboot. There's a free-space guard below.
# Data store matches setup_data_store.sh: $AUTOBOAT_DATA (default ~/autoboat-data)
# with sessions/ for run recordings (frames + CSV) and captures/ for curated
# stills. Raw move-around recordings go to sessions/; curate stills yourself.
DATA_ROOT = os.environ.get("AUTOBOAT_DATA") or os.path.expanduser("~/autoboat-data")
CAPTURE_ROOT = os.path.join(DATA_ROOT, "sessions")
CAPTURE_INTERVAL = 0.5            # seconds between saved stills (~2 fps)
CAPTURE_MIN_FREE_MB = 250         # auto-stop if free space drops below this
CAPTURE_MIN_FRAMES = 20           # need at least this many frames to trust calibration
CAPTURE_FRAME_FORMAT = "png"      # "png": lossless, so Re-analyze reproduces the live
                                  # decision exactly. Texture segmentation is compression-
                                  # sensitive at the water boundary, so a lossy frame can
                                  # flip a near-threshold zone (measured: >20px on ~20% of
                                  # frames, worst case a blocked/open flip), and those
                                  # borderline frames are exactly the ones worth debugging.
                                  # "jpg": ~5x smaller but lossy. Frames save at ~2 fps so
                                  # PNG is cheap in absolute terms (~136KB/frame at 320x240).
CAPTURE_JPEG_QUALITY = 90         # used only when CAPTURE_FRAME_FORMAT == "jpg"
CAPTURE_FRAME_EXT = "png" if CAPTURE_FRAME_FORMAT == "png" else "jpg"


def _frame_file(frames_dir, idx):
    """Path of saved frame `idx`, matching whatever extension it was stored with
    (.png for new sessions, .jpg for older ones), or None if absent."""
    for ext in ("png", "jpg"):
        p = os.path.join(frames_dir, "frame_%06d.%s" % (idx, ext))
        if os.path.isfile(p):
            return p
    return None

cap = {"recording": False, "session": None, "started": 0.0,
       "frames": 0, "last_save": 0.0, "free_mb": None, "error": None,
       "stop_reason": None,
       "last_session": None, "last_frames": 0, "last_duration": 0.0}
caplock = threading.Lock()
_cap_fh = {"f": None}             # open telemetry file handle (guarded by caplock)
_cap_meta = {"d": None}           # session manifest dict in progress (guarded by caplock)
_cap_hsv = {"n": 0, "h": 0.0, "s": 0.0, "v": 0.0,
            "h2": 0.0, "s2": 0.0, "v2": 0.0}   # pixel-level water HSV sums



# ---------- Capture / data-collection helpers ----------
# Ordered telemetry columns. cmd_left/cmd_right are reserved for the autonomy
# controller (blank until it exists) so a session's record already aligns the
# boat's commanded output with what vision saw, frame for frame.
TELEMETRY_COLUMNS = [
    "t_iso", "t_unix", "frame",
    "fps", "vision_ms",
    "best_zone", "center_pct", "z0", "z1", "z2", "z3", "z4",
    "roll_deg", "pitch_deg", "yaw_deg",
    "ax", "ay", "az", "gx", "gy", "gz", "imu_hz",
    "pack_v", "rail_v", "current_ma", "power_w",
    "cpu_pct", "cpu_temp_c", "under_voltage", "throttled_now", "uv_occurred",
    "gps_fix", "gps_valid", "gps_lat", "gps_lon", "gps_alt_m",
    "gps_sats", "gps_hdop", "gps_sog_ms", "gps_cog",
    "cmd_left", "cmd_right", "cmd_mode", "motion", "tof_m",
]


def _fmt(v, nd=None):
    # CSV cell: blank for None, fixed-decimal for floats, str otherwise.
    if v is None:
        return ""
    if nd is not None:
        try:
            return ("%%.%df" % nd) % v
        except (TypeError, ValueError):
            return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", "/home/ben/autoboat", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def _pi_model():
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00").strip()
    except Exception:
        return None


def _pipeline_meta():
    if _vp is None:
        return {"available": False}
    g = lambda k: getattr(_vp, k, None)
    return {"available": True, "method": g("DEFAULT_METHOD"), "num_zones": g("NUM_ZONES"),
            "roi_top_frac": g("ROI_TOP_FRAC"), "gap_tolerance": g("GAP_TOLERANCE"),
            "bottom_skip_max": g("BOTTOM_SKIP_MAX"), "texture_window": g("TEXTURE_WINDOW"),
            "vert_close": g("VERT_CLOSE"), "depth_smooth": g("DEPTH_SMOOTH"),
            "connect_from_bottom": g("CONNECT_FROM_BOTTOM")}


def _session_meta():
    # Static run context: enough to reproduce and interpret the data offline.
    with pwlock:
        pwr_on = bool(power.get("connected"))
    with glock:
        gps_on = bool(gps.get("connected"))
    with alock:
        imu_on = bool(att.get("ok"))
    return {
        "schema_version": 4,   # v3: + motion column; v4: + tof_m column
        "host": {"hostname": (os.uname().nodename if hasattr(os, "uname") else None),
                 "pi_model": _pi_model()},
        "git_commit": _git_commit(),
        "camera": {"resolution": "320x240",
                   "note": "picamera2 RGB888 buffer is BGR-ordered; analyze/annotate use is_rgb=False",
                   "lens": os.environ.get("AUTOBOAT_LENS", "unknown")},
        "pipeline": _pipeline_meta(),
        "imu": {"present": IMU_AVAILABLE, "alpha": ALPHA,
                "roll_sign": ROLL_SIGN, "pitch_sign": PITCH_SIGN, "yaw_sign": YAW_SIGN,
                "accel_units": "m/s^2", "gyro_units": "rad/s"},
        "gps": {"present": GPS_AVAILABLE, "receiver": "VK-162 (u-blox 7)", "baud": GPS_BAUD,
                "note": "consumer GPS ~2.5m CEP; not usable for pool-scale boundaries"},
        "tof": {"present": TOF_LIB_AVAILABLE, "sensor": "VL53L1X", "bus": TOF_BUS,
                "bow_offset_m": TOF_BOW_OFFSET_M,
                "note": "tof_m telemetry is bow-to-obstacle (offset already subtracted)"},
        "capture": {"interval_s": CAPTURE_INTERVAL, "min_free_mb": CAPTURE_MIN_FREE_MB,
                    "frame_format": CAPTURE_FRAME_EXT,
                    "jpeg_quality": (CAPTURE_JPEG_QUALITY if CAPTURE_FRAME_EXT == "jpg" else None),
                    "lossless": CAPTURE_FRAME_EXT == "png", "data_root": DATA_ROOT},
        "sensors_online_at_start": {"camera": camera_available_fn(), "imu": imu_on, "tof": TOF_LIB_AVAILABLE,
                                    "power": pwr_on, "gps": gps_on},
        "controller": dict(autoboat_control.params(), armed=motors.ARMED),
        "motors": {"available": motors.MOTORS_AVAILABLE, "throttle_cap": motors.MOTOR_THROTTLE_CAP, "pivot_cap": motors.MOTOR_PIVOT_CAP,
                   "deadband": motors.MOTOR_DEADBAND, "watchdog_s": motors.MOTOR_WATCHDOG_S,
                   "left_scale": motors.MOTOR_LEFT_SCALE, "right_scale": motors.MOTOR_RIGHT_SCALE,
                   "left_min": motors.MOTOR_LEFT_MIN, "right_min": motors.MOTOR_RIGHT_MIN,
                   "pins": {"ain1": motors.MOT_AIN1, "ain2": motors.MOT_AIN2, "bin1": motors.MOT_BIN1,
                            "bin2": motors.MOT_BIN2, "sleep": motors.MOT_SLP}},
        "telemetry_columns": TELEMETRY_COLUMNS,
    }


def _cap_event(sdir, msg):
    # Append a timestamped line to the session event log. Best effort.
    if not sdir:
        return
    try:
        with open(os.path.join(sdir, "events.log"), "a") as f:
            f.write("%s  %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg))
    except Exception:
        pass


def _write_manifest(sdir, meta):
    if not sdir or meta is None:
        return
    try:
        with open(os.path.join(sdir, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def _free_mb(path):
    try:
        return shutil.disk_usage(path).free / (1024.0 * 1024.0)
    except Exception:
        return None


def _accumulate_hsv(frame_rgb):
    # frame_rgb is actually BGR (picamera2 "RGB888" order). Sample the lower-center
    # patch, most likely open water, and accumulate pixel-level HSV sums.
    try:
        h, w = frame_rgb.shape[:2]
        patch = frame_rgb[int(h * 0.70):h, int(w * 0.35):int(w * 0.65)]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype("float64")
        s = hsv.sum(axis=0)
        sq = (hsv * hsv).sum(axis=0)
        _cap_hsv["n"] += hsv.shape[0]
        _cap_hsv["h"] += s[0]; _cap_hsv["s"] += s[1]; _cap_hsv["v"] += s[2]
        _cap_hsv["h2"] += sq[0]; _cap_hsv["s2"] += sq[1]; _cap_hsv["v2"] += sq[2]
    except Exception:
        pass


def _hsv_calibration():
    # OpenCV HSV: H 0-179, S/V 0-255. Suggests water-mask bounds as mean +/- 2 std.
    # Assumes hue doesn't wrap (pool water sits near H 90-110, so it won't).
    n = _cap_hsv["n"]
    if n <= 0:
        return None
    out = {}
    for ch in ("h", "s", "v"):
        mean = _cap_hsv[ch] / n
        var = max(_cap_hsv[ch + "2"] / n - mean * mean, 0.0)
        out[ch] = {"mean": round(mean, 1), "std": round(var ** 0.5, 1)}
    clamp = lambda x, m: int(max(0, min(m, round(x))))
    lower = [clamp(out["h"]["mean"] - 2 * out["h"]["std"], 179),
             clamp(out["s"]["mean"] - 2 * out["s"]["std"], 255),
             clamp(out["v"]["mean"] - 2 * out["v"]["std"], 255)]
    upper = [clamp(out["h"]["mean"] + 2 * out["h"]["std"], 179),
             clamp(out["s"]["mean"] + 2 * out["s"]["std"], 255),
             clamp(out["v"]["mean"] + 2 * out["v"]["std"], 255)]
    return {"pixels_sampled": int(n), "hsv_lower": lower, "hsv_upper": upper,
            "per_channel": out}


def capture_start():
    with caplock:
        if cap["recording"]:
            return {"ok": True, "message": "already recording",
                    "session": os.path.basename(cap["session"]) if cap["session"] else None}
        try:
            os.makedirs(CAPTURE_ROOT, exist_ok=True)
        except Exception as e:
            return {"ok": False, "error": "cannot create %s: %s" % (CAPTURE_ROOT, e)}
        free = _free_mb(CAPTURE_ROOT)
        if free is not None and free < CAPTURE_MIN_FREE_MB:
            return {"ok": False, "error": "only %.0f MB free, need %d" % (free, CAPTURE_MIN_FREE_MB)}
        name = time.strftime("session_%Y%m%d_%H%M%S")
        sdir = os.path.join(CAPTURE_ROOT, name)
        try:
            os.makedirs(os.path.join(sdir, "frames"), exist_ok=True)
            fh = open(os.path.join(sdir, "telemetry.csv"), "w")
            fh.write(",".join(TELEMETRY_COLUMNS) + "\n")
            fh.flush()
        except Exception as e:
            return {"ok": False, "error": "cannot open session: %s" % e}
        _cap_fh["f"] = fh
        for k in _cap_hsv:
            _cap_hsv[k] = 0 if k == "n" else 0.0
        meta = _session_meta()
        meta["session_id"] = name
        meta["started_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["free_mb_at_start"] = round(free, 1) if free is not None else None
        _cap_meta["d"] = meta
        _write_manifest(sdir, meta)
        _cap_event(sdir, "capture started (method=%s, lens=%s)" % (
            (meta["pipeline"] or {}).get("method"), meta["camera"]["lens"]))
        cap.update({"recording": True, "session": sdir, "started": time.monotonic(),
                    "frames": 0, "last_save": 0.0, "free_mb": free, "error": None,
                    "stop_reason": None})
        return {"ok": True, "message": "recording", "session": name}


def _finalize_session_locked(reason):
    # caplock MUST be held. Closes the telemetry file, writes the HSV calibration
    # and the finalized manifest, logs the stop event. Returns (frames, sdir, calib).
    cap["recording"] = False
    cap["stop_reason"] = reason
    fh = _cap_fh["f"]
    _cap_fh["f"] = None
    sdir = cap["session"]
    frames = cap["frames"]
    started = cap["started"]
    if fh:
        try:
            fh.flush(); fh.close()
        except Exception:
            pass
    calib = _hsv_calibration() if frames >= CAPTURE_MIN_FRAMES else None
    if sdir and calib:
        try:
            with open(os.path.join(sdir, "water_hsv.json"), "w") as cf:
                json.dump(calib, cf, indent=2)
        except Exception:
            pass
    meta = _cap_meta["d"]
    if meta is not None:
        free = _free_mb(sdir) if sdir else None
        meta["stopped_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["duration_s"] = round(time.monotonic() - started, 1) if started else None
        meta["frames_saved"] = frames
        meta["stop_reason"] = reason
        meta["free_mb_at_stop"] = round(free, 1) if free is not None else None
        meta["water_hsv"] = calib
        _write_manifest(sdir, meta)
    _cap_event(sdir, "capture stopped (reason=%s, frames=%d)" % (reason, frames))
    _cap_meta["d"] = None
    if sdir:
        cap["last_session"] = os.path.basename(sdir)
        cap["last_frames"] = frames
        cap["last_duration"] = round(time.monotonic() - started, 1) if started else 0.0
    return frames, sdir, calib


def capture_stop(reason="user"):
    with caplock:
        if not cap["recording"]:
            return {"ok": True, "message": "not recording"}
        frames, sdir, calib = _finalize_session_locked(reason)
    return {"ok": True, "message": "stopped", "frames": frames,
            "session": os.path.basename(sdir) if sdir else None,
            "calibration": calib}



def maybe_capture(frame_rgb, result, fps, vision_ms, cmd=None, motion=None, tof_m=None):
    # Called from camera_loop each frame; saves a still + a full telemetry row at
    # the configured interval while recording. Does its own locking.
    now = time.monotonic()
    with caplock:
        if not cap["recording"] or (now - cap["last_save"]) < CAPTURE_INTERVAL:
            return
        sdir = cap["session"]
        fh = _cap_fh["f"]
        n = cap["frames"]
        with alock:
            roll = math.degrees(att["roll"]); pitch = math.degrees(att["pitch"])
            yaw = math.degrees(att["yaw"]); imu_hz = att["hz"]
            ax = att["ax"]; ay = att["ay"]; az = att["az"]
            gx = att["gx"]; gy = att["gy"]; gz = att["gz"]
        with pwlock:
            pack_v = power["pack_v"]; rail_v = power["rail_v"]
            cur = power["current_ma"]; pwr_w = power["power_w"]
        with slock:
            cpu_pct = sysm.get("cpu_pct"); cpu_temp = sysm.get("temp_c")
            uv = sysm.get("under_voltage"); thr = sysm.get("throttled_now")
            uvo = sysm.get("uv_occurred")
        with glock:
            g_fix = gps["fix_type"]; g_valid = gps["valid"]
            g_lat = gps["lat"]; g_lon = gps["lon"]; g_alt = gps["alt"]
            g_sats = gps["sats_used"]; g_hdop = gps["hdop"]
            g_sog = gps["sog_ms"]; g_cog = gps["cog"]
        fname = "frame_%06d.%s" % (n, CAPTURE_FRAME_EXT)
        try:
            # frame_rgb is already BGR (picamera2 order), which is what imwrite wants.
            if CAPTURE_FRAME_EXT == "png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]   # lossless; 3 = fast/decent
            else:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), CAPTURE_JPEG_QUALITY]
            cv2.imwrite(os.path.join(sdir, "frames", fname), frame_rgb, params)
        except Exception as e:
            cap["error"] = "save failed: %s" % e
            _cap_event(sdir, "frame save failed at #%d: %s" % (n, e))
            return
        z = list(result.zones)
        if fh:
            row = [
                time.strftime("%Y-%m-%dT%H:%M:%S"), _fmt(time.time(), 3), fname,
                _fmt(fps, 1), _fmt(vision_ms, 1),
                _fmt(int(result.best_zone)), _fmt(float(result.center_depth_pct), 1),
                _fmt(z[0]), _fmt(z[1]), _fmt(z[2]), _fmt(z[3]), _fmt(z[4]),
                _fmt(roll, 1), _fmt(pitch, 1), _fmt(yaw, 1),
                _fmt(ax, 3), _fmt(ay, 3), _fmt(az, 3),
                _fmt(gx, 4), _fmt(gy, 4), _fmt(gz, 4), _fmt(imu_hz, 1),
                _fmt(pack_v, 2), _fmt(rail_v, 2),
                ("" if cur is None else _fmt(int(round(cur)))), _fmt(pwr_w, 2),
                _fmt(cpu_pct, 1), _fmt(cpu_temp, 1), _fmt(uv), _fmt(thr), _fmt(uvo),
                _fmt(g_fix), _fmt(g_valid), _fmt(g_lat, 6), _fmt(g_lon, 6), _fmt(g_alt, 1),
                _fmt(g_sats), _fmt(g_hdop, 1), _fmt(g_sog, 2), _fmt(g_cog, 1),
                (_fmt(int(round(cmd["left"] * 100))) if cmd else ""),
                (_fmt(int(round(cmd["right"] * 100))) if cmd else ""),
                (cmd["mode"] if cmd else ""),
                _fmt(motion, 1),
                _fmt(tof_m, 2),
                # cmd_left, cmd_right (signed percent) and cmd_mode from the controller
            ]
            try:
                fh.write(",".join(row) + "\n")
                fh.flush()
            except Exception:
                pass
        _accumulate_hsv(frame_rgb)
        cap["frames"] = n + 1
        cap["last_save"] = now
        if (cap["frames"] % 20) == 0:
            free = _free_mb(sdir)
            cap["free_mb"] = free
            if free is not None and free < CAPTURE_MIN_FREE_MB:
                cap["error"] = "stopped: low card space (%.0f MB)" % free
                _finalize_session_locked("low_space")



# ---------- Past-run analysis helpers ----------
# Read-only access to recorded sessions for the Analyze tab. Session names are
# strictly validated to keep the frame/file routes from escaping CAPTURE_ROOT.
_SESSION_RE = re.compile(r"^session_[0-9_]+$")


def _safe_session_dir(name):
    if not name or not _SESSION_RE.match(name):
        return None
    d = os.path.join(CAPTURE_ROOT, name)
    return d if os.path.isdir(d) else None


def _read_json_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_text_file(path, limit=200000):
    try:
        with open(path) as f:
            return f.read(limit)
    except Exception:
        return None


def _read_telemetry(sdir):
    # Returns (columns, rows). rows is a list of lists of raw string cells.
    try:
        with open(os.path.join(sdir, "telemetry.csv")) as f:
            lines = f.read().splitlines()
    except Exception:
        return None, None
    if not lines:
        return [], []
    return lines[0].split(","), [ln.split(",") for ln in lines[1:] if ln]


def _list_sessions():
    try:
        names = [n for n in os.listdir(CAPTURE_ROOT) if _SESSION_RE.match(n)]
    except Exception:
        names = []
    out = []
    for name in sorted(names, reverse=True):
        sdir = os.path.join(CAPTURE_ROOT, name)
        man = _read_json_file(os.path.join(sdir, "manifest.json")) or {}
        frames = man.get("frames_saved")
        if frames is None:
            try:
                frames = sum(1 for f in os.listdir(os.path.join(sdir, "frames"))
                             if f.startswith("frame_") and f.endswith((".jpg", ".png")))
            except Exception:
                frames = 0
        out.append({
            "name": name, "frames": frames,
            "duration": man.get("duration_s"), "started": man.get("started_iso"),
            "method": (man.get("pipeline") or {}).get("method"),
            "lens": (man.get("camera") or {}).get("lens"),
            "stop_reason": man.get("stop_reason"),
        })
    return out


def _gps_points(cols, rows):
    if not cols or "gps_lat" not in cols or "gps_lon" not in cols:
        return []
    la, lo = cols.index("gps_lat"), cols.index("gps_lon")
    fx = cols.index("gps_fix") if "gps_fix" in cols else None
    sg = cols.index("gps_sog_ms") if "gps_sog_ms" in cols else None
    pts = []
    for i, r in enumerate(rows):
        if len(r) <= max(la, lo) or not r[la] or not r[lo]:
            continue
        try:
            lat, lon = float(r[la]), float(r[lo])
        except ValueError:
            continue
        pts.append({"i": i, "lat": lat, "lon": lon,
                    "fix": r[fx] if fx is not None and len(r) > fx else None,
                    "sog": r[sg] if sg is not None and len(r) > sg else None})
    return pts



def is_recording():
    with caplock:
        return cap["recording"]


def note_event(msg):
    """Append to the current session's event log, if a session is recording."""
    try:
        with caplock:
            sdir = cap["session"] if cap["recording"] else None
        _cap_event(sdir, msg)
    except Exception:
        pass

import time
import math
import json
import glob
import os
import shutil
import threading
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import board
import busio
from adafruit_bus_device.i2c_device import I2CDevice

# Vision stack is optional: if it's missing (e.g. running this dashboard on a
# box without picamera2/opencv), the camera section just shows "--" and the
# IMU + system parts still work.
sys.path.insert(0, "/home/ben/autoboat")
try:
    import cv2
    from picamera2 import Picamera2
    from vision.pipeline import analyze, annotate
    CAMERA_AVAILABLE = True
except Exception as e:
    print(f"[camera] vision stack unavailable, camera section disabled: {e}")
    CAMERA_AVAILABLE = False

# Module handle to the pipeline so a session manifest can record exactly which
# segmentation method and parameters produced its data. Optional.
try:
    import vision.pipeline as _vp
except Exception:
    _vp = None

# GPS is optional too: needs pyserial and a receiver on a USB serial port.
try:
    import serial
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("[gps] pyserial not installed; GPS section disabled "
          "(pip install pyserial --break-system-packages)")

# INA219 pack monitor is optional: it lives on the software I2C bus created by
# the i2c-gpio overlay (SDA=GPIO5, SCL=GPIO6 -> /dev/i2c-3). On a box without
# that bus or the libraries, the Power section just shows "sensor offline".
try:
    from adafruit_extended_bus import ExtendedI2C
    from adafruit_ina219 import INA219
    INA_AVAILABLE = True
except Exception as e:
    INA_AVAILABLE = False
    print(f"[power] INA219 libs unavailable, power section disabled: {e}")

INA_BUS = 3                        # /dev/i2c-3, the i2c-gpio bus on GPIO5/6
INA_ADDR = 0x40
PACK_LOW = 6.6                     # 2S Li, ~3.3 V/cell: head back
PACK_CRITICAL = 6.0               # 2S Li, ~3.0 V/cell: stop

PORT = 8000
TARGET_HZ = 104                    # loop target; matches the sensor's default ODR
PERIOD = 1.0 / TARGET_HZ
GPS_BAUD = 9600                    # VK-162 default (u-blox 7, 8N1)

# ---------- Control actions ----------
# Shell-level actions the dashboard is allowed to run. Each needs a matching
# button entry in the CONTROLS array on the page (see PAGE below).
# These use `sudo -n` (non-interactive): if passwordless sudo is not set up for
# the command, it fails fast and the button reports "sudo: a password is required".
# Verify the systemctl path on your Pi with `which systemctl` and adjust if needed.
ALLOWED_ACTIONS = {
    "reboot": ["sudo", "-n", "systemctl", "reboot"],
    "shutdown": ["sudo", "-n", "systemctl", "poweroff"],
    # Add more shell actions here. Control-loop actions (emergency stop, start/stop
    # recording) are not shell commands; wire those to the control loop / logger
    # once those subsystems exist, then add a matching CONTROLS button.
}

# ---------- IMU ----------
# Minimal MPU-6050 driver that does NOT gate on the WHO_AM_I id, so it works with
# the genuine chip and with the common clone modules that report a different id
# (those make the stock adafruit_mpu6050 raise "Failed to find MPU6050"). Units
# match the old LSM6DSO: acceleration m/s^2, gyro rad/s, so the filter is unchanged.
class MPU6050:
    _G = 9.80665
    _DEG2RAD = math.pi / 180.0

    def __init__(self, i2c, address=0x68):
        self._dev = I2CDevice(i2c, address)
        self._buf = bytearray(6)
        self._write(0x6B, 0x00)   # PWR_MGMT_1: wake from sleep
        self._write(0x1A, 0x04)   # CONFIG: DLPF ~21 Hz, tames vibration
        self._write(0x1B, 0x00)   # GYRO_CONFIG:  +/-250 deg/s  (131 LSB/dps)
        self._write(0x1C, 0x00)   # ACCEL_CONFIG: +/-2 g        (16384 LSB/g)

    def _write(self, reg, val):
        with self._dev as d:
            d.write(bytes([reg, val]))

    def _read3(self, reg):
        with self._dev as d:
            d.write_then_readinto(bytes([reg]), self._buf)
        out = []
        for i in range(3):
            v = (self._buf[2 * i] << 8) | self._buf[2 * i + 1]
            out.append(v - 65536 if v >= 32768 else v)
        return out

    @property
    def acceleration(self):
        x, y, z = self._read3(0x3B)
        return (x / 16384.0 * self._G, y / 16384.0 * self._G, z / 16384.0 * self._G)

    @property
    def gyro(self):
        x, y, z = self._read3(0x43)
        return (x / 131.0 * self._DEG2RAD, y / 131.0 * self._DEG2RAD,
                z / 131.0 * self._DEG2RAD)


sensor = None
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = MPU6050(i2c, address=0x68)
    IMU_AVAILABLE = True
except Exception as e:
    print(f"[imu] MPU6050 not found, attitude section disabled: {e}")
    IMU_AVAILABLE = False

# ---------- Complementary filter config ----------
ALPHA = 0.98          # near 1.0: gyro short-term, accel long-term (drift kill)
ROLL_SIGN = 1         # set to -1 if the boat heels the wrong way
PITCH_SIGN = -1       # flipped: clone IMU reads bow/stern reversed
YAW_SIGN = 1

att = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "hz": 0.0, "ok": False,
       "ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}  # rad + raw
alock = threading.Lock()

sysm = {}                                                   # system metrics, shared
slock = threading.Lock()

# ---------- Camera + vision shared state ----------
proc = {"fps": 0.0, "latency_ms": 0.0, "best_zone": 0,
        "center_pct": 0.0, "zones": [0, 0, 0, 0, 0], "cam_ok": False}
plock = threading.Lock()
frame_buf = {"jpeg": None}
flock = threading.Lock()

# ---------- GPS shared state ----------
gps = {"connected": False, "fix_type": "none", "valid": False,
       "sats_used": None, "sats_view": None, "lat": None, "lon": None,
       "alt": None, "hdop": None, "sog_ms": None, "cog": None,
       "utc": None, "date": None, "last_data": 0.0}
glock = threading.Lock()

# ---------- INA219 pack monitor shared state ----------
power = {"connected": False, "pack_v": None, "rail_v": None,
         "current_ma": None, "power_w": None, "status": "--", "last_data": 0.0}
pwlock = threading.Lock()

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

cap = {"recording": False, "session": None, "started": 0.0,
       "frames": 0, "last_save": 0.0, "free_mb": None, "error": None,
       "stop_reason": None}
caplock = threading.Lock()
_cap_fh = {"f": None}             # open telemetry file handle (guarded by caplock)
_cap_meta = {"d": None}           # session manifest dict in progress (guarded by caplock)
_cap_hsv = {"n": 0, "h": 0.0, "s": 0.0, "v": 0.0,
            "h2": 0.0, "s2": 0.0, "v2": 0.0}   # pixel-level water HSV sums


def imu_loop():
    roll = pitch = yaw = 0.0
    t = time.monotonic()
    count = 0
    win = t
    hz = 0.0
    while True:
        start = time.monotonic()
        try:
            ax, ay, az = sensor.acceleration      # m/s^2
            gx, gy, gz = sensor.gyro              # rad/s
        except OSError:
            time.sleep(0.01)
            continue

        now = time.monotonic()
        dt = now - t
        t = now

        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        roll = ALPHA * (roll + gx * dt) + (1 - ALPHA) * roll_acc
        pitch = ALPHA * (pitch + gy * dt) + (1 - ALPHA) * pitch_acc
        yaw += gz * dt

        count += 1
        if now - win >= 1.0:
            hz = count / (now - win)              # actual measured loop rate
            count = 0
            win = now

        with alock:
            att["roll"] = ROLL_SIGN * roll
            att["pitch"] = PITCH_SIGN * pitch
            att["yaw"] = YAW_SIGN * yaw
            att["hz"] = hz
            att["ok"] = True
            att["ax"] = ax; att["ay"] = ay; att["az"] = az
            att["gx"] = gx; att["gy"] = gy; att["gz"] = gz

        sleep_left = PERIOD - (time.monotonic() - start)
        if sleep_left > 0:
            time.sleep(sleep_left)


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
    while True:
        try:
            frame = picam.capture_array()         # "RGB888" but actually BGR order
        except Exception:
            time.sleep(0.05)
            continue

        t0 = time.monotonic()
        result = analyze(frame, is_rgb=False)
        latency = (time.monotonic() - t0) * 1000.0

        vis = annotate(frame, result, is_rgb=False)  # frame is already BGR
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

        _maybe_capture(frame, result, fps, latency)


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
    "cmd_left", "cmd_right",
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
            "vert_close": g("VERT_CLOSE"), "depth_smooth": g("DEPTH_SMOOTH")}


def _session_meta():
    # Static run context: enough to reproduce and interpret the data offline.
    with pwlock:
        pwr_on = bool(power.get("connected"))
    with glock:
        gps_on = bool(gps.get("connected"))
    with alock:
        imu_on = bool(att.get("ok"))
    return {
        "schema_version": 2,
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
        "capture": {"interval_s": CAPTURE_INTERVAL, "min_free_mb": CAPTURE_MIN_FREE_MB,
                    "jpeg_quality": 90, "data_root": DATA_ROOT},
        "sensors_online_at_start": {"camera": CAMERA_AVAILABLE, "imu": imu_on,
                                    "power": pwr_on, "gps": gps_on},
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
    return frames, sdir, calib


def capture_stop(reason="user"):
    with caplock:
        if not cap["recording"]:
            return {"ok": True, "message": "not recording"}
        frames, sdir, calib = _finalize_session_locked(reason)
    return {"ok": True, "message": "stopped", "frames": frames,
            "session": os.path.basename(sdir) if sdir else None,
            "calibration": calib}


def _maybe_capture(frame_rgb, result, fps, vision_ms):
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
        fname = "frame_%06d.jpg" % n
        try:
            # frame_rgb is already BGR (picamera2 order), which is what imwrite wants.
            cv2.imwrite(os.path.join(sdir, "frames", fname), frame_rgb,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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
                "", "",  # cmd_left, cmd_right: reserved for the autonomy controller
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


# ---------- GPS / NMEA helpers ----------
def _gps_find_port():
    # VK-162 is a CDC-ACM device, so prefer ttyACM; fall back to ttyUSB.
    cands = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    return cands[0] if cands else None


def _gps_checksum_ok(line):
    if not line.startswith("$") or "*" not in line:
        return False
    star = line.rfind("*")
    calc = 0
    for ch in line[1:star]:
        calc ^= ord(ch)
    try:
        return calc == int(line[star + 1:star + 3], 16)
    except ValueError:
        return False


def _to_deg(val, hemi):
    if not val or not hemi or "." not in val:
        return None
    try:
        dot = val.index(".")
        deg = int(val[:dot - 2])
        minutes = float(val[dot - 2:])
        dec = deg + minutes / 60.0
        return -dec if hemi in ("S", "W") else dec
    except (ValueError, IndexError):
        return None


def _fmt_time(t):
    return f"{t[0:2]}:{t[2:4]}:{t[4:6]}" if t and len(t) >= 6 else None


def _fmt_date(d):
    return f"{d[0:2]}-{d[2:4]}-20{d[4:6]}" if d and len(d) >= 6 else None


def _gps_parse(line, st):
    f = line[1:line.rfind("*")].split(",")
    kind = f[0][-3:] if f and len(f[0]) >= 3 else ""
    if kind == "GGA" and len(f) >= 10:
        st["lat"] = _to_deg(f[2], f[3])
        st["lon"] = _to_deg(f[4], f[5])
        st["sats_used"] = int(f[7]) if f[7].isdigit() else None
        st["hdop"] = float(f[8]) if f[8] else None
        st["alt"] = float(f[9]) if f[9] else None
        st["utc"] = _fmt_time(f[1]) or st.get("utc")
    elif kind == "RMC" and len(f) >= 10:
        st["valid"] = (f[2] == "A")
        if f[2] == "A":
            st["lat"] = _to_deg(f[3], f[4])
            st["lon"] = _to_deg(f[5], f[6])
        st["sog_ms"] = (float(f[7]) * 0.514444) if f[7] else None
        st["cog"] = float(f[8]) if f[8] else None
        st["utc"] = _fmt_time(f[1]) or st.get("utc")
        st["date"] = _fmt_date(f[9])
    elif kind == "GSA" and len(f) >= 18:
        st["fix_type"] = {"1": "none", "2": "2D", "3": "3D"}.get(f[2], "?")
    elif kind == "GSV" and len(f) >= 4:
        if f[3].isdigit():
            st["sats_view"] = int(f[3])
    elif kind == "VTG" and len(f) >= 8:
        if f[1]:
            st["cog"] = float(f[1])
        if f[5]:
            st["sog_ms"] = float(f[5]) * 0.514444


def gps_loop():
    # Reconnecting reader: finds the port, streams NMEA into the shared `gps`
    # dict, and on unplug/silence/error backs off and retries so the dashboard
    # recovers on its own when the receiver comes back.
    while True:
        port = _gps_find_port()
        if not port:
            with glock:
                gps["connected"] = False
            time.sleep(3.0)
            continue
        try:
            ser = serial.Serial(port, GPS_BAUD, timeout=1.0)
        except Exception as e:
            print(f"[gps] cannot open {port}: {e}")
            with glock:
                gps["connected"] = False
            time.sleep(3.0)
            continue

        print(f"[gps] reading {port} @ {GPS_BAUD} 8N1")
        work = {"fix_type": "none", "valid": False, "sats_used": None,
                "sats_view": None, "lat": None, "lon": None, "alt": None,
                "hdop": None, "sog_ms": None, "cog": None, "utc": None, "date": None}
        last_data = time.monotonic()
        try:
            while True:
                raw = ser.readline()
                now = time.monotonic()
                if raw:
                    line = raw.decode("ascii", errors="replace").strip()
                    if _gps_checksum_ok(line):
                        try:
                            _gps_parse(line, work)
                        except Exception:
                            pass  # skip a malformed-but-checksummed line
                        last_data = now
                        with glock:
                            gps.update(work)
                            gps["connected"] = True
                            gps["last_data"] = now
                elif now - last_data > 6.0:
                    break  # port open but silent; drop and retry
        except Exception as e:
            print(f"[gps] read error: {e}")
        finally:
            try:
                ser.close()
            except Exception:
                pass
        with glock:
            gps["connected"] = False
        time.sleep(2.0)


def _pack_status(v):
    if v is None:
        return "--"
    if v >= PACK_LOW:
        return "ok"
    if v >= PACK_CRITICAL:
        return "low"
    return "critical"


def power_loop():
    # Reconnecting reader for the high-side INA219. Pack voltage is bus + shunt
    # (the sensor sits in the battery + lead). On any I2C error it backs off and
    # retries so the dashboard recovers if the sensor drops off the bus.
    while True:
        try:
            i2c3 = ExtendedI2C(INA_BUS)
            ina = INA219(i2c3, addr=INA_ADDR)
        except Exception as e:
            print(f"[power] cannot open INA219 on /dev/i2c-{INA_BUS}: {e}")
            with pwlock:
                power["connected"] = False
            time.sleep(3.0)
            continue

        print(f"[power] reading INA219 on /dev/i2c-{INA_BUS} at {hex(INA_ADDR)}")
        try:
            while True:
                bus_v = ina.bus_voltage          # volts at Vin- (rail to loads)
                shunt_v = ina.shunt_voltage       # volts across the shunt
                pack_v = bus_v + shunt_v          # actual battery voltage
                with pwlock:
                    power["connected"] = True
                    power["pack_v"] = round(pack_v, 3)
                    power["rail_v"] = round(bus_v, 3)
                    power["current_ma"] = round(ina.current, 1)
                    power["power_w"] = round(ina.power, 3)
                    power["status"] = _pack_status(pack_v)
                    power["last_data"] = time.monotonic()
                time.sleep(0.5)
        except Exception as e:
            print(f"[power] read error: {e}")
            with pwlock:
                power["connected"] = False
            time.sleep(2.0)


def _cpu_sample():
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + vals[4]          # idle + iowait
    return sum(vals), idle


def sys_loop():
    prev_total, prev_idle = _cpu_sample()
    while True:
        time.sleep(1.0)
        m = {}

        try:
            total, idle = _cpu_sample()
            d_tot, d_idle = total - prev_total, idle - prev_idle
            prev_total, prev_idle = total, idle
            m["cpu_pct"] = round(100.0 * (1 - d_idle / d_tot), 1) if d_tot > 0 else 0.0
        except Exception:
            m["cpu_pct"] = None

        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                m["temp_c"] = round(int(f.read()) / 1000.0, 1)
        except Exception:
            m["temp_c"] = None

        try:
            with open("/proc/loadavg") as f:
                m["load1"] = float(f.read().split()[0])
        except Exception:
            m["load1"] = None

        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    info[k] = int(v.strip().split()[0])
            tot = info["MemTotal"]
            avail = info.get("MemAvailable", info["MemFree"])
            used = tot - avail
            m["mem_total_mb"] = round(tot / 1024)
            m["mem_used_mb"] = round(used / 1024)
            m["mem_pct"] = round(100.0 * used / tot, 1)
        except Exception:
            pass

        try:
            ssid = subprocess.run(["iwgetid", "-r"],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
            m["ssid"] = ssid or None
        except Exception:
            m["ssid"] = None

        try:
            toks = subprocess.run(["hostname", "-I"],
                                  capture_output=True, text=True, timeout=2).stdout.split()
            m["ip"] = next((x for x in toks if ":" not in x), None)
        except Exception:
            m["ip"] = None

        try:
            with open("/proc/uptime") as f:
                m["uptime_s"] = int(float(f.read().split()[0]))
        except Exception:
            m["uptime_s"] = None

        try:
            du = shutil.disk_usage("/")
            m["disk_pct"] = round(100.0 * du.used / du.total, 1)
        except Exception:
            pass

        try:
            with open("/proc/net/wireless") as f:
                for line in f.readlines()[2:]:
                    if ":" in line:
                        lvl = float(line.split()[3].rstrip("."))
                        m["wifi_dbm"] = round(lvl)
                        m["wifi_pct"] = max(0, min(100, round(2 * (lvl + 100))))
                        break
        except Exception:
            pass

        try:
            out = subprocess.run(["vcgencmd", "get_throttled"],
                                 capture_output=True, text=True, timeout=2).stdout
            val = int(out.strip().split("=")[1], 16)
            m["under_voltage"] = bool(val & 0x1)
            m["throttled_now"] = bool(val & 0x4)
            m["uv_occurred"] = bool(val & 0x10000)
        except Exception:
            pass

        with slock:
            sysm.clear()
            sysm.update(m)


PAGE = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoBoat</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0; padding: 18px;
         background: #06090c; color: #c7d2da; }
  h1 { font-size: 14px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
       color: #7d96a3; margin: 0 0 16px; }
  h2 { font-size: 12px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
       color: #6b8794; margin: 24px 0 12px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
             gap: 10px; }
  .card { background: #0c1014; border: 1px solid #1b2530; border-radius: 8px; padding: 10px 12px; }
  .card .k { font-size: 11px; color: #6b8794; letter-spacing: 0.5px; margin-bottom: 6px; }
  .card .v { font-size: 18px; color: #e6eef2; }
  .card .v small { font-size: 12px; color: #7d96a3; }
  .status { margin-top: 12px; padding: 10px 14px; border-radius: 8px; font-size: 13px;
            letter-spacing: 0.5px; border: 1px solid #1b2530; background: #0c1014; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { flex: 1; min-width: 300px; background: #0c1014; border: 1px solid #1b2530;
           border-radius: 10px; padding: 12px; }
  .label { font-size: 12px; color: #6b8794; margin-bottom: 8px; letter-spacing: 0.5px; }
  canvas { width: 100%; height: auto; display: block; }
  #cam { width: 100%; height: auto; display: block; border-radius: 6px; background: #06090c; }
  #g_map { width: 100%; height: 320px; border: 0; border-radius: 8px; margin-top: 12px;
           display: none; background: #0c1014; }
  .readout { margin-top: 14px; font-size: 16px; letter-spacing: 1px; }
  .readout span { color: #e6eef2; }
  .drift { color: #d98a3a; }
  .stale { color: #d9534f; }
  .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .btn { font-family: inherit; font-size: 13px; letter-spacing: 0.5px; color: #c7d2da;
         background: #11181f; border: 1px solid #2a3742; border-radius: 8px;
         padding: 10px 16px; cursor: pointer; transition: background .15s, border-color .15s; }
  .btn:hover { background: #16212b; border-color: #3a4a57; }
  .btn.danger { border-color: #5a2b2b; color: #e0a0a0; }
  .btn.danger:hover { background: #2a1414; }
  .btn.armed { background: #7a1f1f; border-color: #d9534f; color: #fff; }
  .ctrl-status { margin-top: 12px; font-size: 12px; color: #7d96a3; letter-spacing: 0.5px;
                 min-height: 14px; }
</style>
</head>
<body>
<h1>AutoBoat2w</h1>
<div class="metrics">
  <div class="card"><div class="k">IMU RATE</div><div class="v" id="m_hz">--</div></div>
  <div class="card"><div class="k">CPU TEMP</div><div class="v" id="m_temp">--</div></div>
  <div class="card"><div class="k">CPU LOAD</div><div class="v" id="m_cpu">--</div></div>
  <div class="card"><div class="k">MEMORY</div><div class="v" id="m_mem">--</div></div>
  <div class="card"><div class="k">DISK</div><div class="v" id="m_disk">--</div></div>
  <div class="card"><div class="k">UPTIME</div><div class="v" id="m_up">--</div></div>
  <div class="card"><div class="k">SSID</div><div class="v" id="m_ssid">--</div></div>
  <div class="card"><div class="k">WIFI</div><div class="v" id="m_wifi">--</div></div>
  <div class="card"><div class="k">IP</div><div class="v" id="m_ip">--</div></div>
</div>
<div class="status" id="m_power">power: --</div>

<h2>Power</h2>
<div class="panel">
  <div class="metrics" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">
    <div class="card"><div class="k">PACK</div><div class="v" id="pw_pack">--</div></div>
    <div class="card"><div class="k">CURRENT</div><div class="v" id="pw_cur">--</div></div>
    <div class="card"><div class="k">POWER</div><div class="v" id="pw_w">--</div></div>
    <div class="card"><div class="k">RAIL</div><div class="v" id="pw_rail">--</div></div>
  </div>
  <div class="ctrl-status" id="pw_status">pack --</div>
</div>

<h2>Camera &amp; Vision</h2>
<div class="row">
  <div class="panel">
    <div class="label">LIVE VIEW &mdash; water boundary, zones, steering pick</div>
    <img id="cam" alt="waiting for camera...">
  </div>
  <div class="panel" style="max-width: 300px;">
    <div class="label">PROCESSING</div>
    <div class="metrics" style="grid-template-columns: repeat(2, 1fr);">
      <div class="card"><div class="k">PIPELINE</div><div class="v" id="p_fps">--</div></div>
      <div class="card"><div class="k">LATENCY</div><div class="v" id="p_lat">--</div></div>
      <div class="card"><div class="k">BEST ZONE</div><div class="v" id="p_zone">--</div></div>
      <div class="card"><div class="k">CENTER WATER</div><div class="v" id="p_center">--</div></div>
    </div>
  </div>
</div>

<h2>Capture</h2>
<div class="panel">
  <div class="btn-row">
    <button class="btn" id="cap_btn">Start capture</button>
  </div>
  <div class="ctrl-status" id="cap_status">idle</div>
</div>

<h2>GPS</h2>
<div class="panel">
  <div class="metrics" style="grid-template-columns: repeat(auto-fit, minmax(108px, 1fr));">
    <div class="card"><div class="k">FIX</div><div class="v" id="g_fix">--</div></div>
    <div class="card"><div class="k">SATS</div><div class="v" id="g_sats">--</div></div>
    <div class="card"><div class="k">HDOP</div><div class="v" id="g_hdop">--</div></div>
    <div class="card"><div class="k">SPEED</div><div class="v" id="g_speed">--</div></div>
    <div class="card"><div class="k">COURSE</div><div class="v" id="g_course">--</div></div>
  </div>
  <div class="ctrl-status" id="g_pos">position --</div>
  <iframe id="g_map" title="GPS location" loading="lazy"
          referrerpolicy="no-referrer-when-downgrade"></iframe>
</div>

<h2>Attitude</h2>
<div class="row">
  <div class="panel"><div class="label">ROLL &mdash; bow-on, heel off vertical</div><canvas id="cRoll" width="380" height="280"></canvas></div>
  <div class="panel"><div class="label">PITCH &mdash; side, trim off horizontal</div><canvas id="cPitch" width="380" height="280"></canvas></div>
</div>
<div class="readout" id="readout">connecting...</div>

<h2>Controls</h2>
<div class="panel">
  <div class="btn-row" id="ctrl_buttons"></div>
  <div class="ctrl-status" id="ctrl_status"></div>
</div>
<script>
// ---- Controls: add a button by adding one entry here. Danger buttons require
// a second click to confirm. Shell actions need a matching ALLOWED_ACTIONS entry
// on the server; subsystem actions (estop, record) wire up once those exist. ----
const CONTROLS = [
  { id:'reboot', label:'Reboot Pi', action:'reboot', danger:true, confirm:'Click again to reboot' },
  { id:'shutdown', label:'Shut down', action:'shutdown', danger:true, confirm:'Click again to shut down' },
  // { id:'estop',    label:'Emergency stop', action:'estop', danger:true },  // needs control loop
];
// Below this ground speed, show the GPS as stopped. Stationary receivers never
// read exactly zero (Doppler noise floor). Set to 0 to show the raw value.
const SPEED_DEADBAND_MS = 0.15;
// Empty = keyless Google Maps embed (no setup, but unofficial). To use the
// supported Maps Embed API, put a key here (free, needs a Google Cloud project).
const GMAPS_KEY = '';
const _armed = {};
const _armTimers = {};
function makeControls(){
  const wrap = document.getElementById('ctrl_buttons');
  CONTROLS.forEach(function(c){
    const btn = document.createElement('button');
    btn.className = 'btn' + (c.danger ? ' danger' : '');
    btn.textContent = c.label;
    btn.onclick = function(){ onControlClick(c, btn); };
    wrap.appendChild(btn);
  });
}
function disarm(c, btn){
  _armed[c.id] = false;
  clearTimeout(_armTimers[c.id]);
  btn.classList.remove('armed');
  btn.textContent = c.label;
}
function onControlClick(c, btn){
  if (c.danger && !_armed[c.id]){
    _armed[c.id] = true;
    btn.classList.add('armed');
    btn.textContent = c.confirm || 'Click again to confirm';
    clearTimeout(_armTimers[c.id]);
    _armTimers[c.id] = setTimeout(function(){ disarm(c, btn); }, 4000);
    return;
  }
  disarm(c, btn);
  runControl(c, btn);
}
async function runControl(c, btn){
  const st = document.getElementById('ctrl_status');
  st.style.color = '#7d96a3';
  st.textContent = c.label + ': sending...';
  try {
    const r = await fetch('/action/' + c.action, { method: 'POST' });
    let d = {};
    try { d = await r.json(); } catch(e){}
    if (r.ok && d.ok !== false){
      st.style.color = '#3fd0a8';
      st.textContent = (d.message || (c.label + ' sent')) +
        ((c.action === 'reboot' || c.action === 'shutdown') ? '. This page will go offline.' : '.');
    } else {
      st.style.color = '#d9534f';
      st.textContent = c.label + ' failed: ' + (d.error || ('HTTP ' + r.status));
    }
  } catch(e){
    if (c.action === 'reboot' || c.action === 'shutdown'){
      st.style.color = '#d9b13a';
      st.textContent = c.label + ' sent. Connection closed, the Pi is going down.';
    } else {
      st.style.color = '#d9534f';
      st.textContent = c.label + ' failed: ' + e.message;
    }
  }
}

// ---- GPS map: reload the iframe only on first fix or after moving ~8 m, so a
// stationary boat doesn't flicker the map every second. Satellite view. ----
let _mapLat = null, _mapLon = null;
function gmapUrl(lat, lon){
  if (GMAPS_KEY){
    return 'https://www.google.com/maps/embed/v1/place?key=' + GMAPS_KEY +
           '&q=' + lat + ',' + lon + '&zoom=19&maptype=satellite';
  }
  return 'https://maps.google.com/maps?q=' + lat + ',' + lon + '&z=19&t=k&output=embed';
}
function updateMap(lat, lon){
  if (_mapLat !== null){
    const dN = (lat - _mapLat) * 111320;
    const dE = (lon - _mapLon) * 111320 * Math.cos(lat * Math.PI / 180);
    if (Math.hypot(dN, dE) < 8) return;  // moved less than 8 m, leave the map alone
  }
  _mapLat = lat; _mapLon = lon;
  const f = document.getElementById('g_map');
  f.style.display = 'block';
  f.src = gmapUrl(lat, lon);
}

function hullFront(ctx){
  // twin demihulls (bow-on)
  ctx.beginPath();
  ctx.moveTo(-78,-10); ctx.lineTo(-70,14); ctx.lineTo(-58,24);
  ctx.lineTo(-46,22); ctx.lineTo(-38,6); ctx.lineTo(-36,-10); ctx.closePath();
  ctx.moveTo(78,-10); ctx.lineTo(70,14); ctx.lineTo(58,24);
  ctx.lineTo(46,22); ctx.lineTo(38,6); ctx.lineTo(36,-10); ctx.closePath();
  ctx.stroke();
  // bridge deck across the top + tunnel ceiling between the hulls
  ctx.beginPath();
  ctx.moveTo(-78,-10); ctx.lineTo(-78,-22); ctx.lineTo(78,-22); ctx.lineTo(78,-10);
  ctx.moveTo(-36,-10); ctx.lineTo(36,-10);
  ctx.stroke();
  ctx.strokeRect(-16,-35,32,13);                                         // electronics box
  ctx.beginPath(); ctx.moveTo(0,-35); ctx.lineTo(0,-50); ctx.stroke();   // camera/antenna stub
  ctx.beginPath(); ctx.moveTo(-80,10); ctx.lineTo(-94,10);
  ctx.moveTo(80,10); ctx.lineTo(94,10); ctx.stroke();                    // waterline ticks
}
function hullSide(ctx){
  // far demihull, dimmed and offset to imply the second hull
  ctx.save();
  ctx.strokeStyle='rgba(215,227,234,0.38)';
  ctx.beginPath();
  ctx.moveTo(-85,-9); ctx.lineTo(81,-9); ctx.lineTo(115,3);
  ctx.lineTo(77,31); ctx.lineTo(-81,31); ctx.lineTo(-89,9); ctx.closePath();
  ctx.stroke();
  ctx.restore();
  // near demihull (bow to the right)
  ctx.beginPath();
  ctx.moveTo(-90,-16); ctx.lineTo(76,-16); ctx.lineTo(110,-4);
  ctx.lineTo(72,24); ctx.lineTo(-86,24); ctx.lineTo(-94,2); ctx.closePath();
  ctx.stroke();
  ctx.strokeRect(-56,-30,92,14);                                         // wide flat bridge deck
  ctx.beginPath(); ctx.moveTo(20,-30); ctx.lineTo(20,-46); ctx.stroke(); // camera/antenna stub
}
function drawScene(canvas, angleDeg, kind){
  const ctx=canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const cx=W/2, cy=H*0.58, R=Math.min(W,H)*0.40;
  const isRoll = kind==='front';
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0c1014'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(125,150,163,0.09)'; ctx.lineWidth=1;
  for(let g=0; g<=W; g+=30){ ctx.beginPath(); ctx.moveTo(g,0); ctx.lineTo(g,H); ctx.stroke(); }
  for(let g=0; g<=H; g+=30){ ctx.beginPath(); ctx.moveTo(0,g); ctx.lineTo(W,g); ctx.stroke(); }
  ctx.save();
  ctx.translate(cx,cy);
  const range = isRoll?60:45;
  ctx.font='11px ui-monospace, monospace'; ctx.textAlign='center'; ctx.textBaseline='middle';
  for(let v=-range; v<=range; v+=5){
    const maj = (v%15===0), a=v*Math.PI/180;
    const ux = isRoll?Math.sin(a):Math.cos(a);
    const uy = isRoll?-Math.cos(a):-Math.sin(a);
    ctx.strokeStyle = maj?'rgba(174,191,201,0.85)':'rgba(125,150,163,0.4)';
    ctx.lineWidth = maj?1.3:1;
    ctx.beginPath(); ctx.moveTo(ux*R,uy*R); ctx.lineTo(ux*(R+(maj?10:6)),uy*(R+(maj?10:6))); ctx.stroke();
    if(maj){ ctx.fillStyle='rgba(143,163,173,0.9)'; ctx.fillText((v>0?'+':'')+v, ux*(R+22), uy*(R+22)); }
  }
  ctx.strokeStyle='rgba(125,150,163,0.22)'; ctx.setLineDash([4,4]); ctx.lineWidth=1;
  ctx.beginPath();
  if(isRoll){ ctx.moveTo(0,8); ctx.lineTo(0,-R); } else { ctx.moveTo(-R,0); ctx.lineTo(R+12,0); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.save();
  ctx.rotate((isRoll?angleDeg:-angleDeg)*Math.PI/180);
  ctx.strokeStyle='#d7e3ea'; ctx.lineWidth=1.4; ctx.lineJoin='round'; ctx.lineCap='round';
  if(isRoll) hullFront(ctx); else hullSide(ctx);
  ctx.strokeStyle='#3fd0a8'; ctx.lineWidth=2;
  ctx.beginPath();
  if(isRoll){ ctx.moveTo(0,0); ctx.lineTo(0,-R); } else { ctx.moveTo(0,0); ctx.lineTo(R,0); }
  ctx.stroke();
  ctx.fillStyle='#3fd0a8'; ctx.beginPath();
  if(isRoll) ctx.arc(0,-R,3,0,7); else ctx.arc(R,0,3,0,7);
  ctx.fill();
  ctx.restore();
  ctx.restore();
}
const cRoll=document.getElementById('cRoll'), cPitch=document.getElementById('cPitch');
const readout=document.getElementById('readout');
function fmt(v){ const n=v.toFixed(1); return (v>=0?'+':'')+n; }
async function attTick(){
  try{
    const d=await (await fetch('/data',{cache:'no-store'})).json();
    if(!d.ok){
      drawScene(cRoll, 0, 'front');
      drawScene(cPitch, 0, 'side');
      readout.innerHTML='<span class="stale">IMU offline</span>';
      return;
    }
    drawScene(cRoll, d.roll, 'front');
    drawScene(cPitch, d.pitch, 'side');
    readout.innerHTML = 'ROLL <span>'+fmt(d.roll)+'\\u00B0</span> &nbsp; PITCH <span>'+fmt(d.pitch)+
      '\\u00B0</span> &nbsp; <span class="drift">YAW '+fmt(d.yaw)+'\\u00B0 drift</span>';
  }catch(e){
    readout.innerHTML='<span class="stale">link lost, retrying...</span>';
  }
}
function set(id, html, color){
  const el=document.getElementById(id);
  el.innerHTML=html;
  if(color) el.style.color=color;
}
function upt(s){
  if(s==null) return '--';
  const d=Math.floor(s/86400); s%=86400;
  const h=Math.floor(s/3600); s%=3600;
  const m=Math.floor(s/60);
  if(d>0) return d+'d '+h+'h '+m+'m';
  if(h>0) return h+'h '+m+'m';
  return m+'m';
}
// Self-paced camera frame loader: only requests the next frame after the
// current one finishes loading, so it adapts to bandwidth instead of piling up.
function frameTick(){
  const img = new Image();
  img.onload = function(){ document.getElementById('cam').src = img.src; setTimeout(frameTick, 120); };
  img.onerror = function(){ setTimeout(frameTick, 600); };
  img.src = '/frame?t=' + Date.now();
}
async function procTick(){
  let d;
  try { d=await (await fetch('/proc',{cache:'no-store'})).json(); }
  catch(e){ return; }
  if(!d.cam_ok){
    set('p_fps','--','#7d96a3'); set('p_lat','--','#7d96a3');
    set('p_zone','--','#7d96a3'); set('p_center','--','#7d96a3');
    return;
  }
  set('p_fps', d.fps.toFixed(1)+'<small> fps</small>',
      d.fps>=20?'#3fd0a8':d.fps>=10?'#d9b13a':'#d9534f');
  set('p_lat', Math.round(d.latency_ms)+'<small> ms</small>',
      d.latency_ms<=33?'#3fd0a8':d.latency_ms<=66?'#d9b13a':'#d9534f');
  set('p_zone', 'Z'+d.best_zone+'<small> / 4</small>','#e6eef2');
  set('p_center', Math.round(d.center_pct)+'<small> %</small>',
      d.center_pct>=35?'#3fd0a8':d.center_pct>=25?'#d9b13a':'#d9534f');
}
async function gpsTick(){
  let d;
  try { d=await (await fetch('/gps',{cache:'no-store'})).json(); }
  catch(e){ return; }
  const pos = document.getElementById('g_pos');
  if(!d.connected){
    set('g_fix','<small>no device</small>','#7d96a3');
    set('g_sats','--','#7d96a3'); set('g_hdop','--','#7d96a3');
    set('g_speed','--','#7d96a3'); set('g_course','--','#7d96a3');
    pos.textContent = 'GPS not detected. Plug in the receiver; stop gpsd if it is running.';
    return;
  }
  const fix = d.fix_type || 'none';
  if(fix==='3D' && d.valid) set('g_fix','3D','#3fd0a8');
  else if(fix==='2D' && d.valid) set('g_fix','2D','#d9b13a');
  else set('g_fix','NONE','#d9534f');
  set('g_sats', (d.sats_used!=null?d.sats_used:'--')+'<small> / '+(d.sats_view!=null?d.sats_view:'--')+'</small>','#e6eef2');
  if(d.hdop!=null) set('g_hdop', d.hdop, d.hdop<=2?'#3fd0a8':d.hdop<=5?'#d9b13a':'#d9534f');
  else set('g_hdop','--','#7d96a3');
  if(d.sog_ms!=null){
    const v = d.sog_ms < SPEED_DEADBAND_MS ? 0 : d.sog_ms;
    set('g_speed', v.toFixed(2)+'<small> m/s</small>','#e6eef2');
  } else set('g_speed','--','#7d96a3');
  set('g_course', d.cog!=null? Math.round(d.cog)+'<small> deg</small>' : '<small>--</small>', '#e6eef2');
  if(d.lat!=null && d.lon!=null){
    pos.textContent = d.lat.toFixed(6)+', '+d.lon.toFixed(6) +
      (d.alt!=null? '   alt '+Math.round(d.alt)+' m':'') +
      (d.utc? '   UTC '+d.utc:'') +
      (d.age_s!=null && d.age_s>3? '   (stale '+d.age_s+'s)':'');
    updateMap(d.lat, d.lon);
  } else {
    pos.textContent = 'no fix yet' + (d.sats_view!=null? ' ('+d.sats_view+' sats in view, needs sky view)':'');
  }
}
async function sysTick(){
  let d;
  try { d=await (await fetch('/sys',{cache:'no-store'})).json(); }
  catch(e){ return; }
  if(d.imu_hz==null) set('m_hz','--');
  else set('m_hz', d.imu_hz.toFixed(1)+'<small> Hz</small>',
           d.imu_hz>=95?'#3fd0a8':d.imu_hz>=50?'#d9b13a':'#d9534f');
  if(d.temp_c==null) set('m_temp','--','#7d96a3');
  else set('m_temp', d.temp_c.toFixed(1)+'<small> \\u00B0C</small>',
           d.temp_c>=78?'#d9534f':d.temp_c>=65?'#d9b13a':'#3fd0a8');
  if(d.cpu_pct==null) set('m_cpu','--','#e6eef2');
  else set('m_cpu', Math.round(d.cpu_pct)+'<small> %</small>'+(d.load1!=null?' <small>('+d.load1.toFixed(2)+')</small>':''),
           d.cpu_pct>=95?'#d9534f':d.cpu_pct>=85?'#d9b13a':'#e6eef2');
  if(d.mem_pct==null) set('m_mem','--');
  else set('m_mem', d.mem_used_mb+'<small> / '+d.mem_total_mb+' MB</small>',
           d.mem_pct>=92?'#d9534f':d.mem_pct>=80?'#d9b13a':'#e6eef2');
  set('m_disk', d.disk_pct!=null? Math.round(d.disk_pct)+'<small> % used</small>':'--');
  set('m_up', upt(d.uptime_s));
  set('m_ssid', d.ssid? d.ssid : '<small>offline</small>', d.ssid?'#e6eef2':'#d9534f');
  if(d.wifi_dbm==null) set('m_wifi','--');
  else set('m_wifi', d.wifi_dbm+'<small> dBm ('+d.wifi_pct+'%)</small>',
           d.wifi_dbm<=-78?'#d9534f':d.wifi_dbm<=-67?'#d9b13a':'#e6eef2');
  set('m_ip', d.ip? '<small>'+d.ip+'</small>' : '--', '#e6eef2');
  const pw=document.getElementById('m_power');
  if(d.under_voltage){ pw.textContent='POWER: UNDERVOLTAGE NOW';
    pw.style.color='#fff'; pw.style.background='#7a1f1f'; pw.style.borderColor='#d9534f'; }
  else if(d.throttled_now){ pw.textContent='POWER: THROTTLED NOW';
    pw.style.color='#f0c067'; pw.style.background='#0c1014'; pw.style.borderColor='#d9b13a'; }
  else if(d.uv_occurred){ pw.textContent='POWER: undervoltage occurred earlier this session';
    pw.style.color='#d9b13a'; pw.style.background='#0c1014'; pw.style.borderColor='#3a3320'; }
  else if(d.under_voltage===false){ pw.textContent='POWER: OK';
    pw.style.color='#3fd0a8'; pw.style.background='#0c1014'; pw.style.borderColor='#1b2530'; }
  else { pw.textContent='POWER: vcgencmd unavailable'; pw.style.color='#7d96a3'; }
}
async function powerTick(){
  let d;
  try { d=await (await fetch('/power',{cache:'no-store'})).json(); }
  catch(e){ return; }
  const s=document.getElementById('pw_status');
  if(!d.connected){
    set('pw_pack','--','#7d96a3'); set('pw_cur','--','#7d96a3');
    set('pw_w','--','#7d96a3'); set('pw_rail','--','#7d96a3');
    s.textContent='pack: sensor offline'; s.style.color='#7d96a3';
    return;
  }
  if(d.pack_v==null) set('pw_pack','--');
  else set('pw_pack', d.pack_v.toFixed(2)+'<small> V</small>',
           d.pack_v<6.0?'#d9534f':d.pack_v<6.6?'#d9b13a':'#3fd0a8');
  set('pw_cur', d.current_ma!=null? (d.current_ma/1000).toFixed(2)+'<small> A</small>':'--', '#e6eef2');
  set('pw_w', d.power_w!=null? d.power_w.toFixed(2)+'<small> W</small>':'--', '#e6eef2');
  set('pw_rail', d.rail_v!=null? d.rail_v.toFixed(2)+'<small> V</small>':'--', '#7d96a3');
  const age = d.age_s!=null? ' ('+d.age_s+'s ago)':'';
  if(d.status==='critical'){ s.textContent='PACK CRITICAL - stop'+age; s.style.color='#d9534f'; }
  else if(d.status==='low'){ s.textContent='PACK LOW - return to shore'+age; s.style.color='#d9b13a'; }
  else { s.textContent='pack ok'+age; s.style.color='#3fd0a8'; }
}
let _capRec = false;
async function capPoll(){
  try{
    const d = await (await fetch('/capture',{cache:'no-store'})).json();
    _capRec = d.recording;
    const btn = document.getElementById('cap_btn');
    const st = document.getElementById('cap_status');
    btn.textContent = d.recording ? 'Stop capture' : 'Start capture';
    btn.classList.toggle('armed', d.recording);
    if (d.error){ st.style.color = '#d9534f'; st.textContent = d.error; }
    else if (d.recording){
      st.style.color = '#3fd0a8';
      st.textContent = 'recording ' + d.session + ' | ' + d.frames + ' frames | ' +
        d.elapsed + 's | free ' + (d.free_mb != null ? Math.round(d.free_mb) + ' MB' : '--');
    } else {
      st.style.color = '#7d96a3';
      st.textContent = 'idle' + (d.session ? (' | last: ' + d.session) : '');
    }
  }catch(e){}
}
async function capToggle(){
  const btn = document.getElementById('cap_btn');
  btn.disabled = true;
  try{
    const path = _capRec ? '/capture/stop' : '/capture/start';
    const d = await (await fetch(path, { method:'POST' })).json();
    if (d.ok === false){
      const st = document.getElementById('cap_status');
      st.style.color = '#d9534f'; st.textContent = d.error || 'failed';
    }
  }catch(e){}
  btn.disabled = false;
  capPoll();
}
document.getElementById('cap_btn').onclick = capToggle;
makeControls();
setInterval(attTick, 50); attTick();
setInterval(sysTick, 1500); sysTick();
setInterval(procTick, 500); procTick();
setInterval(gpsTick, 1000); gpsTick();
setInterval(powerTick, 1000); powerTick();
frameTick();
setInterval(capPoll, 1000); capPoll();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/capture/start":
            self._json(capture_start())
            return
        if self.path == "/capture/stop":
            self._json(capture_stop())
            return
        if self.path.startswith("/action/"):
            name = self.path[len("/action/"):].strip("/")
            cmd = ALLOWED_ACTIONS.get(name)
            if cmd is None:
                self._json({"ok": False, "error": "unknown action: " + name}, code=404)
                return
            try:
                # sudo -n fails fast if passwordless sudo is not configured, so the
                # button reports the reason instead of silently doing nothing.
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    self._json({"ok": True, "message": name + " sent"})
                else:
                    msg = (r.stderr or r.stdout or "command failed").strip()
                    self._json({"ok": False, "error": msg}, code=500)
            except subprocess.TimeoutExpired:
                # Box is most likely going down (reboot/poweroff); treat as success.
                self._json({"ok": True, "message": name + " sent"})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, code=500)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        if self.path.startswith("/data"):
            with alock:
                body = json.dumps({
                    "ok": att["ok"],
                    "roll": math.degrees(att["roll"]),
                    "pitch": math.degrees(att["pitch"]),
                    "yaw": math.degrees(att["yaw"]),
                }).encode()
            self._send(body, "application/json")
        elif self.path.startswith("/sys"):
            with slock:
                data = dict(sysm)
            with alock:
                data["imu_hz"] = round(att["hz"], 1)
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/proc"):
            with plock:
                body = json.dumps(dict(proc)).encode()
            self._send(body, "application/json")
        elif self.path.startswith("/gps"):
            with glock:
                data = dict(gps)
            ld = data.pop("last_data", 0.0)
            data["age_s"] = round(time.monotonic() - ld, 1) if (data["connected"] and ld) else None
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/power"):
            with pwlock:
                data = dict(power)
            ld = data.pop("last_data", 0.0)
            data["age_s"] = round(time.monotonic() - ld, 1) if (data["connected"] and ld) else None
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/capture"):
            with caplock:
                st = {"recording": cap["recording"], "frames": cap["frames"],
                      "elapsed": round(time.monotonic() - cap["started"], 1) if cap["recording"] else 0,
                      "session": os.path.basename(cap["session"]) if cap["session"] else None,
                      "free_mb": cap["free_mb"], "error": cap["error"]}
            self._send(json.dumps(st).encode(), "application/json")
        elif self.path.startswith("/frame"):
            with flock:
                buf = frame_buf["jpeg"]
            if buf is None:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(buf, "image/jpeg")
        else:
            self._send(PAGE, "text/html; charset=utf-8")


if __name__ == "__main__":
    if IMU_AVAILABLE:
        threading.Thread(target=imu_loop, daemon=True).start()
    else:
        print("imu disabled; attitude section will show offline")
    threading.Thread(target=sys_loop, daemon=True).start()
    if CAMERA_AVAILABLE:
        threading.Thread(target=camera_loop, daemon=True).start()
    else:
        print("camera/vision disabled; dashboard will show -- for those cards")
    if GPS_AVAILABLE:
        threading.Thread(target=gps_loop, daemon=True).start()
    else:
        print("gps disabled; install pyserial to enable the GPS section")
    if INA_AVAILABLE:
        threading.Thread(target=power_loop, daemon=True).start()
    else:
        print("power disabled; install adafruit-circuitpython-ina219 and "
              "adafruit-extended-bus to enable the Power section")
    print("AutoBoat2w dashboard running. From a device on the same network open:")
    print("  http://<this-pi-ip>:%d" % PORT)
    print("Find the Pi IP with:  hostname -I")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

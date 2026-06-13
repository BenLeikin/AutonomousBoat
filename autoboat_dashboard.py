"""AutoBoat dashboard: the HTTP server and nothing else.

Serves the web UI from static/index.html, exposes the read endpoints over each
subsystem's shared state, routes actions to the modules that own them, and
starts the polling threads. All actual function lives in its module:

    control/controller.py   avoidance brain (hardware-free, unit-tested)
    pilot.py                 camera loop: vision -> controller -> motors, Start/Stop
    recording.py             session capture, telemetry, Analyze-tab readers
    hardware/imu.py          MPU-6050 attitude
    hardware/motors.py       DRV8833 actuation, trim, watchdog, ARMED
    hardware/power.py        INA219 pack monitor, critical shutdown, hard guard
    hardware/gps.py          VK-162 NMEA reader
    hardware/tof.py          VL53L1X forward range
    hardware/sysmon.py       CPU/mem/wifi/throttle metrics
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hardware.imu as imu
import hardware.gps as hwgps
import hardware.power as hwpower
import hardware.sysmon as sysmon
import hardware.tof as hwtof
import hardware.motors as motors
import recording
import pilot

PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ---------- Shell actions ----------
# Shell-level actions the dashboard is allowed to run. Each needs a matching
# button entry in the CONTROLS array on the page (static/index.html).
# These use `sudo -n` (non-interactive): if passwordless sudo is not set up for
# the command, it fails fast and the button reports "sudo: a password is required".
ALLOWED_ACTIONS = {
    "reboot": ["sudo", "-n", "systemctl", "reboot"],
    "shutdown": ["sudo", "-n", "systemctl", "poweroff"],
    # --no-block so systemctl queues the restart and returns before systemd kills
    # this very process; otherwise we can get killed mid-stop and the unit never
    # comes back. Needs a sudoers line allowing this exact command.
    "restart": ["sudo", "-n", "systemctl", "--no-block", "restart", "autoboat-dashboard"],
}

# ---------- In-process actions ----------
# Each action lives in the module that owns the machinery; this is just the
# name -> function routing for POST /action/<name>.
PY_ACTIONS = {
    "start": pilot.start_run,
    "stop": pilot.stop_run,
    "arm": motors.arm,
    "disarm": motors.disarm,
    "cap_up": lambda: motors.cap_step(+0.1),
    "cap_down": lambda: motors.cap_step(-0.1),
    "trim_check": motors.trim_check,
    "imu_zero": imu.zero_heading,
    "snapshot": pilot.snapshot_still,
    "critical_override": hwpower._critical_override,
}


def _index_page():
    try:
        with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
            return f.read()
    except Exception as e:
        return ("<html><body><h2>AutoBoat dashboard</h2><p>static/index.html "
                "missing or unreadable (%s). Deploy the static folder next to "
                "autoboat_dashboard.py.</p></body></html>" % e).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, ctype, no_cache=False):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- POST ----------------
    def do_POST(self):
        if self.path == "/capture/start":
            self._json(recording.capture_start())
            return
        if self.path == "/capture/stop":
            self._json(recording.capture_stop())
            return
        if self.path.startswith("/action/"):
            name = self.path[len("/action/"):].strip("/")
            if name in PY_ACTIONS:
                try:
                    self._json(PY_ACTIONS[name]())
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, code=500)
                return
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

    # ---------------- session routes ----------------
    def _handle_session(self):
        full = self.path[len("/session/"):]
        rest, _, query = full.partition("?")
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2:
            self._json({"error": "bad request"}, code=400)
            return
        name, what = parts[0], parts[1]
        sdir = recording._safe_session_dir(name)
        if sdir is None:
            self._json({"error": "no such session"}, code=404)
            return
        method = "texture"
        for kv in query.split("&"):
            if kv.startswith("method="):
                method = kv[len("method="):]
        if method not in ("texture", "color"):
            method = "texture"
        if what == "manifest":
            man = recording._read_json_file(os.path.join(sdir, "manifest.json"))
            self._json(man if man is not None else {})
        elif what == "telemetry":
            cols, rows = recording._read_telemetry(sdir)
            self._send(json.dumps({"columns": cols or [], "rows": rows or []}).encode(),
                       "application/json")
        elif what == "events":
            txt = recording._read_text_file(os.path.join(sdir, "events.log")) or ""
            self._send(txt.encode(), "text/plain; charset=utf-8")
        elif what == "gps":
            cols, rows = recording._read_telemetry(sdir)
            self._send(json.dumps({"points": recording._gps_points(cols, rows or [])}).encode(),
                       "application/json")
        elif what == "frame" and len(parts) >= 3 and parts[2].isdigit():
            fp = recording._frame_file(os.path.join(sdir, "frames"), int(parts[2]))
            if fp and os.path.isfile(fp):
                ctype = "image/png" if fp.endswith(".png") else "image/jpeg"
                try:
                    with open(fp, "rb") as f:
                        self._send(f.read(), ctype)
                except Exception:
                    self.send_response(500); self.send_header("Content-Length", "0"); self.end_headers()
            else:
                self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
        elif what == "analyze" and len(parts) >= 3 and parts[2].isdigit():
            fp = recording._frame_file(os.path.join(sdir, "frames"), int(parts[2]))
            if not fp or not os.path.isfile(fp):
                self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
            try:
                self._send(pilot.analyze_frame_file(fp, method), "image/jpeg")
            except Exception as e:
                code = 503 if "unavailable" in str(e) else 500
                self._json({"error": str(e)}, code=code)
        elif what == "reanalyze":
            try:
                out = pilot.reanalyze_session(os.path.join(sdir, "frames"), method,
                                              recording._frame_file)
                self._json(out)
            except Exception as e:
                code = 503 if "unavailable" in str(e) else 500
                self._json({"error": str(e)}, code=code)
        else:
            self._json({"error": "unknown resource"}, code=404)

    # ---------------- GET ----------------
    def do_GET(self):
        if self.path == "/sessions" or self.path.startswith("/sessions?"):
            self._send(json.dumps(recording._list_sessions()).encode(), "application/json")
            return
        if self.path.startswith("/session/"):
            self._handle_session()
            return
        if self.path.startswith("/data"):
            with imu.alock:
                body = json.dumps({
                    "ok": imu.att["ok"],
                    "roll": math.degrees(imu.att["roll"]),
                    "pitch": math.degrees(imu.att["pitch"]),
                    "yaw": math.degrees(imu.att["yaw"]),
                }).encode()
            self._send(body, "application/json")
        elif self.path.startswith("/sys"):
            with sysmon.slock:
                data = dict(sysmon.sysm)
            with imu.alock:
                data["imu_hz"] = round(imu.att["hz"], 1)
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/proc"):
            with pilot.plock:
                data = dict(pilot.proc)
            data["throttle_cap"] = motors.MOTOR_THROTTLE_CAP
            data["motors_available"] = motors.MOTORS_AVAILABLE
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/tof"):
            with hwtof.tlock:
                data = dict(hwtof.tof)
            ld = data.pop("last_data", 0.0)
            data["age_s"] = round(time.monotonic() - ld, 1) if (data["connected"] and ld) else None
            data["lib"] = hwtof.TOF_LIB_AVAILABLE
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/gps"):
            with hwgps.glock:
                data = dict(hwgps.gps)
            ld = data.pop("last_data", 0.0)
            data["age_s"] = round(time.monotonic() - ld, 1) if (data["connected"] and ld) else None
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/power"):
            with hwpower.pwlock:
                data = dict(hwpower.power)
            ld = data.pop("last_data", 0.0)
            data["age_s"] = round(time.monotonic() - ld, 1) if (data["connected"] and ld) else None
            data["critical"] = hwpower._critical_status()
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path.startswith("/capture"):
            with recording.caplock:
                cap = recording.cap
                st = {"recording": cap["recording"], "frames": cap["frames"],
                      "elapsed": round(time.monotonic() - cap["started"], 1) if cap["recording"] else 0,
                      "session": os.path.basename(cap["session"]) if cap["session"] else None,
                      "free_mb": cap["free_mb"], "error": cap["error"],
                      "last_session": cap["last_session"], "last_frames": cap["last_frames"],
                      "last_duration": cap["last_duration"]}
            self._send(json.dumps(st).encode(), "application/json")
        elif self.path.startswith("/frame"):
            with pilot.flock:
                buf = pilot.frame_buf["jpeg"]
            if buf is None:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(buf, "image/jpeg")
        else:
            # no-store so editing static/index.html shows up on a plain refresh
            self._send(_index_page(), "text/html; charset=utf-8", no_cache=True)


if __name__ == "__main__":
    if imu.IMU_AVAILABLE:
        threading.Thread(target=imu.imu_loop, daemon=True).start()
    else:
        print("imu disabled; attitude section will show offline")
    threading.Thread(target=sysmon.sys_loop, daemon=True).start()
    threading.Thread(target=motors.watchdog, daemon=True).start()
    if pilot.CAMERA_AVAILABLE:
        threading.Thread(target=pilot.camera_loop, daemon=True).start()
    else:
        print("camera/vision disabled; dashboard will show -- for those cards")
    if hwgps.GPS_AVAILABLE:
        threading.Thread(target=hwgps.gps_loop, daemon=True).start()
    else:
        print("gps disabled; install pyserial to enable the GPS section")
    if hwtof.TOF_LIB_AVAILABLE:
        threading.Thread(target=hwtof.tof_loop, daemon=True).start()
    else:
        print("tof disabled; range gate inactive until the VL53L1X + library are present")
    if hwpower.INA_AVAILABLE:
        threading.Thread(target=hwpower.power_loop, daemon=True).start()
    else:
        print("power disabled; install adafruit-circuitpython-ina219 and "
              "adafruit-extended-bus to enable the Power section")
    print("AutoBoat dashboard running. From a device on the same network open:")
    print("  http://<this-pi-ip>:%d" % PORT)
    print("Find the Pi IP with:  hostname -I")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

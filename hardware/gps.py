"""VK-162 GPS over USB serial: NMEA parsing, shared `gps` state, loop."""
import glob
import threading
import time

# GPS is optional too: needs pyserial and a receiver on a USB serial port.
try:
    import serial
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("[gps] pyserial not installed; GPS section disabled "
          "(pip install pyserial --break-system-packages)")

GPS_BAUD = 9600                    # VK-162 default (u-blox 7, 8N1)

# ---------- GPS shared state ----------
gps = {"connected": False, "fix_type": "none", "valid": False,
       "sats_used": None, "sats_view": None, "lat": None, "lon": None,
       "alt": None, "hdop": None, "sog_ms": None, "cog": None,
       "utc": None, "date": None, "last_data": 0.0}
glock = threading.Lock()

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

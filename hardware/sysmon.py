"""System metrics (/proc, vcgencmd, wifi): shared `sysm` state and loop."""
import shutil
import subprocess
import threading
import time

sysm = {}                                                   # system metrics, shared
slock = threading.Lock()

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

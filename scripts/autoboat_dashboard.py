import time
import math
import json
import shutil
import threading
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import board
import busio
from adafruit_lsm6ds import Rate
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS

PORT = 8000
TARGET_HZ = 104                    # loop target; matches the sensor's default ODR
PERIOD = 1.0 / TARGET_HZ

# ---------- IMU ----------
i2c = busio.I2C(board.SCL, board.SDA)
sensor = LSM6DS(i2c, address=0x6A)
# Make the sensor output rate explicit. If you raise TARGET_HZ above 104,
# raise these to the next step (RATE_208_HZ, etc.) or you will reread samples.
sensor.accelerometer_data_rate = Rate.RATE_104_HZ
sensor.gyro_data_rate = Rate.RATE_104_HZ

# ---------- Complementary filter config ----------
ALPHA = 0.98          # near 1.0: gyro short-term, accel long-term (drift kill)
ROLL_SIGN = 1         # set to -1 if the boat heels the wrong way
PITCH_SIGN = 1        # set to -1 if bow/stern is reversed
YAW_SIGN = 1

att = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "hz": 0.0}   # radians + measured Hz
alock = threading.Lock()

sysm = {}                                                   # system metrics, shared
slock = threading.Lock()


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

        sleep_left = PERIOD - (time.monotonic() - start)
        if sleep_left > 0:
            time.sleep(sleep_left)


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
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                m["freq_mhz"] = round(int(f.read()) / 1000)
        except Exception:
            m["freq_mhz"] = None

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
<title>AutoBoat2w</title>
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
           border-radius: 10px; padding: 10px; }
  .label { font-size: 12px; color: #6b8794; margin-bottom: 8px; letter-spacing: 0.5px; }
  canvas { width: 100%; height: auto; display: block; }
  .readout { margin-top: 14px; font-size: 16px; letter-spacing: 1px; }
  .readout span { color: #e6eef2; }
  .drift { color: #d98a3a; }
  .stale { color: #d9534f; }
</style>
</head>
<body>
<h1>AutoBoat2w</h1>
<div class="metrics">
  <div class="card"><div class="k">IMU RATE</div><div class="v" id="m_hz">--</div></div>
  <div class="card"><div class="k">CPU TEMP</div><div class="v" id="m_temp">--</div></div>
  <div class="card"><div class="k">CPU LOAD</div><div class="v" id="m_cpu">--</div></div>
  <div class="card"><div class="k">CPU CLOCK</div><div class="v" id="m_freq">--</div></div>
  <div class="card"><div class="k">MEMORY</div><div class="v" id="m_mem">--</div></div>
  <div class="card"><div class="k">DISK</div><div class="v" id="m_disk">--</div></div>
  <div class="card"><div class="k">WIFI</div><div class="v" id="m_wifi">--</div></div>
  <div class="card"><div class="k">UPTIME</div><div class="v" id="m_up">--</div></div>
</div>
<div class="status" id="m_power">power: --</div>

<h2>Attitude</h2>
<div class="row">
  <div class="panel"><div class="label">ROLL &mdash; bow-on, heel off vertical</div><canvas id="cRoll" width="380" height="280"></canvas></div>
  <div class="panel"><div class="label">PITCH &mdash; side, trim off horizontal</div><canvas id="cPitch" width="380" height="280"></canvas></div>
</div>
<div class="readout" id="readout">connecting...</div>
<script>
function hullFront(ctx){
  ctx.beginPath();
  ctx.moveTo(-82,-15); ctx.lineTo(82,-15); ctx.lineTo(44,30); ctx.lineTo(0,41); ctx.lineTo(-44,30); ctx.closePath();
  ctx.stroke();
  ctx.strokeRect(-24,-15,48,-28);
  ctx.beginPath(); ctx.moveTo(0,-43); ctx.lineTo(0,-72); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-82,-2); ctx.lineTo(-62,-2); ctx.moveTo(82,-2); ctx.lineTo(62,-2); ctx.stroke();
}
function hullSide(ctx){
  ctx.beginPath();
  ctx.moveTo(-92,-17); ctx.lineTo(78,-17); ctx.lineTo(118,-4); ctx.lineTo(70,30); ctx.lineTo(-86,30); ctx.closePath();
  ctx.stroke();
  ctx.strokeRect(-58,-17,52,-26);
  ctx.beginPath(); ctx.moveTo(-8,-43); ctx.lineTo(-8,-70); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(2,-17); ctx.lineTo(6,28); ctx.moveTo(38,-17); ctx.lineTo(44,26); ctx.stroke();
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
  set('m_freq', d.freq_mhz!=null? d.freq_mhz+'<small> MHz</small>':'--');
  if(d.mem_pct==null) set('m_mem','--');
  else set('m_mem', d.mem_used_mb+'<small> / '+d.mem_total_mb+' MB</small>',
           d.mem_pct>=92?'#d9534f':d.mem_pct>=80?'#d9b13a':'#e6eef2');
  set('m_disk', d.disk_pct!=null? Math.round(d.disk_pct)+'<small> % used</small>':'--');
  if(d.wifi_dbm==null) set('m_wifi','--');
  else set('m_wifi', d.wifi_dbm+'<small> dBm ('+d.wifi_pct+'%)</small>',
           d.wifi_dbm<=-78?'#d9534f':d.wifi_dbm<=-67?'#d9b13a':'#e6eef2');
  set('m_up', upt(d.uptime_s));
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
setInterval(attTick, 50); attTick();
setInterval(sysTick, 1500); sysTick();
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

    def do_GET(self):
        if self.path.startswith("/data"):
            with alock:
                body = json.dumps({
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
        else:
            self._send(PAGE, "text/html; charset=utf-8")


if __name__ == "__main__":
    threading.Thread(target=imu_loop, daemon=True).start()
    threading.Thread(target=sys_loop, daemon=True).start()
    print("AutoBoat2w dashboard running. From a device on the same network open:")
    print("  http://<this-pi-ip>:%d" % PORT)
    print("Find the Pi IP with:  hostname -I")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

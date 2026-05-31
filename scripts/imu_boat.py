import time
import math
import board
import busio
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# ---------- IMU ----------
i2c = busio.I2C(board.SCL, board.SDA)
sensor = LSM6DS(i2c, address=0x6A)

# ---------- Complementary filter ----------
# ALPHA close to 1.0 trusts the gyro short-term (smooth, responsive) and
# leans on the accelerometer long-term (kills gyro drift on roll/pitch).
ALPHA = 0.98

# Set these to -1 to flip an axis if the boat tilts the wrong way.
ROLL_SIGN = 1
PITCH_SIGN = 1
YAW_SIGN = 1

state = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "t": time.monotonic()}


def read_attitude():
    ax, ay, az = sensor.acceleration      # m/s^2
    gx, gy, gz = sensor.gyro              # rad/s

    now = time.monotonic()
    dt = now - state["t"]
    state["t"] = now

    # Absolute roll/pitch from gravity vector (valid only because accel
    # measures the 1 g reaction; no yaw info is available from gravity).
    roll_acc = math.atan2(ay, az)
    pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

    # Fuse: integrate the rate gyro, nudge back toward the accel reading.
    state["roll"] = ALPHA * (state["roll"] + gx * dt) + (1 - ALPHA) * roll_acc
    state["pitch"] = ALPHA * (state["pitch"] + gy * dt) + (1 - ALPHA) * pitch_acc
    state["yaw"] += gz * dt               # gyro-only -> drifts, no absolute ref

    return (ROLL_SIGN * state["roll"],
            PITCH_SIGN * state["pitch"],
            YAW_SIGN * state["yaw"])


# ---------- Boat geometry (local coords, origin on the waterline) ----------
hull_front = np.array([[-1.30, 0.15], [1.30, 0.15], [0.70, -0.55], [-0.70, -0.55]])
cabin_front = np.array([[-0.45, 0.15], [0.45, 0.15], [0.45, 0.75], [-0.45, 0.75]])
mast_front = np.array([[0.0, 0.75], [0.0, 1.85]])
flag_front = np.array([[0.0, 1.85], [0.60, 1.65], [0.0, 1.45]])

hull_side = np.array([[-1.55, 0.20], [1.35, 0.20], [2.15, 0.05], [1.25, -0.55], [-1.30, -0.55]])
cabin_side = np.array([[-1.00, 0.20], [-0.05, 0.20], [-0.05, 0.72], [-1.00, 0.72]])
mast_side = np.array([[-0.18, 0.72], [-0.18, 1.78]])
flag_side = np.array([[-0.18, 1.78], [0.45, 1.58], [-0.18, 1.38]])


def rot(points, theta):
    c, s = math.cos(theta), math.sin(theta)
    return points @ np.array([[c, -s], [s, c]]).T


# ---------- Figure ----------
fig, (axR, axP) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("IMU attitude", fontsize=14)


def setup(a, title):
    a.set_xlim(-2.6, 2.6)
    a.set_ylim(-2.0, 2.6)
    a.set_aspect("equal")
    a.axhspan(-2.0, 0.0, color="#9ecbff")        # water
    a.axhline(0.0, color="#1f6feb", lw=2)         # horizon / waterline
    a.set_title(title)
    a.set_xticks([])
    a.set_yticks([])


setup(axR, "Roll (bow-on)")
setup(axP, "Pitch (side)")

hullR = Polygon(hull_front, closed=True, fc="#c0623a", ec="#3a2a20", lw=1.5)
cabinR = Polygon(cabin_front, closed=True, fc="#f2f2f0", ec="#3a2a20", lw=1.5)
flagR = Polygon(flag_front, closed=True, fc="#d4537e", ec="none")
mastR, = axR.plot([], [], color="#3a2a20", lw=2.5)
for p in (hullR, cabinR, flagR):
    axR.add_patch(p)

hullP = Polygon(hull_side, closed=True, fc="#c0623a", ec="#3a2a20", lw=1.5)
cabinP = Polygon(cabin_side, closed=True, fc="#f2f2f0", ec="#3a2a20", lw=1.5)
flagP = Polygon(flag_side, closed=True, fc="#d4537e", ec="none")
mastP, = axP.plot([], [], color="#3a2a20", lw=2.5)
for p in (hullP, cabinP, flagP):
    axP.add_patch(p)

readout = fig.text(0.5, 0.02, "", ha="center", family="monospace", fontsize=11)


def update(_):
    r, p, y = read_attitude()

    hullR.set_xy(rot(hull_front, r))
    cabinR.set_xy(rot(cabin_front, r))
    flagR.set_xy(rot(flag_front, r))
    m = rot(mast_front, r)
    mastR.set_data(m[:, 0], m[:, 1])

    hullP.set_xy(rot(hull_side, p))
    cabinP.set_xy(rot(cabin_side, p))
    flagP.set_xy(rot(flag_side, p))
    m = rot(mast_side, p)
    mastP.set_data(m[:, 0], m[:, 1])

    readout.set_text(
        f"roll {math.degrees(r):+6.1f}    "
        f"pitch {math.degrees(p):+6.1f}    "
        f"yaw {math.degrees(y):+6.1f} (drifts)"
    )
    return hullR, cabinR, flagR, mastR, hullP, cabinP, flagP, mastP, readout


ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()

"""MPU-6050 attitude: driver, complementary filter, shared `att` state, loop."""
import math
import threading
import time

from hardware.bus import get_bus

try:
    from adafruit_bus_device.i2c_device import I2CDevice
except Exception:
    I2CDevice = None

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
    if get_bus() is None or I2CDevice is None:
        raise RuntimeError("no I2C bus / device library")
    sensor = MPU6050(get_bus(), address=0x68)
    IMU_AVAILABLE = True
except Exception as e:
    print(f"[imu] MPU6050 not found, attitude section disabled: {e}")
    IMU_AVAILABLE = False

TARGET_HZ = 104                    # loop target; matches the sensor's default ODR
PERIOD = 1.0 / TARGET_HZ

# ---------- Complementary filter config ----------
ALPHA = 0.98          # near 1.0: gyro short-term, accel long-term (drift kill)
ROLL_SIGN = 1         # set to -1 if the boat heels the wrong way
PITCH_SIGN = -1       # flipped: clone IMU reads bow/stern reversed
YAW_SIGN = 1

att = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "hz": 0.0, "ok": False,
       "ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}  # rad + raw
alock = threading.Lock()
imu_cmd = {"zero_yaw": False}     # set by the Zero-heading control; read in imu_loop

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
            if imu_cmd["zero_yaw"]:
                yaw = 0.0
                imu_cmd["zero_yaw"] = False
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



def zero_heading():
    """Action: zero the yaw integrator (the Zero-heading button)."""
    if not IMU_AVAILABLE:
        return {"ok": False, "error": "IMU offline"}
    with alock:
        imu_cmd["zero_yaw"] = True
    return {"ok": True, "message": "heading zeroed"}

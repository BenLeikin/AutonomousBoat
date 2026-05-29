"""Threaded LSM6DSO reader. Latest sample always available, no locks."""
import threading
import time
import board
import busio
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS


class IMU:
    def __init__(self, address=0x6A, rate_hz=100):
        self._period = 1.0 / rate_hz
        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = LSM6DS(i2c, address=address)

        # Shared state. Single writer (thread), many readers. CPython
        # dict assignment is atomic, so no lock needed for "latest wins".
        self._latest = {
            "t": 0.0,
            "accel": (0.0, 0.0, 0.0),
            "gyro": (0.0, 0.0, 0.0),
            "count": 0,
        }
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest(self):
        # Return reference to the current dict. Reader gets a consistent
        # snapshot because writer replaces the whole dict atomically.
        return self._latest

    def _loop(self):
        next_t = time.monotonic()
        count = 0
        while self._running:
            try:
                accel = self._sensor.acceleration
                gyro = self._sensor.gyro
                count += 1
                self._latest = {
                    "t": time.monotonic(),
                    "accel": accel,
                    "gyro": gyro,
                    "count": count,
                }
            except Exception as e:
                # I2C glitches happen. Skip the sample, don't crash.
                print(f"[imu] read error: {e}")

            next_t += self._period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Falling behind. Reset cadence rather than spiraling.
                next_t = time.monotonic()

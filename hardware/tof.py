"""VL53L1X forward time-of-flight: shared `tof` state and loop.

The sensor lives on its OWN hardware I2C bus (no pin sharing with the IMU):
    /boot/firmware/config.txt:  dtoverlay=i2c5,pins_12_13
    wiring: VIN->3V3 pin 17, GND->pin 34, SDA->GPIO12 pin 32, SCL->GPIO13 pin 33
    verify after reboot: i2cdetect -y 5  ->  0x29
Same ExtendedI2C library the INA219 already uses for its bus. Optional like every
other sensor: absent hardware/overlay/library just disables the range gate and
the controller behaves exactly as before."""
import threading
import time

TOF_BUS = 5                        # /dev/i2c-5: dtoverlay=i2c5,pins_12_13 (GPIO12/13)
TOF_BOW_OFFSET_M = 0.127           # the bow sits 5 in ahead of the sensor. Subtracted
                                   # at the source, so range everywhere (controller
                                   # gate, telemetry tof_m, UI) means distance from
                                   # the BOW to the obstacle - the number that matters.

try:
    import adafruit_vl53l1x as _vl53
    from adafruit_extended_bus import ExtendedI2C as _TofI2C
    TOF_LIB_AVAILABLE = True
except Exception as e:
    _vl53 = None
    _TofI2C = None
    TOF_LIB_AVAILABLE = False
    print(f"[tof] VL53L1X library unavailable, range gate disabled: {e} "
          "(pip install adafruit-circuitpython-vl53l1x --break-system-packages)")

# ---------- VL53L1X forward range shared state ----------
tof = {"connected": False, "range_m": None, "last_data": 0.0}
tlock = threading.Lock()

def tof_loop():
    # Reconnecting reader for the forward time-of-flight. ~10 Hz; on any I2C error
    # it marks the sensor offline (so the controller's range gate disengages rather
    # than acting on stale data) and retries.
    while True:
        try:
            dev = _vl53.VL53L1X(_TofI2C(TOF_BUS))
            dev.distance_mode = 2          # long-range mode, up to ~4 m
            dev.timing_budget = 100        # ms; favors accuracy over rate
            dev.start_ranging()
        except Exception as e:
            print(f"[tof] cannot open VL53L1X on /dev/i2c-%d: {e}" % TOF_BUS)
            with tlock:
                tof["connected"] = False
                tof["range_m"] = None
            time.sleep(3.0)
            continue
        print("[tof] reading VL53L1X (long-range mode)")
        try:
            while True:
                if dev.data_ready:
                    cm = dev.distance          # cm, or None on an invalid reading
                    dev.clear_interrupt()
                    with tlock:
                        tof["connected"] = True
                        # None (out of range / no return) is reported as None: the
                        # gate treats it as "nothing within reach", which is correct
                        # over open water.
                        tof["range_m"] = (max(cm / 100.0 - TOF_BOW_OFFSET_M, 0.0)
                                          if cm is not None else None)
                        tof["last_data"] = time.monotonic()
                time.sleep(0.05)
        except Exception as e:
            print(f"[tof] read error: {e}")
            try:
                dev.stop_ranging()
            except Exception:
                pass
            with tlock:
                tof["connected"] = False
                tof["range_m"] = None
            time.sleep(2.0)


def fresh_range(max_age_s=1.0):
    """Latest range in meters, or None if the sensor is offline, the reading is
    out of range, or the data is stale (a wedged reader must not leave a frozen
    'clear' on the gate)."""
    with tlock:
        if not tof["connected"]:
            return None
        if time.monotonic() - tof["last_data"] >= max_age_s:
            return None
        return tof["range_m"]

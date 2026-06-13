"""Shared hardware I2C bus (bus 1, GPIO2/3). Currently the MPU-6050 lives here;
the VL53L1X has its own bus (see hardware/tof.py). On a box without the
libraries or the bus, get_bus() returns None and dependents report offline.
"""
_bus = None
try:
    import board
    import busio
    _bus = busio.I2C(board.SCL, board.SDA)
except Exception as e:
    print(f"[i2c] hardware I2C bus unavailable: {e}")


def get_bus():
    return _bus

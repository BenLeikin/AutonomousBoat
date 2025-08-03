# imu.py
# Module 9: IMU interface for Bosch LSM6DSOX (Modulin ABX00101)

import time
import math
from smbus2 import SMBus

# I2C address and registers
IMU_ADDR          = 0x6A
WHO_AM_I          = 0x0F
WHO_AM_I_EXPECTED = 0x6C

CTRL1_XL = 0x10  # accel: ODR_XL[7:4], FS_XL[3:2]
CTRL2_G  = 0x11  # gyro:  ODR_G[7:4],  FS_G[3:2]

OUTX_L_G  = 0x22  # gyro X low byte
OUTX_L_XL = 0x28  # accel X low byte

# Sensitivities
ACCEL_SENSITIVITY = 0.061e-3 * 9.80665  # m/s² per LSB
GYRO_SENSITIVITY  = 4.375e-3            # °/s per LSB

# Gravity constant
STANDARD_GRAVITY = 9.80665  # m/s²


def twos_complement(val, bits=16):
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def read_raw(bus, reg_addr):
    lo, hi = bus.read_i2c_block_data(IMU_ADDR, reg_addr, 2)
    raw = (hi << 8) | lo
    return twos_complement(raw)


def read_accel_raw(bus):
    return (
        read_raw(bus, OUTX_L_XL),
        read_raw(bus, OUTX_L_XL + 2),
        read_raw(bus, OUTX_L_XL + 4),
    )


def read_gyro_raw(bus):
    return (
        read_raw(bus, OUTX_L_G),
        read_raw(bus, OUTX_L_G + 2),
        read_raw(bus, OUTX_L_G + 4),
    )

class IMU:
    """
    IMU wraps Bosch LSM6DSOX (ABX00101) over I2C, handles init, calibration, and readings.
    """
    def __init__(self, bus_id=1, calibrate_duration=5.0, calibrate_delay=0.01):
        self.bus = SMBus(bus_id)
        # Verify WHOAMI
        whoami = self.bus.read_byte_data(IMU_ADDR, WHO_AM_I)
        if whoami != WHO_AM_I_EXPECTED:
            raise RuntimeError(f"IMU WHO_AM_I mismatch: 0x{whoami:02X}")
        # Configure sensor: 104 Hz accel/gyro
        self.bus.write_byte_data(IMU_ADDR, CTRL1_XL, 0x40)
        self.bus.write_byte_data(IMU_ADDR, CTRL2_G,  0x40)
        time.sleep(0.1)
        # Calibrate biases
        self._calibrate(calibrate_duration, calibrate_delay)

    def _calibrate(self, duration, delay):
        sums = {'ax':0,'ay':0,'az':0,'gx':0,'gy':0,'gz':0}
        count = 0
        start = time.time()
        while time.time() - start < duration:
            ax, ay, az = read_accel_raw(self.bus)
            gx, gy, gz = read_gyro_raw(self.bus)
            sums['ax'] += ax; sums['ay'] += ay; sums['az'] += az
            sums['gx'] += gx; sums['gy'] += gy; sums['gz'] += gz
            count += 1
            time.sleep(delay)
        # Store raw biases
        self.bias = {k: v/count for k, v in sums.items()}
        # Compute orientation offset
        ax0 = self.bias['ax'] * ACCEL_SENSITIVITY
        ay0 = self.bias['ay'] * ACCEL_SENSITIVITY
        az0 = self.bias['az'] * ACCEL_SENSITIVITY
        self.pitch_offset = math.degrees(math.atan2(ay0, math.sqrt(ax0*ax0 + az0*az0)))
        self.roll_offset  = math.degrees(math.atan2(-ax0, az0))

    def read(self):
        """
        Return a dict with calibrated accel (m/s²), gyro (°/s), pitch and roll (°).
        """
        # Raw
        ax_r, ay_r, az_r = read_accel_raw(self.bus)
        gx_r, gy_r, gz_r = read_gyro_raw(self.bus)
        # Scale
        ax = ax_r * ACCEL_SENSITIVITY
        ay = ay_r * ACCEL_SENSITIVITY
        az = az_r * ACCEL_SENSITIVITY
        gx = (gx_r - self.bias['gx']) * GYRO_SENSITIVITY
        gy = (gy_r - self.bias['gy']) * GYRO_SENSITIVITY
        gz = (gz_r - self.bias['gz']) * GYRO_SENSITIVITY
        # Orientation
        raw_pitch = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        raw_roll  = math.degrees(math.atan2(-ax, az))
        pitch = raw_pitch - self.pitch_offset
        roll  = raw_roll  - self.roll_offset
        return {'ax': ax, 'ay': ay, 'az': az,
                'gx': gx, 'gy': gy, 'gz': gz,
                'pitch': pitch, 'roll': roll}

    def close(self):
        self.bus.close()

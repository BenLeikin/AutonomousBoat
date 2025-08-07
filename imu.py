#!/usr/bin/env python3
"""
IMU interface for Bosch LSM6DSOX (Modulin ABX00101)

Implements:
1. Config-driven parameters (I2C bus/address, register values, calibration settings)
2. Structured logging via Python's logging module
3. Bulk I2C block reads for accel/gyro data
4. Unit conversion during calibration and readings
5. Type hints and docstrings for clarity
6. Context-manager API for automatic resource cleanup
7. Dynamic register configuration from config.yaml
8. I2C error recovery with retries
"""
import os
import yaml
import logging
import time
import math
from smbus2 import SMBus
from typing import Dict, Any, Optional

# Load IMU configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f)
    _cfg: Dict[str, Any] = _full_cfg.get('imu', {}) or {}
except Exception:
    _cfg = {}

# Set up module logger
logger = logging.getLogger(__name__)

# I2C and register constants
IMU_ADDR = _cfg.get('address', 0x6A)
WHO_AM_I_REG = 0x0F
WHO_AM_I_EXPECTED = _cfg.get('who_am_i_expected', 0x6C)
CTRL1_XL_ADDR = 0x10
CTRL2_G_ADDR  = 0x11
OUTX_L_G_ADDR  = 0x22
OUTX_L_XL_ADDR = 0x28

# Block read settings
def read_block(bus: SMBus, reg: int, length: int) -> bytes:
    """
    Read a block of bytes from the IMU with retries on failure.
    """
    retries = int(_cfg.get('retries', 3))
    delay = float(_cfg.get('retry_delay', 0.01))
    for attempt in range(1, retries+1):
        try:
            data = bus.read_i2c_block_data(IMU_ADDR, reg, length)
            return data
        except Exception as e:
            logger.warning("I2C read error reg=0x%02X attempt=%d/%d: %s", reg, attempt, retries, e)
            time.sleep(delay)
    raise IOError(f"Failed to read {length} bytes from reg 0x{reg:02X}")


def twos_complement(val: int, bits: int =16) -> int:
    """Convert raw integer to signed via two's complement."""
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


class IMU:
    """
    Context-manager for LSM6DSOX IMU over I2C.

    Usage:
        with IMU() as imu:
            data = imu.read()
            # data: {'ax':..., 'ay':..., 'az':..., 'gx':..., 'gy':..., 'gz':..., 'pitch':..., 'roll':...}
    """
    def __init__(self) -> None:
        # Configuration parameters
        self.bus_id = int(_cfg.get('bus_id', 1))
        self.ctrl1_xl = int(_cfg.get('ctrl1_xl_value', 0x40))
        self.ctrl2_g  = int(_cfg.get('ctrl2_g_value',  0x40))
        cal = _cfg.get('calibrate', {})
        self.cal_dur   = float(cal.get('duration', 5.0))
        self.cal_delay = float(cal.get('delay', 0.01))
        # Sensitivities
        sens = _cfg.get('sensitivity', {})
        self.acc_sens = float(sens.get('accel', 0.061e-3 * 9.80665))
        self.gyro_sens= float(sens.get('gyro', 4.375e-3))
        # Initialize bus
        self.bus = SMBus(self.bus_id)

    def __enter__(self) -> 'IMU':
        self._verify()
        self._configure()
        self._calibrate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _verify(self) -> None:
        """Verify IMU identity via WHO_AM_I register."""
        who = self.bus.read_byte_data(IMU_ADDR, WHO_AM_I_REG)
        if who != WHO_AM_I_EXPECTED:
            logger.error("IMU WHO_AM_I mismatch: got 0x%02X, expected 0x%02X", who, WHO_AM_I_EXPECTED)
            raise RuntimeError("IMU not detected or wrong device")
        logger.info("IMU detected: WHO_AM_I=0x%02X", who)

    def _configure(self) -> None:
        """Configure accelerometer and gyroscope control registers."""
        self.bus.write_byte_data(IMU_ADDR, CTRL1_XL_ADDR, self.ctrl1_xl)
        self.bus.write_byte_data(IMU_ADDR, CTRL2_G_ADDR,  self.ctrl2_g)
        logger.info("Configured IMU: CTRL1_XL=0x%02X, CTRL2_G=0x%02X", self.ctrl1_xl, self.ctrl2_g)
        time.sleep(0.1)

    def _calibrate(self) -> None:
        """Compute biases over a time window in physical units."""
        samples = int(self.cal_dur / self.cal_delay)
        sums: Dict[str, float] = dict.fromkeys(['ax','ay','az','gx','gy','gz','pitch','roll'], 0.0)
        for _ in range(samples):
            d = self._read_raw_physical()
            for k in sums:
                sums[k] += d.get(k, 0.0)
            time.sleep(self.cal_delay)
        self.bias = {k: sums[k]/samples for k in sums}
        # Offsets for orientation
        self.pitch_offset = self.bias['pitch']
        self.roll_offset  = self.bias['roll']
        logger.info("IMU calibration complete: biases=%s", self.bias)

    def _read_raw(self) -> Dict[str, int]:
        """Read raw accelerometer and gyroscope counts via block reads."""
        # Accelerometer: 6 bytes
        arb = read_block(self.bus, OUTX_L_XL_ADDR, 6)
        ax_r = twos_complement(arb[1] << 8 | arb[0])
        ay_r = twos_complement(arb[3] << 8 | arb[2])
        az_r = twos_complement(arb[5] << 8 | arb[4])
        # Gyroscope: 6 bytes
        grb = read_block(self.bus, OUTX_L_G_ADDR, 6)
        gx_r = twos_complement(grb[1] << 8 | grb[0])
        gy_r = twos_complement(grb[3] << 8 | grb[2])
        gz_r = twos_complement(grb[5] << 8 | grb[4])
        return {'ax':ax_r,'ay':ay_r,'az':az_r,'gx':gx_r,'gy':gy_r,'gz':gz_r}

    def _read_raw_physical(self) -> Dict[str, float]:
        """Convert raw counts to physical units and compute orientation."""
        r = self._read_raw()
        ax = r['ax'] * self.acc_sens
        ay = r['ay'] * self.acc_sens
        az = r['az'] * self.acc_sens
        gx = r['gx'] * self.gyro_sens
        gy = r['gy'] * self.gyro_sens
        gz = r['gz'] * self.gyro_sens
        pitch = math.degrees(math.atan2(ay, math.sqrt(ax*ax+az*az)))
        roll  = math.degrees(math.atan2(-ax, az))
        return {'ax':ax,'ay':ay,'az':az,'gx':gx,'gy':gy,'gz':gz,'pitch':pitch,'roll':roll}

    def read(self) -> Dict[str, float]:
        """Return calibrated sensor readings and orientation."""
        phys = self._read_raw_physical()
        # Subtract biases
        result = {k: phys[k] - self.bias.get(k, 0.0) for k in phys}
        # Adjust orientation offsets
        result['pitch'] = result['pitch'] - self.pitch_offset
        result['roll']  = result['roll']  - self.roll_offset
        return result

    def close(self) -> None:
        """Close the I2C bus connection."""
        self.bus.close()

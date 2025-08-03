#!/usr/bin/env python3
"""
imu_manual_test_plus.py

Manual I²C driver for Bosch LSM6DSOX (Modulin Movement ABX00101)
• Connect SDA → GPIO2, SCL → GPIO3, VCC → 3.3 V, GND → GND
• Enable I²C in raspi-config
• Requires: pip3 install smbus2
"""

import time
import math
import csv
import sys
import logging
import argparse
from smbus2 import SMBus

# I²C address and registers
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

# Constants
STANDARD_GRAVITY = 9.80665  # m/s²


def twos_complement(val, bits=16):
    """Convert unsigned int to signed (two's complement)."""
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def read_raw(bus, reg_addr):
    lo, hi = bus.read_i2c_block_data(IMU_ADDR, reg_addr, 2)
    raw = (hi << 8) | lo
    return twos_complement(raw)


def read_accel(bus):
    return (
        read_raw(bus, OUTX_L_XL),
        read_raw(bus, OUTX_L_XL + 2),
        read_raw(bus, OUTX_L_XL + 4),
    )


def read_gyro(bus):
    return (
        read_raw(bus, OUTX_L_G),
        read_raw(bus, OUTX_L_G + 2),
        read_raw(bus, OUTX_L_G + 4),
    )


def calibrate(bus, duration=10.0, delay=0.01):
    """Collect readings for `duration` seconds while stationary to compute raw biases."""
    logging.info(f"Calibrating for {duration:.1f} seconds (stationary)...")
    sums = {'ax':0, 'ay':0, 'az':0, 'gx':0, 'gy':0, 'gz':0}
    count = 0
    start = time.time()
    while time.time() - start < duration:
        ax_r, ay_r, az_r = read_accel(bus)
        gx_r, gy_r, gz_r = read_gyro(bus)
        sums['ax'] += ax_r; sums['ay'] += ay_r; sums['az'] += az_r
        sums['gx'] += gx_r; sums['gy'] += gy_r; sums['gz'] += gz_r
        count += 1
        time.sleep(delay)
    return {k: v/count for k, v in sums.items()}


def parse_args():
    p = argparse.ArgumentParser(description="LSM6DSOX test with sanity checks and orientation offset")
    p.add_argument("-i","--interval", type=float, default=0.5, help="Seconds between readings")
    p.add_argument("-o","--csv",      type=str,   default=None, help="Path to CSV log file")
    p.add_argument("--accel-tol",      type=float, default=0.5, help="Tolerance for raw accel vs 1g (m/s²)")
    p.add_argument("--residual-tol",   type=float, default=0.1, help="Tolerance for residual accel (m/s²)")
    p.add_argument("--gyro-tol",       type=float, default=0.5, help="Tolerance for gyro drift (°/s)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # I2C init
    try:
        bus = SMBus(1)
    except Exception as e:
        logging.error(f"Failed to open I²C bus: {e}")
        sys.exit(1)

    # Verify sensor
    whoami = bus.read_byte_data(IMU_ADDR, WHO_AM_I)
    if whoami != WHO_AM_I_EXPECTED:
        logging.error(f"Unexpected WHO_AM_I: 0x{whoami:02X}")
        sys.exit(1)
    logging.info(f"WHO_AM_I OK: 0x{whoami:02X}")

    # Sensor config
    bus.write_byte_data(IMU_ADDR, CTRL1_XL, 0x40)
    bus.write_byte_data(IMU_ADDR, CTRL2_G,  0x40)
    time.sleep(0.1)

    # Calibration
    biases = calibrate(bus, duration=10.0)
    logging.info(f"Calibration biases (raw counts): {biases}")

    # Compute initial orientation offsets
    ax0 = biases['ax'] * ACCEL_SENSITIVITY
    ay0 = biases['ay'] * ACCEL_SENSITIVITY
    az0 = biases['az'] * ACCEL_SENSITIVITY
    pitch_offset = math.degrees(math.atan2(ay0, math.sqrt(ax0*ax0 + az0*az0)))
    roll_offset  = math.degrees(math.atan2(-ax0, az0))
    logging.info(f"Initial pitch offset: {pitch_offset:+.2f}°, roll offset: {roll_offset:+.2f}°")

    # CSV setup
    csv_file = None; writer = None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["time","ax","ay","az","gx","gy","gz",
                         "accel_raw_mag","residual_accel","gyro_drift","pitch","roll"] )
        logging.info(f"Logging to {args.csv}")

    logging.info("Starting readings...")
    try:
        while True:
            # Raw data
            ax_r, ay_r, az_r = read_accel(bus)
            gx_r, gy_r, gz_r = read_gyro(bus)

            # Convert raw to units
            axg = ax_r * ACCEL_SENSITIVITY
            ayg = ay_r * ACCEL_SENSITIVITY
            azg = az_r * ACCEL_SENSITIVITY
            gyro_x = (gx_r - biases['gx']) * GYRO_SENSITIVITY
            gyro_y = (gy_r - biases['gy']) * GYRO_SENSITIVITY
            gyro_z = (gz_r - biases['gz']) * GYRO_SENSITIVITY

            # Magnitudes
            accel_raw_mag      = math.sqrt(axg*axg + ayg*ayg + azg*azg)
            accel_residual_mag = math.sqrt(((ax_r-biases['ax'])*ACCEL_SENSITIVITY)**2 +
                                          ((ay_r-biases['ay'])*ACCEL_SENSITIVITY)**2 +
                                          ((az_r-biases['az'])*ACCEL_SENSITIVITY)**2)
            gyro_drift         = math.sqrt(gyro_x*gyro_x + gyro_y*gyro_y + gyro_z*gyro_z)

            # Raw tilt angles
            raw_pitch = math.degrees(math.atan2(ayg, math.sqrt(axg*axg + azg*azg)))
            raw_roll  = math.degrees(math.atan2(-axg, azg))

            # Apply orientation offsets
            pitch = raw_pitch - pitch_offset
            roll  = raw_roll  - roll_offset

            # Sanity checks
            if abs(accel_raw_mag - STANDARD_GRAVITY) > args.accel_tol:
                logging.warning(f"Raw accel mag off 1g: {accel_raw_mag:.3f} m/s²")
            if accel_residual_mag > args.residual_tol:
                logging.warning(f"Residual accel > tol: {accel_residual_mag:.3f} m/s²")
            if gyro_drift > args.gyro_tol:
                logging.warning(f"Gyro drift high: {gyro_drift:.3f} °/s")

            # Logging
            logging.info(f"Accel Raw={accel_raw_mag:.3f} Res={accel_residual_mag:.3f} | Gyro drift={gyro_drift:.3f}")
            logging.info(f"Accel=({axg:.3f},{ayg:.3f},{azg:.3f}) m/s² | Gyro=({gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}) °/s")
            logging.info(f"Pitch={pitch:+.2f}°, Roll={roll:+.2f}°")

            # CSV log
            if writer:
                writer.writerow([time.time(), axg, ayg, azg, gyro_x, gyro_y, gyro_z,
                                  accel_raw_mag, accel_residual_mag, gyro_drift, pitch, roll])
                csv_file.flush()

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("User exit")
    finally:
        bus.close()
        if csv_file:
            csv_file.close()

if __name__ == '__main__':
    main()

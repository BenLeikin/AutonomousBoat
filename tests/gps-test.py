#!/usr/bin/env python3
"""
gps-test.py

Read NMEA from a GPS on UART5 (/dev/ttyAMA5, default 9600 baud)
and read heading from a QMC5883L compass on I²C bus 1 at 0x0D.
Prints GGA/RMC fixes and compass heading in degrees.
"""

import sys
import time
import math
import argparse

import serial
import pynmea2
from smbus2 import SMBus

# I²C settings for QMC5883L
I2C_BUS   = 1
QMC_ADDR  = 0x0D
REG_CTRL1 = 0x09
REG_X_LSB = 0x00
DATA_LEN  = 6

def init_qmc(bus):
    # OSR=512, Range=2G, ODR=50Hz, Continuous mode
    bus.write_byte_data(QMC_ADDR, REG_CTRL1, 0x1D)

def read_qmc(bus):
    data = bus.read_i2c_block_data(QMC_ADDR, REG_X_LSB, DATA_LEN)
    def twos(lsb, msb):
        v = (msb << 8) | lsb
        return v - 0x10000 if v & 0x8000 else v
    x = twos(data[0], data[1])
    y = twos(data[2], data[3])
    # heading from X,Y
    heading = math.degrees(math.atan2(y, x))
    return heading + 360.0 if heading < 0 else heading

def main():
    parser = argparse.ArgumentParser(
        description="GPS on UART5 (/dev/ttyAMA5) + QMC5883L compass on I²C@0x0D"
    )
    parser.add_argument(
        "-p", "--port",
        default="/dev/serial0",
        help="GPS serial port (default: /dev/ttyAMA5)"
    )
    parser.add_argument(
        "-b", "--baud",
        type=int,
        default=9600,
        help="GPS baud rate (default: 9600; adjust if your module runs at 4800 or 38400)"
    )
    args = parser.parse_args()

    # Open GPS serial on UART5
    try:
        gps_ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    # Open I²C bus and init QMC5883L
    try:
        bus = SMBus(I2C_BUS)
        init_qmc(bus)
    except Exception as e:
        print(f"I²C/QMC init failed: {e}", file=sys.stderr)
        gps_ser.close()
        sys.exit(1)

    print(f"GPS → {args.port}@{args.baud}, Compass → I2C/{I2C_BUS}@0x{QMC_ADDR:02X}")
    print("Press Ctrl-C to exit.\n")

    try:
        while True:
            # GPS read
            raw = gps_ser.readline().decode("ascii", errors="ignore").strip()
            gps_str = ""
            if raw:
                try:
                    msg = pynmea2.parse(raw)
                except pynmea2.ParseError:
                    msg = None
                if isinstance(msg, pynmea2.types.talker.GGA):
                    gps_str = (
                        f"[GGA] UTC {msg.timestamp}  "
                        f"Lat {msg.latitude}{msg.lat_dir}  "
                        f"Lon {msg.longitude}{msg.lon_dir}  "
                        f"Fix {msg.gps_qual}  Sats {msg.num_sats}  "
                        f"Alt {msg.altitude}{msg.altitude_units}"
                    )
                elif isinstance(msg, pynmea2.types.talker.RMC):
                    gps_str = (
                        f"[RMC] UTC {msg.timestamp}  Status {msg.status}  "
                        f"Lat {msg.latitude}{msg.lat_dir}  "
                        f"Lon {msg.longitude}{msg.lon_dir}  "
                        f"Speed {msg.spd_over_grnd}kn  Course {msg.true_course}°"
                    )

            # Compass read
            try:
                heading = read_qmc(bus)
                comp_str = f"Heading: {heading:.1f}°"
            except Exception:
                comp_str = "Heading: ERROR"

            # Print if either has data
            if gps_str or comp_str:
                print(gps_str.ljust(80), comp_str)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        gps_ser.close()
        bus.close()

if __name__ == "__main__":
    main()

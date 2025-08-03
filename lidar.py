# lidar.py
# Module 1: TF-Luna LiDAR serial reader with median filtering

import time
import collections
import numpy as np
import serial


def find_frame(ser):
    """
    Read bytes from serial until a valid TF-Luna frame header (0x59 0x59) is found,
    then return the next 7 bytes of the frame.
    """
    while True:
        b1 = ser.read(1)
        if not b1 or b1[0] != 0x59:
            continue
        b2 = ser.read(1)
        if b2 and b2[0] == 0x59:
            return ser.read(7)


def parse_frame(raw):
    """
    Parse a 9-byte TF-Luna frame (including header) into a distance in meters.
    Returns None on checksum failure or invalid length.
    """
    if raw is None or len(raw) != 7:
        return None
    data = bytes([0x59, 0x59]) + raw
    # Compute checksum over first 8 bytes
    checksum = sum(data[0:8]) & 0xFF
    if checksum != data[8]:
        return None
    # Bytes 2-3 contain low/high bytes of distance in cm
    dist_cm = data[2] | (data[3] << 8)
    return dist_cm / 100.0  # convert to meters


class LidarReader:
    """
    LidarReader wraps the TF-Luna serial interface and provides a median-filtered
    distance reading on each call to read().
    """
    def __init__(self, port='/dev/serial0', baud=115200, timeout=1, window=5):
        """
        Initialize serial port and median buffer.
        port: serial device path
        baud: baud rate, default 115200
        timeout: serial read timeout in seconds
        window: number of samples for median filter
        """
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
            # Allow sensor to boot up/stabilize
            time.sleep(2)
            self.buf = collections.deque(maxlen=window)
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open LiDAR port {port}: {e}")

    def read(self):
        """
        Read one TF-Luna frame, apply median filter, and return distance in meters.
        Returns None until enough samples are collected.
        """
        raw = find_frame(self.ser)
        d = parse_frame(raw)
        if d is not None:
            self.buf.append(d)
        if len(self.buf) == self.buf.maxlen:
            # Return median of buffer
            return float(np.median(self.buf))
        return None

    def close(self):
        """
        Close the serial port.
        """
        try:
            self.ser.close()
        except Exception:
            pass

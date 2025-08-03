#!/usr/bin/env python3
"""
TF-Luna + Picamera2 fusion with computer vision for enhanced obstacle detection and logging
- Reads TF-Luna single-point LiDAR via serial
- Captures 720p frames from Pi Camera via Picamera2 API
- Applies median filter to LiDAR readings (window=5)
- Uses background subtraction & contour detection to find visual obstacles
- Overlays LiDAR center-box + range annotation and visual obstacle bounding boxes
- Records annotated video for a configurable-duration (default 30s) trial into timestamped USB directory
- Configurable serial port, output directory, and duration via CLI
- Logs LiDAR data to CSV and console
- Runs headless (no GUI), avoids display errors
"""
import os
import time
import collections
import numpy as np
import cv2
import serial
import argparse
from datetime import datetime
from picamera2 import Picamera2
import math

# === TF-Luna reader: serial + median filter ===
def find_frame(ser):
    while True:
        b1 = ser.read(1)
        if not b1 or b1[0] != 0x59:
            continue
        b2 = ser.read(1)
        if b2 and b2[0] == 0x59:
            return ser.read(7)

def parse_frame(raw):
    if raw is None or len(raw) != 7:
        return None
    data = bytes([0x59, 0x59]) + raw
    checksum = sum(data[0:8]) & 0xFF
    if checksum != data[8]:
        return None
    dist_cm = data[2] | (data[3] << 8)
    return dist_cm / 100.0  # meters

class LidarReader:
    def __init__(self, port, baud=115200, timeout=1, window=5):
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
            time.sleep(2)
            self.buf = collections.deque(maxlen=window)
        except Exception as e:
            raise RuntimeError(f"LiDAR port error ({port}): {e}")

    def read(self):
        try:
            raw = find_frame(self.ser)
        except Exception as e:
            raise RuntimeError(f"LiDAR read error: {e}")
        d = parse_frame(raw)
        if d is not None:
            self.buf.append(d)
        if len(self.buf) == self.buf.maxlen:
            return float(np.median(self.buf))
        return None

    def close(self):
        try:
            self.ser.close()
        except:
            pass

class CameraReader:
    def __init__(self, resolution=(1280, 720), warmup=2):
        print("Initializing Picamera2 with AWB...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"AwbEnable": True}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(warmup)

    def read(self):
        frame = self.picam2.capture_array()
        if frame is None:
            raise RuntimeError("Camera read error: no frame returned from Picamera2")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self.picam2.stop()
        self.picam2.close()

class VisionProcessor:
    def __init__(self, min_area=5000):
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        mask = self.backsub.apply(blur)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            filtered.append(approx)
        return filtered

    def annotate(self, frame, contours):
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
        return frame

class FusionProcessor:
    def __init__(self, threshold=0.25): self.threshold = threshold
    def annotate(self, frame, distance):
        h, w = frame.shape[:2]
        bw, bh = int(w*0.3), int(h*0.3)
        x1, y1 = (w-bw)//2, (h-bh)//2
        x2, y2 = x1+bw, y1+bh
        color = (0,0,255) if distance is not None and distance<self.threshold else (0,255,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        if distance is not None:
            cv2.putText(frame, f"{distance:.2f} m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# === Vector Field Histogram Navigator ===
class VFHNavigator:
    def __init__(self, sectors=72, fov_deg=90, threshold=0.25, frame_width=1280):
        self.sectors = sectors
        self.threshold = threshold
        self.fov = math.radians(fov_deg)
        self.angles = np.linspace(-self.fov/2, self.fov/2, sectors)
        self.frame_w = frame_width

    def compute_histogram(self, contours, lidar_dist):
        hist = np.zeros(self.sectors, dtype=bool)
        center_idx = self.sectors // 2
        if lidar_dist is not None and lidar_dist < self.threshold:
            hist[center_idx] = True
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            angle = (cx - self.frame_w/2)/(self.frame_w/2)*(self.fov/2)
            idx = np.argmin(np.abs(self.angles - angle))
            hist[idx] = True
        return hist

    def find_best_gap(self, hist):
        free = ~hist
        best_start, best_len = 0, 0
        curr_start, curr_len = None, 0
        for i, v in enumerate(free):
            if v:
                if curr_start is None:
                    curr_start, curr_len = i, 1
                else:
                    curr_len += 1
            else:
                if curr_start is not None and curr_len > best_len:
                    best_start, best_len = curr_start, curr_len
                curr_start, curr_len = None, 0
        if curr_start is not None and curr_len > best_len:
            best_start, best_len = curr_start, curr_len
        if best_len == 0:
            return 0.0
        mid = best_start + best_len // 2
        return self.angles[mid]

def main():
    p = argparse.ArgumentParser(description="LiDAR+Vision fusion logging (headless)")
    p.add_argument("--lidar-port", default="/dev/ttyS0")
    p.add_argument("--output-dir", default="/mnt/usb/lidar_fusion_logs")
    p.add_argument("--duration", type=int, default=30, help="Trial duration in seconds")
    args = p.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(args.output_dir, ts)
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, f'fusion_{ts}.mp4')

    cam = CameraReader()
    lidar = LidarReader(port=args.lidar_port)
    vision = VisionProcessor(min_area=5000)
    fuse = FusionProcessor(threshold=0.25)
    # VFH navigator
    vfh = VFHNavigator(sectors=72, fov_deg=90, threshold=0.25, frame_width=1280)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))
    if not out.isOpened():
        print("Error: video writer failed to open.")
        cam.release(); lidar.close(); return

    # LiDAR CSV log
    log_path = os.path.join(session_dir, 'lidar_log.csv')
    try:
        log_file = open(log_path, 'w')
        log_file.write('timestamp,distance_m\n')
    except Exception as e:
        print(f"Warning: could not open LiDAR log file: {e}")
        log_file = None

    start = time.time()
    while time.time() - start < args.duration:
        try:
            # 1. Read sensors
            frame = cam.read()
            dist = lidar.read()

            # 2. Log LiDAR
            ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            if log_file:
                log_file.write(f"{ts_str},{dist if dist is not None else ''}
")
                log_file.flush()
            print(f"[{ts_str}] Obstacles detected: {len(contours)}")

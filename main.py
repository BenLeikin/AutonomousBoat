#!/usr/bin/env python3
"""
Autonomous Boat Navigation Driver
- Integrates LidarReader, CameraReader, VisionProcessor, FusionProcessor, VFHNavigator
- Logs LiDAR distances to CSV
- Reads IMU (pitch/roll) and logs to CSV
- Annotates frames via Annotator
- Controls motors via tested motor_control
- Console output prints only motor action
"""
import os
import time
import math
import cv2
import argparse
from datetime import datetime

from lidar import LidarReader
from camera import CameraReader
from vision import VisionProcessor
from fusion import FusionProcessor
from vfh import VFHNavigator
from logger import Logger
from annotator import Annotator
import motor_control
from imu import IMU

def main():
    parser = argparse.ArgumentParser(description="Autonomous Boat Navigation Driver")
    parser.add_argument("--lidar-port",    default="/dev/serial0",               help="TF-Luna serial port")
    parser.add_argument("--duration",      type=int,   default=30,                help="Run duration in seconds")
    parser.add_argument("--out-dir",       default="/mnt/usb/lidar_fusion_logs", help="Output directory")
    parser.add_argument("--angle-thresh",  type=float, default=15.0,              help="Degrees within which to drive straight")
    parser.add_argument("--imu-calib-duration", type=float, default=5.0,          help="IMU calibration duration in seconds")
    args = parser.parse_args()

    # Session setup
    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir    = os.path.join(args.out_dir, ts)
    os.makedirs(session_dir, exist_ok=True)
    video_path     = os.path.join(session_dir, f"nav_{ts}.mp4")
    lidar_log_path = os.path.join(session_dir, "lidar_log.csv")
    imu_log_path   = os.path.join(session_dir, "imu_log.csv")

    # Initialize IMU
    try:
        imu = IMU(bus_id=1, calibrate_duration=args.imu_calib_duration)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] IMU init failed: {e}")
        imu = None

    # Initialize other modules
    lidar  = LidarReader(port=args.lidar_port)
    cam    = CameraReader()
    vision = VisionProcessor()
    fuse   = FusionProcessor()
    vfh    = VFHNavigator(frame_width=1280)
    logger = Logger(lidar_log_path)

    # Setup IMU CSV log
    imu_log_file = None
    imu_writer   = None
    if imu:
        import csv
        imu_log_file = open(imu_log_path, 'w', newline='')
        imu_writer   = csv.writer(imu_log_file)
        imu_writer.writerow([
            "timestamp","pitch_deg","roll_deg",
            "ax","ay","az","gx","gy","gz"
        ])

    # === Warmup to measure loop throughput ===
    warmup_duration = 2.0  # seconds
    warmup_count    = 0
    warmup_start    = time.time()
    print(f"Warming up for {warmup_duration}s to measure frame rate...")
    while time.time() - warmup_start < warmup_duration:
        # replicate main loop workload minus writing
        cam.read()
        _ = lidar.read()
        cnts = vision.detect(cam.read())
        _ = vfh.compute_histogram(cnts, lidar.read())
        warmup_count += 1
    fps = warmup_count / warmup_duration
    print(f"Measured loop rate: {fps:.1f} FPS")

    # Video writer (use measured fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))
    if not out.isOpened():
        print("Error: video writer failed to open.")
        return

    start = time.time()
    try:
        while time.time() - start < args.duration:
            loop_start = time.time()
            now        = datetime.now().strftime('%H:%M:%S')
            # Sensor reads
            frame = cam.read()
            dist  = lidar.read()

            # Perception & planning
            contours = vision.detect(frame)
            hist     = vfh.compute_histogram(contours, dist)
            hdg_rad  = vfh.find_best_gap(hist)
            hdg_deg  = math.degrees(hdg_rad)

            # Motor action
            if abs(hdg_deg) <= args.angle_thresh:
                motor_control.both_forward(); action = "Forward"
            elif hdg_deg > 0:
                motor_control.pivot_right(); action = "Pivot Right"
            else:
                motor_control.pivot_left();  action = "Pivot Left"

            # Console output: action only
            print(f"[{now}] Action: {action}")

            # Log LiDAR
            logger.log_lidar(dist)

            # Read & log IMU
            imu_text = ""
            if imu:
                data  = imu.read()
                pitch = data['pitch']; roll = data['roll']
                imu_text = f"Pitch:{pitch:+.1f}° Roll:{roll:+.1f}°"
                if imu_writer:
                    imu_writer.writerow([
                        now, pitch, roll,
                        data['ax'], data['ay'], data['az'],
                        data['gx'], data['gy'], data['gz']
                    ])
                    imu_log_file.flush()

            # Annotate frame
            frame = vision.annotate(frame, contours)
            frame = fuse.annotate(frame, dist)
            frame = Annotator.draw_heading(frame, hdg_deg)
            frame = Annotator.draw_histogram(frame, hist)
            Annotator.draw_action(frame, action)
            Annotator.draw_text(frame, imu_text, position=(10,110))
            Annotator.draw_text(frame, f"Heading:{hdg_deg:.1f}°", position=(10,20))
            Annotator.draw_text(frame, now, position=(10, frame.shape[0]-10), scale=0.5)

            # Write video frame
            out.write(frame)

            # Throttle loop to measured FPS
            elapsed = time.time() - loop_start
            if elapsed < 1.0 / fps:
                time.sleep(1.0 / fps - elapsed)

    finally:
        # Cleanup
        motor_control.stop_all()
        motor_control.GPIO.cleanup()
        lidar.close()
        cam.release()
        out.release()
        logger.close()
        if imu:
            imu.close()
        if imu_log_file:
            imu_log_file.close()

if __name__ == '__main__':
    main()

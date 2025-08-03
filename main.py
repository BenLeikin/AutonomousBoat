# main.py
"""
Orchestrates all modules for autonomous boat navigation:
- LidarReader: TF-Luna range readings
- CameraReader: Picamera2 frame capture
- VisionProcessor: background-subtraction contours
- FusionProcessor: LiDAR overlay
- VFHNavigator: local gap-finding
- Logger: CSV and console logging
- Annotator: frame annotation utilities
"""
import os
import time
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


def main():
    parser = argparse.ArgumentParser(description="Autonomous Boat Navigation Driver")
    parser.add_argument("--lidar-port", default="/dev/serial0", help="TF-Luna serial port")
    parser.add_argument("--output-dir", default="/mnt/usb/lidar_fusion_logs", help="Directory for logs and video")
    parser.add_argument("--duration", type=int, default=30, help="Run duration in seconds")
    args = parser.parse_args()

    # Prepare session directory
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(args.output_dir, ts)
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, f"output_{ts}.mp4")
    log_path = os.path.join(session_dir, "lidar_log.csv")

    # Initialize modules
    lidar = LidarReader(port=args.lidar_port)
    cam = CameraReader()
    vision = VisionProcessor()
    fuse = FusionProcessor()
    vfh = VFHNavigator(frame_width=1280)
    logger = Logger(log_path)
    annotator = Annotator()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))
    if not out.isOpened():
        print("Error: video writer failed to open.")
        return

    # Main loop
    start = time.time()
    while time.time() - start < args.duration:
        # 1) Sensor reads
        frame = cam.read()
        dist = lidar.read()

        # 2) Perception
        contours = vision.detect(frame)

        # 3) Planning
        hist = vfh.compute_histogram(contours, dist)
        heading = vfh.find_best_gap(hist)

        # 4) Logging
        logger.log_lidar(dist)
        logger.log_console(len(contours))

        # 5) Annotation
        frame = vision.annotate(frame, contours)
        frame = fuse.annotate(frame, dist)
        frame = annotator.draw_heading(frame, heading)
        frame = annotator.draw_histogram(frame, hist)

        # 6) Output
        out.write(frame)

    # Cleanup
    lidar.close()
    cam.release()
    out.release()
    logger.close()

if __name__ == '__main__':
    main()

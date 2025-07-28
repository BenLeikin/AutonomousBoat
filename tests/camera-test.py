#!/usr/bin/env python3
"""
Simple script to capture a single image from the Raspberry Pi Camera
and save it to the '/mnt/usb/test-photos' directory with a timestamped filename.
"""
import os
import sys
import datetime
import time

from picamera2 import Picamera2
import cv2

# Directory where images will be stored
SAVE_DIR = '/mnt/usb/test-photos'


def main():
    # Discover connected camera devices
    cam_info_list = Picamera2.global_camera_info()
    if not cam_info_list:
        print("Error: No camera device detected. Please connect a camera.", file=sys.stderr)
        sys.exit(1)

    # Use the first detected camera's number
    cam_num = cam_info_list[0].get('Num', 0)

    # Ensure the save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize and configure the camera for 640x480 capture
    camera = Picamera2(camera_num=cam_num)
    config = camera.create_preview_configuration(main={'size': (640, 480)})
    camera.configure(config)

    # Start the camera and give it a moment to warm up
    camera.start()
    time.sleep(2)

    # Capture frame (RGB numpy array)
    frame = camera.capture_array()
    camera.stop()

    # Build timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"photo_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    # Convert RGB to BGR for OpenCV if needed
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write image to disk
    if not cv2.imwrite(filepath, frame):
        print(f"Error: Failed to write image to {filepath}", file=sys.stderr)
        sys.exit(2)

    # Output saved file path
    print(f"Captured image saved to: {filepath}")


if __name__ == '__main__':
    main()

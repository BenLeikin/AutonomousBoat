#!/usr/bin/env python3
"""
Simple script to record video from the Raspberry Pi camera
and save it to '/mnt/usb/test-videos' directory with a timestamped filename.
"""

import os
import sys
import datetime
import time

from picamera2 import Picamera2
import cv2

# Directory where videos will be stored
SAVE_DIR = '/mnt/usb/test-videos'
# Recording duration in seconds
DURATION = 10
# Video resolution (width, height)
RESOLUTION = (640, 480)
# Frames per second
FPS = 24


def main():
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Build filepath with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'video_{timestamp}.mp4'
    filepath = os.path.join(SAVE_DIR, filename)

    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={'size': RESOLUTION})
    picam2.configure(config)
    picam2.start()
    # Warm up camera
    time.sleep(2)

    # Set up video writer
    width, height = RESOLUTION
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))

    print(f'Recording video to {filepath} for {DURATION} seconds')
    start_time = time.time()
    try:
        while time.time() - start_time < DURATION:
            frame = picam2.capture_array()
            # Convert RGB (Picamera2) to BGR (OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
    except Exception as e:
        print(f'Error during recording: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up
        out.release()
        picam2.stop()
        picam2.close()

    print(f'Video saved to: {filepath}')


if __name__ == '__main__':
    main()

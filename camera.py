# camera.py
# Module 2: Picamera2 setup & capture

import time
import cv2
from picamera2 import Picamera2

class CameraReader:
    """
    CameraReader wraps the Pi Camera via Picamera2, captures 720p frames,
    applies auto-white-balance, and provides a BGR image for OpenCV.
    """
    def __init__(self, resolution=(1280, 720), warmup=2):
        """
        Initialize Picamera2 with given resolution and AWB enabled.
        resolution: tuple (width, height)
        warmup: seconds to discard initial frames
        """
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
        """
        Capture a single frame from the camera and return it as a BGR numpy array.
        Raises RuntimeError if no frame is returned.
        """
        frame = self.picam2.capture_array()
        if frame is None:
            raise RuntimeError("Camera read error: no frame returned from Picamera2")
        # Convert RGB (Picamera2) to BGR (OpenCV)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        """
        Stop the camera and release resources.
        """
        self.picam2.stop()
        self.picam2.close()

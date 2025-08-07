#!/usr/bin/env python3
"""
CameraReader for Picamera2 with config-driven parameters,
structured logging, error recovery, and context-manager support.
"""
import os
import yaml
import logging
import time
from typing import Optional
import cv2
from picamera2 import Picamera2

# Load camera configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f) or {}
    _cam_cfg = _full_cfg.get('camera', {}) or {}
except Exception:
    _cam_cfg = {}

# Module logger
logger = logging.getLogger(__name__)

class CameraReader:
    """
    Encapsulates Picamera2 initialization and frame capture.

    Reads parameters from 'camera' section of config.yaml:
      - frame_width, frame_height, fps, warmup

    Methods:
      - read(): return BGR frame or None on failure
      - restart(): attempt to recover camera
      - release(): clean shutdown

    Supports context-manager (with CameraReader()).
    """
    def __init__(self) -> None:
        width = int(_cam_cfg.get('frame_width', 640))
        height = int(_cam_cfg.get('frame_height', 480))
        self.framerate = int(_cam_cfg.get('fps', 30))
        self.warmup = float(_cam_cfg.get('warmup', 2))

        self.resolution = (width, height)
        logger.info("Initializing Picamera2 (resolution=%s, fps=%d, warmup=%.1fs)",
                    self.resolution, self.framerate, self.warmup)
        self.picam2: Picamera2
        self._init_camera()
        time.sleep(self.warmup)

    def _init_camera(self) -> None:
        """Configure and start Picamera2 based on config."""
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            controls={"AwbEnable": True, "FrameRate": self.framerate}
        )
        self.picam2.configure(config)
        self.picam2.start()
        logger.info("Picamera2 started at %d fps", self.framerate)

    def read(self) -> Optional[cv2.Mat]:  # type: ignore
        """
        Capture a single frame and return as BGR array.
        On error, restart camera and return None.
        """
        try:
            frame = self.picam2.capture_array()
        except Exception as e:
            logger.error("Camera capture failed (%s); restarting camera", e)
            self.restart()
            return None

        if frame is None:
            logger.warning("Empty frame returned; restarting camera")
            self.restart()
            return None

        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def restart(self) -> None:
        """Stop and reinitialize the camera on error."""
        try:
            self.release()
        except Exception as e:
            logger.warning("Error releasing camera: %s", e)
        logger.info("Reinitializing camera...")
        self._init_camera()
        time.sleep(self.warmup)

    def release(self) -> None:
        """Stop and close camera, freeing resources."""
        logger.info("Releasing camera resources")
        self.picam2.stop()
        self.picam2.close()

    def __enter__(self) -> 'CameraReader':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

# If run directly, do a simple non-GUI capture test
def _self_test() -> None:
    logging.basicConfig(level=logging.INFO)
    with CameraReader() as cam:
        for i in range(5):
            frame = cam.read()
            if frame is not None:
                logger.info("Captured frame %d; shape=%s", i, frame.shape)
            time.sleep(0.5)

if __name__ == '__main__':
    _self_test()

"""Threaded Picamera2 reader. Latest frame always available, no locks."""
import threading
import time
from picamera2 import Picamera2


class Camera:
    def __init__(self, size=(320, 240), format="RGB888"):
        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"size": size, "format": format}
        )
        self._picam.configure(config)

        self._latest = {
            "t": 0.0,
            "frame": None,
            "count": 0,
        }
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._picam.start()
        time.sleep(0.5)  # auto-exposure settle
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._picam.stop()

    def latest(self):
        return self._latest

    def _loop(self):
        count = 0
        while self._running:
            try:
                frame = self._picam.capture_array()
                count += 1
                self._latest = {
                    "t": time.monotonic(),
                    "frame": frame,
                    "count": count,
                }
            except Exception as e:
                print(f"[camera] capture error: {e}")
                time.sleep(0.1)

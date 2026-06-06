#!/usr/bin/env python3
"""
Camera colour / channel-order test. Run on the Pi.

picamera2's "RGB888" format is famously confusing: the array it hands back may
already be in BGR order (OpenCV-native) or actually RGB. This captures one frame
and saves it BOTH ways so you can see which has correct red/blue, which tells us
how the dashboard should be handling it.

The dashboard holds the camera, so stop it first:
    sudo systemctl stop autoboat-dashboard
    python3 camera_color_test.py
    sudo systemctl start autoboat-dashboard
"""

import os
import sys
import time

try:
    import cv2
    from picamera2 import Picamera2
except Exception as e:
    print("needs picamera2 + opencv on the Pi (run in the venv):", e)
    sys.exit(1)

OUT = os.path.expanduser("~/Documents/cam_test")
os.makedirs(OUT, exist_ok=True)

try:
    picam = Picamera2()
    cfg = picam.create_still_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam.configure(cfg)
    picam.start()
    time.sleep(1.0)  # let auto-exposure and white balance settle
    frame = picam.capture_array()
    picam.stop()
except Exception as e:
    print("capture failed:", e)
    print("is the dashboard still holding the camera? stop it first.")
    sys.exit(1)

print("captured array:", frame.shape, frame.dtype)

# A: treat the array as already-BGR (OpenCV native) and write it straight out.
cv2.imwrite(os.path.join(OUT, "A_as_captured.jpg"), frame)
# B: treat the array as RGB and convert to BGR before writing.
cv2.imwrite(os.path.join(OUT, "B_swapped.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("wrote:")
print(" ", os.path.join(OUT, "A_as_captured.jpg"))
print(" ", os.path.join(OUT, "B_swapped.jpg"))
print("open both. whichever has correct red/blue (blue sky, true colours) is the")
print("right channel order. tell me which, and I'll set the dashboard to match.")

# ~/autoboat/scripts/capture_pool_images.py
"""Capture pool test images. Press Enter to capture, Ctrl+C to quit."""
import sys
import time
import os
from datetime import datetime

sys.path.insert(0, "/home/ben/autoboat")
from sensors.camera import Camera

out_dir = os.path.expanduser("~/autoboat/pool_images")
os.makedirs(out_dir, exist_ok=True)

cam = Camera(size=(320, 240))
cam.start()
print(f"Camera started. Saving to {out_dir}")
print("Press Enter to capture, Ctrl+C to quit.\n")

count = 0
try:
    while True:
        label = input("Label (e.g. 'open_water', 'wall_left', 'sun_glare'): ").strip()
        if not label:
            label = "unlabeled"
        label = label.replace(" ", "_")

        # Grab a fresh frame
        time.sleep(0.3)
        snap = cam.latest()
        frame = snap["frame"]
        if frame is None:
            print("  no frame yet, try again")
            continue

        ts = datetime.now().strftime("%H%M%S")
        path = os.path.join(out_dir, f"{ts}_{label}.jpg")

        # Picamera2 gives RGB but PIL expects RGB so just save directly
        from PIL import Image
        Image.fromarray(frame).save(path, quality=90)

        count += 1
        print(f"  saved {path}  (mean brightness {frame.mean():.0f})")
except KeyboardInterrupt:
    print(f"\nDone. Captured {count} images.")
finally:
    cam.stop()

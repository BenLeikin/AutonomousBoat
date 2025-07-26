# camera_test.py
import os
from picamera2 import picamera2
import time

OUTPUT_DIR = "/mnt/usb/test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    picam2 = Picamera2()
    # Use your same resolution from config
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()

    for i in range(5):
        frame = picam2.capture_array()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"test_{timestamp}.jpg")
        # Save as JPEG
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(path)
        print("Saved", path)
        time.sleep(1)

    picam2.stop()
    print("Done.")

if __name__ == "__main__":
    main()

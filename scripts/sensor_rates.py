"""Run both sensors for 5 seconds, report actual achieved rates."""
import sys
import time
sys.path.insert(0, "/home/ben/autoboat")

from sensors.imu import IMU
from sensors.camera import Camera

imu = IMU(rate_hz=100)
cam = Camera(size=(320, 240))

print("Starting sensors...")
imu.start()
cam.start()

duration = 5.0
start = time.monotonic()
imu_start_count = imu.latest()["count"]
cam_start_count = cam.latest()["count"]

# Sample the latest values periodically so we can confirm both keep updating
while time.monotonic() - start < duration:
    i = imu.latest()
    c = cam.latest()
    age_imu = time.monotonic() - i["t"] if i["t"] else 0
    age_cam = time.monotonic() - c["t"] if c["t"] else 0
    print(f"  imu age {age_imu*1000:5.1f}ms  cam age {age_cam*1000:5.1f}ms  "
          f"imu#{i['count']}  cam#{c['count']}")
    time.sleep(0.5)

elapsed = time.monotonic() - start
imu_samples = imu.latest()["count"] - imu_start_count
cam_frames = cam.latest()["count"] - cam_start_count

print(f"\nElapsed: {elapsed:.2f}s")
print(f"IMU:    {imu_samples} samples = {imu_samples/elapsed:.1f} Hz (target 100)")
print(f"Camera: {cam_frames} frames  = {cam_frames/elapsed:.1f} fps")

print("\nStopping...")
imu.stop()
cam.stop()
print("Done.")

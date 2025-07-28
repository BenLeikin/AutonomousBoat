#!/usr/bin/env python3
import RPi.GPIO as GPIO, time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

print("Toggling GPIO17 five times")
for _ in range(5):
    GPIO.output(18, GPIO.HIGH)
    time.sleep(2.0)
    GPIO.output(18, GPIO.LOW)
    time.sleep(2.0)

GPIO.cleanup()
print("Done")

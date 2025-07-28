#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# ───── CONFIG ────────────────────────────────────────────────────────────────
LEFT_IN1   = 22   # BCM17 → driver IN1 (left motor)
LEFT_IN2   = 23   # BCM18 → driver IN2 (left motor)
RUN_TIME   = 5    # seconds to run the motor

# ───── SETUP ─────────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_IN1, GPIO.OUT)
GPIO.setup(LEFT_IN2, GPIO.OUT)

# ───── RUN LEFT MOTOR FORWARD ────────────────────────────────────────────────
# Ensure your STBY/EEP pin is held high externally
GPIO.output(LEFT_IN1, GPIO.HIGH)
GPIO.output(LEFT_IN2, GPIO.LOW)
print(f"Left motor running forward for {RUN_TIME} seconds")
time.sleep(RUN_TIME)

# ───── STOP & CLEANUP ─────────────────────────────────────────────────────────
GPIO.output(LEFT_IN1, GPIO.LOW)
GPIO.output(LEFT_IN2, GPIO.LOW)
GPIO.cleanup()
print("Left motor stopped, GPIO cleaned up")

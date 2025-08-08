#!/usr/bin/env python3
"""
motor_control.py
BCM GPIO motor control (mirrors tests/motor-test.py pinout).
Pivots are corrected: pivot_left = left_reverse + right_forward,
                      pivot_right = left_forward + right_reverse.
"""
import RPi.GPIO as GPIO

# ── PIN ASSIGNMENTS (BCM) — match your working test ───────────────────────────
LEFT_IN1, LEFT_IN2   = 22, 23   # controls physical RIGHT motor
RIGHT_IN1, RIGHT_IN2 = 17, 18   # controls physical LEFT motor

ALL_PINS = (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2)

# ── SETUP ─────────────────────────────────────────────────────────────────────
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for p in ALL_PINS:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, GPIO.LOW)  # ensure off

# ── HELPERS ───────────────────────────────────────────────────────────────────
def stop_all():
    for p in ALL_PINS:
        GPIO.output(p, GPIO.LOW)

def left_forward():    # drives physical right motor
    GPIO.output(LEFT_IN1, GPIO.HIGH)
    GPIO.output(LEFT_IN2, GPIO.LOW)

def left_reverse():    # reverse physical right motor
    GPIO.output(LEFT_IN1, GPIO.LOW)
    GPIO.output(LEFT_IN2, GPIO.HIGH)

def right_forward():   # drives physical left motor
    GPIO.output(RIGHT_IN1, GPIO.HIGH)
    GPIO.output(RIGHT_IN2, GPIO.LOW)

def right_reverse():   # reverse physical left motor
    GPIO.output(RIGHT_IN1, GPIO.LOW)
    GPIO.output(RIGHT_IN2, GPIO.HIGH)

def both_forward():
    left_forward(); right_forward()

def both_reverse():
    left_reverse(); right_reverse()

# ── PIVOTS (SWAPPED to match your working test) ───────────────────────────────
def pivot_left():      # left motor reverse, right motor forward
    left_reverse(); right_forward()

def pivot_right():     # left motor forward, right motor reverse
    left_forward(); right_reverse()

# Optional: clean up on interpreter exit (only if you call it explicitly)
def cleanup():
    try:
        stop_all()
    finally:
        GPIO.cleanup()

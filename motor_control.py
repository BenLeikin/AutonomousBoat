# motor_control.py
#!/usr/bin/env python3
# Module 8: Motor control interface (BCM GPIO) using tested script

import RPi.GPIO as GPIO

# ───── PIN ASSIGNMENTS ─────────────────────────────────────────────────────────
# BCM numbering, per your wiring:
LEFT_IN1, LEFT_IN2   = 22, 23   # white/green → physical right motor
RIGHT_IN1, RIGHT_IN2 = 17, 18   # grey/purple → physical left motor

ALL_PINS = (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2)

# ───── SETUP ───────────────────────────────────────────────────────────────────
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for p in ALL_PINS:
    GPIO.setup(p, GPIO.OUT)
# ensure all off at start
for p in ALL_PINS:
    GPIO.output(p, GPIO.LOW)

# ───── MOTOR HELPERS ──────────────────────────────────────────────────────────
def stop_all():
    """Halt both motors immediately"""
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
    """Both motors forward"""
    left_forward(); right_forward()

def both_reverse():
    """Both motors reverse"""
    left_reverse(); right_reverse()

def pivot_left():      # left motor reverse, right motor forward
    left_reverse(); right_forward()

def pivot_right():     # left motor forward, right motor reverse
    left_forward(); right_reverse()

# ───── COMMAND MAP ────────────────────────────────────────────────────────────
COMMANDS = {
    'w': (both_forward,  "Both forward"),
    's': (both_reverse,  "Both reverse"),
    'a': (left_forward,  "Turn left  (right motor only)"),
    'd': (right_forward, "Turn right (left motor only)"),
    'q': (pivot_left,    "Pivot left"),
    'e': (pivot_right,   "Pivot right"),
    'x': (stop_all,      "Stopped"),
}

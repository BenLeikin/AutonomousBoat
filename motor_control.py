#!/usr/bin/env python3

# Motor control interface (BCM GPIO) using tested script

import RPi.GPIO as GPIO

# ───── PIN ASSIGNMENTS ─────────────────────────────────────────────────────────
# BCM numbering, per your wiring:
LEFT_IN1, LEFT_IN2   = 22, 23   # white/green → physical right motor
RIGHT_IN1, RIGHT_IN2 = 17, 18   # grey/purple → physical left motor

ALL_PINS = (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2)

# ───── SETUP ───────────────────────────────────────────────────────────────────
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
...

def left_forward():
    # your implementation here
    pass

def left_reverse():
    # your implementation here
    pass

def right_forward():
    # your implementation here
    pass

def right_reverse():
    # your implementation here
    pass

def both_forward():
    left_forward()
    right_forward()

def both_reverse():
    left_reverse()
    right_reverse()

def stop_all():
    for pin in ALL_PINS:
        GPIO.output(pin, GPIO.LOW)

# Swap pivot functions to correct directions

def pivot_left():      # pivot left on the spot
    left_forward()     # left motor forward
    right_reverse()    # right motor reverse

def pivot_right():     # pivot right on the spot
    left_reverse()     # left motor reverse
    right_forward()    # right motor forward

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

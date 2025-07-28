#!/usr/bin/env python3
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

WELCOME = """
Motor control ready.
Commands:
  w → both forward
  s → both reverse
  a → turn left  (right motor only)
  d → turn right (left motor only)
  q → pivot left
  e → pivot right
  x → stop both
  z → quit
"""

# ───── MAIN LOOP ───────────────────────────────────────────────────────────────
def main():
    print(WELCOME)
    try:
        while True:
            cmd = input("cmd> ").strip().lower()
            if cmd == 'z':
                break
            stop_all()  # clear any previous motion
            action = COMMANDS.get(cmd)
            if action:
                action[0]()
                print("→", action[1])
            else:
                print("Unknown cmd, use w/a/s/d/q/e/x/z")
    finally:
        stop_all()
        GPIO.cleanup()
        print("GPIO cleaned up, exiting")

if __name__ == "__main__":
    main()

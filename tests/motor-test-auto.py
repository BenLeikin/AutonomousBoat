#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# ───── PIN ASSIGNMENTS ─────────────────────────────────────────────────────────
# BCM numbering
IN1_PIN = 17  # Driver IN1 → left motor input 1
IN2_PIN = 18  # Driver IN2 → left motor input 2
IN3_PIN = 22  # Driver IN3 → right motor input 1
IN4_PIN = 23  # Driver IN4 → right motor input 2

MOTOR_PINS = {
    'left':  {'in1': IN1_PIN, 'in2': IN2_PIN},
    'right': {'in1': IN3_PIN, 'in2': IN4_PIN},
}

# seconds each action runs
TEST_DURATION = 5
# pause between actions
PAUSE_DURATION = 1

# ───── SETUP & HELPERS ────────────────────────────────────────────────────────

def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for pins in MOTOR_PINS.values():
        GPIO.setup(pins['in1'], GPIO.OUT)
        GPIO.setup(pins['in2'], GPIO.OUT)
    stop_all()

def motor_forward(pins):
    GPIO.output(pins['in1'], GPIO.HIGH)
    GPIO.output(pins['in2'], GPIO.LOW)

def motor_reverse(pins):
    GPIO.output(pins['in1'], GPIO.LOW)
    GPIO.output(pins['in2'], GPIO.HIGH)

def motor_stop(pins):
    GPIO.output(pins['in1'], GPIO.LOW)
    GPIO.output(pins['in2'], GPIO.LOW)

def stop_all():
    for pins in MOTOR_PINS.values():
        motor_stop(pins)

# ───── TEST CYCLE ────────────────────────────────────────────────────────────

def run_cycle(duration):
    steps = [
        ("Left motor forward",  lambda: motor_forward(MOTOR_PINS['left'])),
        ("Right motor forward", lambda: motor_forward(MOTOR_PINS['right'])),
        ("Left motor reverse",  lambda: motor_reverse(MOTOR_PINS['left'])),
        ("Right motor reverse", lambda: motor_reverse(MOTOR_PINS['right'])),
        ("Both motors forward", lambda: (
            motor_forward(MOTOR_PINS['left']),
            motor_forward(MOTOR_PINS['right'])
        )),
        ("Both motors reverse", lambda: (
            motor_reverse(MOTOR_PINS['left']),
            motor_reverse(MOTOR_PINS['right'])
        )),
    ]

    for desc, action in steps:
        print(f"{desc} for {duration} seconds")
        stop_all()                   # ensure we start stopped
        action()
        time.sleep(duration)
        stop_all()
        time.sleep(PAUSE_DURATION)

# ───── MAIN ───────────────────────────────────────────────────────────────────

def main():
    setup_gpio()
    try:
        run_cycle(TEST_DURATION)
        print("Full cycle complete")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        stop_all()
        GPIO.cleanup()
        print("GPIO cleaned up, exiting")

if __name__ == "__main__":
    main()

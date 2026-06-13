#!/usr/bin/env python3
"""
AutoBoat DRV8833 bench test (two motors, differential/tank drive).

RUN WITH PROPS OFF AND THE HULLS SECURED. The motors WILL spin.

Channel A = LEFT motor   (AIN1=GPIO17, AIN2=GPIO27)
Channel B = RIGHT motor  (BIN1=GPIO22, BIN2=GPIO23)
SLP/nSLEEP = GPIO24      (active high; low = driver asleep, outputs off = safe)

The driver is only enabled (SLP high) while a step is actually running, and is
dropped back low between steps and on exit, so the default state is coast.

Usage:
    python3 motor_test.py                 # guided sequence: each motor, each way
    python3 motor_test.py --duty 0.5      # change test speed (0..1, default 0.4)
    python3 motor_test.py --secs 3        # seconds each step runs (default 2)
    python3 motor_test.py --manual        # interactive poke mode
"""

import argparse
import atexit
import sys
import time

try:
    from gpiozero import Motor, OutputDevice
except Exception as e:
    print(f"gpiozero not available: {e}")
    print("Install it in the venv: pip install gpiozero lgpio")
    sys.exit(1)

# DRV8833 wiring (BCM numbering)
AIN1, AIN2 = 17, 27     # channel A -> LEFT motor
BIN1, BIN2 = 22, 23     # channel B -> RIGHT motor
SLP = 24                # nSLEEP / enable, active high

left = Motor(forward=AIN1, backward=AIN2, pwm=True)
right = Motor(forward=BIN1, backward=BIN2, pwm=True)
enable = OutputDevice(SLP, active_high=True, initial_value=False)


def all_stop():
    # Zero the inputs, then put the driver to sleep. Safe to call repeatedly.
    try:
        left.stop()
        right.stop()
    finally:
        enable.off()


atexit.register(all_stop)


def step(motor, name, direction, duty, secs):
    input(f"\nPress Enter to run: {name} {direction} at {int(duty * 100)}% for {secs:.0f}s ...")
    enable.on()
    (motor.forward if direction == "forward" else motor.backward)(duty)
    try:
        time.sleep(secs)
    finally:
        motor.stop()
        enable.off()
    print(f"  done. Did the {name} spin, and in the {direction} direction?")


def guided(duty, secs):
    print(__doc__)
    print("Props off and hulls secured. Starting in 2s, Ctrl-C to abort.")
    time.sleep(2)

    print("\nEnabling driver briefly with both channels idle. Nothing should move.")
    enable.on()
    time.sleep(1.0)
    enable.off()
    print("  If anything spun there, stop and recheck wiring before going on.")

    step(left, "LEFT motor", "forward", duty, secs)
    step(left, "LEFT motor", "reverse", duty, secs)
    step(right, "RIGHT motor", "forward", duty, secs)
    step(right, "RIGHT motor", "reverse", duty, secs)

    # PWM sanity: ramp the left motor up so you can confirm speed tracks duty.
    input("\nPress Enter for a speed ramp on the LEFT motor (0 -> 100%) ...")
    enable.on()
    try:
        for pct in range(0, 101, 10):
            left.forward(pct / 100.0)
            print(f"  left at {pct}%")
            time.sleep(1.0)
    finally:
        left.stop()
        enable.off()

    # Both motors forward together, ramped 0 -> 100%. This is the straight-ahead
    # full-power check: both should track the same speed and pull evenly.
    input("\nPress Enter for BOTH motors forward, ramp 0 -> 100% ...")
    enable.on()
    try:
        for pct in range(0, 101, 10):
            left.forward(pct / 100.0)
            right.forward(pct / 100.0)
            print(f"  both at {pct}%")
            time.sleep(1.0)
    finally:
        left.stop()
        right.stop()
        enable.off()

    print("\nResults:")
    print("- Wrong direction on a motor: swap that motor's two leads, or swap its")
    print("  forward/backward pins (e.g. AIN1<->AIN2) in the wiring.")
    print("- LEFT and RIGHT swapped: swap the motor wires between the A and B outputs.")
    print("- A motor that won't start at low duty: bump --duty; small motors need")
    print("  ~40-50% at this PWM frequency to overcome stiction.")
    all_stop()


def manual(duty, secs):
    print(__doc__)
    print("Manual mode. Commands (run continuously until you stop them):")
    print("  lf / lr   left forward / reverse")
    print("  rf / rr   right forward / reverse")
    print("  ff / bb   both forward / both reverse")
    print("  s         stop both")
    print("  q         quit")
    print(f"  duty is {duty:.2f}; type a number 0-1 to change it")

    cmds = {
        "lf": lambda: left.forward(duty),
        "lr": lambda: left.backward(duty),
        "rf": lambda: right.forward(duty),
        "rr": lambda: right.backward(duty),
        "ff": lambda: (left.forward(duty), right.forward(duty)),
        "bb": lambda: (left.backward(duty), right.backward(duty)),
    }
    while True:
        c = input("> ").strip().lower()
        if c == "q":
            break
        if c == "s":
            left.stop()
            right.stop()
            enable.off()
            continue
        try:
            duty = max(0.0, min(1.0, float(c)))
            print(f"  duty = {duty:.2f}")
            cmds = {
                "lf": lambda: left.forward(duty),
                "lr": lambda: left.backward(duty),
                "rf": lambda: right.forward(duty),
                "rr": lambda: right.backward(duty),
                "ff": lambda: (left.forward(duty), right.forward(duty)),
                "bb": lambda: (left.backward(duty), right.backward(duty)),
            }
            continue
        except ValueError:
            pass
        if c in cmds:
            enable.on()
            cmds[c]()
        else:
            print("  unknown command")
    all_stop()


def main():
    ap = argparse.ArgumentParser(description="DRV8833 two-motor bench test")
    ap.add_argument("--duty", type=float, default=0.4, help="test speed 0..1 (default 0.4)")
    ap.add_argument("--secs", type=float, default=2.0, help="seconds per step (default 2)")
    ap.add_argument("--manual", action="store_true", help="interactive poke mode")
    args = ap.parse_args()

    duty = max(0.0, min(1.0, args.duty))
    try:
        if args.manual:
            manual(duty, args.secs)
        else:
            guided(duty, args.secs)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        all_stop()


if __name__ == "__main__":
    main()



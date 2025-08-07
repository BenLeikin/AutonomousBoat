#!/usr/bin/env python3
"""
Launcher for AutonomousBoat navigation only.
Removes independent camera, vision, and IMU processes to avoid resource conflicts.
"""
import subprocess
import sys
import os
import signal

# Only start navigation.py, which drives camera, vision, IMU, and motors
SCRIPTS = [
    "navigation.py",
]

processes = []

def launch(script: str) -> subprocess.Popen | None:
    """Launch a Python script as a subprocess."""
    path = os.path.join(os.path.dirname(__file__), script)
    if not os.path.isfile(path):
        print(f"[Warning] Missing module: {script}")
        return None
    return subprocess.Popen([sys.executable, path])


def shutdown(signum, frame) -> None:
    print("\n[Info] Terminating navigation module...")
    for p in processes:
        if p and p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    sys.exit(0)


def main() -> None:
    # Clean shutdown on SIGINT/SIGTERM
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[Info] Launching navigation module...")
    for script in SCRIPTS:
        p = launch(script)
        if p:
            processes.append(p)
            print(f"[Info] Launched {script} (PID {p.pid})")

    # Wait for navigation to exit
    for p in processes:
        if p:
            p.wait()

    print("[Info] Navigation module has exited.")

if __name__ == '__main__':
    main()

"""
Motor driver interface for the DRV8833 motor driver using Raspberry Pi GPIO.
Attempts to import RPi.GPIO; if unavailable, provides a dummy GPIO for testing.
"""

# Try importing RPi.GPIO; fall back to dummy for non-Pi environments
try:
    import RPi.GPIO as GPIO
except ImportError:
    class _DummyGPIO:
        BCM = 'BCM'
        OUT = 'OUT'
        def __init__(self):
            self.mode = None
            self.setup_calls = []
            self.output_calls = []

        def setmode(self, mode):
            self.mode = mode

        def setup(self, pin, mode):
            self.setup_calls.append((pin, mode))

        def output(self, pin, value):
            self.output_calls.append((pin, value))

    GPIO = _DummyGPIO()

class MotorDriver:
    """Controls two DC motors (port and starboard) via a DRV8833 driver."""
    def __init__(self, pins: dict):
        """
        Initialize the GPIO pins.

        :param pins: A dict mapping 'port_fwd', 'port_rev', 'star_fwd', 'star_rev' to BCM pin numbers.
        """
        GPIO.setmode(GPIO.BCM)
        for pin in pins.values():
            GPIO.setup(pin, GPIO.OUT)
        self.pins = pins

    def set_port(self, forward: bool, reverse: bool):
        """
        Control the port (left) motor.

        :param forward: True to spin forward
        :param reverse: True to spin reverse
        """
        GPIO.output(self.pins['port_fwd'], forward)
        GPIO.output(self.pins['port_rev'], reverse)

    def set_star(self, forward: bool, reverse: bool):
        """
        Control the starboard (right) motor.

        :param forward: True to spin forward
        :param reverse: True to spin reverse
        """
        GPIO.output(self.pins['star_fwd'], forward)
        GPIO.output(self.pins['star_rev'], reverse)

    def stop(self):
        """
        Stop both motors immediately.
        """
        self.set_port(False, False)
        self.set_star(False, False)

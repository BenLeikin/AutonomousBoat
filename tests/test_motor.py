# tests/test_motor.py
import pytest
import autoboat.motor as motor_mod

class DummyGPIO:
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

def test_motordriver_initialization_and_stop(monkeypatch):
    dummy_gpio = DummyGPIO()
    # Monkey-patch the GPIO used in motor.py
    monkeypatch.setattr(motor_mod, 'GPIO', dummy_gpio)

    pins = {'port_fwd': 1, 'port_rev': 2, 'star_fwd': 3, 'star_rev': 4}
    md = motor_mod.MotorDriver(pins)

    # Should set mode to BCM and call setup for each pin
    assert dummy_gpio.mode == dummy_gpio.BCM
    assert set(dummy_gpio.setup_calls) == set((p, dummy_gpio.OUT) for p in pins.values())

    # Test stop(): last output to each pin must be False
    md.stop()
    # Two motors × two outputs → 4 calls, all value False
    assert all(val is False for _, val in dummy_gpio.output_calls)

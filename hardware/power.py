"""INA219 pack monitor: pack/current readout, the critical-battery auto-shutdown
state machine, and the hard overcurrent guard (last resort behind the controller's
stall logic). `recording` is imported lazily inside functions: recording imports
this module for telemetry, so a top-level import here would be a cycle."""
import subprocess
import threading
import time

import hardware.motors as motors

SHUTDOWN_CMD = ["sudo", "-n", "systemctl", "poweroff"]

# INA219 pack monitor is optional: it lives on the software I2C bus created by
# the i2c-gpio overlay (SDA=GPIO5, SCL=GPIO6 -> /dev/i2c-3). On a box without
# that bus or the libraries, the Power section just shows "sensor offline".
try:
    from adafruit_extended_bus import ExtendedI2C
    from adafruit_ina219 import INA219
    INA_AVAILABLE = True
except Exception as e:
    INA_AVAILABLE = False
    print(f"[power] INA219 libs unavailable, power section disabled: {e}")

INA_BUS = 3                        # /dev/i2c-3, the i2c-gpio bus on GPIO5/6
INA_ADDR = 0x40
PACK_LOW = 6.6                     # 2S Li, ~3.3 V/cell: head back
PACK_CRITICAL = 6.0               # 2S Li, ~3.0 V/cell: stop

# Auto-shutdown on a critically low pack, so a dying battery cannot brown out the
# Pi mid-write and corrupt the SD / session (which is exactly what happened on the
# 5.7V run). The Pi owns this loop: it counts down and powers itself off even with
# no browser open. The dashboard overlay only mirrors the state and offers an
# override. A debounce avoids tripping on a momentary sag, and a snooze lets a
# watching operator buy time without disabling the protection outright.
CRITICAL_DEBOUNCE = 5.0           # pack must stay below critical this long before arming
CRITICAL_GRACE = 30.0             # countdown (s) before poweroff once armed
CRITICAL_SNOOZE = 120.0           # override delays re-arming this long, then re-checks
CRITICAL_RECOVER = PACK_CRITICAL + 0.15   # pack must rise above this to fully reset

crit = {"below_since": None, "deadline": None, "snooze_until": None, "fired": False}
critlock = threading.Lock()

# ---------- INA219 pack monitor shared state ----------
power = {"connected": False, "pack_v": None, "rail_v": None,
         "current_ma": None, "power_w": None, "status": "--", "last_data": 0.0}
pwlock = threading.Lock()

def _pack_status(v):
    if v is None:
        return "--"
    if v >= PACK_LOW:
        return "ok"
    if v >= PACK_CRITICAL:
        return "low"
    return "critical"


def _do_critical_shutdown():
    # Finalize any recording first so we do not leave a half-written session
    # (the brownout signature), then halt the Pi cleanly.
    try:
        import recording
        if recording.is_recording():
            recording.capture_stop(reason="battery_critical")
    except Exception:
        pass
    try:
        subprocess.run(SHUTDOWN_CMD, capture_output=True, text=True, timeout=10)
    except Exception:
        pass


def _critical_tick(pack_v):
    # Called from power_loop each read. Arms a countdown after the pack has been
    # below critical for CRITICAL_DEBOUNCE, fires the shutdown at the deadline,
    # and fully resets only once the pack recovers above CRITICAL_RECOVER.
    if pack_v is None:
        return
    now = time.monotonic()
    fire = False
    with critlock:
        snoozed = crit["snooze_until"] is not None and now < crit["snooze_until"]
        if pack_v < PACK_CRITICAL:
            if crit["below_since"] is None:
                crit["below_since"] = now
            sustained = (now - crit["below_since"]) >= CRITICAL_DEBOUNCE
            if sustained and not snoozed and crit["deadline"] is None and not crit["fired"]:
                crit["deadline"] = now + CRITICAL_GRACE
                motors.force_stop()   # cut autonomy immediately when the pack goes critical
            if crit["deadline"] is not None and not crit["fired"] and now >= crit["deadline"]:
                crit["fired"] = True
                fire = True
        elif pack_v >= CRITICAL_RECOVER:
            crit["below_since"] = None
            crit["deadline"] = None
            crit["snooze_until"] = None
            crit["fired"] = False
    if fire:
        _do_critical_shutdown()


def _critical_status():
    now = time.monotonic()
    with critlock:
        if crit["fired"]:
            return {"armed": True, "seconds_left": 0, "fired": True}
        if crit["deadline"] is not None:
            return {"armed": True, "fired": False,
                    "seconds_left": max(0, int(round(crit["deadline"] - now)))}
        snoozed = crit["snooze_until"] is not None and now < crit["snooze_until"]
        return {"armed": False, "fired": False, "seconds_left": None,
                "snoozed": snoozed,
                "snooze_left": (int(round(crit["snooze_until"] - now)) if snoozed else None)}


def _critical_override():
    now = time.monotonic()
    with critlock:
        crit["deadline"] = None
        crit["snooze_until"] = now + CRITICAL_SNOOZE
    return {"ok": True, "message": "shutdown overridden; battery re-checked in %ds" % int(CRITICAL_SNOOZE)}






def power_loop():
    # Reconnecting reader for the high-side INA219. Pack voltage is bus + shunt
    # (the sensor sits in the battery + lead). On any I2C error it backs off and
    # retries so the dashboard recovers if the sensor drops off the bus.
    # Also hosts the hard overcurrent guard: the controller's stall logic should
    # back off a pinned boat within ~2s, so this is the dumb last resort that
    # protects the DRV8833 (rated 1.5A/ch continuous; stalls measured at 3.2A)
    # if the smart path ever fails. Sustained extreme current while running =>
    # force-stop the motors.
    hard_hits = 0
    while True:
        try:
            i2c3 = ExtendedI2C(INA_BUS)
            ina = INA219(i2c3, addr=INA_ADDR)
        except Exception as e:
            print(f"[power] cannot open INA219 on /dev/i2c-{INA_BUS}: {e}")
            with pwlock:
                power["connected"] = False
            time.sleep(3.0)
            continue

        print(f"[power] reading INA219 on /dev/i2c-{INA_BUS} at {hex(INA_ADDR)}")
        try:
            while True:
                bus_v = ina.bus_voltage          # volts at Vin- (rail to loads)
                shunt_v = ina.shunt_voltage       # volts across the shunt
                pack_v = bus_v + shunt_v          # actual battery voltage
                cur_ma = round(ina.current, 1)
                with pwlock:
                    power["connected"] = True
                    power["pack_v"] = round(pack_v, 3)
                    power["rail_v"] = round(bus_v, 3)
                    power["current_ma"] = cur_ma
                    power["power_w"] = round(ina.power, 3)
                    power["status"] = _pack_status(pack_v)
                    power["last_data"] = time.monotonic()
                _critical_tick(pack_v)
                if motors.ARMED and cur_ma is not None and cur_ma > MOTOR_HARD_STALL_MA:
                    hard_hits += 1
                    if hard_hits >= MOTOR_HARD_STALL_N:
                        print("[power] HARD STALL: %.0f mA sustained, force-stopping motors" % cur_ma)
                        import recording
                        recording.note_event("HARD overcurrent guard tripped at %.0f mA; motors stopped" % cur_ma)
                        motors.force_stop()
                        hard_hits = 0
                else:
                    hard_hits = 0
                time.sleep(0.5)
        except Exception as e:
            print(f"[power] read error: {e}")
            with pwlock:
                power["connected"] = False
            time.sleep(2.0)

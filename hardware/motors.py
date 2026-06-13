"""DRV8833 motor actuation: pins, duty mapping (trim + kickstart + caps),
arm/disarm, run watchdog, and the IMU-based trim-check calibration aid.

ARMED is the single source of truth for "the controller drives the motors";
the pilot reads it each frame and the power module force-stops through it."""
import atexit
import math
import threading
import time

from hardware.imu import att, alock, YAW_SIGN, IMU_AVAILABLE

ARMED = False


def _clamp(x, lo=-1.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x

# ---------------- Motor actuation (DRV8833) ----------------
# Turns the controller's left/right command into motor drive. Boots DISARMED.
# Output is scaled by a throttle cap that starts low, so a wrong side/direction
# mapping is a slow, recoverable nudge in the water rather than a full-speed run
# at the wall; confirm the boat steers the right way, then raise the cap. A
# watchdog cuts the motors if the vision loop stops feeding fresh decisions.
MOT_AIN1, MOT_AIN2 = 17, 27       # channel A -> LEFT motor (bench-confirmed pins)
MOT_BIN1, MOT_BIN2 = 22, 23       # channel B -> RIGHT motor
MOT_SLP = 24                      # nSLEEP / enable (HIGH = driver awake)
MOTOR_THROTTLE_CAP = 0.35         # 0..1, master speed cap (top of each motor's duty range)
MOTOR_DEADBAND = 0.05             # |intent| below this coasts (ignore tiny commands)
MOTOR_WATCHDOG_S = 0.6            # no fresh decision within this => cut motors
MOTOR_BACKUP_CAP = 0.60           # duty cap while in backup mode, independent of (and
                                  # higher than) the cruising throttle cap. The cap
                                  # limits forward ramming speed; the escape reverse is
                                  # the safe direction and needs real authority - props
                                  # make far less thrust in reverse, and 35% reverse
                                  # duty measurably failed to move the boat off a wall.
MOTOR_PIVOT_CAP = 0.60            # duty cap while pivoting. A tank turn is in-place,
                                  # not charging anything, so it can spin harder than
                                  # the cruise cap safely. At 35% the turns were very
                                  # slow (session 194246: 6-16 s per 90 deg); more
                                  # authority speeds them up without raising forward
                                  # speed or changing any turn logic.
MOTOR_HARD_STALL_MA = 2600        # last-resort guard: current above this...
MOTOR_HARD_STALL_N = 8            #   ...for this many 0.5s samples (4s) => force-stop.
                                  # The controller's stall logic backs off in ~2s, so
                                  # this only fires if that path is broken or disabled.

# Per-motor calibration. Two motors/props/drivetrains are never identical, so without
# this a "straight" command veers and turns are lopsided. The controller emits a
# normalized thrust intent in [-1,1]; this layer maps it to each motor's real PWM.
#   *_SCALE   trim. Trim the STRONGER motor DOWN (scale < 1.0); do not scale the weak
#             one up, that would blow past the throttle cap. 1.0 = no trim.
#   *_MIN     kickstart: the duty below which that motor won't reliably turn (stiction).
#             Intent past the deadband maps onto [*_MIN, cap] instead of [0, cap]. 0 = off.
# Defaults are no-ops, so behavior is unchanged until you measure per boat. Use the
# "trim_check" action (drives straight, reads IMU yaw) to find which side is stronger.
MOTOR_LEFT_SCALE = 1.0
MOTOR_RIGHT_SCALE = 1.0
MOTOR_LEFT_MIN = 0.0              # 0..cap, e.g. 0.12 if the left motor needs ~12% to start
MOTOR_RIGHT_MIN = 0.0

try:
    from gpiozero import Motor as _Motor, OutputDevice as _OutputDevice
    _mleft = _Motor(forward=MOT_AIN1, backward=MOT_AIN2, pwm=True)
    _mright = _Motor(forward=MOT_BIN1, backward=MOT_BIN2, pwm=True)
    _menable = _OutputDevice(MOT_SLP, active_high=True, initial_value=False)
    MOTORS_AVAILABLE = True
except Exception:
    _mleft = _mright = _menable = None
    MOTORS_AVAILABLE = False

armlock = threading.Lock()
_arm_state = {"last_cmd_t": 0.0}
_trim_busy = False                # set while the trim-check helper owns the motors


def _set_motor(m, signed_duty):
    # Drive one motor at a signed duty already in PWM terms (calibration applied
    # upstream). Coast on ~zero. forward()/backward() expect a magnitude in [0,1].
    if m is None:
        return
    if abs(signed_duty) < 1e-3:
        m.stop()                       # coast
    elif signed_duty > 0:
        m.forward(min(signed_duty, 1.0))
    else:
        m.backward(min(-signed_duty, 1.0))


def _motor_duty(intent, scale, min_duty, cap=None):
    # Map a controller thrust intent in [-1,1] to a signed PWM duty for one motor.
    # |intent| under the deadband coasts; otherwise it maps onto [min_duty, cap] (the
    # kickstart floor up to the duty cap), then the trim scale pulls a strong motor
    # down. Sign (direction) is preserved. `cap` defaults to the cruising throttle
    # cap; backup mode passes the higher MOTOR_BACKUP_CAP for real reverse authority.
    if cap is None:
        cap = MOTOR_THROTTLE_CAP
    f = min(abs(intent), 1.0)
    if f < MOTOR_DEADBAND:
        return 0.0
    span = max(cap - min_duty, 0.0)
    duty = (min_duty + span * f) * scale
    duty = min(duty, cap)
    return duty if intent >= 0 else -duty


def _motors_off():
    # Coast both motors and sleep the driver. Safe to call anytime, repeatedly.
    try:
        if _mleft:
            _mleft.stop()
        if _mright:
            _mright.stop()
    finally:
        if _menable:
            _menable.off()


def actuate(cmd, now):
    # Every frame from camera_loop. No-op unless armed with motors present, or while
    # the trim-check helper holds the motors.
    if _trim_busy or not (MOTORS_AVAILABLE and ARMED):
        return
    with armlock:
        _arm_state["last_cmd_t"] = now
    if _menable:
        _menable.on()
    # Backup gets its own, higher duty cap: reversing off a wall is the safe
    # direction, and the cruise cap throttled the escape into uselessness.
    # Backup and pivot are in-place maneuvers (not charging an obstacle), so each
    # gets a higher duty cap than the cruise speed limit: real reverse authority to
    # clear a wall, real rotation authority for a brisk tank turn.
    mode = cmd.get("mode")
    if mode == "backup":
        cap = max(MOTOR_BACKUP_CAP, MOTOR_THROTTLE_CAP)
    elif mode == "pivot":
        cap = max(MOTOR_PIVOT_CAP, MOTOR_THROTTLE_CAP)
    else:
        cap = MOTOR_THROTTLE_CAP
    _set_motor(_mleft, _motor_duty(_clamp(cmd["left"]), MOTOR_LEFT_SCALE, MOTOR_LEFT_MIN, cap))
    _set_motor(_mright, _motor_duty(_clamp(cmd["right"]), MOTOR_RIGHT_SCALE, MOTOR_RIGHT_MIN, cap))


def _motor_watchdog():
    # Cut motors if the control loop stops feeding fresh commands while armed.
    while True:
        time.sleep(0.1)
        if not (MOTORS_AVAILABLE and ARMED):
            continue
        with armlock:
            stale = (time.monotonic() - _arm_state["last_cmd_t"]) > MOTOR_WATCHDOG_S
        if stale:
            _motors_off()


def _trim_check(secs=2.5):
    # Calibration aid: drive BOTH motors straight forward at equal raw duty (no trim,
    # no kickstart) and measure the IMU yaw rate. A matched pair holds yaw near zero;
    # a mismatch yaws the boat. The boat turns toward the WEAKER motor, so trim the
    # other (stronger) side down. Runs only while stopped, owns the motors for `secs`,
    # then coasts. Blocks the calling request for the duration (it's a manual action).
    global _trim_busy
    if not MOTORS_AVAILABLE:
        return {"ok": False, "error": "motor driver not available"}
    if not IMU_AVAILABLE:
        return {"ok": False, "error": "IMU offline; cannot measure yaw drift"}
    if ARMED:
        return {"ok": False, "error": "stop the boat first; trim check runs while stopped"}
    duty = MOTOR_THROTTLE_CAP
    samples = []
    _trim_busy = True               # keep _actuate off the motors even if armed mid-check
    try:
        if _menable:
            _menable.on()
        _set_motor(_mleft, duty)
        _set_motor(_mright, duty)
        settle = time.monotonic() + 0.4        # let it spin up before sampling
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            if time.monotonic() >= settle:
                with alock:
                    samples.append(YAW_SIGN * att["gz"])    # rad/s, controller's sign convention
            time.sleep(0.02)
    finally:
        _motors_off()
        _trim_busy = False
    if not samples:
        return {"ok": False, "error": "no IMU samples captured"}
    deg = math.degrees(sum(samples) / len(samples))
    if abs(deg) < 2.0:
        msg = "tracks straight (%.1f deg/s yaw); motors look matched, no trim needed" % deg
        stronger = None
    else:
        # Boat turns toward the weaker side, so the opposite side is stronger.
        turned = "right" if deg > 0 else "left"
        stronger = "left" if deg > 0 else "right"
        weaker = "right" if deg > 0 else "left"
        msg = ("veered %s at %.1f deg/s -> %s motor is stronger; lower MOTOR_%s_SCALE "
               "(e.g. 0.9) until it tracks straight. If it actually swung the other "
               "way, your IMU yaw sign is flipped." % (turned, abs(deg), stronger,
                                                        stronger.upper()))
    return {"ok": True, "yaw_deg_s": round(deg, 2), "samples": len(samples),
            "test_duty": round(duty, 2), "stronger_side": stronger, "message": msg}


def _arm():
    global ARMED
    if not MOTORS_AVAILABLE:
        return {"ok": False, "message": "motor driver not available; cannot arm"}
    with armlock:
        _arm_state["last_cmd_t"] = time.monotonic()
    ARMED = True
    return {"ok": True, "armed": True,
            "message": "ARMED at %d%% throttle cap" % int(round(MOTOR_THROTTLE_CAP * 100))}


def _disarm():
    global ARMED
    ARMED = False
    _motors_off()
    return {"ok": True, "armed": False, "message": "disarmed; motors coasting"}


def _cap_step(delta):
    global MOTOR_THROTTLE_CAP
    MOTOR_THROTTLE_CAP = max(0.1, min(1.0, round(MOTOR_THROTTLE_CAP + delta, 2)))
    return {"ok": True, "cap": MOTOR_THROTTLE_CAP,
            "message": "throttle cap %d%%" % int(round(MOTOR_THROTTLE_CAP * 100))}

def force_stop():
    """Silent emergency stop for protective paths (hard overcurrent, critical
    battery): disarm and coast without the action-result dict."""
    global ARMED
    ARMED = False
    _motors_off()


# Public aliases for the action registry / other modules.
arm = _arm
disarm = _disarm
cap_step = _cap_step
trim_check = _trim_check
watchdog = _motor_watchdog
atexit.register(_motors_off)

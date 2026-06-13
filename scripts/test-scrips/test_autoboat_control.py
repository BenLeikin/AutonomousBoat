"""Tests for the control package. No hardware needed; runnable directly from
anywhere under the repo: `python3 scripts/test_autoboat_control.py` (or via pytest).
The controller takes `now` as an argument, so time is driven explicitly and tests
are deterministic.

Zone convention: zones[0] is the leftmost column, zones[-1] the rightmost; higher =
more open water (farther to an obstacle). `center` is the center water-depth percent.
"""
import os
import sys

# Find the repo root (the dir that holds the `control` package) by walking up from
# this file, so the import resolves no matter how deep the test is placed and no
# matter the current working directory.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "control")):
    _d = os.path.dirname(_d)
if _d not in sys.path:
    sys.path.insert(0, _d)

from control import controller as A

DT = 0.0625   # ~16 fps

# Representative vision fields.
CLEAR = [55, 60, 62, 60, 55]    # wide open ahead
BOXED = [6, 7, 8, 7, 6]         # walled in on every side
OPEN_R = [5, 8, 15, 40, 70]     # blocked center, opening to the right
OPEN_L = [70, 40, 15, 8, 5]     # blocked center, opening to the left


class Driver:
    """Drives a controller with an advancing clock."""
    def __init__(self, armed=False, t0=1000.0):
        self.c = A.Controller()
        self.c.reset(t0)
        self.t = t0
        self.armed = armed

    def step(self, zones, center, yaw=0.5, current=None, motion=None, rng=None):
        self.t += DT
        return self.c.step(zones, center, yaw, self.t, armed=self.armed,
                           depths=zones, current_ma=current, motion=motion,
                           range_m=rng)

    def run_until(self, zones, center, yaw, want_mode, max_s=5.0, current=None,
                  motion=None, rng=None):
        n = int(max_s / DT)
        for _ in range(n):
            cmd = self.step(zones, center, yaw, current=current, motion=motion, rng=rng)
            if cmd["mode"] == want_mode:
                return cmd, self.t
        return None, self.t


def test_fresh_controller_starts_in_assess():
    c = A.Controller()
    assert c.state["mode"] == "assess"
    c.reset(0.0)
    assert c.state["mode"] == "assess"


def test_assess_clear_goes_to_run():
    d = Driver()
    cmd = d.step(CLEAR, 60)
    assert cmd["mode"] == "run"
    assert cmd["throttle"] > 0          # easing forward


def test_assess_boxed_backs_up():
    d = Driver()
    cmd = d.step(BOXED, 6)
    assert cmd["mode"] == "backup"
    assert cmd["left"] < 0 and cmd["right"] < 0          # both reverse
    assert abs(cmd["left"] - cmd["right"]) < 1e-6        # straight back (no bias)


def test_assess_blocked_with_open_side_pivots():
    d = Driver()
    cmd = d.step(OPEN_L, 7)
    assert cmd["mode"] == "pivot"


def test_pivot_is_a_tank_turn_toward_the_opening():
    # Opening on the LEFT -> rotate left -> left motor reverses, right forward.
    d = Driver()
    cmd = d.step(OPEN_L, 7)
    assert cmd["left"] < 0 < cmd["right"], cmd
    # Opening on the RIGHT -> rotate right -> right reverses, left forward.
    d2 = Driver()
    cmd2 = d2.step(OPEN_R, 7)
    assert cmd2["right"] < 0 < cmd2["left"], cmd2


def test_run_hard_steer_becomes_a_tank_turn():
    # Marginal center (not blocked, not clear) with a strong right opening: the
    # controller should commit to a tight turn that drops the inside (right) motor
    # below zero -- an arc tightened into a tank turn.
    d = Driver(armed=True)
    cmd = None
    for _ in range(80):
        cmd = d.step(OPEN_R, 18)
    assert cmd["mode"] == "run", cmd
    assert cmd["turn"] > 0.5, cmd          # committed right turn
    assert cmd["right"] < 0 < cmd["left"], cmd   # inside motor reversed


def test_backup_then_pivot():
    d = Driver()
    d.step(BOXED, 6)                       # assess -> backup
    assert d.c.state["mode"] == "backup"
    # hold boxed; after CTL_BACKUP_SEC the reverse finishes and it pivots
    cmd, _ = d.run_until(BOXED, 6, 0.5, "pivot", max_s=A.CTL_BACKUP_SEC + 1.0)
    assert cmd is not None, "never left backup"


def test_pinned_pivot_backs_up_when_armed():
    # Blocked center but an open side (so NOT boxed): pivot. Commanded to pivot but
    # not yawing (pinned) while armed -> back up after ~CTL_YAW_STUCK_SEC.
    d = Driver(armed=True)
    cmd, t = d.run_until(OPEN_L, 7, 0.01, "backup", max_s=A.CTL_YAW_STUCK_SEC + 1.0)
    assert cmd is not None, "pinned pivot never backed up"


def test_unarmed_pivot_ignores_yaw_stuck():
    # Same pinned situation but unarmed: the yaw check is armed-only, so before the
    # long-pivot timeout (CTL_STUCK_SEC) it must still be pivoting, not backing up.
    d = Driver(armed=False)
    held = A.CTL_YAW_STUCK_SEC + 0.3       # past yaw-stuck, before long-pivot timeout
    assert held < A.CTL_STUCK_SEC
    cmd = None
    for _ in range(int(held / DT)):
        cmd = d.step(OPEN_L, 7, yaw=0.01)
    assert cmd["mode"] == "pivot", cmd


def test_analyze_zones_centroid_sign():
    assert A.analyze_zones(OPEN_R)["centroid"] > 0    # right opening -> positive
    assert A.analyze_zones(OPEN_L)["centroid"] < 0    # left opening  -> negative
    assert abs(A.analyze_zones(CLEAR)["centroid"]) < 0.05   # symmetric -> ~straight


def test_steer_sign_matches_bench_confirmed_fix():
    # The run-1 vs run-2 saga: open-right must drive the LEFT motor faster.
    d = Driver(armed=True)
    cmd = None
    for _ in range(60):
        cmd = d.step(OPEN_R, 18)
    assert cmd["left"] > cmd["right"], cmd


def test_current_stall_forces_backup_even_when_vision_says_clear():
    # The 20260609 wall sessions: vision pegged CLEAR while pinned at 3200 mA.
    d = Driver(armed=True)
    for _ in range(40):                      # cruise at normal current; must stay run
        cmd = d.step(CLEAR, 50, current=550)
    assert cmd["mode"] == "run"
    # Pin it: vision still CLEAR, current slams up. Must hit backup within ~3s.
    cmd, t = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.5, current=3000)
    assert cmd is not None, "never detected the stall"
    assert cmd["stalled"] and "STALLED" in cmd["reason"], cmd
    # The smoothed command reverses within the short MOVE_TAU window, not instantly.
    for _ in range(int(0.8 / DT)):
        cmd = d.step(CLEAR, 50, current=3000)
        if cmd["mode"] != "backup":
            break
    assert cmd["throttle"] < 0 and cmd["left"] < 0, cmd   # actually reversing


def test_no_stall_false_positive_at_cruise_current():
    d = Driver(armed=True)
    cmd = None
    for _ in range(160):                     # 10s of normal cruise, spiky-free
        cmd = d.step(CLEAR, 50, current=600)
    assert cmd["mode"] == "run", cmd


def test_single_current_spike_does_not_trigger():
    # One 3200 sample between normal readings (prop bite, spin-up) must ride through.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, current=550)
    for _ in range(8):                       # one 0.5s telemetry sample ~ 8 frames
        d.step(CLEAR, 50, current=3200)
    cmd = None
    for _ in range(40):
        cmd = d.step(CLEAR, 50, current=550)
    assert cmd["mode"] == "run", cmd


def test_motion_stall_triggers_without_current():
    # No current sensor: frame-motion ~1 (pinned) while commanding forward and
    # vision claims clear must still force a backup.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, motion=30.0)       # moving normally
    cmd, t = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.0, motion=1.0)
    assert cmd is not None, "motion stall never detected"


def test_stall_snaps_to_reverse_immediately():
    # Session 171347 bug: the EMA eased out of full-forward, so the first beat of
    # "backup" still commanded +50 into the wall at stall current. On the trigger
    # frame the smoothed throttle must already be <= 0 and reversing next frame.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, current=550)
    cmd, _ = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.5, current=3000)
    assert cmd is not None
    assert cmd["throttle"] <= 0.0, cmd          # no forward shove in backup, ever
    cmd = d.step(CLEAR, 50, current=600)
    assert cmd["throttle"] < -0.1, cmd          # committed reverse within one frame


def test_backup_holds_reverse_until_motion_shows_movement():
    # A weak reverse that doesn't move the boat (motion ~2) must extend past the
    # minimum backup time; once motion shows real movement it may exit.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, current=550, motion=30)
    cmd, _ = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.5, current=3000, motion=30)
    assert cmd is not None
    t_backup = d.t
    # pinned: motion stays ~2. Must still be backing up past the minimum duration.
    while d.t - t_backup < A.CTL_BACKUP_SEC + 0.5:
        cmd = d.step(CLEAR, 50, current=600, motion=2.0)
    assert cmd["mode"] == "backup", cmd
    # now it breaks free: motion jumps, backup may end
    for _ in range(int(1.0 / DT)):
        cmd = d.step(CLEAR, 50, current=600, motion=30.0)
        if cmd["mode"] != "backup":
            break
    assert cmd["mode"] == "pivot", cmd


def test_backup_gives_up_at_hard_cap_when_wedged():
    # Truly wedged (motion never rises): the reverse must not run forever.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, current=550, motion=30)
    cmd, _ = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.5, current=3000, motion=30)
    assert cmd is not None
    t_backup = d.t
    while d.t - t_backup < A.CTL_BACKUP_MAX_SEC + 0.5:
        cmd = d.step(CLEAR, 50, current=600, motion=2.0)
        if cmd["mode"] != "backup":
            break
    assert cmd["mode"] == "pivot", cmd
    assert d.t - t_backup <= A.CTL_BACKUP_MAX_SEC + 0.5


def test_pivot_grinding_at_stall_current_backs_up():
    # Session 171347: pivots scraped the wall at 3000-3200 mA while partially
    # rotating, so the gyro yaw-stuck check never fired. Sustained stall current
    # during a pivot must force a backup (yaw is healthy here, current is not).
    d = Driver(armed=True)
    cmd = d.step(OPEN_L, 7, yaw=0.5, current=550)       # assess -> pivot
    assert cmd["mode"] == "pivot"
    cmd, _ = d.run_until(OPEN_L, 7, 0.5, "backup", max_s=3.0, current=3000)
    assert cmd is not None, "grinding pivot never backed up"
    assert "STALLED" in cmd["reason"], cmd


def test_tof_near_blocks_even_when_vision_says_clear():
    # The whole point: vision pegs CLEAR at this pool's walls; a 0.3m range reading
    # must stop the approach anyway. Sides look open -> expect a pivot, not a ram.
    d = Driver(armed=True)
    for _ in range(20):
        cmd = d.step(CLEAR, 50, rng=2.5)
    assert cmd["mode"] == "run"
    cmd = d.step(CLEAR, 50, rng=0.3)
    assert cmd["mode"] in ("pivot", "backup"), cmd
    assert "rangefinder" in cmd["reason"] or cmd["mode"] == "backup", cmd


def test_tof_slow_zone_caps_cruise():
    d = Driver(armed=True)
    for _ in range(60):
        cmd = d.step(CLEAR, 50, rng=3.0)
    full = cmd["throttle"]
    for _ in range(60):
        cmd = d.step(CLEAR, 50, rng=0.9)        # inside slow zone, above block
    assert cmd["mode"] == "run"
    assert cmd["throttle"] <= A.CTL_TOF_SLOW_THROTTLE + 0.02 < full, (full, cmd)


def test_tof_hysteresis_no_chatter_at_threshold():
    # Dithering right at the block threshold must not flap run<->pivot: once
    # latched blocked, it stays blocked until range exceeds CTL_TOF_CLEAR_M.
    d = Driver(armed=True)
    for _ in range(20):
        d.step(CLEAR, 50, rng=2.0)
    d.step(CLEAR, 50, rng=0.45)                  # latch blocked
    modes = set()
    for rng in (0.52, 0.48, 0.55, 0.49, 0.6, 0.51):   # all below CLEAR_M=0.7
        modes.add(d.step(CLEAR, 50, rng=rng)["mode"])
    assert "run" not in modes, modes             # still latched
    for _ in range(30):
        cmd = d.step(CLEAR, 50, rng=1.0)         # genuinely clear
    assert cmd["mode"] == "run", cmd


def test_no_tof_means_unchanged_behavior():
    # Sensor absent (None): controller must behave exactly as before.
    d = Driver(armed=True)
    cmd = None
    for _ in range(60):
        cmd = d.step(CLEAR, 50, current=550)
    assert cmd["mode"] == "run" and cmd["throttle"] > 0.8, cmd


def test_post_stall_pivot_ignores_clear_vision_for_a_while():
    # After a stall backup, vision's "clear" got us here; the pivot must hold for
    # at least CTL_STALL_PIVOT_SEC before trusting it again.
    d = Driver(armed=True)
    for _ in range(40):
        d.step(CLEAR, 50, current=550)
    cmd, _ = d.run_until(CLEAR, 50, 0.0, "backup", max_s=3.5, current=3000)
    assert cmd is not None
    # ride out the backup into pivot (current back to normal once reversing free)
    cmd, t_pivot = d.run_until(CLEAR, 50, 0.5, "pivot", max_s=A.CTL_BACKUP_SEC + 1.0, current=500)
    assert cmd is not None
    # vision says CLEAR the whole time; must stay pivoting until the forced window ends
    held = 0.0
    while held < A.CTL_STALL_PIVOT_SEC - 2 * DT:
        cmd = d.step(CLEAR, 50, yaw=0.5, current=500)
        held = d.t - t_pivot
        assert cmd["mode"] == "pivot", ("left pivot early at %.2fs" % held, cmd)
    # and shortly after the window it may return to run
    cmd, _ = d.run_until(CLEAR, 50, 0.5, "run", max_s=1.0, current=500)
    assert cmd is not None, "never returned to run after forced pivot"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except AssertionError as e:
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:
            print("ERROR", fn.__name__, "->", repr(e))
    print("\n%d/%d passed" % (passed, len(fns)))
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)

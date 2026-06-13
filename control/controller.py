"""AutoBoat reactive obstacle-avoidance controller.

Pure decision logic with no hardware dependencies (only numpy + stdlib), so it
imports and runs anywhere: unit tests, telemetry replay, an eventual headless
control process. The dashboard owns the camera, sensors, motors, and HTTP
server; each frame it feeds the vision result to Controller.step() and then
actuates / displays the returned command dict.

Modes: assess -> run / pivot / backup.
  assess  - just started (Start pressed). Look first, decide before moving, so the
            boat never charges forward into whatever it happened to be pointed at.
  run     - path ahead clear: cruise and steer toward the opening.
  pivot   - blocked ahead but a side is open: tank-turn in place (motors counter-
            rotate) toward the opening until the path clears.
  backup  - boxed in (nothing ahead, no open side) or pinned: reverse to gain
            standoff, then pivot toward the most open side.

All tuning is the CTL_* constants below. The Controller owns its own state and a
lock (step is called from the camera thread, reset from the HTTP thread), so the
class has no module-global mutable state and tests can spin up fresh instances.
"""
import math
import time
import threading

import numpy as np

CTL_TURN_TAU = 0.45        # steering smoothing time constant (s) in run mode. dt-derived
                           # EMA weight, so behavior is the same at 16fps or 2fps. Lower
                           # than before for a snappier, tighter turn-in.
CTL_MOVE_TAU = 0.18        # smoothing for pivot/backup: short, so a tank turn or a
                           # reverse commits quickly instead of easing in over half a second.
CTL_TURN_DWELL = 0.4       # min seconds to hold a turn/straight decision (anti-chatter)
CTL_DEADBAND = 0.08        # |turn| below this snaps to straight (no twitch)

CTL_BLOCKED_PCT = 12.0     # center water below this => path ahead is blocked
CTL_CLEAR_PCT = 22.0       # center water at/above this => open enough to run
CTL_SIDE_OPEN_PCT = 20.0   # a zone this open means there's somewhere to pivot to.
                           # blocked ahead + a side this open => pivot; blocked with
                           # no side this open => boxed in => back up.
CTL_COMFORT_PCT = 38.0     # this open AND the field flat => hold straight, stop
CTL_COMFORT_SPREAD = 0.18  #   chasing marginally-deeper edges (the weaving fix)

CTL_TURN_GAIN = 1.7        # openness centroid [-1,1] -> turn command. Higher = tighter
                           # turns; combined with the slowdown below, a strong steer
                           # drops the inside motor through zero into a tank turn.
CTL_STEER_SIGN = -1.0      # motor steering polarity (bench-confirmed). Flip (+1/-1) if
                           # the boat steers toward obstacles instead of away.
CTL_TURN_ENTER = 0.22      # break into a turn when the opening is this off-center
CTL_TURN_EXIT = 0.10       # once turning, hold it until the opening recenters below this
CTL_CRUISE = 1.0           # forward throttle when the path is clear (full ahead)
CTL_TURN_SLOWDOWN = 1.5    # forward backs off this much per unit |turn|. >1 means a hard
                           # steer takes the inside motor below zero: an arc becomes a
                           # true tank turn for the tightest avoids.
CTL_MIN_THROTTLE = 0.0     # no forward floor, so run-mode steering can pivot fully in place

CTL_PIVOT_TURN = 0.9       # tank-turn magnitude while pivoting toward the opening
# A pivot is a tank turn in place: given time it sweeps every heading, so the
# policy is to KEEP PIVOTING until a free path is actually found (session 192948).
# Vision impatience ("still looks boxed/blocked") is not a reason to back up;
# backup is reserved for PHYSICAL evidence the rotation isn't working (gyro says
# no yaw, current says grinding) plus a long positional cap.
CTL_PIVOT_MAX_SEC = 10.0   # pivoted more than a full sweep without finding a path =>
                           # back up once to change position, then resume scanning
CTL_PIVOT_EXIT_CLEAR_SEC = 0.4   # path must read clear this long (consecutively)
                           # before leaving the pivot; a single noisy clear frame was
                           # flickering pivot->run->pivot at frame rate
CTL_TURN_MIN_DEG = 90.0    # when the ToF is the authority, a turn is COMMITTED: once
                           # blocked, tank-turn until heading has swept at least this
                           # far (gyro-integrated), then resume forward regardless of
                           # what vision claims mid-turn. A corner just turns again.
CTL_TURN_MAX_SEC = 6.0     # safety cap on a committed turn if yaw integration stalls
CTL_REVERSE = 1.0          # reverse throttle intent while backing up. Full: props are far
                           # weaker in reverse and the boat starts from a dead stop
                           # against the wall (0.6 measurably failed to move it).
CTL_BACKUP_SEC = 1.4       # minimum backup duration
CTL_BACKUP_MAX_SEC = 3.5   # extend the reverse up to this long if the boat hasn't
                           # actually started moving yet (frame-motion check below)
CTL_BACKUP_FREE_MOTION = 12.0   # frame-motion at/above this during backup = actually
                           # moving; below it the reverse hasn't freed the boat yet
                           # (failed backups measured 1-12, free movement 15+)
CTL_BACKUP_TURN = 0.0      # turn bias while reversing. 0 = straight back (predictable);
                           # reverse flips effective steering, so leave 0 unless bench-tuned.
CTL_YAW_MIN = 0.15         # rad/s; pivoting but yawing less than this...
CTL_YAW_STUCK_SEC = 1.5    #   ...for this long counts as pinned => back up (armed only)

# Forward-stall detection. Vision is an appearance classifier, not a rangefinder:
# a smooth wall that matches the water (measured on session_20260609_*: pinned at
# the coping with every zone maxed and center pegged "clear") is invisible to it.
# These two motion-truth signals catch the boat physically not going anywhere while
# commanding forward, and force a backup regardless of what vision claims.
#   Current: pool cruise draws ~450-650 mA; a prop-loaded stall slams 1800-3200 mA
#   (DRV8833 current-limit territory). A dt-aware EMA over the bursty 2 Hz samples
#   rides through single spikes and fires ~1.5-2 s into a real stall (validated
#   against both wall-test sessions: catches every pinned interval, no cruise fires).
#   Motion: mean |frame difference| vs a ~0.5 s-old frame. Pinned reads ~1, moving
#   never measured below 15 (same sessions), so <8 sustained is unambiguous.
CTL_STALL_MA = 1400        # ABSOLUTE backstop: current EMA above this is treated as a
                           # stall no matter the baseline. Kept as a hard ceiling, but
                           # the primary trigger below is relative, so a shifted DC
                           # baseline (e.g. a sense-wiring offset) can't peg this.
CTL_STALL_TAU = 1.5        # EMA time constant (s) for the responsive current signal
CTL_STALL_BASE_TAU = 8.0   # slow baseline EMA: tracks where cruise current SITS, so a
                           # stall is detected as a jump above recent normal rather than
                           # an absolute number. Long enough that a real multi-second
                           # grind rises above it before the baseline catches up.
CTL_STALL_RISE_MA = 700    # primary trigger: responsive EMA exceeding the slow baseline
                           # by this much = a real load spike (a wall grind pulls ~2-3x
                           # cruise). Works whether cruise is 550 mA or offset to 1600.
CTL_STALL_THR = 0.4        # run-mode stall checks apply only above this commanded throttle
CTL_MOTION_MIN = 8.0       # frame-motion below this while driving forward...
CTL_MOTION_SEC = 1.5       #   ...for this long => stalled. 0 disables the motion check.
CTL_STALL_PIVOT_SEC = 2.2  # after a stall backup, pivot at least this long before
                           # trusting vision again (it just said "clear" into a wall)

# Forward time-of-flight (VL53L1X). The camera infers distance from appearance and
# this pool's walls are appearance-camouflaged (proven across 8 sessions and 3 ML
# approaches); the ToF measures it. A single forward beam, so it gates the path
# AHEAD; vision still owns left/right steering. None = sensor absent: behavior is
# unchanged, so the controller runs identically until the hardware exists.
CTL_TOF_BLOCK_M = 0.50     # range below this => path ahead is blocked, full stop on
                           # vision's opinion. Sized to current speeds and ~1s of
                           # decision latency; raise it if approaches still feel hot.
CTL_TOF_CLEAR_M = 0.70     # hysteresis: ahead only counts as clear again above this,
                           # so a reading dithering at the threshold can't chatter
CTL_TOF_SLOW_M = 1.00      # within this range, cap cruise to the slow throttle below
CTL_TOF_SLOW_THROTTLE = 0.45   # approach speed inside the slow zone


def params():
    """Controller tuning as a flat dict, for the session manifest. Mirrors the
    CTL_* constants so a recording records exactly which gains produced its data."""
    return {
        "turn_tau": CTL_TURN_TAU, "move_tau": CTL_MOVE_TAU,
        "turn_dwell": CTL_TURN_DWELL, "deadband": CTL_DEADBAND,
        "blocked_pct": CTL_BLOCKED_PCT, "clear_pct": CTL_CLEAR_PCT,
        "side_open_pct": CTL_SIDE_OPEN_PCT,
        "comfort_pct": CTL_COMFORT_PCT, "comfort_spread": CTL_COMFORT_SPREAD,
        "turn_gain": CTL_TURN_GAIN, "turn_enter": CTL_TURN_ENTER,
        "turn_exit": CTL_TURN_EXIT, "cruise": CTL_CRUISE,
        "turn_slowdown": CTL_TURN_SLOWDOWN,
        "min_throttle": CTL_MIN_THROTTLE, "pivot_turn": CTL_PIVOT_TURN,
        "pivot_max_sec": CTL_PIVOT_MAX_SEC,
        "pivot_exit_clear_sec": CTL_PIVOT_EXIT_CLEAR_SEC, "reverse": CTL_REVERSE,
        "backup_max_sec": CTL_BACKUP_MAX_SEC, "backup_free_motion": CTL_BACKUP_FREE_MOTION,
        "stall_ma": CTL_STALL_MA, "stall_tau": CTL_STALL_TAU,
        "stall_base_tau": CTL_STALL_BASE_TAU, "stall_rise_ma": CTL_STALL_RISE_MA,
        "stall_thr": CTL_STALL_THR, "motion_min": CTL_MOTION_MIN,
        "motion_sec": CTL_MOTION_SEC, "stall_pivot_sec": CTL_STALL_PIVOT_SEC,
        "tof_block_m": CTL_TOF_BLOCK_M, "tof_clear_m": CTL_TOF_CLEAR_M,
        "tof_slow_m": CTL_TOF_SLOW_M, "tof_slow_throttle": CTL_TOF_SLOW_THROTTLE,
        "turn_min_deg": CTL_TURN_MIN_DEG, "turn_max_sec": CTL_TURN_MAX_SEC,
        "backup_sec": CTL_BACKUP_SEC, "backup_turn": CTL_BACKUP_TURN,
    }


def _clamp(x, lo=-1.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def analyze_zones(zones, depths=None):
    """Pure. Produce the steering signal and a flatness measure.

    The steering centroid is computed from the full per-column depth curve when
    `depths` is given (it is in normal operation), not from the 5 zone medians.
    The full curve is a better-conditioned heading: it weighs the whole opening
    profile instead of letting one of five coarse buckets dominate, which on real
    runs cuts the raw frame-to-frame steering jitter by ~20-25% before smoothing.
    Weights are squared so the boat commits to the dominant opening rather than
    averaging two gaps into the obstacle between them, and because the positions
    are symmetric about 0, a uniform/saturated field yields centroid 0 (straight).
    `spread` (flatness) stays on the zone medians, which are robust to single
    spiked columns; the comfort gate uses it to decide the field is uniform.
    Falls back to the zone centroid if no curve is supplied.
    """
    z = [max(0.0, float(v)) for v in zones]
    n = len(z)
    if n == 0:
        return {"centroid": 0.0, "zmax": 0.0, "spread": 0.0}
    zmax = max(z)
    if zmax <= 0.0:
        return {"centroid": 0.0, "zmax": zmax, "spread": 0.0}
    spread = (zmax - min(z)) / zmax

    if depths is not None and len(depths) > 1:
        d = np.asarray(depths, dtype=np.float64)
        w = d * d
        wsum = w.sum()
        if wsum > 0:
            m = len(d)
            pos = -1.0 + 2.0 * np.arange(m) / (m - 1)   # column positions on [-1,1]
            centroid = float((pos * w).sum() / wsum)
        else:
            centroid = 0.0
    else:                                                # fallback: zone centroid
        pos = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
        w = [v * v for v in z]
        wsum = sum(w)
        centroid = sum(p * wi for p, wi in zip(pos, w)) / wsum if wsum > 0 else 0.0
    return {"centroid": centroid, "zmax": zmax, "spread": spread}


class Controller:
    """Stateful reactive avoidance controller. One instance owns its state and a
    lock. Feed it vision each frame via step(); reset() it on Start."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = self._initial_state()

    @staticmethod
    def _initial_state():
        return {"mode": "assess", "turn": 0.0, "throttle": 0.0, "left": 0.0,
                "right": 0.0, "reason": "init", "pivot_dir": 1, "turning": False,
                "pivot_since": None, "backup_until": None, "backup_hard_until": None,
                "yaw_low_since": None, "last_t": None, "turn_t": 0.0,
                "cur_ema": 0.0, "cur_base": 0.0, "turn_swept": 0.0, "motion_low_since": None, "min_pivot_until": 0.0,
                "stall_pending": False, "stall_src": None, "tof_blocked": False, "pivot_clear_since": None}

    def reset(self, now=None):
        """Reset into `assess` and zero motion/timers. Called on Start so a run
        begins by reading the surroundings rather than resuming stale motion."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            s = self.state
            s["mode"] = "assess"
            s["turn"] = 0.0
            s["throttle"] = 0.0
            s["left"] = 0.0
            s["right"] = 0.0
            s["turning"] = False
            s["pivot_since"] = None
            s["backup_until"] = None
            s["backup_hard_until"] = None
            s["yaw_low_since"] = None
            s["turn_t"] = now
            s["last_t"] = None
            s["cur_ema"] = 0.0
            s["cur_base"] = 0.0
            s["turn_swept"] = 0.0
            s["motion_low_since"] = None
            s["min_pivot_until"] = 0.0
            s["stall_pending"] = False
            s["stall_src"] = None
            s["tof_blocked"] = False
            s["pivot_clear_since"] = None
            s["reason"] = "assessing what's ahead"

    def snapshot(self):
        """Thread-safe copy of the current command, for display without a step."""
        with self._lock:
            s = self.state
            return {"mode": s["mode"], "turn": round(s["turn"], 3),
                    "throttle": round(s["throttle"], 3), "left": round(s["left"], 3),
                    "right": round(s["right"], 3), "reason": s["reason"]}

    def step(self, zones, center, yaw_rate, now, armed=False, depths=None,
             current_ma=None, motion=None, range_m=None):
        """Advance one frame. Returns the command dict
        (mode/turn/throttle/left/right/reason/armed/stalled). `center` is
        center_depth_pct, `yaw_rate` is rad/s (used for the pinned check while
        running), `armed` gates the motion-truth checks (they need real motion to
        be meaningful), `depths` is the full per-column water-depth curve used for
        the steering centroid. `current_ma` is total motor-side current and
        `motion` is the mean |frame difference| vs a ~0.5s-old frame; both are
        optional stall signals (None disables that check). `range_m` is the
        forward time-of-flight distance in meters (None = sensor absent): it
        gates the path AHEAD over vision's opinion, with hysteresis, because the
        camera provably cannot range this pool's walls.

        Modes assess -> run / pivot / backup. `assess` is entered on reset() so
        the boat reads what's ahead before it moves. `pivot` is a tank turn in
        place toward the opening. `backup` reverses to gain standoff when boxed
        in, pinned, or physically stalled against an obstacle vision can't see,
        then pivots."""
        a = analyze_zones(zones, depths)
        centroid = a["centroid"]
        spread = a["spread"]
        zmax = a["zmax"]
        # Vision's opinion of the path ahead. When the ToF is present it is the
        # obstacle authority (it can range this pool's walls; the camera can't), so
        # vision is demoted to STEERING only and these no longer trigger avoidance.
        v_blocked = center < CTL_BLOCKED_PCT
        v_clear = center >= CTL_CLEAR_PCT
        side_open = zmax >= CTL_SIDE_OPEN_PCT      # somewhere off-center looks open

        with self._lock:
            s = self.state
            # ---- ToF forward gate ----
            tof_active = range_m is not None        # sensor present and fresh
            tof_near = False
            if tof_active:
                if s["tof_blocked"]:
                    tof_near = range_m < CTL_TOF_CLEAR_M     # latched: need real clearance
                else:
                    tof_near = range_m < CTL_TOF_BLOCK_M
                s["tof_blocked"] = tof_near

            if tof_active:
                # ToF is the authority: go forward until IT says block. Vision does
                # not block or clear the path, it only steers (centroid below).
                blocked = tof_near
                clear = not tof_near
            else:
                # No sensor: fall back to the original vision-only behavior.
                blocked = v_blocked
                clear = v_clear
            boxed = blocked and not side_open      # nothing ahead and no clear side

            mode = s["mode"]
            open_dir = 1 if centroid >= 0 else -1   # +1 = opening to the right

            def enter_backup():
                s["backup_until"] = now + CTL_BACKUP_SEC
                s["backup_hard_until"] = now + CTL_BACKUP_MAX_SEC
                s["pivot_dir"] = open_dir
                s["pivot_since"] = None
                s["yaw_low_since"] = None
                s["motion_low_since"] = None

            def enter_pivot():
                s["pivot_since"] = now
                s["pivot_dir"] = open_dir
                s["yaw_low_since"] = None
                s["pivot_clear_since"] = None
                s["turn_swept"] = 0.0          # start the committed-turn heading count
                s["cur_ema"] = 0.0; s["cur_base"] = 0.0   # fresh stall watch for the pivot

            # ---- stall detection (motion truth vs vision) ----
            # The current EMA warms only while run/pivot is actively driving, so
            # backup current (reversing off a wall is legitimately heavy) never
            # poisons the next mode's stall watch. Triggers while armed in:
            #   run   - commanding real forward throttle: current EMA or no-visual-
            #           motion means we're pushing an unseen wall.
            #   pivot - current EMA only: a tank turn grinding a prop against the
            #           wall pegs the limiter even while partially rotating, which
            #           the gyro yaw-stuck check alone misses.
            driving_fwd = (armed and mode == "run" and s["throttle"] > CTL_STALL_THR)
            pivoting = (armed and mode == "pivot")
            if current_ma is not None and (driving_fwd or pivoting):
                dt_c = (now - s["last_t"]) if s["last_t"] is not None else 0.0625
                dt_c = min(max(dt_c, 0.005), 0.5)
                # Seed both EMAs to the first sample of a fresh watch (both 0) so
                # neither ramps up from zero and reads a spurious rise on entry.
                if s["cur_base"] <= 0.0:
                    # Seed the responsive EMA to the live sample, but cap the baseline
                    # seed at the absolute stall line: if a watch happens to begin
                    # while already grinding (e.g. straight into a pivot against the
                    # wall), the baseline must not seed high and mask the stall.
                    s["cur_ema"] = float(current_ma)
                    s["cur_base"] = min(float(current_ma), CTL_STALL_MA)
                else:
                    a_c = dt_c / (CTL_STALL_TAU + dt_c)
                    s["cur_ema"] += a_c * (float(current_ma) - s["cur_ema"])
                    a_b = dt_c / (CTL_STALL_BASE_TAU + dt_c)
                    s["cur_base"] += a_b * (float(current_ma) - s["cur_base"])
            # Stall = a real RISE above the recent baseline (robust to a shifted DC
            # offset). The absolute ceiling is kept only as a backstop AND lifted by
            # the baseline, so a flat-but-offset cruise (session 191057's 1625 mA)
            # can't peg it; only a genuine excursion above normal does.
            rise = s["cur_ema"] - s["cur_base"]
            abs_ceiling = max(CTL_STALL_MA, s["cur_base"] + CTL_STALL_RISE_MA)
            stall_cur = ((driving_fwd or pivoting)
                         and (rise > CTL_STALL_RISE_MA or s["cur_ema"] > abs_ceiling))
            stall_mot = False
            if driving_fwd and motion is not None and CTL_MOTION_MIN > 0:
                if motion < CTL_MOTION_MIN:
                    if s["motion_low_since"] is None:
                        s["motion_low_since"] = now
                    stall_mot = (now - s["motion_low_since"]) > CTL_MOTION_SEC
                else:
                    s["motion_low_since"] = None
            else:
                s["motion_low_since"] = None
            stalled = stall_cur or stall_mot

            # ---- transitions ----
            if stalled and mode in ("run", "pivot"):
                # Physically pinned/grinding while vision says go: back off hard, and
                # don't trust vision's "clear" again until we've pivoted well away.
                enter_backup(); mode = "backup"
                s["stall_pending"] = True
                s["stall_src"] = "high current" if stall_cur else "no visual motion"
                s["cur_ema"] = 0.0; s["cur_base"] = 0.0
                # Snap the smoothed command to zero so the reverse starts NOW. The
                # EMA otherwise eases out of full-forward, spending the first beat
                # of "backup" still shoving the boat into the wall at stall current
                # (visible in session 171347 as backup rows commanding +50 forward).
                s["turn"] = 0.0
                s["throttle"] = 0.0
            elif mode == "assess":
                # First look after Start: decide before moving.
                if clear:
                    mode = "run"
                elif boxed and not tof_active:
                    enter_backup(); mode = "backup"   # vision-only: nowhere to go
                elif blocked:
                    enter_pivot(); mode = "pivot"     # ToF-blocked or vision-blocked
                else:
                    mode = "run"                    # marginal but not blocked: ease forward
            elif mode == "backup":
                min_done = not (s["backup_until"] is not None and now < s["backup_until"])
                hard_done = not (s["backup_hard_until"] is not None and now < s["backup_hard_until"])
                # Reverse for at least the minimum, then keep reversing until the
                # frame-motion shows the boat actually moving (a 21%-duty reverse from
                # a dead stop against the wall measurably went nowhere), up to the
                # hard cap so a truly wedged boat can't reverse forever.
                freed = (motion is None or motion >= CTL_BACKUP_FREE_MOTION)
                if min_done and (freed or hard_done):
                    # reverse finished: pivot toward whatever side looks most open now
                    enter_pivot(); mode = "pivot"
                    if s["stall_pending"]:
                        # vision just drove us into a wall; force a real heading change
                        s["min_pivot_until"] = now + CTL_STALL_PIVOT_SEC
                        s["stall_pending"] = False
            elif mode == "pivot":
                pivoting_for = (now - s["pivot_since"]) if s["pivot_since"] is not None else 0.0
                past_forced = now >= s["min_pivot_until"]
                # Integrate heading swept during this pivot (gyro). dt clamped so a
                # stale frame can't dump a huge step.
                if s["last_t"] is not None:
                    dt_y = min(max(now - s["last_t"], 0.0), 0.5)
                    s["turn_swept"] += abs(yaw_rate) * dt_y
                swept_deg = math.degrees(s["turn_swept"])

                if tof_active:
                    # COMMITTED TURN: ToF is the authority. Hold the tank turn until
                    # we've swept the minimum heading, ignoring vision entirely; then
                    # resume forward. If still blocked after the turn, run will
                    # immediately re-enter pivot (a corner = another 90).
                    turn_done = swept_deg >= CTL_TURN_MIN_DEG
                    timed_out = pivoting_for > CTL_TURN_MAX_SEC
                    if turn_done or timed_out:
                        mode = "run"
                        s["pivot_since"] = None
                        s["yaw_low_since"] = None
                        s["pivot_clear_since"] = None
                        s["cur_ema"] = 0.0; s["cur_base"] = 0.0
                    else:
                        # only escalate to backup if physically not rotating
                        yaw_stuck = False
                        if armed:
                            if abs(yaw_rate) < CTL_YAW_MIN:
                                if s["yaw_low_since"] is None:
                                    s["yaw_low_since"] = now
                                yaw_stuck = (now - s["yaw_low_since"]) > CTL_YAW_STUCK_SEC
                            else:
                                s["yaw_low_since"] = None
                        if yaw_stuck:
                            enter_backup(); mode = "backup"
                        # else: hold the committed direction. Do NOT recompute
                        # pivot_dir from vision here: near a wall the camera's idea of
                        # "open side" flickers frame to frame, which flipped the tank
                        # turn back and forth so the boat rocked in place and never
                        # swept its 90deg (session 193807: 7s pivot, only 25deg). The
                        # direction was chosen once at enter_pivot and stays put.
                else:
                    # No ToF: original vision-driven persistence. Exit to run only on
                    # SUSTAINED clear (a single noisy clear frame was flickering
                    # pivot->run->pivot at frame rate, session 192948).
                    if clear and past_forced:
                        if s["pivot_clear_since"] is None:
                            s["pivot_clear_since"] = now
                    else:
                        s["pivot_clear_since"] = None
                    if (s["pivot_clear_since"] is not None
                            and (now - s["pivot_clear_since"]) >= CTL_PIVOT_EXIT_CLEAR_SEC):
                        mode = "run"
                        s["pivot_since"] = None
                        s["yaw_low_since"] = None
                        s["pivot_clear_since"] = None
                        s["cur_ema"] = 0.0; s["cur_base"] = 0.0
                    else:
                        # Keep pivoting until a free path appears. Vision still reading
                        # boxed/blocked is EXPECTED mid-rotation and never aborts the
                        # scan; escalate to backup only on physical evidence the
                        # rotation isn't working (no yaw / grinding) or a full sweep.
                        yaw_stuck = False
                        if armed:
                            if abs(yaw_rate) < CTL_YAW_MIN:
                                if s["yaw_low_since"] is None:
                                    s["yaw_low_since"] = now
                                yaw_stuck = (now - s["yaw_low_since"]) > CTL_YAW_STUCK_SEC
                            else:
                                s["yaw_low_since"] = None
                        forced_extra = 0.0
                        if s["pivot_since"] is not None:
                            forced_extra = max(0.0, s["min_pivot_until"] - s["pivot_since"])
                        full_sweep = pivoting_for > (CTL_PIVOT_MAX_SEC + forced_extra)
                        if full_sweep or yaw_stuck:
                            enter_backup(); mode = "backup"
                        else:
                            s["pivot_dir"] = open_dir   # keep re-aiming at the open side
            else:  # run
                if blocked:
                    if boxed and not tof_active:
                        enter_backup(); mode = "backup"   # vision-only, walled in
                    else:
                        enter_pivot(); mode = "pivot"     # ToF block => committed turn
                        if tof_active:
                            # Snap to an in-place tank turn now; don't let the throttle
                            # EMA ease down from cruise and carry the bow into the wall
                            # during the first beat of the turn.
                            s["turn"] = 0.0
                            s["throttle"] = 0.0

            s["mode"] = mode

            # ---- act on the (possibly updated) mode ----
            snappy = False        # short time constant for decisive moves (pivot/backup)
            cause = None
            if mode == "backup":
                turn_raw, thr_raw = CTL_BACKUP_TURN * s["pivot_dir"], -CTL_REVERSE
                snappy = True
                if s["stall_pending"]:
                    reason = "backup: STALLED against unseen obstacle (%s)" % s.get("stall_src", "?")
                elif motion is not None and motion < CTL_BACKUP_FREE_MOTION:
                    reason = "backup: still pinned, holding reverse"
                else:
                    reason = "backup: reversing to clear the wall"
            elif mode == "pivot":
                d = s["pivot_dir"]
                turn_raw, thr_raw = CTL_PIVOT_TURN * d, 0.0     # pure tank turn, motors counter-rotate
                snappy = True
                reason = "pivot: tank-turning " + ("right" if d > 0 else "left") + " toward opening"
                if tof_near:
                    reason += " (rangefinder: wall ahead)"
            elif mode == "assess":
                turn_raw, thr_raw = 0.0, 0.0
                reason = "assessing what's ahead"
            else:  # run
                turn_target = _clamp(CTL_TURN_GAIN * centroid)
                comfort = (center >= CTL_COMFORT_PCT and spread < CTL_COMFORT_SPREAD)
                if comfort:
                    turn_target = 0.0
                # Turn hysteresis with a dwell floor: only break into a turn when the
                # opening is clearly off-center (ENTER); once turning, keep it until it
                # recenters (EXIT). The dwell forbids flipping faster than CTL_TURN_DWELL.
                if (now - s["turn_t"]) >= CTL_TURN_DWELL:
                    if s["turning"]:
                        if abs(turn_target) < CTL_TURN_EXIT:
                            s["turning"] = False
                            s["turn_t"] = now
                    elif abs(turn_target) > CTL_TURN_ENTER:
                        s["turning"] = True
                        s["turn_t"] = now
                if s["turning"]:
                    turn_raw = turn_target
                    cause = "steer"
                else:
                    turn_raw = 0.0
                    cause = "comfort" if comfort else "cruise"
                reason = ""
                # Full ahead when straight; back off hard with the turn so a sharp avoid
                # tightens to a pivot instead of arcing into the wall it is dodging.
                thr_raw = max(CTL_MIN_THROTTLE,
                              CTL_CRUISE * (1.0 - CTL_TURN_SLOWDOWN * abs(turn_raw)))
                # ToF approach zone: something solid is genuinely ahead within the
                # slow radius, so close at a speed the block threshold can stop.
                if range_m is not None and range_m < CTL_TOF_SLOW_M:
                    thr_raw = min(thr_raw, CTL_TOF_SLOW_THROTTLE)
                    if cause == "cruise":
                        cause = "tof_slow"

            # ---- per-mode smoothing (time constant), then tank mix ----
            last_t = s["last_t"]
            dt = (now - last_t) if last_t is not None else (1.0 / 16.0)
            dt = min(max(dt, 0.005), 0.5)          # clamp first call and any stall
            s["last_t"] = now
            tau = CTL_MOVE_TAU if snappy else CTL_TURN_TAU
            a_s = dt / (tau + dt)                   # EMA weight from dt and tau
            turn = s["turn"] + a_s * (turn_raw - s["turn"])
            throttle = s["throttle"] + a_s * (thr_raw - s["throttle"])
            steer = CTL_STEER_SIGN * turn
            left = _clamp(throttle - steer)         # tank mix; clamps allow a motor to reverse
            right = _clamp(throttle + steer)        #   for a true tank turn when |steer|>throttle
            if mode == "run":
                settled = abs(turn) < CTL_DEADBAND
                if cause == "comfort":
                    reason = "run: open and flat, " + ("straight" if settled else "easing straight")
                elif cause == "cruise":
                    reason = "run: clear ahead" if settled else "run: easing straight"
                elif cause == "tof_slow":
                    reason = "run: obstacle %.1fm ahead, slowing" % range_m
                else:
                    reason = "run: steering toward opening"
            s["turn"] = turn
            s["throttle"] = throttle
            s["left"] = left
            s["right"] = right
            s["reason"] = reason
            return {"mode": mode, "turn": round(turn, 3), "throttle": round(throttle, 3),
                    "left": round(left, 3), "right": round(right, 3),
                    "reason": reason, "armed": armed, "stalled": bool(stalled)}

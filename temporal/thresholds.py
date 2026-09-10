"""Every runtime constant in the segmenter, in one place, with its provenance.

No number here is a guess. Each default is a percentile of a signal measured over the
committed archive (RandomForest/data/, 24 held signs, 2,378 usable frames, 157 s of signing
at ~15 fps), reproduced by temporal/calibrate.py. Values marked NEEDS-GESTURE-DATA cannot be
derived from held signs and must be recalibrated once J/Z footage exists -- they are the only
guesses in the system and they are labelled as such.

Units are SECONDS and PALM-WIDTHS throughout, never frames or pixels. Frame rate is not
constant on a webcam, and pixel distances change with camera distance; both would make these
constants camera-specific. In palm-seconds they transfer.
"""
from dataclasses import dataclass, asdict, fields
import json


@dataclass
class Thresholds:
    # --- stillness / motion, palm-widths per second -------------------------------------
    # Measured on the archive's held signs: p95 = 0.841, p99 = 1.254, max = 2.028.
    V_STILL: float = 0.85           # p95 of held-sign speed: below this the hand is "parked"
    V_MOVE_ARMED: float = 1.30      # p99; low is safe because a rising edge is also required
    V_MOVE_ARMED_SUSTAIN: float = 0.13   # seconds the crossing must persist (jitter rejection)
    V_MOVE_UNARMED: float = 2.10    # archive max 2.028; zero false starts at this value
    V_SMOOTH_WINDOW: float = 0.33   # SECONDS of smoothing behind v_bar. Was a 5-SAMPLE average,
                                    # which is a different amount of smoothing at every frame
                                    # rate: 0.33 s on the 15 fps archive these thresholds were
                                    # calibrated on, but 0.17 s at 30 fps. The under-smoothed
                                    # signal oscillates across V_MOVE_ARMED and never holds it
                                    # for V_MOVE_ARMED_SUSTAIN, so no track ever starts. Measured
                                    # on 30 fps footage: 46-81% of frames above the threshold,
                                    # zero sustained rising edges. 0.33 s reproduces the archive
                                    # calibration exactly while making v_bar frame-rate invariant.

    # --- shape stability, palm units, vs the TRAILING 0.4 s median shape ----------------
    # Measured on the archive: p95 = 0.120, p99 = 0.204.
    SHAPE_STABLE: float = 0.14      # p95, rounded up: below this the handshape is settled
    SHAPE_WINDOW: float = 0.40      # the trailing window sigma is measured against
    RIGID_VETO: float = 1.35        # CALIBRATED on 60 real J gestures: p95 of sigma_max, per the
                                    # spec rule. The old 0.45 was extrapolated from held signs and
                                    # rejected 90% of genuine gestures -- it was the single largest
                                    # cause of missed J's, and it failed silently.
                                    #
                                    # Why it was so far out: shape42 is deliberately NOT
                                    # rotation-normalized, because orientation is what separates P
                                    # from K and H from U. A J hook rotates the wrist, so sigma
                                    # rises from the hand TURNING, not from the handshape changing,
                                    # and the veto conflates the two. 1.35 still sits 3.8x above
                                    # the worst held sign in the archive (0.35), so it keeps
                                    # rejecting stable-shape transitions. The principled fix is a
                                    # rotation-invariant rigidity measure (align shapes before
                                    # comparing); this threshold is the empirical stand-in.

    # --- gate arming --------------------------------------------------------------------
    GATE_ARM_FRAC: float = 0.50     # gate must hold on this fraction of the pre-onset window
    GATE_ARM_WINDOW: float = 0.40   # seconds of pre-onset history examined
    # J_GATE passes 100/100 held-I frames and 0/2278 of every other letter (aspect-corrected),
    # so 0.50 is pure margin for a hand still settling into the launch pose.

    # --- event acceptance ---------------------------------------------------------------
    T_MIN: float = 0.30             # NEEDS-GESTURE-DATA. J/Z run 0.5-1.5 s; band widened 0.2 s
    T_MAX: float = 1.80
    L_MIN: float = 1.50             # palm units. Archive's held-sign path over any 1.2 s
                                    # window maxes at 1.295, so this excludes every hold.
    L_MAX: float = 12.0             # rejects a reach across the desk
    STRAIGHT_VETO: float = 0.90     # |net| / path. Transport is ~0.95+; neither J nor Z is straight.

    # --- classifier operating point -----------------------------------------------------
    P_EMIT: float = 0.70            # NEEDS-GESTURE-DATA. Sweep on S1 GroupKFold only.
    MARGIN: float = 0.25            # winner must beat runner-up by this much

    # --- static branch ------------------------------------------------------------------
    VOTE_WINDOW: float = 0.50       # seconds of votes before a static letter may be emitted
    VOTE_AGREE: float = 0.80        # fraction of votes that must agree
    VOTE_PROB: float = 0.60         # mean winner probability; this is the unknown-rejection floor
    HOLD_SETTLE: float = 0.30       # seconds of stillness required to enter HOLD
    D_WAIT: float = 0.35            # I and D are deferred this long, because they are also the
                                    # launch poses for J and Z. Without this every J emits "IJ".

    # --- housekeeping -------------------------------------------------------------------
    COOLDOWN_MOTION: float = 0.60   # after emitting J or Z
    COOLDOWN_STATIC: float = 0.50   # after emitting a static letter
    GAP_INTERP: float = 0.20        # detection gaps up to this are interpolated
    GAP_RESET: float = 0.50         # no hand for this long clears the buffer and last_emitted
    BUFFER: float = 4.0             # seconds of history; must exceed
                                    # T_MAX + lead-in + fall-confirm + arming window
    FALL_CONFIRM: float = 0.15      # seconds below V_STILL that ends a track

    # --- tier 2 (co-articulated gestures with no preceding pause) -----------------------
    TIER2_ENABLED: bool = False     # ships disabled; enable only once its false-fire rate on
                                    # held-out negative footage is measured below 1/min
    TIER2_GATE_FRAC: float = 0.70
    TIER2_P: float = 0.85
    TIER2_STRIDE: int = 5

    def __post_init__(self):
        need = self.T_MAX + 0.25 + self.FALL_CONFIRM + self.GATE_ARM_WINDOW
        if self.BUFFER <= need:
            raise ValueError(f"BUFFER={self.BUFFER}s cannot hold a maximum-length episode "
                             f"({need:.2f}s needed)")
        if not (self.V_STILL < self.V_MOVE_ARMED < self.V_MOVE_UNARMED):
            raise ValueError("speed thresholds must be strictly ordered")

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown thresholds in {path}: {sorted(unknown)}")
        return cls(**data)


#: Constants that cannot be derived from held-sign footage and are currently educated guesses.
NEEDS_GESTURE_DATA = ("RIGID_VETO", "T_MIN", "T_MAX", "P_EMIT", "MARGIN", "V_SMOOTH_WINDOW")

# V_SMOOTH_WINDOW is on that list for a reason worth stating. 0.33 s is what the archive
# effectively used at 15 fps, so it reproduces the calibration exactly -- but held signs cannot
# say whether it is right for a GESTURE. Smoothing suppresses the jitter that causes spurious
# triggers, and also blunts the onset of a real one: raising it from an effective 0.17 s to
# 0.33 s on 19 minutes of third-party video took candidate events from 15 to 11, recovering a Z
# that had been undetectable while losing several J candidates. Both directions are real. The
# window and V_MOVE_ARMED must be swept together against labelled J/Z footage, choosing the pair
# that maximises recall at a fixed false-fire rate. Until then this is the frame-rate-invariant
# choice, not the tuned one.

DEFAULT = Thresholds()

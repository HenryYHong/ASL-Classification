"""Every runtime constant in the segmenter, in one place, with its provenance.

No number here is a guess. Each default is a percentile of a signal measured over the
committed archive (RandomForest/data/, 24 held signs, 2,378 usable frames, 157 s of signing
at ~15 fps), reproduced by temporal/calibrate.py. Values marked NEEDS-GESTURE-DATA cannot be
derived from held signs and must be recalibrated once J/Z footage exists -- they are the only
guesses in the system and they are labeled as such.

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
    V_FALL: float = 1.30            # "decelerated" -- see the two-tier ending below.
    LEAD_IN: float = 0.25           # seconds of history prepended to a track. v_bar is smoothed
                                    # over V_SMOOTH_WINDOW, so by the time the rise is confirmed
                                    # the stroke is already underway; without this the recorded
                                    # path starts late and misses the opening of the gesture.
                                    # BUFFER was always sized to allow it; it was never applied.
    REQUIRE_PARKED: bool = True     # the arming window must contain a genuinely PARKED frame
                                    # (v_bar < V_STILL), not merely one below V_MOVE_ARMED. The
                                    # design always said "parked, then moving", but only the
                                    # second half was enforced, so tracks fired during ordinary
                                    # fingerspelling transitions -- and a running track blocks
                                    # the static branch, which is why other letters degraded.
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
    RIGID_VETO: float = 1.15        # CALIBRATED on 90 real J/Z gestures: p95 of ROTATION-ALIGNED
                                    # sigma_max. Measured on the same gestures the plain measure
                                    # needed 1.35, so aligning buys a tighter bound -- though far
                                    # less than expected: aligned p95 1.13 against plain 1.20,
                                    # meaning the deviation during a real gesture is genuine
                                    # finger movement and motion-blur noise, NOT wrist rotation.
                                    # The rotation hypothesis was tested and largely refuted.
                                    # Held signs score at most 0.35 under the aligned measure.
                                    # (superseded note) p95 of sigma_max, per the
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
    T_MIN: float = 0.30             # J/Z run 0.5-1.5 s; band widened 0.2 s below
    T_MAX: float = 2.10             # MEASURED. 1.80 was guessed before any gesture existed and
                                    # then sat BELOW the training data's own Z durations, whose
                                    # p95 is 1.81 and max 1.83 -- the cap was clipping real Z's
                                    # out of the distribution the classifier was fitted on, and
                                    # live Z attempts aborted at 1.81 s and 1.83 s. Trained MOVE
                                    # events top out at 1.81 s, so a longer window admits no new
                                    # kind of junk; the length and straightness vetoes still run.
    L_MIN: float = 1.50             # palm units. Archive's held-sign path over any 1.2 s
                                    # window maxes at 1.295, so this excludes every hold.
    L_MAX: float = 20.0             # MEASURED, not assumed. 12.0 was a guess at "a reach across
                                    # the desk" and it silently killed 5 of 12 genuine Z gestures
                                    # in live use, whose paths ran 12.9-14.8 palm -- this signer's
                                    # Z is simply large. The straightness veto is what actually
                                    # rejects transport (a reach is 0.9+ straight; a Z is 0.2-0.45),
                                    # so the length ceiling only needs to exclude the absurd.
                                    # Training MOVE events top out at 7.6 palm, so raising this
                                    # admits nothing new to the classifier.
    STRAIGHT_VETO: float = 0.90     # |net| / path. Transport is ~0.95+; neither J nor Z is straight.

    # --- classifier operating point -----------------------------------------------------
    P_EMIT: float = 0.55            # MEASURED by sweeping out-of-fold probabilities, split by
                                    # clip, over 182 real events. 0.70 was the pre-data guess and
                                    # it silently dropped roughly one genuine gesture in three
                                    # (J+Z recall 0.69). 0.55 raises recall to 0.76 for two false
                                    # fires in 55 held-out MOVE events; below 0.55 buys nothing.
                                    # The signer reports missed letters, not spurious ones, so
                                    # recall is the side worth paying for here.
    MARGIN: float = 0.20            # winner must beat runner-up by this much

    # --- static branch ------------------------------------------------------------------
    # Latency budget for one static letter is HOLD_SETTLE + VOTE_WINDOW, and every 0.1s here is
    # 0.1s the signer waits. Swept against 90 recorded still-hand phases: 0.30/0.50/0.80 gave a
    # median 1.22s and emitted on 30 of them; 0.15/0.25/0.70 gives 0.83s and emits on 39 -- both
    # faster and more willing. Shortening further backfires (0.10/0.20/0.65 falls back to 1.20s)
    # because too few votes land inside the window to reach agreement at all.
    VOTE_WINDOW: float = 0.30       # seconds of votes before a static letter may be emitted
    VOTE_MIN: int = 4               # ...or this many votes, whichever completes first. Without
                                    # the fallback the window is frame-rate fragile: at 15 fps
                                    # four votes span 0.199s (too short to satisfy a 0.9x0.25s
                                    # span test) and a fifth pushes the first out of the window,
                                    # so the span can never land in the required band and the
                                    # vote NEVER completes. It happened to fit at 30 fps, so the
                                    # bug was invisible live and cost 8 of 24 archive letters.
    VOTE_AGREE: float = 0.70        # fraction of votes that must agree
    VOTE_PROB: float = 0.70         # mean winner probability: the unknown-rejection floor.
                                    # Raised from 0.35 once a SECOND capture session made the
                                    # model genuinely confident on this signer. Measured over a
                                    # live alphabet run, correct emissions scored 0.72-0.99 while
                                    # every transitional misfire scored 0.36-0.65, so the floor
                                    # now separates them cleanly. 0.35 was the right value for a
                                    # one-session model that was never confident about anything;
                                    # keeping it after fixing the data would have been tuning
                                    # around a problem that no longer exists.
                                    # (superseded note follows)
                                    # -- was 0.35:
                                    # MEASURED from 856 live holds, not carried over from the
                                    # benchmark. The forest is trained on near-duplicate frames
                                    # from one capture burst, so it is badly over-confident
                                    # in-session -- mean winner probability 0.82 there against
                                    # 0.44 live. A floor set on the in-session scale rejected
                                    # 98% of live holds: S won 65 holds at a mean 0.35 and U won
                                    # 34 at 0.33, both recognized correctly and both silently
                                    # discarded. At 0.35 with the margin below, 60% of live holds
                                    # emit and 21 of the 24 letters become reachable.
                                    # The right long-term fix is calibrating the probabilities
                                    # (or a second capture session), not a lower floor.
    VOTE_MARGIN: float = 0.10       # the winner must always beat the runner-up by at least this
    # A letter may emit by EITHER route, because absolute probability and margin measure
    # different things and gating only on the first was rejecting correct answers:
    #   confident   mean probability >= VOTE_PROB
    #   decisive    margin >= VOTE_MARGIN_CLEAR and probability >= VOTE_PROB_FLOOR
    # Measured on live holds: D was classified correctly on 100 of 100 frames yet peaked at 0.41
    # probability, because the forest splits its mass across D/X/C -- no absolute floor could
    # ever emit it. Its margin over the runner-up was 0.14, while genuine junk (a hand mid
    # transition) won by 0.05-0.07. The margin separates them where the probability cannot.
    VOTE_MARGIN_CLEAR: float = 0.12
    VOTE_PROB_FLOOR: float = 0.30   # even a decisive winner needs this much mass to be an answer
    HOLD_SETTLE: float = 0.25       # seconds of stillness required to enter HOLD. 0.15 was too
                                    # permissive once a second session raised confidence: a brief
                                    # pause while moving between letters counted as a hold, and
                                    # a hand in transit genuinely passes through other letters'
                                    # shapes. Measured live, moving B->C emitted D,D,D,O,D before
                                    # the real C ever landed.
    D_WAIT: float = 0.35            # I and D are deferred this long, because they are also the
                                    # launch poses for J and Z. Without this every J emits "IJ".

    # --- housekeeping -------------------------------------------------------------------
    COOLDOWN_MOTION: float = 0.60   # after emitting J or Z
    COOLDOWN_STATIC: float = 0.80   # after emitting a static letter. Longer than feels natural
                                    # because duplicate suppression keys on the LAST letter, and a
                                    # prediction flickering D->O->D defeats it entirely -- each
                                    # differs from its predecessor. The cooldown is what actually
                                    # stops one held letter producing a stream of emissions.
    GAP_INTERP: float = 0.30        # detection gaps up to this are interpolated. 0.20 aborted
                                    # 4 of 12 live Z gestures at gaps of 0.20-0.25s: MediaPipe
                                    # loses a fast hand to motion blur mid-stroke, and a Z is the
                                    # fastest thing it is asked to follow. Interpolating ~9 frames
                                    # of a smooth arc is safer than discarding the gesture.
    GAP_RESET: float = 0.50         # no hand for this long clears the buffer and last_emitted
    BUFFER: float = 4.0             # seconds of history; must exceed
                                    # T_MAX + lead-in + fall-confirm + arming window
    # A track ends on EITHER tier, because the two natural ways to finish a gesture look
    # nothing alike to a speed signal:
    #   STOPPED   the hand comes to rest and stays there. Confirmed quickly, because a full
    #             stop is unambiguous and the path is already complete.
    #   SLOWED    the hand decelerates but keeps drifting. Needs a LONGER confirmation, because
    #             a Z reverses direction twice and its speed dips at every corner -- cutting on
    #             the first dip truncates the gesture mid-stroke and it misclassifies.
    # Ending only on STOPPED forces the signer to freeze after every gesture. Ending only on
    # SLOWED truncates anyone who decelerates smoothly into a hold. Both were reported.
    FALL_CONFIRM: float = 0.15      # seconds below V_STILL (stopped) that ends a track
    FALL_CONFIRM_SLOW: float = 0.35  # seconds below V_FALL (merely slowed) that ends a track

    # --- words: segmentation and the dictionary hint ------------------------------------
    SPACE_GAP: float = 1.20         # seconds with no hand in frame before a word break is
                                    # inserted. MEASURED against the committed recordings, which
                                    # separate cleanly into two populations: while a hand is up
                                    # and tracked, consecutive frames sit 0.041s apart at the
                                    # median and 0.076s at p99, and the longest dropout across 13
                                    # minutes of continuous recording is 0.996s; deliberate
                                    # hand-down rests between prompts start at 1.008s and cluster
                                    # at 1.5-5s. 1.20 clears every observed dropout and still
                                    # falls under the shortest rest anyone actually took.
    WORD_MIN_RATIO: float = 0.02    # a dictionary word is offered only if the frames make it at
                                    # least this likely relative to the letters actually read.
                                    # The emitted string is the per-position argmax, so it always
                                    # scores highest; this asks how far behind a real word is
                                    # allowed to be before the hint is worth showing.
    WORD_DOMINANCE: float = 10.0    # ...and only if that word is this many times likelier than
                                    # the next candidate. With 150k words in the list, most
                                    # letter strings have some same-length neighbour, and an
                                    # ambiguous field should abstain rather than guess.

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

#: The dictionary hint's two constants. SPACE_GAP is measured; these are not, because nothing in
#: the repository is a recording of somebody spelling a word. Sweeping them needs a session of
#: real words with their intended spellings written down -- the same discipline that retired the
#: pre-data guesses at P_EMIT and T_MAX, both of which were wrong in the direction that loses
#: letters silently.
NEEDS_WORD_DATA = ("WORD_MIN_RATIO", "WORD_DOMINANCE")

# V_SMOOTH_WINDOW is on that list for a reason worth stating. 0.33 s is what the archive
# effectively used at 15 fps, so it reproduces the calibration exactly -- but held signs cannot
# say whether it is right for a GESTURE. Smoothing suppresses the jitter that causes spurious
# triggers, and also blunts the onset of a real one: raising it from an effective 0.17 s to
# 0.33 s on 19 minutes of third-party video took candidate events from 15 to 11, recovering a Z
# that had been undetectable while losing several J candidates. Both directions are real. The
# window and V_MOVE_ARMED must be swept together against labeled J/Z footage, choosing the pair
# that maximizes recall at a fixed false-fire rate. Until then this is the frame-rate-invariant
# choice, not the tuned one.

DEFAULT = Thresholds()

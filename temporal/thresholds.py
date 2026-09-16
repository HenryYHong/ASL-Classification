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
from dataclasses import dataclass, asdict, fields, replace
import json


@dataclass
class Thresholds:
    # --- stillness / motion, palm-widths per second -------------------------------------
    # Measured on the archive's held signs with the segmenter's own v_bar (calibrate.py
    # --offline, whose arithmetic is now the segmenter's): p95 = 0.859, p99 = 1.297,
    # max = 2.231. The earlier 0.841 / 1.254 / 2.028 were of a differently smoothed signal
    # (one step more per window) that the thresholds are never applied to.
    V_STILL: float = 0.85           # held-sign v_bar p95 is 0.859; 0.85 sits 0.01 under it and
                                    # is kept: replaying the archive, all 24 held signs still
                                    # park and emit exactly their letter, and the S2-S4 holds
                                    # 47/47. Below this the hand is "parked".
    V_MOVE_ARMED: float = 1.30      # p99 (1.297); low is safe because a rising edge is also
                                    # required
    V_MOVE_ARMED_SUSTAIN: float = 0.13   # seconds the crossing must persist (jitter rejection)
    V_MOVE_UNARMED: float = 2.10    # archive max 2.028 (calibrate.py's v_bar; the segmenter's
                                    # own v_bar peaks at 2.231 on the same frames, see
                                    # calibrate.py). NOT READ BY THE STATE MACHINE: every
                                    # track starts on the armed rising edge, and the "unarmed"
                                    # start the name implies never existed -- the one test
                                    # that referenced it was implied by V_STILL and was
                                    # removed. Retained because the overlay (live_demo.py,
                                    # docs/app.js) draws it as the red tick on the speed meter,
                                    # calibrate.py measures it, and the ordering check below
                                    # keeps the three ticks in a sensible order.
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
    # J_GATE at features.J_THUMB_MAX 1.30 passes 100/100 held-I frames of the archive and 13 of
    # the 2,278 frames of every other letter (aspect-corrected) -- all 13 are Y (13 of Y's 100
    # frames), and replaying the whole 157 s archive through the segmenter starts 0 tracks
    # from them, because arming also needs a parked frame and a sustained rising edge. So
    # 0.50 is margin for a hand still settling into the launch pose. A Y hold followed by
    # motion could arm a track no MOVE example resembles; no such footage exists yet.

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
    P_EMIT: float = 0.55            # MEASURED by sweeping out-of-fold probabilities. Re-swept on
                                    # the current label set (one credited gesture per prompted
                                    # item, GroupKFold(5) by item over the S1 takes: 102 events
                                    # in 80 items, 28 runtime-reachable MOVE): J+Z recall at the
                                    # operating point / false J on MOVE = 0.955 / 3 at 0.40,
                                    # 0.940 / 2 at 0.50-0.60, 0.925 / 2 at 0.65, 0.910 / 1 at
                                    # 0.70, 0.851 / 1 at 0.80; the shipped forest at 0.55 reads
                                    # 0.946 / 2 of 28. 0.55 stays. History: 0.70 was the
                                    # pre-data guess and on the first (fragment-labeled) 182
                                    # events it silently dropped roughly one genuine gesture in
                                    # three; the signer reports missed letters, not spurious
                                    # ones, so recall is the side worth paying for here.
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
    VOTE_PROB: float = 0.70         # mean winner probability, the "confident" route. UNCHANGED
                                    # and, at the VOTE_MARGIN_CLEAR / VOTE_PROB_FLOOR below,
                                    # inoperative: a mean winner probability of 0.70 leaves at
                                    # most 0.30 for the runner-up, so its margin is at least
                                    # 0.40 and the decisive route (margin >= 0.20, probability
                                    # >= 0.50) already admits it. 0.70 and 0.80 give identical
                                    # positive and negative tables at every grid point of the
                                    # sweep (vote-thresholds/sweep_tables.txt), i.e. it decided
                                    # nothing. Kept so the route still exists if
                                    # VOTE_MARGIN_CLEAR is ever raised past 0.40 or the floor
                                    # past 0.70. The history below explains the value.
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
                                    # (not swept)
    # A letter may emit by EITHER route, because absolute probability and margin measure
    # different things and gating only on the first was rejecting correct answers:
    #   confident   mean probability >= VOTE_PROB
    #   decisive    margin >= VOTE_MARGIN_CLEAR and probability >= VOTE_PROB_FLOOR
    # Measured on live holds: D was classified correctly on 100 of 100 frames yet peaked at 0.41
    # probability, because the forest splits its mass across D/X/C -- no absolute floor could
    # ever emit it. Its margin over the runner-up was 0.14, while genuine junk (a hand mid
    # transition) won by 0.05-0.07. The margin separates them where the probability cannot.
    #
    # The two values below were re-measured for the jitter-trained 112-D forest that ships
    # (static/v4, sigma 0.12 x4, strong-rules filter, RF100 min_samples_leaf 5) by
    # vote-thresholds/{02_votesim,03_negatives,04_tables,05_replay}.py. They are NOT right for
    # the previous rotation-augmented RF400: on that forest the floor was harmless on idle (it
    # emitted on 2 of the same 43 idle holds at 0.30) but cost 14 points of first-vote correct.
    #
    # THE CLEAN-IDLE METRIC the floor is set against. docs/browser_log.jsonl holds 2,890 hold
    # records from one ~2-minute live session of one signer (Tasks landmarks, the page's own
    # segmenter). Consecutive records with gaps <= 0.2 s form 148 holds. 79 of them the page
    # emitted on. Of the 69 it never emitted on, 26 sit within 2.5 s before or 1 s after a
    # J/Z track record: those are the I/D LAUNCH POSES, read correctly at mean probabilities
    # up to 1.00 and canceled by D_WAIT or by the motion emission, so they are excluded --
    # counting them would call a correct read of an I a false emission. The remaining 43 are
    # the clean idle holds: 2,180 records, 2,064 four-frame votes of a relaxed hand the page
    # was right to stay silent on (2,058 sliding four-record votes, plus 6 holds with fewer
    # than 4 records -- sizes 1, 1, 2, 2, 2, 3 -- that are one vote each).
    # The motion recordings' REST windows are NOT used as negatives: the signer re-forms the
    # next item's launch I/D within about 1 s of every rest, so a vote there is mostly a
    # correct I. The legacy single-frame rate over all 2,890 records is not this metric; it
    # counts the page's own 79 emissions (a 2.7% floor) and the 26 launch-adjacent holds.
    #
    # HOW THE TWO VALUES WERE SET, in two rounds on two draws of the same recipe (the jitter
    # RNG is seeded by block content, static_aug.content_rng, so a re-implementation is a
    # different draw). Round one, the design sweep on the candidate draw, chose 0.20 / 0.50:
    # clean idle 0 of 43 holds, 0 of 2,064 votes (the previous forest at its own 0.12 / 0.30:
    # 2 of 43), the floor 0.04 above that draw's most confident clean-idle vote (max 0.46, p90
    # 0.36, of the votes that pass agree and margin), 63 of the 79 live page-emitted holds
    # still emitting. Round two re-ran the same gate on the COMMITTED model_static.p before
    # shipping, per the rule that any clean-idle hold emitting raises the floor by 0.05: at
    # 0.50 this draw emitted on 1 of 43 -- six Q votes on one relaxed hold, max mean
    # probability 0.509, 0.009 over the floor -- so the floor is 0.55. At 0.55: 0 of 43 holds,
    # 0 of 2,064 votes, 0.041 above this forest's most confident clean-idle vote (sweep on
    # this draw: 0.40 -> 3/43, 0.45 -> 1/43, 0.50 -> 1/43, 0.55 -> 0/43, 0.60 -> 0/43). The
    # floor sits within 0.05 of a relaxed hand on both draws, so a different pose or camera
    # could cross it: re-measure after the next live session.
    #
    # Cost, on the 71 held-out holds of the four sessions (leave-one-session-out fold models,
    # 4-frame first vote). Candidate draw at 0.20 / 0.50: correct 0.817 / wrong 0.056,
    # IDENTICAL to that forest at the old 0.12 / 0.30 -- the idle emissions all sat below 0.50
    # and the correct first votes all above it (min 0.53, p5 0.58); its end-to-end replay
    # (real timestamps, one fresh Segmenter per hold) gave exactly the right letter 64/71
    # (0.901; old rule 63/71), first emission wrong 4/71, silent 1/71, cross-day fold 18/24.
    # Committed draw, the numbers that describe what ships: first vote correct 57 -> 56 of 71
    # for 0.50 -> 0.55, wrong 2/71 and silent 2/71 unchanged, first emitted letter right 65/71
    # unchanged; end-to-end replay at 0.55 exactly the right letter 65/71 (0.915), first
    # emission wrong 3/71 (2/71 at 0.50), silent 2/71 either way (S1-E and S2-K; K is silent
    # under every model), letter among emissions 66/71, 0 track starts, median first-emission
    # latency 0.46 s, p90 0.96 s, cross-day fold (S1, 24 holds) exact 19/24. Of the 79 holds
    # the page emitted on live, 59 (0.747) still emit at 0.55 (62 at 0.50 with this draw);
    # live Tasks-landmark confidences run lower than leave-one-session-out ones, and that
    # 59/79 is the number to watch.
    #
    # Alternatives measured and rejected (candidate draw). FLOOR 0.40 leaves 3 of 43 idle
    # holds (0.070) emitting; 0.60 costs about 10 points of first-vote correct (0.817 -> 0.718
    # at 0.12 / 0.60, 0.704 at 0.40 / 0.60) and raises silent holds in the replay (1/71 ->
    # 5/71) for nothing more on idle. MARGIN_CLEAR 0.30 costs correct first votes (0.817 ->
    # 0.803) and buys nothing on an idle hand already at zero; 0.20 rather than 0.12 costs
    # this forest nothing and removes one wrong first vote in 71 for the 101-D jitter
    # forests. A unanimity route (agree == 1.0 over >= 8 votes, margin >= VOTE_MARGIN) was
    # rejected: at 24 fps the 0.30 s window holds 6-7 votes so it almost never fires, and
    # where it can it adds nothing on the 71 holds (0.817 / 0.817) while raising non-launch
    # REST emissions 0.012 -> 0.061 per vote and 0.025 -> 0.083 per hold -- a unanimous
    # low-probability vote is exactly a relaxed hand read consistently as the wrong letter.
    VOTE_MARGIN_CLEAR: float = 0.20 # MEASURED, was 0.12 (see above)
    VOTE_PROB_FLOOR: float = 0.55   # MEASURED, was 0.30, then 0.50 on the candidate draw and
                                    # 0.55 on the committed one (see above); the knob that
                                    # separates a held letter from an idle hand under the
                                    # jitter-trained forest. The margin does not. The word
                                    # layer's constants below were swept at 0.50 and only
                                    # the point configuration was re-measured at 0.55.
    HOLD_SETTLE: float = 0.25       # seconds of stillness required to enter HOLD. 0.15 was too
                                    # permissive once a second session raised confidence: a brief
                                    # pause while moving between letters counted as a hold, and
                                    # a hand in transit genuinely passes through other letters'
                                    # shapes. Measured live, moving B->C emitted D,D,D,O,D before
                                    # the real C ever landed.
    D_WAIT: float = 0.35            # I and D are deferred this long, because they are also the
                                    # launch poses for J and Z. Without this every J emits "IJ".
                                    # The timer alone was not enough: a J's track needs the
                                    # rise, V_MOVE_ARMED_SUSTAIN and a fall-confirm, never
                                    # inside 0.35 s, so the parked I was released while the
                                    # track was in flight (38 releases, 0 cancels on the
                                    # committed takes). A parked I/D whose own launch pose arms
                                    # a track is now HELD until that track resolves: a J/Z
                                    # emission cancels it, an abort or an abstention releases
                                    # it. The timer decides only when no track is in flight
                                    # and no rise is being confirmed: a rise that began
                                    # inside D_WAIT but whose V_MOVE_ARMED_SUSTAIN (0.13 s)
                                    # had not elapsed at the deadline defers the release by
                                    # at most that sustain, so the arm decision -- not the
                                    # timer -- settles it (on the takes 2 of the 4 'IJ'
                                    # residues were exactly this: I voted at 111.20 s and
                                    # 117.40 s, rise at +0.30 / +0.32 s, arm at +0.44 /
                                    # +0.46 s). The remaining leading I's are I's held
                                    # 0.5-2.4 s before the stroke, which the timer releases
                                    # by design (the deferred-until-hold-exit alternative
                                    # was rejected for its latency: S1 I 0.92 -> 4.01 s).

    # --- housekeeping -------------------------------------------------------------------
    COOLDOWN_MOTION: float = 0.60   # after EMITTING J or Z (not after every scored track: set
                                    # before the abstention test, it silently blocked the
                                    # static branch for 0.6 s after most motion). It rate-limits
                                    # the static branch only; a second motion letter is not
                                    # held back by it, and the double J's seen on the takes were
                                    # 1.0-3.5 s apart, which the relabeled forest removed
                                    # (9 -> 0 doubles), not a cooldown.
    COOLDOWN_STATIC: float = 0.80   # after emitting a static letter. Longer than feels natural
                                    # because duplicate suppression keys on the LAST letter, and a
                                    # prediction flickering D->O->D defeats it entirely -- each
                                    # differs from its predecessor. The cooldown is what actually
                                    # stops one held letter producing a stream of emissions.
    GAP_INTERP: float = 0.30        # a detection gap LONGER than this while TRACKING aborts the
                                    # track (nothing is emitted rather than classify a half-seen
                                    # path). A shorter gap does nothing: the segmenter never
                                    # interpolates, the missing frames are simply absent from
                                    # the span and the event feature is computed over the
                                    # frames that exist (the arc-length resampling and the
                                    # time-windowed smoothing tolerate that). The name is
                                    # historical. 0.20 aborted 4 of 12 live Z gestures at gaps
                                    # of 0.20-0.25 s: MediaPipe loses a fast hand to motion
                                    # blur mid-stroke, and a Z is the fastest thing it is asked
                                    # to follow. Only train_motion.py --whole-clips (a
                                    # diagnostic, not the runtime path) interpolates gaps up to
                                    # this length.
    GAP_RESET: float = 0.50         # no hand for this long clears the buffer and last_emitted
    HAND_SWITCH_S: float = 0.50     # seconds MediaPipe's handedness label must disagree with
                                    # the one the segmenter is canonicalizing with before it
                                    # switches (Segmenter._latch_handedness). MEASURED on the
                                    # five committed prompted takes (13,692 detected frames,
                                    # 166 carrying the other label in 51 runs): every flip
                                    # inside an unbroken detection is short -- 32 of the 51
                                    # runs are one frame, 47 are under 0.25 s, the longest is
                                    # 5 frames / 0.20 s (take 2, mid-Z-stroke at 12 palm/s)
                                    # and the longest inside a parked launch pose 2 frames /
                                    # 0.06 s. The four runs of 0.73-1.63 s are a hand dropped
                                    # to the bottom edge of the frame between detection gaps
                                    # (every frame of the episode "Right", "None" on both
                                    # sides), i.e. a fresh episode the latch adopts on its
                                    # first frame, not a flip. 0.50 is 2.5x the longest flip
                                    # and equal to GAP_RESET, the drop-and-raise it takes to
                                    # change hands for real. Without the latch a one-frame
                                    # flip mirrored the hand mid-track, sigma_rigid crossed
                                    # RIGID_VETO and the gesture was lost: replaying the takes
                                    # with the raw per-frame label credited 79 of 113 items
                                    # against 85 with one modal label per take; with the latch
                                    # the per-frame replay credits 85 (the six recovered are
                                    # take 0 item 28, take 2 items 1/5/11/28, take 4 item 2).
    BUFFER: float = 4.0             # seconds of history; must exceed T_MAX + LEAD_IN +
                                    # FALL_CONFIRM_SLOW + GATE_ARM_WINDOW (the longest episode
                                    # the buffer has to hold in full; checked in __post_init__)
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
    WORD_MIN_RATIO: float = 0.10    # a dictionary word is offered only if the frames make it at
                                    # least this likely relative to the letters actually read.
                                    # The emitted string is the per-position argmax, so it always
                                    # scores highest; this asks how far behind a real word is
                                    # allowed to be before the hint is worth showing. Was 0.02,
                                    # tuned at the old 0.12 / 0.30 gate. 0.10 / 10 / 2.5 come
                                    # from a 112-configuration sweep of (min ratio, dominance,
                                    # prior) -- min ratio {0.02, 0.05, 0.1, 0.2} x dominance
                                    # {3, 5, 10, 20} x prior {0, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0},
                                    # which temporal/simulate_words.py --sweep runs by default,
                                    # so the committed script reproduces the table -- on
                                    # leave-one-session-out letter posteriors at 0.20 / 0.50,
                                    # before VOTE_PROB_FLOOR was raised to 0.55 for the
                                    # committed forest; the grid has not been re-run at 0.55.
                                    # No cell won every table (consecutive-window retries:
                                    # (0.10, 20, 3.0) and (0.20, 3, 3.0); fresh-hold retries:
                                    # (0.10, 5, 2.5)); (0.10, 10, 2.5) is within 0.015 utility
                                    # (recovered - C x shown-wrong, C in {2, 3}, under one
                                    # seed sd of ~0.01) of the best cell in every table and
                                    # was chosen as the robust compromise. At 0.20 / 0.50 the
                                    # letters that survive are more confident than at 0.12 /
                                    # 0.30, so the layer can demand a closer candidate.
    WORD_DOMINANCE: float = 10.0    # ...and only if that word is this many times likelier than
                                    # the next candidate, prior included. The list is now the
                                    # 34.7k most frequent Google Books words plus proper names
                                    # (was 150k Webster headwords), so fewer strings have a
                                    # same-length neighbor, but a word field that is still
                                    # ambiguous after the prior should abstain rather than
                                    # guess. Kept at 10: the sweep's per-table bests used 3,
                                    # 5 and 20, and 10 is within 0.015 of each (WORD_MIN_RATIO).
    WORD_PRIOR: float = 2.5         # weight of the rank prior: each candidate's log-likelihood
                                    # gets WORD_PRIOR * log-prior added, where the prior is its
                                    # position in the frequency-ordered list (the file order of
                                    # docs/words.txt IS the prior; it must never be sorted).
                                    # 0 would score every word equally; the sweep's table bests
                                    # sit at 2.5-3.0 (3.0 wins the consecutive tables, 2.5 the
                                    # fresh-hold ones) and 2.5 is the compromise (see
                                    # WORD_MIN_RATIO), favoring common words over rare ones
                                    # that happen to be a closer letter-by-letter fit.

    # --- tier 2 (co-articulated gestures with no preceding pause) -----------------------
    TIER2_ENABLED: bool = False     # ships disabled; enable only once its false-fire rate on
                                    # held-out negative footage is measured below 1/min
    TIER2_GATE_FRAC: float = 0.70
    TIER2_P: float = 0.85
    TIER2_STRIDE: int = 5

    def __post_init__(self):
        # The longest episode the buffer must hold in full: the arming window before the rise,
        # the lead-in, a track that runs to T_MAX and the slow-tier fall confirmation. The
        # check used to hard-code 0.25 for LEAD_IN and read the short FALL_CONFIRM, so raising
        # LEAD_IN past 1.35 s would have passed while the buffer could no longer hold a track.
        need = (self.GATE_ARM_WINDOW + self.LEAD_IN + self.T_MAX
                + max(self.FALL_CONFIRM, self.FALL_CONFIRM_SLOW))
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

#: The dictionary hint's constants (WORD_MIN_RATIO, WORD_DOMINANCE, WORD_PRIOR) are no longer
#: guesses: temporal/simulate_words.py spells 2,000 frequency-weighted common words, 600 proper
#: names and 1,000 rare words from random held-out holds of the leave-one-session-out letter
#: posteriors (one fresh Segmenter vote per letter, the vote gate above, 5 seeds) and scores
#: what the layer would show. At the shipped gate (VOTE_MARGIN_CLEAR 0.20 / VOTE_PROB_FLOOR
#: 0.55) and constants, on the committed forest's out-of-fold posteriors (make_oof.py, seed
#: 0): common words recovered 0.44 / shown wrong 0.07 when a misread letter is retried from
#: consecutive windows of the same hold (3 tries), and 0.63 / 0.06 when every retry is a
#: fresh hold (names 0.31/0.07 and 0.49/0.04, rare 0.18/0.05 and 0.34/0.03); the two retry
#: models bracket what a signer does. The constants were swept at 0.20 / 0.50, where the same
#: run gives 0.48 / 0.07 and 0.68 / 0.05: the floor raise costs the simulator's random-window
#: retries 0.04-0.06 recovered while the real-segmenter replay of the same 71 holds is
#: unchanged at 65/71 exact. That is a simulator of the segmenter, not a recording of spelled
#: words.
#: A session of real words with their intended spellings written down is still the missing
#: measurement -- the same discipline that retired the pre-data guesses at P_EMIT and T_MAX --
#: so nothing is listed here, and nothing about the word layer is claimed beyond the simulation.
NEEDS_WORD_DATA = ()

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

#: Numbers mode (the 10-class digit forest, static/v4) runs the same segmenter with two vote
#: constants raised. Kept OUT of the dataclass on purpose: a Thresholds json from calibrate.py
#: and the browser's REQUIRED_THRESHOLDS list must not change, so digits mode is
#: dataclasses.replace(DEFAULT, **DIGITS_OVERRIDES) at the point the mode is entered (live_demo.py
#: on --mode digits or the M key; docs/app.js on the toggle, from payload['digits'].thresholds).
#: Provenance (temporal/train_digits.py, the same checks re-run on the shipped static/v4
#: forest; the static/v3 candidate they were first measured on read 0.943 -> 0.267 and 0.953
#: -> 0.910 on the same rows and does not ship): the digit forest was trained on 218 signers of
#: a public photo set and never on this project's author, and at the letter gate of the
#: previous release (0.12 / 0.30) it emits a digit on 0.883 of the 2,890 logged hold records
#: of the author's letters-mode browser session (single-frame rate; any digit there is a false
#: one; three quarters of them '0', a quarter '1'). At 0.40 / 0.60 that falls to 0.162 (468
#: records; 0.27 on the holds where the letter forest itself emitted), split 57% '0' / 40%
#: '1', while the author's letters that share a handshape with a digit (O/V/W/F/B as
#: 0/2/6/9/4, the only proxy that exists for the author's digits; 0.956 of their frames read
#: as that digit) still emit correctly on 0.922 of their four-frame votes and wrongly on 0.000
#: (0.942 / 0.016 at 0.12 / 0.30). C, R, X and U also read as 0/2/1/2 confidently, which is
#: what makes the proxy weak evidence. One spurious digit per six idle holds is still high:
#: duplicate suppression bounds it to one per hand-raise, and the page says so. Nothing here
#: was verified on the author's hand signing a digit; recording those is the first follow-up.
DIGITS_OVERRIDES = {"VOTE_MARGIN_CLEAR": 0.40, "VOTE_PROB_FLOOR": 0.60}


def digits_thresholds(base=None):
    """The Thresholds numbers mode runs: `base` (DEFAULT when None) with DIGITS_OVERRIDES applied."""
    return replace(DEFAULT if base is None else base, **DIGITS_OVERRIDES)

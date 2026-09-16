"""The live-stream state machine: decides WHEN a letter has been signed.

Classifying a pre-cut clip of a J is easy. The actual problem is that a webcam feed has no
clip boundaries: 30 times a second this has to decide whether the hand is holding a static
letter, performing a J, performing a Z, or just moving between signs. That decision is this
file, and it is the part of the design that everything else serves.

Driven entirely by the timestamps passed into step(). Nothing reads a clock, so replaying
recorded footage through a Segmenter reproduces a live run exactly -- which is what lets the
same code auto-label training data and evaluate held-out streams. Every constant is in
seconds or palm widths with one exception: VOTE_MIN is a frame COUNT (the vote completes on
VOTE_MIN votes or on VOTE_WINDOW seconds, whichever comes first, thresholds.py says why), so a
static vote lands HOLD_SETTLE plus up to (VOTE_MIN - 1) / fps after the hold forms -- 0.05 s
at 60 fps, 0.10 s at 30, 0.20 s at 15 -- and anything clocked from the vote (D_WAIT delivery,
the pending-launch grace below) shifts by that much across frame rates.

    S0 NO_HAND   -> no detection for GAP_RESET; clears history and last_emitted
    S1 SETTLING  -> hand present but moving or reshaping. NOTHING is ever emitted here.
    S2 HOLD      -> parked and settled; the static branch votes here
    S3 TRACKING  -> a gesture is underway; the writing fingertip is being recorded

Two mechanisms deserve attention because they are what make the thing usable:

RISING EDGE. A track starts only if the hand was parked in a gate-passing launch pose and THEN
began to move. Incidental hand travel almost never begins from a parked, gate-passing pose, so
most motion never creates a scoring opportunity at all. That is what allows V_MOVE_ARMED to be
set low enough (1.30 palm/s) that a slow, deliberate J still triggers it.

EDGE-TRIGGERED STATIC EMISSION. A static letter is emitted when a stable hold is first
established, not on every frame where the classifier has an opinion. Holding A for ten seconds
emits one A. Signing a doubled letter means leaving the hold and coming back -- which is the
physical act anyway.

Two rules tie the branches together, because the launch pose of a motion letter is itself a
static letter (I for J, D for Z) and so is the pose it finishes in:

PENDING LAUNCH LETTER. A voted I or D is parked for D_WAIT. If a track arms from that same
pose while it is parked, the timer no longer decides: the letter is HELD until the track
resolves -- a J/Z emission cancels it, an abort or an abstention releases it. Before this the
parked I was released mid-track on every J ("IJ"). A rise that began inside D_WAIT but whose
V_MOVE_ARMED_SUSTAIN has not elapsed at the deadline is treated the same way for at most that
sustain (0.13 s): the timer waits for the arm decision instead of releasing an I whose J is
0.1 s from arming (on the takes 2 of the 4 remaining "IJ" were exactly this). An I held
longer than D_WAIT before the stroke is still released by the timer, by design: deferring
I and D to hold exit was measured and rejected for its latency (S1 I 0.92 -> 4.01 s).

POST-MOTION SETTLING. After a J/Z emission the static branch stays silent while the hand rests
in the finishing pose. The suppression clears on a reshape (sigma > SHAPE_STABLE) at any time,
or on the hand moving again (v_bar > V_STILL) after it has once been still; a hand that morphs
straight into the next letter is therefore not delayed. The vote reports why it did not emit
("post_motion") like every other verdict, from one predicate (_vote_verdict).

HANDEDNESS LATCH. features.canonicalize_handedness mirrors a "Left" hand onto the right-hand
convention, and MediaPipe assigns that label per frame. A single frame labeled the other way
mid-track mirrors the hand for that frame alone, sigma_rigid jumps past RIGID_VETO and the
track aborts -- 166 of the 13,692 detected frames on the committed takes carry the other
label, and replaying with the raw per-frame label lost 6 of the 113 prompted gestures that the
modal label per take (which training uses) credits. So step() keeps the label it has and
switches only once the other label has persisted for HAND_SWITCH_S (thresholds.py: 0.50 s;
the longest flip inside an unbroken detection on the takes is 5 frames / 0.20 s). The latch
is cleared with the rest of the history on GAP_RESET, which is the only way a signer changes
hands anyway, so the first frame after a gap adopts its own label. A detected frame with no
real label keeps the latched one. Training-side harvest (label_events.py) still feeds one
modal label per recording; the latch makes the live path agree with it on every frame that
is not a genuine change of hand.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import features as F
from thresholds import Thresholds, DEFAULT


NO_HAND, SETTLING, HOLD, TRACKING = "NO_HAND", "SETTLING", "HOLD", "TRACKING"

#: The two handshapes that are also the launch poses for the motion letters. Emitting these
#: immediately would turn every J into "IJ" and every Z into "DZ".
DEFERRED = {"I", "D"}

#: The static letter each motion letter launches from. A parked I is canceled by a J that
#: starts from it, and the pose a J finishes in must not be read back as an I.
LAUNCH_POSE = {"J": "I", "Z": "D"}


@dataclass
class Emission:
    letter: str
    kind: str             # "static" | "motion"
    t: float
    confidence: float
    detail: dict = field(default_factory=dict)


@dataclass
class _Frame:
    t: float
    P: np.ndarray         # (21,2) isotropic, handedness-canonicalized
    S: float
    m: np.ndarray         # palm centre
    shape: np.ndarray     # 42-D, palm-normalized: drives sigma / stability / rigidity
    j_gate: bool
    z_gate: bool
    static_feat: np.ndarray = None  # what the static classifier consumes: 101-D static/v3 or
                                    # 112-D static/v4, chosen by the model's feature tag
    v: float = 0.0        # smoothed palm speed at this frame, kept so the arming window can ask
                          # whether the hand was ever actually parked


class Segmenter:
    def __init__(self, thresholds: Thresholds = DEFAULT, static_model=None, motion_model=None,
                 static_classes=None, motion_classes=None, on_event=None, on_hold=None,
                 static_feature_tag=None, static_featfn=None, arm_gates=True):
        self.th = thresholds
        #: The static feature is chosen by the static model's 'feature' tag through
        #: features.STATIC_FEATURES (None = static/v3, the tag pickles carried before the
        #: registry existed). A forest fed a vector of another layout does not crash; it just
        #: reads wrong letters, so the tag is resolved here and the width is checked against
        #: the estimator when it says what it was fitted on. static_featfn overrides the tag
        #: for callers that build the vector some other way (tests, offline sweeps).
        if static_featfn is not None:
            self.static_featfn, dim = static_featfn, None
            self.static_feature_tag = static_feature_tag
        else:
            self.static_featfn, dim = F.static_feature_for(static_feature_tag)
            self.static_feature_tag = (F.DEFAULT_STATIC_TAG if static_feature_tag is None
                                       else str(static_feature_tag))
        fitted = getattr(static_model, "n_features_in_", None)
        if dim is not None and fitted is not None and int(fitted) != dim:
            raise ValueError(f"static model was fitted on {int(fitted)}-D vectors but feature "
                             f"tag {self.static_feature_tag!r} builds {dim}-D; pass the tag "
                             "the model's pickle carries")
        #: False in digits mode. z_gate passes 88% of digit-1 frames, so leaving a '1' would
        #: arm a Z track and park the machine in TRACKING, where nothing static is ever voted,
        #: until T_MAX, the rigidity veto or a fall-confirm ends it. With arming off no track
        #: can start; the gate fractions are still computed for the overlay.
        self.arm_gates = arm_gates
        #: Optional callback receiving every cut motion span, before the vetoes and before any
        #: model runs. Used by label_events.py to build training data with runtime boundaries.
        self.on_event = on_event
        #: Optional callback fired once per completed static vote, whether or not it emitted.
        #: A hold that abstains leaves no other trace, so without this a letter that silently
        #: fails the probability floor is indistinguishable from one never attempted.
        self.on_hold = on_hold
        self.static_model = static_model
        self.motion_model = motion_model
        self.static_classes = list(static_classes) if static_classes is not None else None
        self.motion_classes = list(motion_classes) if motion_classes is not None else None

        self.buf: deque[_Frame] = deque()
        self.state = NO_HAND
        self.last_seen: Optional[float] = None

        self._still_since: Optional[float] = None
        self._rise_start: Optional[float] = None
        self._fall_start: Optional[float] = None
        self._fall_stop: Optional[float] = None
        self._track_from: Optional[float] = None
        self._arm: Optional[str] = None
        self._hold_consumed = False
        self._votes: list = []
        self._cooldown_until = -np.inf
        self._pending: Optional[Emission] = None
        self._pending_until = 0.0
        #: A track armed from the pending letter's own launch pose: the pending letter waits
        #: for the track to resolve (canceled by a J/Z emission, released by an abort or an
        #: abstention) instead of being released on the D_WAIT timer while the track is in
        #: flight -- which is how "IJ" was produced.
        self._pending_held = False
        #: Set by a motion emission. While set, the static branch does not emit: the hand is
        #: resting in the pose the gesture finished in (an I after a J), which is not a new
        #: letter. Cleared when the hand reshapes (sigma > SHAPE_STABLE), when it moves again
        #: after having been still (v_bar > V_STILL), or when it leaves the frame.
        #: Values: None | "settling" | "still".
        self._post_motion = None
        #: The debounced chirality ("Left" / "Right") every frame is canonicalized with, and
        #: the time the other label has been reported since (None while the labels agree).
        #: See HANDEDNESS LATCH in the module docstring.
        self.hand: Optional[str] = None
        self._hand_other_since: Optional[float] = None
        self.last_emitted: Optional[str] = None

        # Exposed for the debug overlay; the only defense against a silently non-firing gate.
        self.v_bar = 0.0
        self.sigma = 0.0
        self.sigma_rigid = 0.0

    # ------------------------------------------------------------------ helpers

    def _trim(self):
        while self.buf and self.buf[-1].t - self.buf[0].t > self.th.BUFFER:
            self.buf.popleft()

    def _reset_history(self):
        self.buf.clear()
        self._still_since = self._rise_start = self._fall_start = None
        self._fall_stop = None
        self._track_from = None
        self._arm = None
        self._hold_consumed = False
        self._votes = []
        self.last_emitted = None
        self._pending = None
        self._pending_held = False
        self._post_motion = None
        self.hand = None
        self._hand_other_since = None

    def _since(self, t0):
        return [f for f in self.buf if f.t >= t0]

    def _update_signals(self):
        """v_bar over the last V_SMOOTH_WINDOW seconds, sigma against the trailing
        SHAPE_WINDOW median; both via features.trailing_window / window_speed, which
        calibrate.py measures (docs/segmenter.js _updateSignals is the twin)."""
        n = len(self.buf)
        if n < 2:
            self.v_bar = 0.0
            self.sigma = 0.0
            self.sigma_rigid = 0.0
            return
        # Smoothing window in SECONDS, not frames: a fixed sample count is a different amount
        # of smoothing at every frame rate, and an under-smoothed v_bar never holds
        # V_MOVE_ARMED long enough to trigger, so nothing is ever detected. The window
        # selection and the arithmetic live in features.py so calibrate.py measures the same
        # v_bar the thresholds are applied to; features.palm_speed is this, per frame.
        times = np.array([f.t for f in self.buf])
        sel = F.trailing_window(times, times[-1], self.th.V_SMOOTH_WINDOW)
        tail = [self.buf[int(k)] for k in sel]
        self.v_bar = F.window_speed(times[sel], np.stack([f.m for f in tail]),
                                    np.array([f.S for f in tail]))

        win = self._since(self.buf[-1].t - self.th.SHAPE_WINDOW)
        if len(win) >= 2:
            sh = np.stack([f.shape for f in win])
            ref = np.median(sh, axis=0)
            self.sigma = float(np.linalg.norm((sh[-1] - ref).reshape(21, 2), axis=-1).mean())
            # The rigidity veto uses the ROTATION-ALIGNED deviation, so a turning hand is not
            # mistaken for a reshaping one. Stability (SHAPE_STABLE) keeps the plain measure:
            # a rotating hand is genuinely not parked.
            cur = sh[-1].reshape(21, 2); r = ref.reshape(21, 2)
            H = cur.T @ r
            U, _, Vt = np.linalg.svd(H)
            dsign = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, dsign]) @ U.T
            self.sigma_rigid = float(np.linalg.norm((R @ cur.T).T - r, axis=-1).mean())
        else:
            self.sigma = 0.0
            self.sigma_rigid = 0.0

    def _gate_fraction(self, t_end, which):
        win = [f for f in self.buf if t_end - self.th.GATE_ARM_WINDOW <= f.t <= t_end]
        if not win:
            return 0.0
        return sum(getattr(f, which) for f in win) / len(win)

    # ------------------------------------------------------------------ main entry

    def step(self, t, landmarks, handedness=None, width=None, height=None) -> Optional[Emission]:
        """Advance one frame. `landmarks` is (21,3) raw MediaPipe output, or None."""
        out = None

        if landmarks is None:
            if self.last_seen is not None and t - self.last_seen > self.th.GAP_RESET:
                # Dropping the hand is how you sign the same letter twice in a row.
                self._reset_history()
                self.state = NO_HAND
            elif self.state == TRACKING and self.last_seen is not None \
                    and t - self.last_seen > self.th.GAP_INTERP:
                # Emit nothing rather than classify a half-seen path.
                self._report_span(f"abort:GAP {t - self.last_seen:.2f}s", t)
                self._abort_track()
            return self._check_pending(t)

        raw = np.asarray(landmarks, dtype=np.float64)
        if not np.isfinite(raw[:, :2]).all():
            # A partially-NaN landmark set reaches the rigidity SVD and raises
            # LinAlgError("SVD did not converge"), killing the process mid-session. Treat it as
            # a non-detection, which is what it is.
            return self.step(t, None, handedness, width, height)
        P = F.to_isotropic(raw[:, :2], width, height)
        P = F.canonicalize_handedness(P, self._latch_handedness(t, handedness))
        fr = _Frame(t=t, P=P, S=float(F.palm_scale(P)), m=F.palm_centre(P),
                    shape=F.shape42(P), static_feat=self.static_featfn(P),
                    j_gate=bool(F.j_gate(P)), z_gate=bool(F.z_gate(P)))
        self.buf.append(fr)
        self._trim()
        self.last_seen = t
        self._update_signals()
        fr.v = self.v_bar
        self._update_post_motion()

        if self.state == NO_HAND:
            self.state = SETTLING

        if self.state == TRACKING:
            out = self._step_tracking(t)
        elif self.state == HOLD:
            out = self._step_hold(t)
        else:
            out = self._step_settling(t)

        return out or self._check_pending(t)

    def _latch_handedness(self, t, label):
        """The label this frame is canonicalized with (HANDEDNESS LATCH, module docstring).

        A real label ("Left"/"Right") is adopted at once when nothing is latched, kept while it
        agrees with the latch, and replaces the latch only after HAND_SWITCH_S of the other
        label without interruption; a frame carrying no real label (None / "Unknown") keeps
        the latched hand. Anything MediaPipe never emits still raises in canonicalize."""
        text = "" if label is None else str(label).strip().lower()
        if text[:1] not in ("l", "r"):
            return self.hand if self.hand is not None else label
        lr = "Left" if text[0] == "l" else "Right"
        if self.hand is None or lr == self.hand:
            self.hand = lr
            self._hand_other_since = None
        elif self._hand_other_since is None:
            self._hand_other_since = t
        elif t - self._hand_other_since >= self.th.HAND_SWITCH_S:
            self.hand = lr
            self._hand_other_since = None
        return self.hand

    def _update_post_motion(self):
        """Clear the post-motion static suppression once the hand has left the pose the
        gesture finished in: a reshape (sigma > SHAPE_STABLE) at any time, or a move
        (v_bar > V_STILL) after a frame on which the hand was still. A gesture that ends on
        the SLOWED tier is still drifting when it scores, so plain speed must not clear the
        flag until the hand has parked once; a reshape is unambiguous either way.

        Measured on the recorded takes: freezing the finishing pose for 1.5 s after each of
        75 emissions produced a static letter 15 times with the shipped code and once with
        this rule; morphing straight into a B on the frame after the emission still emitted
        the B 75/75 times (a rule that waited for a still frame first emitted it 18/75)."""
        if self._post_motion is None:
            return
        th = self.th
        if self.sigma > th.SHAPE_STABLE:
            self._post_motion = None
            return
        if self._post_motion == "settling":
            if self.v_bar < th.V_STILL:
                self._post_motion = "still"
        elif self.v_bar > th.V_STILL:
            self._post_motion = None

    # ------------------------------------------------------------------ states

    def _step_settling(self, t):
        th = self.th

        # rising edge: v_bar crossed V_MOVE_ARMED from below and stayed there. Every track
        # starts here and nowhere else: there is no "unarmed" start, and V_MOVE_UNARMED is
        # not read by the state machine (see thresholds.py).
        if self.arm_gates and self.v_bar >= th.V_MOVE_ARMED:
            if self._rise_start is None:
                self._rise_start = t
            elif t - self._rise_start >= th.V_MOVE_ARMED_SUSTAIN and self._arm is None:
                # Arm on the window BEFORE the rise began: was the hand parked in a launch pose?
                jf = self._gate_fraction(self._rise_start, "j_gate")
                zf = self._gate_fraction(self._rise_start, "z_gate")
                parked = True
                if th.REQUIRE_PARKED:
                    win = [f for f in self.buf
                           if self._rise_start - th.GATE_ARM_WINDOW <= f.t <= self._rise_start]
                    parked = any(f.v < th.V_STILL for f in win)
                if parked and max(jf, zf) >= th.GATE_ARM_FRAC:
                    self._arm = "J" if jf >= zf else "Z"
                    if (self._pending is not None
                            and self._pending.letter == LAUNCH_POSE[self._arm]):
                        # The parked I is this track's launch pose: it is either the start
                        # of a J (cancel it) or a plain I that moved on (release it). Neither
                        # is known until the track resolves, so the timer must not decide.
                        self._pending_held = True
                    # Back-date the start: the smoothed speed confirms the rise only after the
                    # stroke has begun, so without a lead-in the path misses its own opening.
                    self._track_from = self._rise_start - th.LEAD_IN
                    self.state = TRACKING
                    self._fall_start = None
                    return None
        else:
            self._rise_start = None

        if self.v_bar < th.V_STILL and self.sigma < th.SHAPE_STABLE:
            if self._still_since is None:
                self._still_since = t
            elif t - self._still_since >= th.HOLD_SETTLE:
                self.state = HOLD
                self._hold_consumed = False
                self._votes = []
        else:
            self._still_since = None
        return None

    def _step_hold(self, t):
        th = self.th
        if self.v_bar > th.V_STILL or self.sigma > th.SHAPE_STABLE:
            self.state = SETTLING
            self._still_since = None
            self._votes = []
            return None

        if self.static_model is None or self._hold_consumed:
            return None

        proba = self.static_model.predict_proba(self.buf[-1].static_feat.reshape(1, -1))[0]
        self._votes.append((t, proba))
        self._votes = [(vt, p) for vt, p in self._votes if t - vt <= th.VOTE_WINDOW]
        spanned = self._votes and (t - self._votes[0][0]) >= th.VOTE_WINDOW * 0.9
        if not (spanned or len(self._votes) >= th.VOTE_MIN):
            return None

        idx = [int(np.argmax(p)) for _, p in self._votes]
        win = max(set(idx), key=idx.count)
        agree = idx.count(win) / len(idx)
        mean_probs = np.mean([p for _, p in self._votes], axis=0)
        meanp = float(mean_probs[win])
        runner = float(np.sort(mean_probs)[-2]) if len(mean_probs) > 1 else 0.0
        margin = meanp - runner
        letter = self.static_classes[win] if self.static_classes else str(win)
        # ONE predicate decides both the emission and the diagnostic: the on_hold "blocked"
        # flag used to test meanp < VOTE_PROB alone, which reported holds that the decisive
        # route went on to emit as blocked.
        why = self._vote_verdict(t, letter, agree, meanp, margin)
        if self.on_hold is not None:
            top = int(np.argmax(np.mean([p for _, p in self._votes], axis=0)))
            probs = np.mean([p for _, p in self._votes], axis=0)
            order = np.argsort(probs)[::-1][:3]
            fr = self.buf[-1]
            _e = F.extension_ratios(fr.P)
            geom = {"idx_straight": float(F.finger_straightness(fr.P, "index")),
                    "mid_straight": float(F.finger_straightness(fr.P, "mid")),
                    "thumb_midtip": float(np.linalg.norm(fr.P[4] - fr.P[12]) / F.palm_scale(fr.P)),
                    "thumb_pinkymcp": float(F.thumb_pinkymcp(fr.P)),
                    "ext": {k: float(v) for k, v in _e.items()}}
            self.on_hold({"t": t, "geom": geom,
                          # The landmarks themselves, so a live failure can be replayed against
                          # any candidate model offline instead of costing another live session.
                          "P": fr.P.tolist(),
                          "winner": self.static_classes[win] if self.static_classes else str(win),
                          "agree": float(agree), "meanp": float(meanp),
                          "top3": [(self.static_classes[int(j)] if self.static_classes else str(j),
                                    float(probs[int(j)])) for j in order],
                          "margin": float(margin),
                          "blocked": why != "emitted",
                          "why": why})
        if why in ("agree", "margin", "undecided", "cooldown"):
            return None            # keep voting; a later frame may pass
        if why == "post_motion":
            # The hand is resting in the pose the gesture finished in. Nothing in this hold
            # can change that verdict, so stop voting until the hand leaves.
            self._hold_consumed = True
            return None

        self._hold_consumed = True
        self._cooldown_until = t + th.COOLDOWN_STATIC

        # Suppress consecutive duplicates. A held sign jitters in and out of HOLD several times
        # over a few seconds -- measured on the archive, 16 of 24 held letters break and re-form
        # their hold at least once -- and without this each re-entry re-emits the same letter.
        # last_emitted is cleared only when the hand leaves the frame (GAP_RESET), so signing a
        # doubled letter means dropping the hand between the two, which is the physical act anyway.
        if why == "duplicate":
            return None

        em = Emission(letter, "static", t, meanp, {"agree": agree,
                                                   # The whole vote, not just the winner: the
                                                   # word layer scores candidate spellings
                                                   # against every letter's distribution.
                                                   "probs": [float(x) for x in mean_probs]})
        if letter in DEFERRED:
            # Park it: a J may be about to start from this exact pose.
            self._pending = em
            self._pending_until = t + th.D_WAIT
            self._pending_held = False
            return None
        self.last_emitted = letter
        return em

    def _vote_verdict(self, t, letter, agree, meanp, margin):
        """Why a completed vote does or does not emit. The single source of truth for both
        the emission decision and the on_hold diagnostic."""
        th = self.th
        if agree < th.VOTE_AGREE:
            return "agree"
        if margin < th.VOTE_MARGIN:
            return "margin"
        confident = meanp >= th.VOTE_PROB
        decisive = margin >= th.VOTE_MARGIN_CLEAR and meanp >= th.VOTE_PROB_FLOOR
        if not (confident or decisive):
            return "undecided"
        if t < self._cooldown_until:
            return "cooldown"
        if self._post_motion is not None:
            return "post_motion"
        if letter == self.last_emitted:
            return "duplicate"
        return "emitted"

    def _step_tracking(self, t):
        th = self.th
        elapsed = t - self._track_from

        if self.sigma_rigid > th.RIGID_VETO:
            self._report_span(f"abort:RIGID_VETO sigma={self.sigma_rigid:.2f}>{th.RIGID_VETO}", t)
            # The handshape itself changed, so this was a transition between letters, not a
            # rigid-hand sign. This replaces a gate-persistence check, which would have killed
            # real Js whose pinky curls at the bottom of the hook.
            self._abort_track()
            return None
        if elapsed > th.T_MAX:
            self._report_span(f"abort:T_MAX {elapsed:.2f}s", t)
            self._abort_track()
            return None

        # Tier 1: actually stopped. Fast to confirm; the path is complete.
        if self.v_bar < th.V_STILL:
            if self._fall_stop is None:
                self._fall_stop = t
            if t - self._fall_stop >= th.FALL_CONFIRM:
                return self._score(t, "stopped")
        else:
            self._fall_stop = None

        # Tier 2: slowed but still drifting. Confirmed slowly, so a corner of a Z -- where the
        # tip reverses and the smoothed speed dips -- does not end the track mid-gesture.
        if self.v_bar < th.V_FALL:
            if self._fall_start is None:
                self._fall_start = t
            elif t - self._fall_start >= th.FALL_CONFIRM_SLOW:
                return self._score(t, "slowed")
        else:
            self._fall_start = None
        return None

    def _report_span(self, reason, t):
        """Hand an observer the span of a track that is ending, however it ends.

        Aborted tracks matter more than scored ones for diagnosis: a gesture killed by the
        rigidity veto, by T_MAX, or by a detection gap leaves no other trace at all. Logging
        only the scored path makes precisely the failures one is hunting invisible.
        """
        if self.on_event is None or self._track_from is None:
            return
        frames = self._since(self._track_from)
        if len(frames) < 3:
            return
        times = np.array([f.t for f in frames])
        self.on_event({"times": times, "P": np.stack([f.P for f in frames]),
                       "arm": self._arm, "duration": float(times[-1] - times[0]),
                       "t_end": t, "accepted": None, "reason": reason})

    def _abort_track(self):
        self.state = SETTLING
        self._track_from = None
        self._arm = None
        # Whatever ended the track, the parked launch letter is no longer waiting on it. A
        # J/Z emission cancels it in _score; every other ending releases it.
        self._pending_held = False
        self._rise_start = None
        self._fall_start = None
        self._fall_stop = None
        self._still_since = None

    # ------------------------------------------------------------------ scoring

    def _score(self, t, ending="flush"):
        th = self.th
        arm = self._arm
        frames = self._since(self._track_from)
        self._abort_track()
        if len(frames) < 3:
            return None

        times = np.array([f.t for f in frames])
        P = np.stack([f.P for f in frames])
        duration = float(times[-1] - times[0])

        # Hand the raw span to any observer before the vetoes run. This is what lets training
        # data be cut by the very same code path that will cut events at runtime: label a
        # recording by replaying it through a Segmenter and harvesting these spans, rather than
        # feeding whole clips to the classifier. A 2.0 s recorded clip is longer than T_MAX, so
        # training on whole clips would produce a model whose examples the runtime can never
        # reproduce -- train/serve skew of exactly the kind this project exists to document.
        if self.on_event is not None:
            self.on_event({"times": times, "P": P, "arm": arm, "duration": duration,
                           "t_end": t, "accepted": None, "reason": "scored"})

        if not (th.T_MIN <= duration <= th.T_MAX):
            return None

        # Cheap vetoes first, so most motion is rejected before any model runs and never
        # receives a probability at all -- the failure mode that sank Approach A.
        tip = P[:, F.TIP_FOR_ARM[arm], :]
        S_evt = float(np.median(F.palm_scale(P)))
        L = F.path_length(tip) / S_evt
        if not (th.L_MIN <= L <= th.L_MAX):
            return None
        net = float(np.linalg.norm(tip[-1] - tip[0])) / S_evt
        if L > 1e-9 and net / L > th.STRAIGHT_VETO:
            return None            # a straight line is transport, not a letter

        if self.motion_model is None:
            return None
        feat = F.event_features(times, P, arm).reshape(1, -1)
        proba = self.motion_model.predict_proba(feat)[0]
        order = np.argsort(proba)[::-1]
        win = int(order[0])
        label = self.motion_classes[win] if self.motion_classes else str(win)
        margin = float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0

        if label not in ("J", "Z") or proba[win] < th.P_EMIT or margin < th.MARGIN:
            return None            # abstention: the correct and most common answer

        # Only an EMISSION starts the cooldown. Setting it before the test above meant every
        # abstained track -- most motion -- silently blocked the static branch for 0.6 s.
        self._cooldown_until = t + th.COOLDOWN_MOTION
        # A real J started from an I; cancel the parked I so the output is "J", not "IJ".
        self._pending = None
        self._pending_held = False
        # The hand now rests in the pose the gesture finished in; that pose is not a letter.
        self._post_motion = "settling"
        self.last_emitted = label
        return Emission(label, "motion", t, float(proba[win]),
                        {"arm": arm, "duration": duration, "path": L, "margin": margin,
                         "end": ending})

    def flush(self, t):
        """End of stream: score a track still in flight. Offline replay only.

        A live webcam has no end, so nothing is ever in flight when it stops mattering. A
        recorded file does, and third-party footage is routinely trimmed tight to the gesture --
        the video simply stops the instant the hand finishes moving. Such a track never receives
        its fall-confirm (v_bar below V_STILL for FALL_CONFIRM seconds) and would be dropped in
        silence, which looks identical to "the gesture was never recognized".

        The vetoes still run, so this cannot admit anything the runtime would have rejected; it
        only stops a complete gesture being lost to the file ending a fifth of a second early.
        """
        if self.state != TRACKING:
            return None
        return self._score(t)

    def _check_pending(self, t):
        if self._pending is None or t < self._pending_until:
            return None
        if self._pending_held:
            return None            # a track armed from this pose; wait for it to resolve
        if (self.state == SETTLING and self._arm is None and self._rise_start is not None
                and t - self._rise_start < self.th.V_MOVE_ARMED_SUSTAIN):
            # A rise is being confirmed at the deadline: the arm decision, not the timer,
            # settles whether this is the launch of a J/Z. step() runs the settling step
            # first, so on the frame the sustain elapses either the arm sets _pending_held
            # or the gate fails and this frame releases; the deferral is bounded by one
            # V_MOVE_ARMED_SUSTAIN and only happens while the hand is already moving.
            return None
        em, self._pending = self._pending, None
        self._pending_held = False
        self.last_emitted = em.letter
        return em

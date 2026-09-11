"""The live-stream state machine: decides WHEN a letter has been signed.

Classifying a pre-cut clip of a J is easy. The actual problem is that a webcam feed has no
clip boundaries: 30 times a second this has to decide whether the hand is holding a static
letter, performing a J, performing a Z, or just moving between signs. That decision is this
file, and it is the part of the design that everything else serves.

Driven entirely by the timestamps passed into step(). Nothing reads a clock and nothing counts
frames, so replaying recorded footage through a Segmenter reproduces a live run exactly --
which is what lets the same code auto-label training data and evaluate held-out streams.

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
    static_feat: np.ndarray = None  # 143-D, what the static classifier consumes
    v: float = 0.0        # smoothed palm speed at this frame, kept so the arming window can ask
                          # whether the hand was ever actually parked


class Segmenter:
    def __init__(self, thresholds: Thresholds = DEFAULT, static_model=None, motion_model=None,
                 static_classes=None, motion_classes=None, on_event=None, on_hold=None):
        self.th = thresholds
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

    def _since(self, t0):
        return [f for f in self.buf if f.t >= t0]

    def _update_signals(self):
        """v_bar over the last 5 samples, sigma against the trailing SHAPE_WINDOW median."""
        n = len(self.buf)
        if n < 2:
            self.v_bar = 0.0
            self.sigma = 0.0
            self.sigma_rigid = 0.0
            return
        # Smoothing window in SECONDS, not frames: a fixed sample count is a different amount
        # of smoothing at every frame rate, and an under-smoothed v_bar never holds
        # V_MOVE_ARMED long enough to trigger, so nothing is ever detected.
        tail = self._since(self.buf[-1].t - self.th.V_SMOOTH_WINDOW)
        if len(tail) < 2:
            tail = list(self.buf)[-2:]
        t = np.array([f.t for f in tail])
        m = np.stack([f.m for f in tail])
        S = np.array([f.S for f in tail])
        dt = np.maximum(np.diff(t), 1e-6)
        step = np.linalg.norm(np.diff(m, axis=0), axis=-1) / (S[1:] * dt)
        self.v_bar = float(step.mean())

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
        P = F.canonicalize_handedness(P, handedness)
        fr = _Frame(t=t, P=P, S=float(F.palm_scale(P)), m=F.palm_centre(P),
                    shape=F.shape42(P), static_feat=F.static_feature(P),
                    j_gate=bool(F.j_gate(P)), z_gate=bool(F.z_gate(P)))
        self.buf.append(fr)
        self._trim()
        self.last_seen = t
        self._update_signals()
        fr.v = self.v_bar

        if self.state == NO_HAND:
            self.state = SETTLING

        if self.state == TRACKING:
            out = self._step_tracking(t)
        elif self.state == HOLD:
            out = self._step_hold(t)
        else:
            out = self._step_settling(t)

        return out or self._check_pending(t)

    # ------------------------------------------------------------------ states

    def _step_settling(self, t):
        th = self.th
        moving = self.v_bar > th.V_MOVE_UNARMED or self.sigma > th.SHAPE_STABLE

        # rising edge: v_bar crossed V_MOVE_ARMED from below and stayed there
        if self.v_bar >= th.V_MOVE_ARMED:
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
                    # Back-date the start: the smoothed speed confirms the rise only after the
                    # stroke has begun, so without a lead-in the path misses its own opening.
                    self._track_from = self._rise_start - th.LEAD_IN
                    self.state = TRACKING
                    self._fall_start = None
                    return None
        else:
            self._rise_start = None

        if not moving and self.v_bar < th.V_STILL and self.sigma < th.SHAPE_STABLE:
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
                          "blocked": (agree < th.VOTE_AGREE or meanp < th.VOTE_PROB
                                      or margin < th.VOTE_MARGIN or t < self._cooldown_until),
                          "why": ("agree" if agree < th.VOTE_AGREE else
                                  "margin" if margin < th.VOTE_MARGIN else
                                  "undecided" if not (meanp >= th.VOTE_PROB or
                                                      (margin >= th.VOTE_MARGIN_CLEAR and
                                                       meanp >= th.VOTE_PROB_FLOOR)) else
                                  "cooldown" if t < self._cooldown_until else "emitted")})
        confident = meanp >= th.VOTE_PROB
        decisive = margin >= th.VOTE_MARGIN_CLEAR and meanp >= th.VOTE_PROB_FLOOR
        if (agree < th.VOTE_AGREE or margin < th.VOTE_MARGIN or not (confident or decisive)
                or t < self._cooldown_until):
            return None

        letter = self.static_classes[win] if self.static_classes else str(win)
        self._hold_consumed = True
        self._cooldown_until = t + th.COOLDOWN_STATIC

        # Suppress consecutive duplicates. A held sign jitters in and out of HOLD several times
        # over a few seconds -- measured on the archive, 16 of 24 held letters break and re-form
        # their hold at least once -- and without this each re-entry re-emits the same letter.
        # last_emitted is cleared only when the hand leaves the frame (GAP_RESET), so signing a
        # doubled letter means dropping the hand between the two, which is the physical act anyway.
        if letter == self.last_emitted:
            return None

        em = Emission(letter, "static", t, meanp, {"agree": agree,
                                                   # The whole vote, not just the winner: the
                                                   # word layer scores candidate spellings
                                                   # against every letter's distribution.
                                                   "probs": [float(x) for x in mean]})
        if letter in DEFERRED:
            # Park it: a J may be about to start from this exact pose.
            self._pending = em
            self._pending_until = t + th.D_WAIT
            return None
        self.last_emitted = letter
        return em

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
                return self._score(t)
        else:
            self._fall_stop = None

        # Tier 2: slowed but still drifting. Confirmed slowly, so a corner of a Z -- where the
        # tip reverses and the smoothed speed dips -- does not end the track mid-gesture.
        if self.v_bar < th.V_FALL:
            if self._fall_start is None:
                self._fall_start = t
            elif t - self._fall_start >= th.FALL_CONFIRM_SLOW:
                return self._score(t)
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
        self._rise_start = None
        self._fall_start = None
        self._fall_stop = None
        self._still_since = None

    # ------------------------------------------------------------------ scoring

    def _score(self, t):
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

        self._cooldown_until = t + th.COOLDOWN_MOTION
        if label not in ("J", "Z") or proba[win] < th.P_EMIT or margin < th.MARGIN:
            return None            # abstention: the correct and most common answer

        # A real J started from an I; cancel the parked I so the output is "J", not "IJ".
        self._pending = None
        self.last_emitted = label
        return Emission(label, "motion", t, float(proba[win]),
                        {"arm": arm, "duration": duration, "path": L, "margin": margin})

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
        if self._pending is not None and t >= self._pending_until:
            em, self._pending = self._pending, None
            self.last_emitted = em.letter
            return em
        return None

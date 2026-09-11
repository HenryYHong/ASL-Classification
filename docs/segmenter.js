// The live-stream state machine: decides WHEN a letter has been signed.
//
// A line-by-line port of temporal/segmenter.py. Names are the Python's, mechanically
// camel-cased (v_bar -> vBar, _step_hold -> _stepHold), so the two files diff by eye; where
// this file departs from the Python at all, a comment says so and why.
//
//     S0 NO_HAND   -> no detection for GAP_RESET; clears history and lastEmitted
//     S1 SETTLING  -> hand present but moving or reshaping. NOTHING is ever emitted here.
//     S2 HOLD      -> parked and settled; the static branch votes here
//     S3 TRACKING  -> a gesture is underway; the writing fingertip is being recorded
//
// Driven entirely by the timestamps passed into step(). Nothing reads a clock and nothing counts
// frames, so replaying recorded footage reproduces a live run exactly -- and so the browser at
// 60 fps and the archive at 15 fps behave identically, which is the whole reason every constant
// in thresholds is in seconds and palm-widths.

import {
  toIsotropic, canonicalizeHandedness, palmScale, palmCentre, shape42, staticFeature,
  jGate, zGate, eventFeatures, pathLength, extensionRatios, fingerStraightness, thumbPinkyMcp,
  TIP_FOR_ARM,
} from './features.js';
import { predictProba } from './forest.js';

export const NO_HAND = 'NO_HAND';
export const SETTLING = 'SETTLING';
export const HOLD = 'HOLD';
export const TRACKING = 'TRACKING';

// The two handshapes that are also the launch poses for the motion letters. Emitting these
// immediately would turn every J into "IJ" and every Z into "DZ".
const DEFERRED = new Set(['I', 'D']);

// Every threshold this file reads, so a missing one fails at construction instead of at 3 a.m.
// in a live session. An absent key is `undefined`, and every comparison against `undefined` is
// false -- a gate that silently never fires, which is the exact class of bug this port exists
// to avoid repeating.
const REQUIRED_THRESHOLDS = [
  'V_STILL', 'V_MOVE_ARMED', 'V_MOVE_ARMED_SUSTAIN', 'V_MOVE_UNARMED', 'V_FALL', 'LEAD_IN',
  'REQUIRE_PARKED', 'V_SMOOTH_WINDOW', 'SHAPE_STABLE', 'SHAPE_WINDOW', 'RIGID_VETO',
  'GATE_ARM_FRAC', 'GATE_ARM_WINDOW', 'T_MIN', 'T_MAX', 'L_MIN', 'L_MAX', 'STRAIGHT_VETO',
  'P_EMIT', 'MARGIN', 'VOTE_WINDOW', 'VOTE_MIN', 'VOTE_AGREE', 'VOTE_PROB', 'VOTE_MARGIN',
  'VOTE_MARGIN_CLEAR', 'VOTE_PROB_FLOOR', 'HOLD_SETTLE', 'D_WAIT', 'COOLDOWN_MOTION',
  'COOLDOWN_STATIC', 'GAP_INTERP', 'GAP_RESET', 'BUFFER', 'FALL_CONFIRM', 'FALL_CONFIRM_SLOW',
];

function checkThresholds(th) {
  if (th === null || typeof th !== 'object') {
    throw new Error('Segmenter: thresholds object is required (models.json carries one)');
  }
  const missing = REQUIRED_THRESHOLDS.filter((k) => th[k] === undefined || th[k] === null);
  if (missing.length) {
    throw new Error(`Segmenter: thresholds missing [${missing.join(', ')}]`);
  }
  // The Python dataclass asserts these in __post_init__; the same two invariants, because a
  // BUFFER too short to hold a maximum-length episode truncates gestures rather than erroring,
  // and unordered speed thresholds disable the rising edge entirely.
  const need_ = th.T_MAX + 0.25 + th.FALL_CONFIRM + th.GATE_ARM_WINDOW;
  if (th.BUFFER <= need_) {
    throw new Error(`Segmenter: BUFFER=${th.BUFFER}s cannot hold a maximum-length episode ` +
                    `(${need_.toFixed(2)}s needed)`);
  }
  if (!(th.V_STILL < th.V_MOVE_ARMED && th.V_MOVE_ARMED < th.V_MOVE_UNARMED)) {
    throw new Error('Segmenter: speed thresholds must be strictly ordered');
  }
  return th;
}

export class Emission {
  constructor(letter, kind, t, confidence, detail = {}) {
    this.letter = letter;
    this.kind = kind;                 // "static" | "motion"
    this.t = t;
    this.confidence = confidence;
    this.detail = detail;
  }
}

// ---------------------------------------------------------------- small numeric helpers
//
// These reproduce the numpy calls the Python makes. np.median averages the two middle elements
// on an even count; a "pick the upper" median is a different signal and would shift sigma.

function median(values) {
  const a = Float64Array.from(values);
  a.sort();
  const n = a.length;
  if (n === 0) return 0;
  const h = n >> 1;
  return n % 2 ? a[h] : (a[h - 1] + a[h]) / 2;
}

/** Per-column median of `rows` (each a length-`dim` shape vector): np.median(sh, axis=0). */
function medianColumns(rows, dim) {
  const out = new Float64Array(dim);
  const col = new Float64Array(rows.length);
  for (let d = 0; d < dim; d++) {
    for (let i = 0; i < rows.length; i++) col[i] = rows[i][d];
    const c = col.slice();
    c.sort();
    const n = c.length;
    const h = n >> 1;
    out[d] = n % 2 ? c[h] : (c[h - 1] + c[h]) / 2;
  }
  return out;
}

/** np.argmax: first index of the maximum, so ties go to the lowest index. */
function argmax(p) {
  let bi = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i;
  return bi;
}

/** Second-largest value: np.sort(p)[-2]. Note this is the runner-up over ALL classes, which
 *  is not necessarily the runner-up to the vote winner -- see _stepHold. */
function secondLargest(p) {
  const a = Float64Array.from(p);
  a.sort();
  return a.length > 1 ? a[a.length - 2] : 0.0;
}

function meanProbs(votes, nClasses) {
  const out = new Float64Array(nClasses);
  for (const [, p] of votes) for (let c = 0; c < nClasses; c++) out[c] += p[c];
  for (let c = 0; c < nClasses; c++) out[c] /= votes.length;
  return out;
}

/** Finite check over x and y only: np.isfinite(raw[:, :2]).all().
 *
 * Indexed exactly as features.js indexes a landmark ([0], [1]), not by .x/.y, so a caller that
 * hands over MediaPipe's {x, y, z} objects without converting them fails here -- as a stream of
 * non-detections -- rather than reaching toIsotropic and producing NaN features that the forest
 * would happily turn into a confident letter.
 */
function landmarksFinite(landmarks) {
  if (!landmarks || landmarks.length !== 21) return false;
  for (const p of landmarks) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return false;
  }
  return true;
}

export class Segmenter {
  constructor(thresholds, opts = {}) {
    this.th = checkThresholds(thresholds);
    const {
      staticModel = null, motionModel = null, staticClasses = null, motionClasses = null,
      onEvent = null, onHold = null,
    } = opts;

    // Optional callback receiving every cut motion span, before the vetoes and before any model
    // runs. Used by label_events.py to build training data with runtime boundaries.
    this.onEvent = onEvent;
    // Optional callback fired once per completed static vote, whether or not it emitted. A hold
    // that abstains leaves no other trace, so without this a letter that silently fails the
    // probability floor is indistinguishable from one never attempted.
    this.onHold = onHold;
    this.staticModel = staticModel;
    this.motionModel = motionModel;
    // The Python callers always pass model["classes"]; forest.js keeps that same list on the
    // prepared model, so defaulting to it cannot produce a different label set -- it only stops
    // a forgotten argument turning every emission into a bare class index.
    this.staticClasses = staticClasses ? Array.from(staticClasses)
      : (staticModel && staticModel.classes ? Array.from(staticModel.classes) : null);
    this.motionClasses = motionClasses ? Array.from(motionClasses)
      : (motionModel && motionModel.classes ? Array.from(motionModel.classes) : null);

    this.buf = [];
    this.state = NO_HAND;
    this.lastSeen = null;

    this._stillSince = null;
    this._riseStart = null;
    this._fallStart = null;
    this._fallStop = null;
    this._trackFrom = null;
    this._arm = null;
    this._holdConsumed = false;
    this._votes = [];
    this._cooldownUntil = -Infinity;
    this._pending = null;
    this._pendingUntil = 0.0;
    this.lastEmitted = null;

    // Exposed for the debug overlay; the only defense against a silently non-firing gate.
    this.vBar = 0.0;
    this.sigma = 0.0;
    this.sigmaRigid = 0.0;
  }

  // ------------------------------------------------------------------ helpers

  _trim() {
    while (this.buf.length && this.buf[this.buf.length - 1].t - this.buf[0].t > this.th.BUFFER) {
      this.buf.shift();
    }
  }

  _resetHistory() {
    this.buf.length = 0;
    this._stillSince = null;
    this._riseStart = null;
    this._fallStart = null;
    this._fallStop = null;
    this._trackFrom = null;
    this._arm = null;
    this._holdConsumed = false;
    this._votes = [];
    this.lastEmitted = null;
    this._pending = null;
  }

  _since(t0) {
    return this.buf.filter((f) => f.t >= t0);
  }

  /** vBar over the last V_SMOOTH_WINDOW seconds, sigma against the trailing SHAPE_WINDOW median. */
  _updateSignals() {
    const th = this.th;
    const n = this.buf.length;
    if (n < 2) {
      this.vBar = 0.0;
      this.sigma = 0.0;
      this.sigmaRigid = 0.0;
      return;
    }
    // Smoothing window in SECONDS, not frames: a fixed sample count is a different amount of
    // smoothing at every frame rate, and an under-smoothed vBar never holds V_MOVE_ARMED long
    // enough to trigger, so nothing is ever detected.
    let tail = this._since(this.buf[n - 1].t - th.V_SMOOTH_WINDOW);
    if (tail.length < 2) tail = this.buf.slice(-2);
    let sum = 0.0;
    for (let i = 1; i < tail.length; i++) {
      const dt = Math.max(tail[i].t - tail[i - 1].t, 1e-6);
      const d = Math.hypot(tail[i].m[0] - tail[i - 1].m[0], tail[i].m[1] - tail[i - 1].m[1]);
      sum += d / (tail[i].S * dt);
    }
    this.vBar = sum / (tail.length - 1);

    const win = this._since(this.buf[n - 1].t - th.SHAPE_WINDOW);
    if (win.length >= 2) {
      const cur = win[win.length - 1].shape;
      const dim = cur.length;                       // 42
      const k = dim / 2;                            // 21 landmarks
      const ref = medianColumns(win.map((f) => f.shape), dim);

      let acc = 0.0;
      for (let i = 0; i < k; i++) {
        acc += Math.hypot(cur[2 * i] - ref[2 * i], cur[2 * i + 1] - ref[2 * i + 1]);
      }
      this.sigma = acc / k;

      // The rigidity veto uses the ROTATION-ALIGNED deviation, so a turning hand is not mistaken
      // for a reshaping one. Stability (SHAPE_STABLE) keeps the plain measure: a rotating hand is
      // genuinely not parked.
      //
      // The Python takes the 2x2 Kabsch rotation through an SVD. In two dimensions the proper
      // rotation has a closed form -- theta = atan2(sum(cx*ry - cy*rx), sum(cx*rx + cy*ry)) --
      // and it is the same rotation, because numpy's `diag([1, sign(det)])` correction is exactly
      // the constraint det(R) = +1. Verified rather than argued: replaying a stream of a hand
      // TURNING ON THE SPOT through both implementations agrees to 2.3e-16 on values up to
      // 2.8e-3 (test_segmenter.mjs --dump, then the numpy replay). A stream where the hand only
      // translates would not have tested this at all -- the rotation is near-identity there.
      let A = 0.0;
      let B = 0.0;
      for (let i = 0; i < k; i++) {
        const cx = cur[2 * i], cy = cur[2 * i + 1];
        const rx = ref[2 * i], ry = ref[2 * i + 1];
        A += cx * rx + cy * ry;
        B += cx * ry - cy * rx;
      }
      const ang = Math.atan2(B, A);
      const co = Math.cos(ang), si = Math.sin(ang);
      let racc = 0.0;
      for (let i = 0; i < k; i++) {
        const cx = cur[2 * i], cy = cur[2 * i + 1];
        racc += Math.hypot(co * cx - si * cy - ref[2 * i], si * cx + co * cy - ref[2 * i + 1]);
      }
      this.sigmaRigid = racc / k;
    } else {
      this.sigma = 0.0;
      this.sigmaRigid = 0.0;
    }
  }

  _gateFraction(tEnd, which) {
    const lo = tEnd - this.th.GATE_ARM_WINDOW;
    const win = this.buf.filter((f) => f.t >= lo && f.t <= tEnd);
    if (!win.length) return 0.0;
    let s = 0;
    for (const f of win) s += f[which] ? 1 : 0;
    return s / win.length;
  }

  // ------------------------------------------------------------------ main entry

  /**
   * Advance one frame. `landmarks` is 21 raw MediaPipe landmarks, or null.
   * Returns an Emission or null.
   */
  step(t, landmarks, handedness = null, width = null, height = null) {
    let out = null;
    const th = this.th;

    if (landmarks === null || landmarks === undefined) {
      if (this.lastSeen !== null && t - this.lastSeen > th.GAP_RESET) {
        // Dropping the hand is how you sign the same letter twice in a row.
        this._resetHistory();
        this.state = NO_HAND;
      } else if (this.state === TRACKING && this.lastSeen !== null
                 && t - this.lastSeen > th.GAP_INTERP) {
        // Emit nothing rather than classify a half-seen path.
        this._reportSpan(`abort:GAP ${(t - this.lastSeen).toFixed(2)}s`, t);
        this._abortTrack();
      }
      return this._checkPending(t);
    }

    if (!landmarksFinite(landmarks)) {
      // A partially-NaN landmark set reaches the rigidity alignment and produces NaN signals,
      // which compare false against every threshold and wedge the machine in SETTLING for as
      // long as it lasts. Treat it as a non-detection, which is what it is.
      return this.step(t, null, handedness, width, height);
    }

    let P = toIsotropic(landmarks, width, height);
    P = canonicalizeHandedness(P, handedness);
    const fr = {
      t,
      P,
      S: Number(palmScale(P)),
      m: palmCentre(P),
      shape: shape42(P),
      staticFeat: staticFeature(P),
      jGate: Boolean(jGate(P)),
      zGate: Boolean(zGate(P)),
      // Smoothed palm speed at this frame, kept so the arming window can ask whether the hand
      // was ever actually parked.
      v: 0.0,
    };
    this.buf.push(fr);
    this._trim();
    this.lastSeen = t;
    this._updateSignals();
    fr.v = this.vBar;

    if (this.state === NO_HAND) this.state = SETTLING;

    if (this.state === TRACKING) out = this._stepTracking(t);
    else if (this.state === HOLD) out = this._stepHold(t);
    else out = this._stepSettling(t);

    return out || this._checkPending(t);
  }

  // ------------------------------------------------------------------ states

  _stepSettling(t) {
    const th = this.th;
    const moving = this.vBar > th.V_MOVE_UNARMED || this.sigma > th.SHAPE_STABLE;

    // rising edge: vBar crossed V_MOVE_ARMED from below and stayed there
    if (this.vBar >= th.V_MOVE_ARMED) {
      if (this._riseStart === null) {
        this._riseStart = t;
      } else if (t - this._riseStart >= th.V_MOVE_ARMED_SUSTAIN && this._arm === null) {
        // Arm on the window BEFORE the rise began: was the hand parked in a launch pose?
        const jf = this._gateFraction(this._riseStart, 'jGate');
        const zf = this._gateFraction(this._riseStart, 'zGate');
        let parked = true;
        if (th.REQUIRE_PARKED) {
          const lo = this._riseStart - th.GATE_ARM_WINDOW;
          // A frame below V_STILL, not merely below V_MOVE_ARMED: the design always said
          // "parked, then moving", and enforcing only the second half fired tracks during
          // ordinary fingerspelling transitions -- and a running track blocks the static branch.
          parked = this.buf.some((f) => f.t >= lo && f.t <= this._riseStart && f.v < th.V_STILL);
        }
        if (parked && Math.max(jf, zf) >= th.GATE_ARM_FRAC) {
          this._arm = jf >= zf ? 'J' : 'Z';
          // Back-date the start: the smoothed speed confirms the rise only after the stroke has
          // begun, so without a lead-in the path misses its own opening.
          this._trackFrom = this._riseStart - th.LEAD_IN;
          this.state = TRACKING;
          this._fallStart = null;
          return null;
        }
      }
    } else {
      this._riseStart = null;
    }

    if (!moving && this.vBar < th.V_STILL && this.sigma < th.SHAPE_STABLE) {
      if (this._stillSince === null) {
        this._stillSince = t;
      } else if (t - this._stillSince >= th.HOLD_SETTLE) {
        this.state = HOLD;
        this._holdConsumed = false;
        this._votes = [];
      }
    } else {
      this._stillSince = null;
    }
    return null;
  }

  _stepHold(t) {
    const th = this.th;
    if (this.vBar > th.V_STILL || this.sigma > th.SHAPE_STABLE) {
      this.state = SETTLING;
      this._stillSince = null;
      this._votes = [];
      return null;
    }

    if (this.staticModel === null || this._holdConsumed) return null;

    const proba = predictProba(this.staticModel, this.buf[this.buf.length - 1].staticFeat);
    this._votes.push([t, proba]);
    this._votes = this._votes.filter(([vt]) => t - vt <= th.VOTE_WINDOW);
    // EITHER a time span OR VOTE_MIN votes, whichever completes first. Gating on the span alone
    // is frame-rate fragile: at 15 fps four votes span 0.199 s, too short for the 0.9x window
    // test, and a fifth pushes the first out of the window -- so the vote never completes at all.
    const spanned = this._votes.length > 0
      && (t - this._votes[0][0]) >= th.VOTE_WINDOW * 0.9;
    if (!(spanned || this._votes.length >= th.VOTE_MIN)) return null;

    const nClasses = proba.length;
    const idx = this._votes.map(([, p]) => argmax(p));
    const counts = new Map();
    for (const i of idx) counts.set(i, (counts.get(i) || 0) + 1);
    // max(set(idx), key=idx.count). A tie is broken HERE by the lowest class index. The Python
    // breaks it by CPython's set iteration order, which is not ascending and is not a language
    // guarantee: set([3, 9]) yields 9 first, and of the 276 two-way ties over 24 classes, 93
    // name the higher index. So the two files genuinely differ on a tie -- and cannot differ on
    // an emission, because a tie cannot survive the agreement test. If k classes share the top
    // count m then len(idx) >= k*m, so agree = m/len(idx) <= 1/k <= 0.5, and `agree < VOTE_AGREE`
    // rejects the vote whichever class it named. VOTE_AGREE is 0.70; anything above 0.5 holds.
    // The residue is the onHold diagnostic's `winner` on a vote that was always going to be
    // blocked, and a VOTE_AGREE dropped to 0.5 or below, which would make this observable.
    let win = -1;
    let bestCount = -1;
    for (const i of Array.from(counts.keys()).sort((a, b) => a - b)) {
      if (counts.get(i) > bestCount) { bestCount = counts.get(i); win = i; }
    }
    const agree = bestCount / idx.length;
    const mean = meanProbs(this._votes, nClasses);
    const meanp = mean[win];
    // The runner-up is the second-largest MEAN over all classes, so when the majority winner is
    // not the argmax of the mean this margin can go negative -- and VOTE_MARGIN then blocks it,
    // which is the intent: the vote disagreed with itself.
    const runner = secondLargest(mean);
    const margin = meanp - runner;

    const confident = meanp >= th.VOTE_PROB;
    // D is classified correctly on 100 of 100 live frames yet peaks at 0.41, because the forest
    // splits its mass across D/X/C. No absolute floor can ever emit it; its margin (0.14) over
    // genuine junk (0.05-0.07) is what separates them.
    const decisive = margin >= th.VOTE_MARGIN_CLEAR && meanp >= th.VOTE_PROB_FLOOR;

    if (this.onHold !== null) {
      const order = Array.from(mean.keys())
        .sort((a, b) => mean[a] - mean[b] || a - b)
        .reverse()
        .slice(0, 3);
      const fr = this.buf[this.buf.length - 1];
      const e = extensionRatios(fr.P);
      const S = Number(palmScale(fr.P));
      const ext = {};
      for (const k of Object.keys(e)) ext[k] = Number(e[k]);
      const geom = {
        idx_straight: Number(fingerStraightness(fr.P, 'index')),
        mid_straight: Number(fingerStraightness(fr.P, 'mid')),
        thumb_midtip: Math.hypot(fr.P[4][0] - fr.P[12][0], fr.P[4][1] - fr.P[12][1]) / S,
        thumb_pinkymcp: Number(thumbPinkyMcp(fr.P)),
        ext,
      };
      this.onHold({
        t,
        geom,
        // The landmarks themselves, so a live failure can be replayed against any candidate
        // model offline instead of costing another live session.
        P: fr.P,
        winner: this.staticClasses ? this.staticClasses[win] : String(win),
        agree,
        meanp,
        top3: order.map((j) => [this.staticClasses ? this.staticClasses[j] : String(j), mean[j]]),
        margin,
        // Ported as the Python writes it: `blocked` still asks meanp >= VOTE_PROB alone, so a
        // letter emitted by the decisive route is reported blocked while `why` says "emitted".
        // It is a diagnostic field, and a log that disagreed with the Python's log would be
        // worse than one that reproduces its quirk.
        blocked: (agree < th.VOTE_AGREE || meanp < th.VOTE_PROB || margin < th.VOTE_MARGIN
                  || t < this._cooldownUntil),
        why: (agree < th.VOTE_AGREE ? 'agree'
          : margin < th.VOTE_MARGIN ? 'margin'
            : !(confident || decisive) ? 'undecided'
              : t < this._cooldownUntil ? 'cooldown' : 'emitted'),
      });
    }

    if (agree < th.VOTE_AGREE || margin < th.VOTE_MARGIN || !(confident || decisive)
        || t < this._cooldownUntil) {
      return null;
    }

    const letter = this.staticClasses ? this.staticClasses[win] : String(win);
    this._holdConsumed = true;
    this._cooldownUntil = t + th.COOLDOWN_STATIC;

    // Suppress consecutive duplicates. A held sign jitters in and out of HOLD several times over
    // a few seconds -- measured on the archive, 16 of 24 held letters break and re-form their
    // hold at least once -- and without this each re-entry re-emits the same letter. lastEmitted
    // is cleared only when the hand leaves the frame (GAP_RESET), so signing a doubled letter
    // means dropping the hand between the two, which is the physical act anyway.
    if (letter === this.lastEmitted) return null;

    const em = new Emission(letter, 'static', t, meanp, { agree });
    if (DEFERRED.has(letter)) {
      // Park it: a J may be about to start from this exact pose.
      this._pending = em;
      this._pendingUntil = t + th.D_WAIT;
      return null;
    }
    this.lastEmitted = letter;
    return em;
  }

  _stepTracking(t) {
    const th = this.th;
    const elapsed = t - this._trackFrom;

    if (this.sigmaRigid > th.RIGID_VETO) {
      this._reportSpan(`abort:RIGID_VETO sigma=${this.sigmaRigid.toFixed(2)}>${th.RIGID_VETO}`, t);
      // The handshape itself changed, so this was a transition between letters, not a rigid-hand
      // sign. This replaces a gate-persistence check, which would have killed real Js whose pinky
      // curls at the bottom of the hook.
      this._abortTrack();
      return null;
    }
    if (elapsed > th.T_MAX) {
      this._reportSpan(`abort:T_MAX ${elapsed.toFixed(2)}s`, t);
      this._abortTrack();
      return null;
    }

    // Tier 1: actually stopped. Fast to confirm; the path is complete.
    if (this.vBar < th.V_STILL) {
      if (this._fallStop === null) this._fallStop = t;
      if (t - this._fallStop >= th.FALL_CONFIRM) return this._score(t);
    } else {
      this._fallStop = null;
    }

    // Tier 2: slowed but still drifting. Confirmed slowly, so a corner of a Z -- where the tip
    // reverses and the smoothed speed dips -- does not end the track mid-gesture.
    if (this.vBar < th.V_FALL) {
      if (this._fallStart === null) {
        this._fallStart = t;
      } else if (t - this._fallStart >= th.FALL_CONFIRM_SLOW) {
        return this._score(t);
      }
    } else {
      this._fallStart = null;
    }
    return null;
  }

  /**
   * Hand an observer the span of a track that is ending, however it ends.
   *
   * Aborted tracks matter more than scored ones for diagnosis: a gesture killed by the rigidity
   * veto, by T_MAX, or by a detection gap leaves no other trace at all. Logging only the scored
   * path makes precisely the failures one is hunting invisible.
   */
  _reportSpan(reason, t) {
    if (this.onEvent === null || this._trackFrom === null) return;
    const frames = this._since(this._trackFrom);
    if (frames.length < 3) return;
    const times = frames.map((f) => f.t);
    this.onEvent({
      times,
      P: frames.map((f) => f.P),
      arm: this._arm,
      duration: times[times.length - 1] - times[0],
      t_end: t,
      accepted: null,
      reason,
    });
  }

  _abortTrack() {
    this.state = SETTLING;
    this._trackFrom = null;
    this._arm = null;
    this._riseStart = null;
    this._fallStart = null;
    this._fallStop = null;
    this._stillSince = null;
  }

  // ------------------------------------------------------------------ scoring

  _score(t) {
    const th = this.th;
    const arm = this._arm;
    const frames = this._since(this._trackFrom);
    this._abortTrack();
    if (frames.length < 3) return null;

    const times = frames.map((f) => f.t);
    const P = frames.map((f) => f.P);
    const duration = times[times.length - 1] - times[0];

    // Hand the raw span to any observer before the vetoes run. This is what lets training data be
    // cut by the very same code path that cuts events at runtime: a 2.0 s recorded clip is longer
    // than T_MAX, so training on whole clips would produce a model whose examples the runtime can
    // never reproduce -- train/serve skew of exactly the kind this project exists to document.
    if (this.onEvent !== null) {
      this.onEvent({ times, P, arm, duration, t_end: t, accepted: null, reason: 'scored' });
    }

    if (!(th.T_MIN <= duration && duration <= th.T_MAX)) return null;

    // Cheap vetoes first, so most motion is rejected before any model runs and never receives a
    // probability at all -- the failure mode that sank Approach A.
    const tipIdx = TIP_FOR_ARM[arm];
    const tip = frames.map((f) => f.P[tipIdx]);
    // np.median(F.palm_scale(P)) over the span; each frame's scale is already in the buffer.
    const S_evt = median(frames.map((f) => f.S));
    const L = pathLength(tip) / S_evt;
    if (!(th.L_MIN <= L && L <= th.L_MAX)) return null;
    const net = Math.hypot(tip[tip.length - 1][0] - tip[0][0],
                           tip[tip.length - 1][1] - tip[0][1]) / S_evt;
    if (L > 1e-9 && net / L > th.STRAIGHT_VETO) return null;  // a straight line is transport

    if (this.motionModel === null) return null;
    const feat = eventFeatures(times, P, arm);
    const proba = predictProba(this.motionModel, feat);
    const order = Array.from(proba.keys())
      .sort((a, b) => proba[a] - proba[b] || a - b)
      .reverse();
    const win = order[0];
    const label = this.motionClasses ? this.motionClasses[win] : String(win);
    const margin = order.length > 1 ? proba[order[0]] - proba[order[1]] : 1.0;

    this._cooldownUntil = t + th.COOLDOWN_MOTION;
    if ((label !== 'J' && label !== 'Z') || proba[win] < th.P_EMIT || margin < th.MARGIN) {
      return null;            // abstention: the correct and most common answer
    }

    // A real J started from an I; cancel the parked I so the output is "J", not "IJ".
    this._pending = null;
    this.lastEmitted = label;
    return new Emission(label, 'motion', t, proba[win],
                        { arm, duration, path: L, margin });
  }

  /**
   * End of stream: score a track still in flight. Offline replay only.
   *
   * A live webcam has no end, so nothing is ever in flight when it stops mattering. A recorded
   * file does, and third-party footage is routinely trimmed tight to the gesture -- the video
   * stops the instant the hand finishes moving. Such a track never receives its fall-confirm and
   * would be dropped in silence, which looks identical to "the gesture was never recognized".
   *
   * The vetoes still run, so this cannot admit anything the runtime would have rejected.
   */
  flush(t) {
    if (this.state !== TRACKING) return null;
    return this._score(t);
  }

  _checkPending(t) {
    if (this._pending !== null && t >= this._pendingUntil) {
      const em = this._pending;
      this._pending = null;
      this.lastEmitted = em.letter;
      return em;
    }
    return null;
  }
}

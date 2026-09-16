// Behavioral check for segmenter.js.  Run:  node docs/test_segmenter.mjs
//
// Reading the port next to temporal/segmenter.py is not evidence. This drives synthetic streams
// built from the real landmarks in golden.json through the real forests in models.json, and
// asserts the behavior the Python specifies -- including the mechanisms that broke silently
// in the Python and would break just as silently here:
//
//   * the static vote completes on EITHER a time span OR VOTE_MIN votes, so a held letter emits
//     at 15 fps, 30 fps and 60 fps alike. Gating on the span alone cost 8 of 24 archive letters
//     and was invisible at 30 fps, where it happened to fit.
//   * I and D are held back by D_WAIT, because they are the launch poses for J and Z -- and a
//     parked I whose own pose arms a track is HELD until the track resolves, so a J is "J" and
//     not "IJ" (section 8).
//   * a parked I whose J stroke begins inside D_WAIT but whose rise is still being confirmed
//     at the deadline waits for the arm decision, not the timer (section 8, the gray zone);
//     an I held longer than D_WAIT before the stroke still reads "IJ" by design.
//   * after a J/Z emission the hand rests in the finishing pose, which is not a letter: the
//     static branch stays silent until the hand reshapes or moves again (section 9).
//   * with arming off (numbers mode) no track can start, however gate-passing the pose
//     (section 10).
//   * the handedness label is latched: a one-frame flip mid-stroke is canonicalized with the
//     latched hand and the J survives; a flip that persists for HAND_SWITCH_S switches; a
//     GAP_RESET clears the latch (section 12).
//
// Sections 8, 9 and 12 need a J to be EMITTED, and the motion forest abstains on the synthetic
// arc (correctly: it is not a J), so they run with a stub motion forest that answers J p=1.0
// for every span. The vetoes, the cut and the state machine are the real ones.
//
// With `--dump <path>` it also writes the generated frames and its own per-frame signal trace,
// which is what temporal/segmenter.py was replayed against to confirm the two implementations
// agree numerically (vBar, sigma, sigmaRigid and every emission).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { prepareModel, predictProba } from './forest.js';
import { eventFeatures, toIsotropic, canonicalizeHandedness } from './features.js';
import { Segmenter, NO_HAND, SETTLING, HOLD, TRACKING, LAUNCH_POSE } from './segmenter.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(fs.readFileSync(path.join(HERE, 'models.json'), 'utf8'));
const golden = JSON.parse(fs.readFileSync(path.join(HERE, 'golden.json'), 'utf8'));

const TH = payload.thresholds;
const STATIC = prepareModel(payload.static);
const MOTION = prepareModel(payload.motion);
// The numbers forest and its override block, when the export carried them (phase 2).
const DIGITS = payload.digits ? prepareModel(payload.digits) : null;
const TH_DIGITS = payload.digits ? { ...TH, ...payload.digits.thresholds } : null;
const W = 1920;
const H = 1080;

// A one-leaf forest over {J, MOVE, Z} that answers J with probability 1.0 for any 79-D input.
// It stands in for the motion forest where a test needs an EMISSION rather than the forest's
// (correct) abstention on a synthetic arc; everything upstream of predictProba is unchanged.
const ALWAYS_J = prepareModel({
  classes: ['J', 'MOVE', 'Z'], feature: 'event/v1', dim: 79,
  trees: [{ f: [-2], t: [-2], l: [-1], r: [-1], v: { 0: [1, 0, 0] } }],
});

// The first A, D, I and V among golden.json's letter cases (the solutions-API ones; the file may
// also carry Tasks-API and digit cases, which are not used here). D and I are the deferred
// pair; I is also the J launch pose (its j_gate is true), which is what lets the motion phase
// below arm a track at all.
const letterCases = golden.cases.filter((c) => (c.model || 'static') === 'static'
                                              && (c.api || 'solutions') === 'solutions');
const firstOf = (letter) => {
  const c = letterCases.find((x) => x.predicted === letter);
  if (!c) throw new Error(`golden.json has no solutions-API case predicted ${letter}`);
  return c;
};
const A = firstOf('A');
const D = firstOf('D');
const I = firstOf('I');
const V = firstOf('V');

let failures = 0;
function check(name, ok, detail = '') {
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? `   ${detail}` : ''}`);
}

// ---------------------------------------------------------------- synthetic streams
//
// Landmarks are MediaPipe-normalized: x by frame width, y by frame height. A translation
// measured in PALM units therefore has to be un-squashed on x before it is applied, or the
// synthetic hand moves further in y than in x and the arc below stops being an arc. This is the
// same aspect correction features.toIsotropic applies in the other direction.
const ASPECT = W / H;

function translate(lm, du, dv) {
  return lm.map((p) => [p[0] + du / ASPECT, p[1] + dv, p[2]]);
}

/** `n` frames at `fps` starting at `t0`, each the same hand, optionally displaced by move(s). */
function span(caseObj, t0, seconds, fps, move = null) {
  const n = Math.round(seconds * fps);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fps;
    const s = n > 1 ? i / (n - 1) : 0;
    const [du, dv] = move ? move(s) : [0, 0];
    out.push({ t, landmarks: move ? translate(caseObj.landmarks, du, dv) : caseObj.landmarks });
  }
  return out;
}

/** Rotate a hand about its own palm center by `ang` radians, IN ISOTROPIC SPACE.
 *
 * Rotating the MediaPipe-normalized coordinates directly would shear the hand, because x is
 * divided by frame width and y by frame height: the rotation has to happen in the square-aspect
 * space the features are measured in, and be squashed back afterwards.
 */
function rotate(lm, ang) {
  const u = lm.map((p) => [p[0] * ASPECT, p[1]]);
  const palm = [0, 5, 9, 13, 17];
  let uc = 0, vc = 0;
  for (const i of palm) { uc += u[i][0] / palm.length; vc += u[i][1] / palm.length; }
  const c = Math.cos(ang), si = Math.sin(ang);
  return u.map(([x, y], i) => {
    const dx = x - uc, dy = y - vc;
    return [(uc + c * dx - si * dy) / ASPECT, vc + si * dx + c * dy, lm[i][2]];
  });
}

/** A parked hand that turns on the spot: the palm center never moves, so vBar stays at zero. */
function turning(t0, seconds, fps, sweep = Math.PI / 2) {
  const n = Math.round(seconds * fps);
  return Array.from({ length: n }, (_, i) => ({
    t: t0 + i / fps,
    landmarks: rotate(I.landmarks, sweep * (i / (n - 1))),
  }));
}

/** `seconds` of non-detection at `fps`, so a gap is sampled at the same rate as the video. */
function gap(t0, seconds, fps) {
  const n = Math.round(seconds * fps);
  const out = [];
  for (let i = 0; i < n; i++) out.push({ t: t0 + i / fps, landmarks: null });
  return out;
}

// The J launch pose carried along a half-circle of one palm width: the handshape never changes,
// so this is the case the rigidity veto must NOT kill, and it starts from a pose j_gate passes.
const R = I.palm_scale;                          // one palm width, in isotropic units
const arc = (s) => [R * (Math.cos(Math.PI * s) - 1), R * Math.sin(Math.PI * s)];

/** parked launch pose, then a stroke, then parked again -- the shape a scored track has. */
function stroke(t0, fps, { park = 0.8, move = 0.6, rest = 1.0 } = {}) {
  return [
    ...span(I, t0, park, fps),
    ...span(I, t0 + park, move, fps, arc),
    ...(rest ? span(I, t0 + park + move, rest, fps, () => arc(1)) : []),
  ];
}

/** parked launch pose, then travel that never stops: the track dies of T_MAX. */
function runaway(t0, fps, seconds = 2.6) {
  return [
    ...span(I, t0, 0.8, fps),
    ...span(I, t0 + 0.8, seconds, fps, (s) => [3 * R * seconds * s, 0]),
  ];
}

function run(frames, opts = {}) {
  const { th = TH, ...rest } = opts;
  const events = [];
  const holds = [];
  const seg = new Segmenter(th, {
    staticModel: STATIC,
    motionModel: MOTION,
    staticClasses: payload.static.classes,
    motionClasses: payload.motion.classes,
    onEvent: (e) => events.push(e),
    onHold: (h) => holds.push(h),
    ...rest,
  });
  const emissions = [];
  const trace = [];
  const states = new Set();
  for (const f of frames) {
    // Every synthetic hand is a Left hand unless the frame says otherwise (section 12 flips
    // the label per frame to exercise the latch).
    const em = seg.step(f.t, f.landmarks, f.handedness === undefined ? 'Left' : f.handedness, W, H);
    states.add(seg.state);
    trace.push({ t: f.t, state: seg.state, vBar: seg.vBar, sigma: seg.sigma,
                 sigmaRigid: seg.sigmaRigid, hand: seg.hand,
                 P: seg.buf.length ? seg.buf[seg.buf.length - 1].P : null });
    // `delivered` is the frame the emission came back on, which is NOT em.t: a deferred letter
    // keeps the timestamp of the vote that produced it and is handed over D_WAIT later.
    if (em) emissions.push(Object.assign(em, { delivered: f.t }));
  }
  return { seg, emissions, events, holds, trace, states };
}

// ---------------------------------------------------------------- 1. a still hand for 2 s
//
// Exactly one static letter, no motion event, at every frame rate. The Python emits on the first
// completed vote and then sets _holdConsumed, so holding A for ten seconds is still one A.

console.log('--- a still hand for 2 s ---');
for (const fps of [15, 30, 60]) {
  const { seg, emissions, events } = run(span(A, 0, 2.0, fps));
  const statics = emissions.filter((e) => e.kind === 'static');
  const motions = emissions.filter((e) => e.kind === 'motion');
  check(`${fps} fps: exactly one emission`, emissions.length === 1,
        `got ${emissions.length} [${emissions.map((e) => e.letter).join(',')}]`);
  check(`${fps} fps: it is the static letter Python predicts`,
        statics.length === 1 && statics[0].letter === A.predicted,
        `got ${statics.map((e) => `${e.letter}@${e.t.toFixed(3)}s p=${e.confidence.toFixed(3)}`)}`);
  check(`${fps} fps: zero motion events`, motions.length === 0 && events.length === 0,
        `${motions.length} motion emissions, ${events.length} cut spans`);
  check(`${fps} fps: ends parked in HOLD`, seg.state === HOLD, `state ${seg.state}`);
}

// The vote must also be able to complete on the SPAN, not only on the count: at 15 fps four
// votes span 0.199 s, which is why VOTE_MIN exists. Both routes are exercised above; this pins
// the latency the span route produces so a regression in either shows up as a number.
{
  const { emissions } = run(span(A, 0, 2.0, 30));
  const latency = emissions[0].t;
  check('emission lands within HOLD_SETTLE + VOTE_WINDOW of the first frame',
        latency <= TH.HOLD_SETTLE + TH.VOTE_WINDOW + 1 / 30,
        `${latency.toFixed(3)}s (budget ${(TH.HOLD_SETTLE + TH.VOTE_WINDOW).toFixed(2)}s)`);
}

// ---------------------------------------------------------------- 2. I and D are deferred
//
// Same stream, different letter: D must arrive D_WAIT later than a non-deferred letter would,
// because a Z may be about to start from that exact pose.

console.log('\n--- deferral of the launch poses ---');
{
  const a = run(span(A, 0, 2.0, 30)).emissions;
  const d = run(span(D, 0, 2.0, 30)).emissions;
  const i = run(span(I, 0, 2.0, 30)).emissions;
  check('D emits exactly once', d.length === 1 && d[0].letter === 'D',
        `got [${d.map((e) => e.letter).join(',')}]`);
  check('I emits exactly once', i.length === 1 && i[0].letter === 'I',
        `got [${i.map((e) => e.letter).join(',')}]`);
  // The deferred letter keeps the vote's timestamp and is HANDED OVER D_WAIT later, so the
  // delay is visible in the frame it came back on, not in em.t.
  check('a deferred letter keeps the timestamp of its vote', d[0].t === d[0].delivered - TH.D_WAIT
        || Math.abs(d[0].delivered - d[0].t - TH.D_WAIT) <= 1 / 30 + 1e-9,
        `voted ${d[0].t.toFixed(3)}s, delivered ${d[0].delivered.toFixed(3)}s`);
  for (const [name, e] of [['D', d[0]], ['I', i[0]]]) {
    const held = e.delivered - a[0].delivered;
    check(`${name} is held back by D_WAIT, because it is a launch pose`,
          held >= TH.D_WAIT - 1e-9 && held < TH.D_WAIT + 2 / 30,
          `${held.toFixed(3)}s later than the undeferred A, D_WAIT ${TH.D_WAIT}s`);
  }
}

// ---------------------------------------------------------------- 3. the gap rules
//
// GAP_RESET clears the buffer, the state and lastEmitted; anything shorter does not. Dropping
// the hand is how the signer spells a doubled letter, so this is user-visible behavior, not
// housekeeping.

console.log('\n--- detection gaps ---');
{
  const fps = 30;
  const long = [
    ...span(A, 0, 2.0, fps),
    ...gap(2.0, TH.GAP_RESET + 0.2, fps),
  ];
  const { seg, emissions } = run(long);
  check('gap longer than GAP_RESET resets to NO_HAND', seg.state === NO_HAND, `state ${seg.state}`);
  check('...and clears the history', seg.buf.length === 0, `${seg.buf.length} frames buffered`);
  check('...and clears lastEmitted', seg.lastEmitted === null, `lastEmitted ${seg.lastEmitted}`);
  check('...emitting nothing on the way out', emissions.length === 1,
        `${emissions.length} emissions`);

  const short = [
    ...span(A, 0, 2.0, fps),
    ...gap(2.0, TH.GAP_RESET - 0.1, fps),
  ];
  const brief = run(short).seg;
  check('gap shorter than GAP_RESET does not reset', brief.state === HOLD && brief.lastEmitted === 'A',
        `state ${brief.state}, lastEmitted ${brief.lastEmitted}`);

  // The point of the reset: the same letter twice, which duplicate suppression otherwise eats.
  const twice = [
    ...span(A, 0, 2.0, fps),
    ...gap(2.0, TH.GAP_RESET + 0.2, fps),
    ...span(A, 2.8, 1.5, fps),
  ];
  const again = run(twice).emissions;
  check('the same letter re-emits after the hand leaves and returns',
        again.length === 2 && again.every((e) => e.letter === 'A'),
        `[${again.map((e) => `${e.letter}@${e.t.toFixed(2)}`).join(', ')}]`);

  // Non-finite landmarks are a non-detection, not an exception and not a frame.
  const nan = [
    ...span(A, 0, 0.5, fps),
    ...Array.from({ length: 20 }, (_, i) => ({
      t: 0.5 + i / fps,
      landmarks: A.landmarks.map((p, k) => (k === 3 ? [NaN, p[1], p[2]] : p)),
    })),
  ];
  const nanRun = run(nan).seg;
  check('non-finite landmarks are treated as a non-detection',
        nanRun.state === NO_HAND && nanRun.buf.length === 0, `state ${nanRun.state}`);
}

// ---------------------------------------------------------------- 4. the rising edge
//
// A track arms only from a hand that was PARKED in a gate-passing pose. The I hand is carried
// rigidly along a half-circle of one palm width: the handshape never changes, so this is exactly
// the case the rigidity veto must NOT kill, and the pose it starts from is the J launch pose.

console.log('\n--- the rising edge ---');
{
  const fps = 30;
  const frames = stroke(0, fps);
  const { seg, emissions, events, states, trace } = run(frames);
  const scored = events.filter((e) => e.reason === 'scored');
  check('a parked, gate-passing hand that moves enters TRACKING', states.has(TRACKING),
        `states seen: ${[...states].join(',')}`);
  check('the track is cut and scored exactly once', scored.length === 1,
        `${events.length} spans: ${events.map((e) => e.reason).join(' | ')}`);
  if (scored.length === 1) {
    const e = scored[0];
    check('the cut is back-dated by LEAD_IN', e.times[0] <= 0.8 + 1e-9,
          `span starts at ${e.times[0].toFixed(3)}s, stroke starts at 0.800s`);
    check('the scored span is within [T_MIN, T_MAX]',
          e.duration >= TH.T_MIN && e.duration <= TH.T_MAX, `${e.duration.toFixed(3)}s`);
    check('the arm is the gate that passed', e.arm === 'J', `arm ${e.arm}`);
  }
  // A rigid hand carried through an arc is what sigmaRigid exists to forgive: the rotation-
  // aligned deviation stays near zero however far the palm travels, so RIGID_VETO never fires.
  const worst = Math.max(...trace.map((r) => r.sigmaRigid));
  check('a rigidly translated hand never trips RIGID_VETO', worst < TH.RIGID_VETO,
        `max sigmaRigid ${worst.toExponential(2)} vs ${TH.RIGID_VETO}`);
  console.log(`      emissions: [${emissions.map((e) => `${e.letter}/${e.kind}@${e.t.toFixed(2)}s`).join(', ')}]`);
  console.log(`      peak vBar: ${Math.max(...trace.map((r) => r.vBar)).toFixed(2)} palm/s`);

  // A hand that TURNS on the spot is the case the two sigmas were split for. The plain measure
  // must see it (a rotating hand is not parked, so SHAPE_STABLE breaks the hold); the rotation-
  // aligned one must not (a rigid hand is rigid however far it turns), or RIGID_VETO kills real
  // Js for turning their wrist -- which it did, and which cost 90% of genuine gestures.
  const turn = run([...span(I, 0, 0.6, fps), ...turning(0.6, 1.0, fps)]);
  const late = turn.trace.filter((r) => r.t >= 0.9);
  const maxSigma = Math.max(...late.map((r) => r.sigma));
  const maxRigid = Math.max(...late.map((r) => r.sigmaRigid));
  check('a turning hand breaks the hold on the plain sigma', maxSigma > TH.SHAPE_STABLE,
        `max sigma ${maxSigma.toFixed(3)} vs SHAPE_STABLE ${TH.SHAPE_STABLE}`);
  // Not identically zero: the reference is a per-column median over the trailing window, and the
  // componentwise median of a set of ROTATED shapes is not itself a rotation of the shape, so a
  // small residual survives the alignment. Two orders of magnitude below the plain sigma is the
  // claim, and the exact value is what the Python cross-check compares.
  check('...while the rotation-aligned sigma stays near zero',
        maxRigid < maxSigma / 10 && maxRigid < TH.RIGID_VETO / 100,
        `max sigmaRigid ${maxRigid.toExponential(2)} against plain sigma ` +
        `${maxSigma.toFixed(3)} and RIGID_VETO ${TH.RIGID_VETO}`);

  // A hand that never parked must not arm, however gate-passing the pose. This is the case
  // REQUIRE_PARKED was added for: ordinary fingerspelling travel used to fire tracks, and a
  // running track blocks the static branch, which is why the other letters degraded.
  //
  // The drift runs at ~1 palm/s -- above V_STILL, below V_MOVE_ARMED -- for longer than
  // GATE_ARM_WINDOW before the stroke, so the pre-onset window holds no parked frame. It has to
  // be that long: the very first frame of any stream reports vBar 0 (there is nothing to
  // difference against yet) and so counts as parked, in this port and in the Python alike.
  const drift = (s) => [-1.2 * R * s, 0];
  const transit = [
    ...span(I, 0, 1.2, fps, drift),
    ...span(I, 1.2, 0.6, fps, (s) => [-1.2 * R + arc(s)[0], arc(s)[1]]),
    ...span(I, 1.8, 0.8, fps, () => [-1.2 * R + arc(1)[0], arc(1)[1]]),
  ];
  const never = run(transit);
  check('a hand already in transit never arms a track',
        !never.states.has(TRACKING) && never.events.length === 0,
        `states seen: ${[...never.states].join(',')}`);
  // ...and the same stream DOES arm with REQUIRE_PARKED off, which is what proves the check
  // above is testing the parked rule rather than a stroke that simply never crossed V_MOVE_ARMED.
  const unparked = run(transit, { th: { ...TH, REQUIRE_PARKED: false } });
  check('...and would have armed without REQUIRE_PARKED', unparked.states.has(TRACKING),
        `states seen: ${[...unparked.states].join(',')}`);
}

// ---------------------------------------------------------------- 5. flush
//
// Offline replay only: a file trimmed tight to the gesture ends before the fall-confirm, and the
// track would otherwise be dropped in silence.

console.log('\n--- flush ---');
{
  const fps = 30;
  const frames = stroke(0, fps, { rest: 0 });   // the stream stops as the stroke ends
  const { seg, events } = run(frames);
  check('the stream ends mid-track', seg.state === TRACKING, `state ${seg.state}`);
  const before = events.length;
  seg.flush(1.4);
  check('flush scores the track still in flight', events.length === before + 1,
        `${events.length - before} span(s) cut by flush`);
  check('flush on a settled segmenter returns null', seg.flush(1.5) === null);
}

// ---------------------------------------------------------------- 6. tracks that end badly
//
// An aborted track leaves no other trace, which is why _reportSpan hands the observer the span
// however it ends. Both aborts here are purely temporal, so a synthetic can reach them honestly.
//
// RIGID_VETO is deliberately NOT exercised here: morphing the I hand all the way into an A peaks
// at sigmaRigid 0.32 against a threshold of 1.15, exactly as thresholds.py records (held signs
// top out at 0.35 under the aligned measure; 1.15 was calibrated on 90 real gestures). A
// synthetic that tripped it would have to distort the hand past anything MediaPipe emits, and
// would test the number rather than the behavior.

console.log('\n--- tracks that end badly ---');
{
  const fps = 30;
  const { events, seg } = run(runaway(0, fps));
  check('a track that never stops is aborted at T_MAX', events.length === 1
        && events[0].reason.startsWith('abort:T_MAX'), `[${events.map((e) => e.reason).join(', ')}]`);
  check('...and the segmenter goes back to SETTLING, not HOLD',
        seg.state === SETTLING, `state ${seg.state}`);

  // A detection gap longer than GAP_INTERP but shorter than GAP_RESET: emit nothing rather than
  // classify a half-seen path, but do not clear the history -- the hand has not left.
  const lost = [
    ...span(I, 0, 0.8, fps),
    ...span(I, 0.8, 0.3, fps, (s) => arc(0.5 * s)),
    ...gap(1.1, 0.4, fps),
  ];
  const g = run(lost);
  check('a track interrupted by a detection gap is aborted, not scored',
        g.events.length === 1 && g.events[0].reason.startsWith('abort:GAP'),
        `[${g.events.map((e) => e.reason).join(', ')}]`);
  check('...and the gap being shorter than GAP_RESET keeps the history',
        g.seg.state === SETTLING && g.seg.buf.length > 0,
        `state ${g.seg.state}, ${g.seg.buf.length} frames buffered`);
  check('...emitting nothing', g.emissions.filter((e) => e.kind === 'motion').length === 0);
}

// ---------------------------------------------------------------- 7. two letters in a row
//
// Emission is edge-triggered on the hold, so spelling two different letters needs no gap between
// them: the handshape change breaks the hold and the next one forms a new one. The onHold
// callback is checked here too -- a hold that abstains leaves no other trace, so if that
// diagnostic is broken a silently non-emitting letter looks exactly like one never attempted.

console.log('\n--- two letters in a row ---');
{
  const fps = 30;
  const { emissions, holds } = run([
    ...span(A, 0, 0.9, fps),
    ...span(V, 0.9, 1.6, fps),
  ]);
  check('A then V spells "AV"', emissions.map((e) => e.letter).join('') === 'AV',
        `[${emissions.map((e) => `${e.letter}@${e.t.toFixed(2)}`).join(', ')}]`);
  check('one completed vote is reported per letter', holds.length === 2,
        `${holds.length} holds: ${holds.map((h) => `${h.winner}/${h.why}`).join(', ')}`);
  check('the vote diagnostic agrees with what was emitted',
        holds.every((h, k) => h.winner === emissions[k].letter && h.why === 'emitted' && !h.blocked),
        holds.map((h) => `${h.winner} why=${h.why} agree=${h.agree.toFixed(2)} ` +
                         `meanp=${h.meanp.toFixed(3)}`).join(' | '));
  // One predicate decides the emission and the diagnostic. The flag used to test
  // meanp < VOTE_PROB alone and reported decisive-route emissions as blocked.
  const decisive = run([...span(A, 0, 0.9, fps), ...span(V, 0.9, 1.6, fps)],
                       { th: { ...TH, VOTE_PROB: 1.01 } }).holds;
  check('blocked is (why !== "emitted"), also on the decisive route',
        decisive.length > 0 && decisive.every((h) => h.blocked === (h.why !== 'emitted'))
        && decisive.some((h) => h.why === 'emitted'),
        decisive.map((h) => `${h.winner} why=${h.why} blocked=${h.blocked}`).join(' | '));
  check('...and carries the landmarks that produced it, for offline replay',
        holds.every((h) => h.P.length === 21 && h.top3.length === 3
                           && Number.isFinite(h.geom.thumb_pinkymcp)));
}

// ---------------------------------------------------------------- 8. the pending launch letter
//
// A voted I is parked for D_WAIT. If a track arms from that same pose while it is parked, the
// timer no longer decides: the I is HELD until the track resolves. A J/Z emission cancels it
// (the output is "J", not "IJ"); an abort or an abstention releases it (a plain I that moved
// on is still an I). Measured on the committed takes before this rule: 38 releases mid-track,
// 0 cancels -- every J read "IJ".

console.log('\n--- the pending launch letter ---');
{
  const fps = 30;
  // Park just long enough for the vote to complete (HOLD_SETTLE, then four votes: ~0.40 s at
  // 30 fps) and start the stroke at once, so the track arms BEFORE the D_WAIT timer would have
  // released the I: the smoothed speed confirms a rise about 0.23 s after the stroke begins
  // (V_SMOOTH_WINDOW, then V_MOVE_ARMED_SUSTAIN). The premise is asserted, not assumed: the
  // first TRACKING frame must come before the vote's t + D_WAIT.
  const park = 0.48;
  const armedWithin = (trace, holds) => {
    const vote = holds.find((h) => h.why === 'emitted' && h.winner === 'I');
    const first = trace.find((r) => r.state === TRACKING);
    return { vote, first, ok: Boolean(vote && first) && first.t < vote.t + TH.D_WAIT };
  };

  // (a) the track resolves as a J: the parked I is canceled.
  const j = run(stroke(0, fps, { park }), { motionModel: ALWAYS_J });
  const pa = armedWithin(j.trace, j.holds);
  check('premise: the track arms inside D_WAIT of the I vote', pa.ok,
        pa.vote && pa.first ? `vote ${pa.vote.t.toFixed(3)}s, D_WAIT ${TH.D_WAIT}, first TRACKING ` +
                              `${pa.first.t.toFixed(3)}s` : 'no vote or no TRACKING frame');
  check('a J that starts from the parked I cancels it: "J", not "IJ"',
        j.emissions.map((e) => e.letter).join('') === 'J',
        `[${j.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
  check('the launch pose is the one the rule names', LAUNCH_POSE.J === 'I' && LAUNCH_POSE.Z === 'D');

  // (b) the track is aborted by a detection gap: the I is released, not lost.
  const lost = [
    ...span(I, 0, park, fps),
    ...span(I, park, 0.3, fps, (s) => arc(0.5 * s)),
    ...gap(park + 0.3, 0.4, fps),
  ];
  const g = run(lost, { motionModel: ALWAYS_J });
  check('a track aborted by a gap releases the held I',
        g.events.length === 1 && g.events[0].reason.startsWith('abort:GAP')
        && g.emissions.map((e) => e.letter).join('') === 'I',
        `[${g.events.map((e) => e.reason).join(', ')}] -> [${g.emissions.map((e) => e.letter).join(',')}]`);
  check('...after the abort, not on the D_WAIT timer',
        g.emissions.length === 1 && g.emissions[0].delivered >= g.events[0].t_end - 1e-9,
        g.emissions.length ? `released ${g.emissions[0].delivered.toFixed(3)}s, abort ` +
                             `${g.events[0].t_end.toFixed(3)}s` : '');

  // (c) the track scores and the real forest abstains (the synthetic arc is not a J): released.
  const a = run(stroke(0, fps, { park }));
  const scoredAt = a.events.find((e) => e.reason === 'scored');
  check('an abstained track releases the held I',
        Boolean(scoredAt) && a.emissions.map((e) => e.letter).join('') === 'I'
        && a.emissions[0].delivered >= scoredAt.t_end - 1e-9,
        `[${a.emissions.map((e) => `${e.letter}@${e.delivered.toFixed(2)}`).join(', ')}]` +
        (scoredAt ? ` scored at ${scoredAt.t_end.toFixed(2)}s` : ' (no scored span)'));

  // (a) again at the browser's common rate. VOTE_MIN is a frame count, so the I vote lands
  // earlier at 60 fps (0.317 s against 0.400 s) and the deadline with it; the stroke still has
  // to arm inside D_WAIT of the vote for this route, and at park 0.40 it does at both rates.
  const j60 = run(stroke(0, 60, { park: 0.40 }), { motionModel: ALWAYS_J });
  const p60 = armedWithin(j60.trace, j60.holds);
  check('60 fps, premise: the track arms inside D_WAIT of the I vote', p60.ok,
        p60.vote && p60.first ? `vote ${p60.vote.t.toFixed(3)}s, first TRACKING ${p60.first.t.toFixed(3)}s`
                              : 'no vote or no TRACKING frame');
  check('60 fps: "J", not "IJ"', j60.emissions.map((e) => e.letter).join('') === 'J',
        `[${j60.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);

  // (d) THE GRAY ZONE. The stroke begins inside D_WAIT but the smoothed speed is still
  // confirming the rise when the deadline arrives (rise seen, V_MOVE_ARMED_SUSTAIN not yet
  // elapsed, no arm). The timer then waits for the arm decision instead of releasing the I:
  // the arm lands within one sustain of the deadline and the string is "J". Under the timer
  // alone it read "IJ" -- on the takes 2 of the 4 remaining "IJ" were exactly this. Swept at
  // 30 fps (scratch sweep_park: park 0.52-0.68 s is the zone, 0.70 s and beyond is late) and
  // 60 fps (0.46-0.58 s). The premise is asserted: the first TRACKING frame must come at or
  // after the deadline and no later than one sustain plus a frame after it.
  const grayZone = (trace, holds, rate) => {
    const vote = holds.find((h) => h.why === 'emitted' && h.winner === 'I');
    const first = trace.find((r) => r.state === TRACKING);
    if (!vote || !first) return { vote, first, ok: false };
    const deadline = vote.t + TH.D_WAIT;
    return { vote, first, deadline,
             ok: first.t >= deadline - 1e-9
                 && first.t <= deadline + TH.V_MOVE_ARMED_SUSTAIN + 1 / rate + 1e-9 };
  };
  // The timer-only rule, for the control: a subclass whose _checkPending has no gray zone.
  // Everything else (the hold, the pending park, the arm and its _pendingHeld) is inherited.
  class TimerOnly extends Segmenter {
    _checkPending(t) {
      if (this._pending === null || t < this._pendingUntil) return null;
      if (this._pendingHeld) return null;
      const em = this._pending;
      this._pending = null;
      this._pendingHeld = false;
      this.lastEmitted = em.letter;
      return em;
    }
  }
  for (const [rate, grayPark] of [[30, 0.60], [60, 0.52]]) {
    const frames = stroke(0, rate, { park: grayPark });
    const gz = run(frames, { motionModel: ALWAYS_J });
    const pg = grayZone(gz.trace, gz.holds, rate);
    check(`${rate} fps, premise: the arm lands after the deadline but within one sustain`, pg.ok,
          pg.vote && pg.first ? `vote ${pg.vote.t.toFixed(3)}s, deadline ${pg.deadline.toFixed(3)}s, ` +
                                `first TRACKING ${pg.first.t.toFixed(3)}s, sustain ${TH.V_MOVE_ARMED_SUSTAIN}`
                              : 'no vote or no TRACKING frame');
    check(`${rate} fps: a rise still being confirmed at the deadline holds the I: "J", not "IJ"`,
          gz.emissions.map((e) => e.letter).join('') === 'J',
          `[${gz.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
    // The control: the same stream through the timer-only rule reads "IJ", so the check above
    // is testing the gray zone and not a stroke that happened to arm inside D_WAIT.
    const ctl = [];
    const timer = new TimerOnly(TH, { staticModel: STATIC, motionModel: ALWAYS_J });
    for (const f of frames) { const em = timer.step(f.t, f.landmarks, 'Left', W, H); if (em) ctl.push(em.letter); }
    check(`${rate} fps: ...and would have read "IJ" on the timer alone`, ctl.join('') === 'IJ',
          `[${ctl.join(',')}]`);
  }

  // (e) An I held well past D_WAIT before the stroke is released by the timer, by design:
  // deferring I and D to hold exit was measured and rejected for its latency (S1 I 0.92 ->
  // 4.01 s). This is "IJ" and stays "IJ".
  const late = run(stroke(0, fps, { park: 1.0 }), { motionModel: ALWAYS_J });
  const pl = grayZone(late.trace, late.holds, fps);
  check('premise: the stroke arms more than one sustain after the deadline',
        Boolean(pl.vote && pl.first) && pl.first.t > pl.deadline + TH.V_MOVE_ARMED_SUSTAIN + 1 / fps,
        pl.vote && pl.first ? `deadline ${pl.deadline.toFixed(3)}s, first TRACKING ${pl.first.t.toFixed(3)}s`
                            : 'no vote or no TRACKING frame');
  check('an I held 1 s before the stroke is released on the timer: "IJ" by design',
        late.emissions.map((e) => e.letter).join('') === 'IJ',
        `[${late.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
}

// ---------------------------------------------------------------- 9. after a motion letter
//
// The pose a J finishes in is an I, and it is not a new letter. The static branch stays silent
// while the hand rests there; a reshape (sigma > SHAPE_STABLE) clears the suppression at once,
// and so does moving again after having been still. On the recorded takes, freezing the
// finishing pose for 1.5 s after each of 75 emissions produced a static letter 15 times with
// the shipped code and once with this rule; morphing straight into the next letter still
// emitted it 75/75.

console.log('\n--- after a motion letter ---');
{
  const fps = 30;
  // Parked long enough that the I is delivered before the stroke (this is not section 8), so
  // the expected string is exactly I J and nothing after.
  const pre = [...span(I, 0, 1.5, fps), ...span(I, 1.5, 0.6, fps, arc)];
  const end = () => arc(1);

  // (a) freeze the finishing pose 1.5 s, then drop the hand: no static letter after the J.
  const frozen = run([...pre, ...span(I, 2.1, 1.5, fps, end), ...gap(3.6, 1.0, fps)],
                     { motionModel: ALWAYS_J });
  const fl = frozen.emissions.map((e) => e.letter);
  check('the finishing pose held 1.5 s emits nothing after the J', fl.join('') === 'IJ',
        `[${frozen.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
  // The verdict order is the Python's: COOLDOWN_MOTION answers the first votes after the
  // emission, post_motion the ones after it expires; nothing is emitted by either.
  const after = frozen.holds.filter((h) => h.t > frozen.emissions[fl.indexOf('J')]?.t);
  check('...and the vote says why: cooldown, then post_motion', after.length > 0
        && after.every((h) => (h.why === 'post_motion' || h.why === 'cooldown') && h.blocked)
        && after[after.length - 1].why === 'post_motion',
        after.map((h) => `${h.winner}/${h.why}`).join(', ') || 'no vote completed');
  check('the motion emission records how the track ended',
        ['stopped', 'slowed'].includes(frozen.emissions[fl.indexOf('J')]?.detail.end),
        `end=${frozen.emissions[fl.indexOf('J')]?.detail.end}`);

  // (b) morph into V 0.3 s after the emission: the reshape clears the suppression, V emits.
  const jAt = frozen.emissions[fl.indexOf('J')].t;
  const linger = Math.max(0, jAt + 0.3 - 2.1);
  const from = I.landmarks.map((p, k) => translate([p], ...end())[0]);
  const morph = (s) => from.map((p, k) => [
    p[0] + (V.landmarks[k][0] - p[0]) * s, p[1] + (V.landmarks[k][1] - p[1]) * s, p[2]]);
  const t1 = 2.1 + linger;
  const morphed = [
    ...pre,
    ...span(I, 2.1, linger, fps, end),
    ...Array.from({ length: Math.round(0.3 * fps) }, (_, i) => ({
      t: t1 + i / fps, landmarks: morph((i + 1) / Math.round(0.3 * fps)) })),
    ...span(V, t1 + 0.3, 1.5, fps),
  ];
  const m = run(morphed, { motionModel: ALWAYS_J });
  check('morphing into V 0.3 s after the J still emits the V',
        m.emissions.map((e) => e.letter).join('') === 'IJV',
        `[${m.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
  const vEm = m.emissions.find((e) => e.letter === 'V');
  // The ordinary latency of a letter that forms after a reshape: sigma is measured against
  // the trailing SHAPE_WINDOW median, which still holds morph frames for that long, then
  // HOLD_SETTLE and the vote. No post-motion delay on top of that.
  check('...within the ordinary hold latency of the V forming',
        Boolean(vEm) && vEm.t - (t1 + 0.3) <= TH.SHAPE_WINDOW + TH.HOLD_SETTLE + TH.VOTE_WINDOW + 2 / fps,
        vEm ? `${(vEm.t - (t1 + 0.3)).toFixed(3)}s after the morph ended (budget ` +
              `${(TH.SHAPE_WINDOW + TH.HOLD_SETTLE + TH.VOTE_WINDOW).toFixed(2)}s)` : 'no V');

  // (c) moving again after being still clears it too: freeze, then carry the I away and park.
  const carried = [
    ...pre,
    ...span(I, 2.1, 0.8, fps, end),
    ...span(I, 2.9, 0.5, fps, (s) => [end()[0] + 2.0 * R * s, end()[1]]),
    ...span(I, 3.4, 1.8, fps, () => [end()[0] + 2.0 * R, end()[1]]),
  ];
  const c = run(carried, { motionModel: ALWAYS_J });
  const cl = c.emissions.map((e) => e.letter).join('');
  // The carried I arms a second track (it starts from a parked, gate-passing pose); the
  // straightness veto rejects that straight carry before the stub is asked, so no second J.
  // Either way the I parked at the end (delivered D_WAIT after its vote) is a NEW letter, and
  // the static branch must be free to say so.
  check('an I parked again after moving away is a new letter', /^IJ(J)?I$/.test(cl),
        `[${c.emissions.map((e) => `${e.letter}/${e.kind}@${e.delivered.toFixed(2)}`).join(', ')}]`);
}

// ---------------------------------------------------------------- 10. arming off
//
// Numbers mode runs the same machine with armGates false: zGate passes most '1' frames, so a
// '1' carried across the frame would arm a Z track and park the machine in TRACKING, where no
// static vote happens, until T_MAX, the veto or a fall-confirm ends it. With arming off no
// track can start. The gate fractions are still computed for the overlay.

console.log('\n--- arming off ---');
{
  const fps = 30;
  const on = run(stroke(0, fps));
  const off = run(stroke(0, fps), { armGates: false });
  check('the same stroke arms with the default', on.states.has(TRACKING) && on.events.length === 1,
        `states ${[...on.states].join(',')}`);
  check('...and never enters TRACKING with armGates false',
        !off.states.has(TRACKING) && off.events.length === 0,
        `states ${[...off.states].join(',')}`);
  check('the gate fractions are still computed for the overlay',
        off.seg._gateFraction(0.7, 'jGate') >= TH.GATE_ARM_FRAC,
        `jf ${off.seg._gateFraction(0.7, 'jGate').toFixed(2)}`);

  // The stream the digits-mode measurement used: hold a '1' 1.2 s, carry it 0.5 s at
  // 2.5 palm/s, hold 1.2 s, drop the hand, hold a '0' 1.5 s (Ankara landmarks, 100x100,
  // handedness Left). Python: arm_gates True 17 TRACKING frames, False 0; '1' then '0' both.
  const one = golden.cases.find((c) => c.model === 'digits' && c.label === '1');
  const zero = golden.cases.find((c) => c.model === 'digits' && c.label === '0');
  if (!DIGITS || !one || !zero) {
    console.log('skip  the \'1\'-carry stream needs models.json\'s digits forest and golden.json\'s '
                + 'digit cases (export_models.py with temporal/model_digits.p present)');
  } else {
    const Rd = one.palm_scale;                         // 100x100: isotropic == normalized
    const carry = (s) => [2.5 * Rd * 0.5 * s, 0];
    const move = (c, t0, sec, mv) => Array.from({ length: Math.round(sec * fps) }, (_, i) => {
      const s = Math.round(sec * fps) > 1 ? i / (Math.round(sec * fps) - 1) : 0;
      const [du, dv] = mv(s);
      return { t: t0 + i / fps, landmarks: c.landmarks.map((p) => [p[0] + du, p[1] + dv, p[2]]) };
    });
    const digitStream = [
      ...move(one, 0, 1.2, () => [0, 0]),
      ...move(one, 1.2, 0.5, carry),
      ...move(one, 1.7, 1.2, () => carry(1)),
      ...gap(2.9, 0.6, fps),
      ...move(zero, 3.5, 1.5, () => [0, 0]),
    ];
    const drive = (armGates) => {
      const seg = new Segmenter(TH_DIGITS, { staticModel: DIGITS, motionModel: null, armGates });
      const emissions = [];
      const states = new Set();
      let tracking = 0;
      for (const f of digitStream) {
        const em = seg.step(f.t, f.landmarks, 'Left', one.width, one.height);
        states.add(seg.state);
        if (seg.state === TRACKING) tracking += 1;
        if (em) emissions.push(em);
      }
      return { emissions, states, tracking, seg };
    };
    const gatesOn = drive(true);
    const gatesOff = drive(false);
    check("a carried '1' arms a Z track with the gates on (why the switch exists)",
          gatesOn.states.has(TRACKING) && gatesOn.tracking > 0,
          `${gatesOn.tracking} TRACKING frames`);
    check("...and never enters TRACKING with armGates false", !gatesOff.states.has(TRACKING),
          `states ${[...gatesOff.states].join(',')}`);
    check("'1' then '0' are emitted, once each",
          gatesOff.emissions.map((e) => e.letter).join('') === '10'
          && gatesOn.emissions.map((e) => e.letter).join('') === '10',
          `off [${gatesOff.emissions.map((e) => `${e.letter}@${e.t.toFixed(2)}`).join(', ')}] ` +
          `on [${gatesOn.emissions.map((e) => `${e.letter}@${e.t.toFixed(2)}`).join(', ')}]`);
    check('the digits segmenter votes with the digit forest\'s feature',
          gatesOff.seg.staticFeatureTag === payload.digits.feature
          && gatesOff.seg.staticClasses.join('') === '0123456789',
          `${gatesOff.seg.staticFeatureTag}, classes ${gatesOff.seg.staticClasses.join('')}`);
  }
}

// ---------------------------------------------------------------- 11. the frame size is required
//
// The Python raises on a missing width/height (float(None)); this port used to divide null by
// null and every signal went NaN, which compares false against every threshold and wedged the
// machine in SETTLING with no error anywhere -- a replay script that forgot the size looked
// like a segmenter that never emits.

console.log('\n--- the frame size is required ---');
{
  const seg = new Segmenter(TH, { staticModel: STATIC });
  let threw = null;
  try { seg.step(0.0, A.landmarks, 'Left'); } catch (e) { threw = e; }
  check('step() without width/height throws', threw instanceof TypeError,
        threw ? threw.message : 'no error');
  check('...and a non-detection frame does not need them', seg.step(0.1, null, null) === null);
  let bad = null;
  try { seg.step(0.2, A.landmarks, 'Left', 640, 0); } catch (e) { bad = e; }
  check('a zero height throws too', bad instanceof TypeError, bad ? bad.message : 'no error');
  check('the static feature follows the model\'s tag',
        seg.staticFeatureTag === (payload.static.feature || 'static/v3'),
        `${seg.staticFeatureTag} for a ${payload.static.feature} forest of dim ${STATIC.dim}`);
  let mis = null;
  try {
    new Segmenter(TH, { staticModel: STATIC,
                        staticFeatureTag: STATIC.dim === 112 ? 'static/v3' : 'static/v4' });
  } catch (e) { mis = e; }
  check('a tag whose width disagrees with the forest is refused', mis !== null,
        mis ? mis.message : 'accepted');
}

// ---------------------------------------------------------------- 12. the handedness latch
//
// MediaPipe labels the hand per frame, and canonicalizeHandedness mirrors a "Left" hand. One
// frame labeled the other way mid-track mirrors the hand for that frame alone: sigmaRigid
// jumps past RIGID_VETO and the track aborts. On the committed takes 166 of 13,692 detected
// frames carry the other label (longest flip inside an unbroken detection 5 frames / 0.20 s)
// and the raw per-frame label lost 6 of 113 gestures that the modal label credits. The
// segmenter keeps the label it has and switches only after HAND_SWITCH_S of the other label;
// GAP_RESET clears the latch.

console.log('\n--- the handedness latch ---');
{
  const fps = 30;
  const canon = (lm, hand) => canonicalizeHandedness(toIsotropic(lm, W, H), hand);
  const samePoints = (P, Q) => P.length === Q.length
    && P.every((p, i) => Math.abs(p[0] - Q[i][0]) < 1e-12 && Math.abs(p[1] - Q[i][1]) < 1e-12);

  // (a) one frame labeled "Right" in the middle of a J stroke (the rest "Left", the section-8
  // stream so the string is exactly "J"): the frame is canonicalized with the latched Left,
  // the track survives and the J is emitted.
  const park = 0.48;
  const frames = stroke(0, fps, { park });
  const mid = frames.findIndex((f) => f.t >= park + 0.3);
  const flipped = frames.map((f, i) => (i === mid ? { ...f, handedness: 'Right' } : f));
  const one = run(flipped, { motionModel: ALWAYS_J });
  check('a one-frame flip mid-stroke still emits the J',
        one.trace[mid].state === TRACKING
        && one.emissions.map((e) => e.letter).join('') === 'J'
        && one.events.filter((e) => e.reason === 'scored').length === 1
        && !one.events.some((e) => e.reason.startsWith('abort')),
        `flip at ${frames[mid].t.toFixed(3)}s (${one.trace[mid].state}) -> ` +
        `[${one.emissions.map((e) => e.letter).join(',')}] spans: ${one.events.map((e) => e.reason).join(' | ')}`);
  check('...the latch never moved', one.trace.every((r) => r.hand === 'Left'),
        `hands seen: ${[...new Set(one.trace.map((r) => r.hand))].join(',')}`);
  check('...and the flipped frame was canonicalized with the latched hand, not mirrored',
        samePoints(one.trace[mid].P, canon(frames[mid].landmarks, 'Left'))
        && !samePoints(one.trace[mid].P, canon(frames[mid].landmarks, 'Right'))
        && one.trace[mid].sigma < 1e-9,
        `sigma ${one.trace[mid].sigma.toExponential(1)} at the flipped frame`);
  // The control: what that one frame looks like when the mirror actually happens. Its
  // landmarks are mirrored in place (x -> -x, the negation canonicalizeHandedness applies
  // to u) under the unchanged "Left" label, so the latch cannot intervene. The rigidly
  // carried hand that reads sigma 0 above becomes a 0.57 reshape and a ~20 palm/s jump for
  // that frame. This frontal synthetic fist keeps its mirror's rotation-aligned deviation at
  // 0.43, under RIGID_VETO 1.15 (measured over every golden case: 0.35-0.92), so the synthetic
  // cannot reproduce the abort the flips caused on the takes; what it pins is that the
  // latched frame is the SAME points, not a mirror the machine then has to survive.
  const mirrored = frames.map((f, i) => (i === mid
    ? { ...f, landmarks: f.landmarks.map((p) => [-p[0], p[1], p[2]]) } : f));
  const ctl = run(mirrored, { motionModel: ALWAYS_J });
  check('control: the same frame actually mirrored is a reshape and a jump',
        ctl.trace[mid].sigma > TH.SHAPE_STABLE && ctl.trace[mid].vBar > 10
        && samePoints(ctl.trace[mid].P, canon(frames[mid].landmarks, 'Right')),
        `sigma ${ctl.trace[mid].sigma.toFixed(3)} (SHAPE_STABLE ${TH.SHAPE_STABLE}), vBar ` +
        `${ctl.trace[mid].vBar.toFixed(1)} palm/s, sigmaRigid ${ctl.trace[mid].sigmaRigid.toFixed(3)}`);

  // (b) a flip shorter than HAND_SWITCH_S never switches, however many frames it spans.
  const brief = span(I, 0, 2.0, fps).map((f) => (f.t >= 1.0 && f.t < 1.0 + TH.HAND_SWITCH_S - 0.1
    ? { ...f, handedness: 'Right' } : f));
  const b = run(brief);
  check('a flip shorter than HAND_SWITCH_S never switches the latch',
        b.trace.every((r) => r.hand === 'Left') && b.seg._handOtherSince === null,
        `hands seen: ${[...new Set(b.trace.map((r) => r.hand))].join(',')}`);

  // (c) a sustained flip switches after HAND_SWITCH_S, and every frame from then on is
  // canonicalized with the new label.
  const held = span(I, 0, 2.0, fps);
  const sustained = held.map((f) => (f.t >= 1.0 ? { ...f, handedness: 'Right' } : f));
  const s = run(sustained);
  const switchedAt = s.trace.findIndex((r) => r.hand === 'Right');
  const dt = switchedAt >= 0 ? s.trace[switchedAt].t - 1.0 : NaN;
  check('a sustained flip switches the latch after HAND_SWITCH_S', switchedAt >= 0
        && dt >= TH.HAND_SWITCH_S - 1e-9 && dt < TH.HAND_SWITCH_S + 1 / fps + 1e-9,
        switchedAt >= 0 ? `switched ${dt.toFixed(3)}s after the flip began (HAND_SWITCH_S ${TH.HAND_SWITCH_S})`
                        : 'never switched');
  check('...Left before the switch, Right from then on',
        s.trace.every((r, i) => r.hand === (i < switchedAt ? 'Left' : 'Right')));
  const last = s.trace[s.trace.length - 1];
  check('...and frames after the switch are canonicalized with the new label',
        samePoints(last.P, canon(held[held.length - 1].landmarks, 'Right'))
        && !samePoints(last.P, canon(held[held.length - 1].landmarks, 'Left')));

  // (d) GAP_RESET clears the latch; the first frame after the gap adopts its own label at
  // once; a detected frame with no real label keeps the latched hand.
  const seg = s.seg;
  seg.step(2.0 + TH.GAP_RESET + 0.05, null, null, W, H);
  check('GAP_RESET clears the latch', seg.state === NO_HAND && seg.hand === null
        && seg._handOtherSince === null, `hand ${seg.hand}`);
  seg.step(2.0 + TH.GAP_RESET + 0.10, I.landmarks, 'Left', W, H);
  check('the first frame after the gap adopts its own label at once', seg.hand === 'Left',
        `hand ${seg.hand}`);
  seg.step(2.0 + TH.GAP_RESET + 0.15, I.landmarks, 'Unknown', W, H);
  seg.step(2.0 + TH.GAP_RESET + 0.20, I.landmarks, null, W, H);
  check('a detected frame with no real label keeps the latched hand', seg.hand === 'Left',
        `hand ${seg.hand}`);
  const P = seg.buf[seg.buf.length - 1].P;
  check('...and is canonicalized with it', samePoints(P, canon(I.landmarks, 'Left')));
  // The digits segmenter runs the same latch (armGates off changes nothing here).
  const dseg = new Segmenter(TH, { staticModel: STATIC, armGates: false });
  dseg.step(0, I.landmarks, 'Right', W, H);
  dseg.step(1 / fps, I.landmarks, 'Left', W, H);
  check('the latch applies with armGates off too', dseg.hand === 'Right'
        && dseg._handOtherSince === 1 / fps, `hand ${dseg.hand}, other since ${dseg._handOtherSince}`);
}

// ---------------------------------------------------------------- optional trace dump

const dumpAt = process.argv.indexOf('--dump');
if (dumpAt >= 0 && process.argv[dumpAt + 1]) {
  const fps = 30;
  // One stream through every path the synthetics can reach: a held letter, a reset, the same
  // letter again, a deferred letter, a scored track, a T_MAX abort and a GAP abort. The long
  // gaps between phases are what put the machine back in NO_HAND between them.
  const frames = [
    ...span(A, 0, 2.0, fps),
    ...gap(2.0, 0.7, fps),
    ...span(A, 2.7, 1.5, fps),
    ...gap(4.2, 0.7, fps),
    ...stroke(4.9, fps),                        // 4.9 -> 7.3, scored
    ...gap(7.3, 0.7, fps),
    ...runaway(8.0, fps),                       // 8.0 -> 11.4, abort:T_MAX
    ...gap(11.4, 0.7, fps),
    ...span(I, 12.1, 0.8, fps),
    ...span(I, 12.9, 0.3, fps, (s) => arc(0.5 * s)),
    ...gap(13.2, 0.4, fps),                     // abort:GAP
    ...gap(13.6, 0.7, fps),
    ...span(A, 14.3, 0.9, fps),                 // two letters with no gap between them
    ...span(V, 15.2, 1.6, fps),
    ...gap(16.8, 0.7, fps),
    ...span(I, 17.5, 0.6, fps),
    ...turning(18.1, 1.0, fps),                 // the rotation-aligned sigma, against numpy's SVD
  ];
  const { emissions, trace, events } = run(frames);
  fs.writeFileSync(process.argv[dumpAt + 1], JSON.stringify({
    // The per-frame P is a reference into the segmenter's buffer for section 12's checks, not
    // part of the signal trace; the latched hand is.
    width: W, height: H, handedness: 'Left', frames, trace: trace.map(({ P, ...r }) => r),
    emissions: emissions.map((e) => ({ letter: e.letter, kind: e.kind, t: e.t,
                                       confidence: e.confidence })),
    // The motion feature and its probabilities for every cut span, recomputed from the span the
    // segmenter reported. _score consumes exactly these, so a cross-check against Python's puts
    // numbers on the branch that a synthetic arc otherwise only exercises as an abstention.
    events: events.map((e) => {
      const feat = Array.from(eventFeatures(e.times, e.P, e.arm));
      return { reason: e.reason, t_end: e.t_end, arm: e.arm, duration: e.duration,
               t0: e.times[0], n: e.times.length, feat,
               probs: Array.from(predictProba(MOTION, feat)) };
    }),
  }));
  console.log(`\ntrace dumped to ${process.argv[dumpAt + 1]} ` +
              `(${frames.length} frames, ${emissions.length} emissions)`);
}

console.log(`\n${failures ? `${failures} FAILING checks` : 'all checks pass'}`);
process.exit(failures ? 1 : 0);

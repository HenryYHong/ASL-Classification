// Behavioural check for segmenter.js.  Run:  /usr/local/bin/node web/test_segmenter.mjs
//
// Reading the port next to temporal/segmenter.py is not evidence. This drives synthetic streams
// built from the real landmarks in golden.json through the real forests in models.json, and
// asserts the behaviour the Python specifies -- including the two mechanisms that broke silently
// in the Python and would break just as silently here:
//
//   * the static vote completes on EITHER a time span OR VOTE_MIN votes, so a held letter emits
//     at 15 fps, 30 fps and 60 fps alike. Gating on the span alone cost 8 of 24 archive letters
//     and was invisible at 30 fps, where it happened to fit.
//   * I and D are held back by D_WAIT, because they are the launch poses for J and Z.
//
// With `--dump <path>` it also writes the generated frames and its own per-frame signal trace,
// which is what temporal/segmenter.py was replayed against to confirm the two implementations
// agree numerically (vBar, sigma, sigmaRigid and every emission).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { prepareModel, predictProba } from './forest.js';
import { eventFeatures } from './features.js';
import { Segmenter, NO_HAND, SETTLING, HOLD, TRACKING } from './segmenter.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(fs.readFileSync(path.join(HERE, 'models.json'), 'utf8'));
const golden = JSON.parse(fs.readFileSync(path.join(HERE, 'golden.json'), 'utf8'));

const TH = payload.thresholds;
const STATIC = prepareModel(payload.static);
const MOTION = prepareModel(payload.motion);
const W = 1920;
const H = 1080;

// Cases 0 and 3 and 6 are the first A, the first D and the first I of golden.json. D and I are
// the deferred pair; I is also the J launch pose (its j_gate is true), which is what lets the
// motion phase below arm a track at all.
const A = golden.cases[0];
const D = golden.cases[3];
const I = golden.cases[6];
const V = golden.cases[12];

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

/** Rotate a hand about its own palm centre by `ang` radians, IN ISOTROPIC SPACE.
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

/** A parked hand that turns on the spot: the palm centre never moves, so vBar stays at zero. */
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
const R = golden.cases[6].palm_scale;             // one palm width, in isotropic units
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
    const em = seg.step(f.t, f.landmarks, 'Left', W, H);
    states.add(seg.state);
    trace.push({ t: f.t, state: seg.state, vBar: seg.vBar, sigma: seg.sigma,
                 sigmaRigid: seg.sigmaRigid });
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
// the hand is how the signer spells a doubled letter, so this is user-visible behaviour, not
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
// would test the number rather than the behaviour.

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
  check('...and carries the landmarks that produced it, for offline replay',
        holds.every((h) => h.P.length === 21 && h.top3.length === 3
                           && Number.isFinite(h.geom.thumb_pinkymcp)));
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
    width: W, height: H, handedness: 'Left', frames, trace,
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

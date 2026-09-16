/* Verify features.js against the golden vectors Python produced. Run: node docs/test_features.mjs
 *
 * Each golden case names its forest (`model`: "static" by default, or "digits"), and the feature
 * function checked for it is the one that forest's tag in models.json selects through
 * STATIC_FEATURES: static/v4 (112-D) for both shipped forests, letters and digits. Cases
 * tagged api "tasks" store the raw Tasks-API handedness label and go through app.js's
 * swapTasksHandedness first, exactly as a live label does.
 *
 * Reading the two implementations and agreeing they look equivalent is not evidence -- every
 * expensive bug in this project was a silent divergence between two implementations that looked
 * equivalent. Only the numbers count.
 *
 * export_models.make_case rounds the landmarks to 6 decimals BEFORE computing every stored
 * field, so a correct port reproduces each field to the stored outputs' own 6-decimal rounding
 * half-step, 0.5e-6, inside the file's 1e-6 tolerance: measured 5.0e-7 worst over the 41 cases,
 * on shape42, static_feature and palm_scale alike. Every field is therefore held to
 * golden.tolerance.features with no allowance on top. Any allowance above that hides a real
 * error: the finite-difference bound this file once carried (from an older export that
 * computed the outputs from unrounded landmarks) let a +6e-6 injected on one component pass.
 *
 * Independently confirmed twice, outside this file: fed the same rounded landmarks, this port and
 * temporal/features.py agree to 4.4e-16 on shape42, 8.9e-16 on static_feature, 5.6e-17 on
 * palm_scale and on every gate -- and over 400 randomized hands (both handedness labels, five
 * aspect ratios, 75 j_gate and 59 z_gate firings) to 1.3e-15 with zero gate disagreements.
 *
 * The golden vectors reach only the five per-frame functions. The temporal half is checked at
 * the bottom of this file against the properties its docstrings claim -- see the note there.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  toIsotropic, canonicalizeHandedness, palmScale, palmCentre, shape42, staticFeature,
  staticFeatureV4, staticFeatureFor, thumbStraightness, tipPalmDistances, thumbPinkyMcp,
  pairDistances, jGate, zGate, rollingShapeSigma, rollingShapeSigmaAligned, movingAverageTime,
  palmSpeed, trailingWindow, windowSpeed, resampleArclength, turningAngles, pathLength,
  eventFeatures, KEY_POINTS, STATIC_DIM, STATIC_DIM_V4, STATIC_FEATURES, THUMB_TARGETS,
  J_THUMB_MAX, SHAPE_DIM, EVENT_DIM, K_RESAMPLE,
} from './features.js';
import { swapTasksHandedness } from './app.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(HERE, 'golden.json'), 'utf8'));
const TOL = golden.tolerance.features;
// The feature tag of each forest, from models.json: what selects the function per case.
const payload = JSON.parse(readFileSync(join(HERE, 'models.json'), 'utf8'));
const TAGS = { static: payload.static.feature };
if (payload.digits) TAGS.digits = payload.digits.feature;

let failures = 0;

/** Report one field: PASS only if EVERY component sits inside golden.tolerance.features. The
 *  printed number is the maximum over components. */
function check(caseIdx, field, got, want, extra = '') {
  let err = got.length === want.length ? 0 : Infinity;
  let ok = got.length === want.length;
  for (let i = 0; i < got.length && got.length === want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > TOL) ok = false;
    if (d > err) err = d;
  }
  if (!ok) failures += 1;
  const shown = Number.isFinite(err) ? err.toExponential(2) : 'LENGTH MISMATCH';
  console.log(`case ${String(caseIdx).padStart(2)}  ${field.padEnd(14)} ` +
              `${ok ? 'PASS' : 'FAIL'}  max|d| = ${shown}  ` +
              `allowed ${TOL.toExponential(2)}${extra}`);
}

// The exact call order export_models.py uses: isotropic first, then chirality. A Tasks-API case
// carries the label the Tasks API reported, which the page swaps before canonicalizing.
const labelOf = (c) => (c.api === 'tasks' ? swapTasksHandedness(c.reported_handedness)
  : c.handedness);
const prepare = (c) => (lm) => canonicalizeHandedness(toIsotropic(lm, c.width, c.height),
                                                      labelOf(c));

let skipped = 0;
golden.cases.forEach((c, idx) => {
  const which = c.model || 'static';
  if (!(which in TAGS)) {
    // The export writes golden.json and models.json together, so a case for a forest the
    // payload does not carry means the two files are from different runs.
    console.log(`case ${String(idx).padStart(2)}  ${which.padEnd(14)} FAIL  models.json carries no ${which} forest`);
    failures += 1;
    return;
  }
  const [featureFn, dim] = staticFeatureFor(TAGS[which]);
  const P = prepare(c)(c.landmarks);

  const s42 = shape42(P);
  const feat = featureFn(P);
  if (s42.length !== SHAPE_DIM) throw new Error(`shape42 built ${s42.length}-D`);
  if (feat.length !== dim) throw new Error(`${TAGS[which]} built ${feat.length}-D, expected ${dim}`);

  const tagNote = `  [${which} ${TAGS[which]} ${dim}-D${c.api === 'tasks' ? ', tasks label swapped' : ''}]`;
  check(idx, 'shape42', s42, c.shape42);
  check(idx, 'static_feature', feat, c.static_feature, tagNote);
  check(idx, 'palm_scale', [palmScale(P)], [c.palm_scale]);

  // The gates are inequalities; they are required to match exactly.
  const j = jGate(P);
  const z = zGate(P);
  check(idx, 'j_gate', [j === c.j_gate ? 0 : 1], [0], `  (got ${j}, want ${c.j_gate})`);
  check(idx, 'z_gate', [z === c.z_gate ? 0 : 1], [0], `  (got ${z}, want ${c.z_gate})`);
});

// The tolerance must stay tight enough to see a port error of the size it once let through:
// a +5e-6 on one component of the first case's own feature vector has to fail.
{
  const c = golden.cases[0];
  const [featureFn] = staticFeatureFor(TAGS[c.model || 'static']);
  const feat = Array.from(featureFn(prepare(c)(c.landmarks)));
  feat[7] += 5e-6;
  const before = failures;
  const quiet = console.log;
  console.log = () => {};
  check(0, 'perturbed', feat, c.static_feature);
  console.log = quiet;
  const caught = failures === before + 1;
  failures = before + (caught ? 0 : 1);
  console.log(`case --  +5e-6 mutant   ${caught ? 'PASS' : 'FAIL'}  a 5e-6 error on one component ` +
              `${caught ? 'fails' : 'PASSES'} the ${TOL.toExponential(0)} tolerance`);
}

// The letter forest ships on static/v4, so every letter case must carry a 112-D vector once the
// export has been regenerated; until then the cases are v3 and this says so instead of failing.
{
  const letterCases = golden.cases.filter((c) => (c.model || 'static') === 'static');
  if (TAGS.static !== 'static/v4') {
    console.log(`skip    112-D golden vectors: models.json's letter forest is ${TAGS.static}, ` +
                'not static/v4 yet (regenerate with export_models.py after train_static.py)');
    skipped += 1;
  } else {
    const all112 = letterCases.every((c) => c.static_feature.length === STATIC_DIM_V4);
    check('--', '112-D letters', [all112 ? 0 : 1], [0],
          `  (${letterCases.length} letter cases, every static_feature ${STATIC_DIM_V4}-D: ${all112})`);
  }
  const tasksCases = golden.cases.filter((c) => c.api === 'tasks');
  const digitCases = golden.cases.filter((c) => c.model === 'digits');
  console.log(`        ${letterCases.length} letter cases (${tasksCases.length} from the Tasks API), ` +
              `${digitCases.length} digit cases`);
}

// A handedness ARRAY must throw rather than silently doing nothing, which is the whole reason
// the Python raises: str(ndarray) starts with "[" and matched neither branch, leaving genuinely
// mirrored clips unflipped. Verify the port kept the guard.
let threw = false;
try {
  canonicalizeHandedness([[0, 0]], ['Left', 'Left']);
} catch (err) {
  threw = err instanceof TypeError;
}
check('--', 'array label', [threw ? 0 : 1], [0],
      threw ? '  (throws TypeError)' : '  (no TypeError raised)');

// --------------------------------------------------------------------------------------
// The golden vectors cover five of this module's functions: everything downstream of one
// still frame. The temporal half -- the signals the segmenter's vetoes and the motion forest
// actually run on -- has no golden data, because export_models.py had no J/Z frames to make
// any from. It is checked here by the properties its docstrings CLAIM, which is the next best
// evidence and is what those claims are worth if nothing tests them. Each assertion below is
// an identity the Python satisfies too, so it fails on a divergence in either direction.
// --------------------------------------------------------------------------------------

let nInvariants = 0;
function invariant(name, ok, detail) {
  nInvariants += 1;
  if (!ok) failures += 1;
  console.log(`inv --  ${name.padEnd(38)} ${ok ? 'PASS' : 'FAIL'}  ${detail}`);
}
const maxAbsDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

// One real hand, carried along an arc while the wrist turns: a synthetic J in all but name.
const HAND = prepare(golden.cases[6])(golden.cases[6].landmarks);
const CENTRE = palmCentre(HAND);
const rotated = [], translated = [], times = [];
for (let i = 0; i < 12; i++) {
  const a = 0.25 * i, cs = Math.cos(a), sn = Math.sin(a);
  rotated.push(HAND.map((p) => {
    const x = p[0] - CENTRE[0], y = p[1] - CENTRE[1];
    return [cs * x - sn * y + CENTRE[0] + 0.05 * i, sn * x + cs * y + CENTRE[1] + 0.03 * i];
  }));
  translated.push(HAND.map((p) => [p[0] + 0.05 * i, p[1] + 0.03 * i]));
  times.push(i * 0.04);
}

// shape42 subtracts the palm center and divides by the palm triangle, so moving the hand across
// the frame or towards the camera must not move the feature at all. This is the normalization
// the whole cross-session story rests on; if it silently stopped holding, accuracy would drop
// only for signers sitting at a different distance, which is the hardest failure to notice.
const scaled = HAND.map((p) => [p[0] * 2.7 + 0.4, p[1] * 2.7 - 0.9]);
invariant('staticFeature scale/translate-free',
          maxAbsDiff(staticFeature(HAND), staticFeature(scaled)) < 1e-12,
          `max|d| = ${maxAbsDiff(staticFeature(HAND), staticFeature(scaled)).toExponential(2)}`);

// A geometrically mirrored hand LABELED Left must land on exactly the same feature as the
// original labeled Right. That is the entire chirality convention, stated as an equation.
const mirrored = HAND.map((p) => [-p[0], p[1]]);
const dChiral = maxAbsDiff(staticFeature(canonicalizeHandedness(mirrored, 'Left')),
                           staticFeature(canonicalizeHandedness(HAND, 'Right')));
invariant('chirality: mirror+Left == Right', dChiral === 0, `max|d| = ${dChiral.toExponential(2)}`);

// static/v4 = static/v3 followed by the 11-value thumb block, in the Python's order. The
// registry is what both sides key on; the block is rebuilt here from its definition (the
// thumb chain, the five tips to the palm center, the thumb tip to 6, 7, 10, 11, 3) and
// required positionally, because a reordering is a silent break the forest reads confidently.
{
  const v4 = staticFeatureV4(HAND);
  const S4 = palmScale(HAND);
  const m = palmCentre(HAND);
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const straight = d(HAND[4], HAND[1]) / (d(HAND[2], HAND[1]) + d(HAND[3], HAND[2]) + d(HAND[4], HAND[3]));
  const tips = [4, 8, 12, 16, 20].map((i) => d(HAND[i], m) / S4);
  const thumbTo = [6, 7, 10, 11, 3].map((j) => d(HAND[4], HAND[j]) / S4);
  const manual = staticFeature(HAND).concat([straight], tips, thumbTo);
  invariant('staticFeatureV4 layout + length',
            v4.length === STATIC_DIM_V4 && manual.length === STATIC_DIM_V4
            && maxAbsDiff(v4, manual) === 0 && v4[101] === thumbStraightness(HAND)
            && maxAbsDiff(v4.slice(102, 107), tipPalmDistances(HAND)) === 0
            && THUMB_TARGETS.join(',') === '6,7,10,11,3',
            `${v4.length}-D = 101 + 1 + 5 + 5, max|d| vs the definition ${maxAbsDiff(v4, manual).toExponential(2)}`);
  const v4s = staticFeatureV4(scaled);
  invariant('staticFeatureV4 scale/translate-free', maxAbsDiff(v4, v4s) < 1e-12,
            `max|d| = ${maxAbsDiff(v4, v4s).toExponential(2)}`);
  const c4 = maxAbsDiff(staticFeatureV4(canonicalizeHandedness(mirrored, 'Left')),
                        staticFeatureV4(canonicalizeHandedness(HAND, 'Right')));
  invariant('staticFeatureV4 chirality', c4 === 0, `max|d| = ${c4.toExponential(2)}`);
  const reg = STATIC_FEATURES;
  invariant('STATIC_FEATURES registry',
            reg['static/v3'][0] === staticFeature && reg['static/v3'][1] === STATIC_DIM
            && reg['static/v4'][0] === staticFeatureV4 && reg['static/v4'][1] === STATIC_DIM_V4
            && staticFeatureFor(null)[0] === staticFeature && Object.keys(reg).length === 2,
            `v3 -> ${STATIC_DIM}, v4 -> ${STATIC_DIM_V4}, null -> v3`);
  let unknown = false;
  try { staticFeatureFor('static/v9'); } catch (e) { unknown = /unknown static feature tag/.test(e.message); }
  invariant('an unregistered tag throws', unknown, 'static/v9 -> Error');
}

// The J gate's thumb ceiling is J_THUMB_MAX (1.30; was 1.20), measured on the prompted takes:
// 1.20 lost 13 of 60 J items at the gate. Pinned here by moving the I hand's thumb tip along the
// line away from the pinky MCP so thumbPinkyMcp lands just under and just over the ceiling;
// nothing else the gate reads (pinky, index, middle, ring extension) moves with it.
{
  const setThumb = (target) => {
    const P = HAND.map((p) => p.slice());
    const S = palmScale(P);
    const dir = [P[4][0] - P[17][0], P[4][1] - P[17][1]];
    const n = Math.hypot(dir[0], dir[1]);
    P[4] = [P[17][0] + (dir[0] / n) * target * S, P[17][1] + (dir[1] / n) * target * S];
    return P;
  };
  const under = setThumb(J_THUMB_MAX - 0.05);
  const over = setThumb(J_THUMB_MAX + 0.05);
  invariant('jGate thumb ceiling is J_THUMB_MAX', J_THUMB_MAX === 1.30 && jGate(HAND)
            && jGate(under) && !jGate(over)
            && Math.abs(thumbPinkyMcp(under) - (J_THUMB_MAX - 0.05)) < 1e-9,
            `1.30: ${thumbPinkyMcp(under).toFixed(2)} passes, ${thumbPinkyMcp(over).toFixed(2)} fails`);
}

// ORDER IS THE FEATURE. Rebuild the distance block from itertools.combinations' own rule --
// i ascending, j > i, over KEY_POINTS AS WRITTEN -- and require it positionally. A sorted or
// transposed loop produces a vector the forest reads confidently and wrongly.
const S = palmScale(HAND);
const manualPairs = [];
for (let i = 0; i < KEY_POINTS.length; i++) {
  for (let j = i + 1; j < KEY_POINTS.length; j++) {
    const a = HAND[KEY_POINTS[i]], b = HAND[KEY_POINTS[j]];
    manualPairs.push(Math.hypot(a[0] - b[0], a[1] - b[1]) / S);
  }
}
const pd = pairDistances(HAND);
invariant('pairDistances order + length',
          pd.length === manualPairs.length + 4 && maxAbsDiff(pd.slice(0, manualPairs.length), manualPairs) === 0
          && SHAPE_DIM + pd.length === STATIC_DIM,
          `${manualPairs.length} pairs + 4 straightness, ${SHAPE_DIM} + ${pd.length} = ${STATIC_DIM}`);

// Both sigmas are deviations of a normalized shape from a normalized reference, so a hand that
// only TRAVELS must read exactly zero. If it did not, the segmenter would never see a settled
// handshape and no static letter could ever be emitted.
const sigTrans = rollingShapeSigma(translated.map(shape42), times);
invariant('sigma: pure translation is zero', Math.max(...sigTrans) < 1e-12,
          `max = ${Math.max(...sigTrans).toExponential(2)}`);

// ...and a hand that only TURNS must read large under the plain measure and much smaller once
// the rotation is removed. This gap is the whole reason rollingShapeSigmaAligned exists: the
// rigidity veto calibrated on the plain measure needed 1.35 palm and stopped rejecting
// anything. (Not zero: the trailing reference is a MEDIAN of shapes at different angles, which
// is not itself a rotation of any one frame.)
const shapesRot = rotated.map(shape42);
const plain = Math.max(...rollingShapeSigma(shapesRot, times));
const aligned = Math.max(...rollingShapeSigmaAligned(shapesRot, times));
invariant('sigma: Kabsch removes wrist rotation', aligned < 0.3 * plain,
          `plain ${plain.toFixed(3)} -> aligned ${aligned.toFixed(3)}`);

// Absolute frame position never enters any model: the same gesture traced elsewhere on screen
// must produce a bit-identical 79-D descriptor, every component of it.
const evA = eventFeatures(times, rotated, 'Z');
const evB = eventFeatures(times, rotated.map((P) => P.map((p) => [p[0] + 0.37, p[1] - 0.21])), 'Z');
invariant('eventFeatures position-free',
          evA.length === EVENT_DIM && maxAbsDiff(evA, evB) < 1e-12,
          `${evA.length}-D, max|d| = ${maxAbsDiff(evA, evB).toExponential(2)}`);
invariant('eventFeatures arm flag [78]',
          evA[78] === 1.0 && eventFeatures(times, rotated, 'J')[78] === 0.0, 'J -> 0, Z -> 1');

// Arc-length resampling, not time resampling: on a straight path the k points must come out
// equally spaced whatever the input sampling was.
const ladder = resampleArclength([[0, 0], [3, 4], [6, 8], [9, 12]]);
const gaps = ladder.slice(1).map((p, i) => Math.hypot(p[0] - ladder[i][0], p[1] - ladder[i][1]));
invariant('resampleArclength equal spacing',
          ladder.length === K_RESAMPLE && Math.max(...gaps) - Math.min(...gaps) < 1e-12
          && Math.abs(gaps[0] - pathLength(ladder) / (K_RESAMPLE - 1)) < 1e-12,
          `${K_RESAMPLE} points, gap spread ${(Math.max(...gaps) - Math.min(...gaps)).toExponential(2)}`);

// Sign convention: a counter-clockwise corner in (u, v) is POSITIVE. Component [58] is a signed
// net turn, so an inverted cross product would mirror J against Z in the motion forest.
invariant('turningAngles sign', Math.abs(turningAngles([[0, 0], [1, 0], [1, 1]])[0] - Math.PI / 2) < 1e-15,
          'left turn -> +pi/2');

// Smoothing is over SECONDS, and the window is the trailing one. At dt = 0.1 s a 0.25 s window
// holds three samples, so out[5] = mean(3,4,5) = 4 -- and out[0] = 0, never a lookahead.
const mat = movingAverageTime([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                              Array.from({ length: 10 }, (_, i) => i * 0.1), 0.25);
invariant('movingAverageTime is causal + in seconds', mat[0] === 0 && mat[5] === 4,
          `out[0] = ${mat[0]}, out[5] = ${mat[5]}`);

// palmSpeed is in palm-widths per second, so a hand crossing one palm width per second reads
// 1.0 regardless of frame rate. Checked at two very different rates against the same motion,
// which is the failure that silently disabled the detector at 30 fps.
for (const dt of [1 / 15, 1 / 60]) {
  const n = Math.round(2.0 / dt);
  const c = [], sc = [], tt = [];
  for (let i = 0; i < n; i++) { c.push([0.25 * dt * i, 0]); sc.push(0.25); tt.push(i * dt); }
  const sp = palmSpeed(c, sc, tt);
  invariant(`palmSpeed frame-rate invariant @${Math.round(1 / dt)}fps`,
            Math.abs(sp[n - 1] - 1.0) < 1e-12 && sp[0] === 0.0,
            `steady state = ${sp[n - 1].toFixed(12)} palm/s, out[0] = ${sp[0]}`);
}

// palmSpeed IS the segmenter's vBar: the window holds every frame with t >= t_end - window and
// averages the k steps between them, not the k+1 steps a per-step moving average over the same
// window would (the step entering the window from before it is not counted). At dt = 0.1 s
// and a 0.25 s window the window at i = 5 is frames 3, 4, 5 -> two steps. And a gap longer
// than the window falls back to the last two frames rather than to one (no steps at all).
{
  const tt = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
  const c = tt.map((t, i) => [i * i * 0.01, 0]);         // accelerating, so k vs k+1 differ
  const sc = tt.map(() => 1.0);
  const [j, k] = trailingWindow(tt, 0.5, 0.25);
  const two = (Math.hypot(c[4][0] - c[3][0], 0) / 0.1 + Math.hypot(c[5][0] - c[4][0], 0) / 0.1) / 2;
  const sp = palmSpeed(c, sc, tt, 0.25);
  invariant('palmSpeed averages the k steps INSIDE the window',
            j === 3 && k === 6 && Math.abs(sp[5] - two) < 1e-12
            && Math.abs(windowSpeed(tt.slice(j, k), c.slice(j, k), sc.slice(j, k)) - two) < 1e-12,
            `window [${j}, ${k}), out[5] = ${sp[5].toFixed(4)} (two steps: ${two.toFixed(4)})`);
  const gapT = [0, 0.1, 0.2, 1.0];
  const [gj, gk] = trailingWindow(gapT, 1.0, 0.33);
  invariant('a gap longer than the window keeps the last two frames', gj === 2 && gk === 4,
            `window [${gj}, ${gk})`);
}

console.log(`\n${golden.cases.length} golden cases + ${nInvariants} invariants: ` +
            (failures ? `${failures} FAILING` : 'all pass') + (skipped ? ` (${skipped} skipped)` : ''));
console.log('allowance = golden tolerance 1e-6 (golden.json\'s landmarks are rounded before its ' +
            'outputs are computed, so the only residual is the 6-decimal rounding of the stored ' +
            'values, <= 5e-7)');
process.exit(failures ? 1 : 0);

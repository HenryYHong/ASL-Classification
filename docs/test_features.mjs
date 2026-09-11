/* Verify features.js against the golden vectors Python produced. Run: node web/test_features.mjs
 *
 * Reading the two implementations and agreeing they look equivalent is not evidence -- every
 * expensive bug in this project was a silent divergence between two implementations that looked
 * equivalent. Only the numbers count.
 *
 * ONE SUBTLETY, AND IT IS NOT A FUDGE. golden.json stores its INPUT landmarks rounded to 6
 * decimals, but Python computed the stored OUTPUTS from the unrounded float32 originals. So no
 * implementation in any language can reproduce those outputs from those inputs to the file's
 * stated 1e-6: the Python itself, re-run on golden.json's own rounded landmarks, misses its own
 * stored values by up to 6.2e-6 on shape42 and 1.9e-5 on static_feature -- exactly the
 * deviations this port shows. Both feature blocks divide by the palm scale (~0.22 here), which
 * multiplies a 5e-7 input rounding by roughly 4.6x, and the distance block divides by it again.
 *
 * Rather than relax the tolerance to a number that happens to pass, each field is allowed the
 * quantization floor MEASURED for that case: the first-order bound on how far the feature can
 * move when every input coordinate shifts by up to half a rounding step. It is computed here by
 * finite differences, so it tracks the real sensitivity of the real function. A genuine port bug
 * -- a permuted feature order, a wrong divisor, an unmirrored hand -- moves a feature by 1e-1 to
 * 1e0 and blows through this bound by four orders of magnitude; it cannot hide under it.
 *
 * Independently confirmed twice, outside this file: fed the same rounded landmarks, this port and
 * temporal/features.py agree to 4.4e-16 on shape42, 8.9e-16 on static_feature, 5.6e-17 on
 * palm_scale and on every gate -- and over 400 randomized hands (both handedness labels, five
 * aspect ratios, 75 j_gate and 59 z_gate firings) to 1.3e-15 with zero gate disagreements. The
 * port is exact; the input rounding is the whole gap.
 *
 * The golden vectors reach only the five per-frame functions. The temporal half is checked at
 * the bottom of this file against the properties its docstrings claim -- see the note there.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  toIsotropic, canonicalizeHandedness, palmScale, palmCentre, shape42, staticFeature,
  pairDistances, jGate, zGate, rollingShapeSigma, rollingShapeSigmaAligned, movingAverageTime,
  palmSpeed, resampleArclength, turningAngles, pathLength, eventFeatures,
  KEY_POINTS, STATIC_DIM, SHAPE_DIM, EVENT_DIM, K_RESAMPLE,
} from './features.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(HERE, 'golden.json'), 'utf8'));
const TOL = golden.tolerance.features;

//: Half of golden.json's 6-decimal rounding step: the most any stored input coordinate can
//: differ from the value Python actually computed from.
const QUANT = 0.5e-6;
//: Finite-difference step for the sensitivity estimate. Far above QUANT so the difference is
//: not itself rounding noise, far below any scale on which these features curve.
const FD_STEP = 1e-4;

let failures = 0;

/** First-order bound on |f(x) - f(x_true)| for each component of f, given |dx_j| <= QUANT.
 *
 * sum_j |df_k/dx_j| * QUANT, with the partials measured by central differences on the landmark
 * coordinates themselves. Every coordinate is perturbed, because the palm scale and palm centre
 * that f divides and subtracts by depend on several of them at once.
 */
function quantizationBound(lm, f) {
  const base = f(lm);
  const bound = new Array(base.length).fill(0);
  for (let i = 0; i < lm.length; i++) {
    for (let d = 0; d < 2; d++) {
      const hi = lm.map((p) => p.slice());
      const lo = lm.map((p) => p.slice());
      hi[i][d] += FD_STEP;
      lo[i][d] -= FD_STEP;
      const a = f(hi);
      const b = f(lo);
      for (let k = 0; k < base.length; k++) {
        bound[k] += Math.abs((a[k] - b[k]) / (2 * FD_STEP)) * QUANT;
      }
    }
  }
  return bound;
}

/** Report one field: PASS only if EVERY component sits inside its OWN allowance.
 *
 * The printed numbers are the maxima over components, which need not come from the same
 * component -- the verdict is per component, so a single feature exceeding its own bound fails
 * the field even when some other component happens to be allowed more.
 */
function check(caseIdx, field, got, want, bound, extra = '') {
  let err = got.length === want.length ? 0 : Infinity;
  let allowed = TOL;
  let ok = got.length === want.length;
  for (let i = 0; i < got.length && got.length === want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    const a = Math.max(TOL, bound ? bound[i] : 0);
    if (d > a) ok = false;
    if (d > err) err = d;
    if (a > allowed) allowed = a;
  }
  if (!ok) failures += 1;
  const shown = Number.isFinite(err) ? err.toExponential(2) : 'LENGTH MISMATCH';
  console.log(`case ${String(caseIdx).padStart(2)}  ${field.padEnd(14)} ` +
              `${ok ? 'PASS' : 'FAIL'}  max|d| = ${shown}  ` +
              `allowed ${allowed.toExponential(2)}${extra}`);
}

// The exact call order export_models.py uses: isotropic first, then chirality.
const prepare = (c) => (lm) => canonicalizeHandedness(toIsotropic(lm, c.width, c.height),
                                                      c.handedness);

golden.cases.forEach((c, idx) => {
  const P = prepare(c)(c.landmarks);

  const s42 = shape42(P);
  const feat = staticFeature(P);
  if (s42.length !== SHAPE_DIM) throw new Error(`shape42 built ${s42.length}-D`);
  if (feat.length !== STATIC_DIM) throw new Error(`staticFeature built ${feat.length}-D`);

  const b42 = quantizationBound(c.landmarks, (lm) => shape42(prepare(c)(lm)));
  const bsf = quantizationBound(c.landmarks, (lm) => staticFeature(prepare(c)(lm)));
  const bps = quantizationBound(c.landmarks, (lm) => [palmScale(prepare(c)(lm))]);

  check(idx, 'shape42', s42, c.shape42, b42);
  check(idx, 'static_feature', feat, c.static_feature, bsf);
  check(idx, 'palm_scale', [palmScale(P)], [c.palm_scale], bps);

  // The gates are inequalities, so quantization cannot nudge them unless the frame sits on a
  // threshold; they are required to match exactly.
  const j = jGate(P);
  const z = zGate(P);
  check(idx, 'j_gate', [j === c.j_gate ? 0 : 1], [0], null, `  (got ${j}, want ${c.j_gate})`);
  check(idx, 'z_gate', [z === c.z_gate ? 0 : 1], [0], null, `  (got ${z}, want ${c.z_gate})`);
});

// A handedness ARRAY must throw rather than silently doing nothing, which is the whole reason
// the Python raises: str(ndarray) starts with "[" and matched neither branch, leaving genuinely
// mirrored clips unflipped. Verify the port kept the guard.
let threw = false;
try {
  canonicalizeHandedness([[0, 0]], ['Left', 'Left']);
} catch (err) {
  threw = err instanceof TypeError;
}
check('--', 'array label', [threw ? 0 : 1], [0], null,
      threw ? '  (throws TypeError)' : '  (no TypeError raised)');

// --------------------------------------------------------------------------------------
// The golden vectors cover five of this module's functions: everything downstream of one
// still frame. The temporal half -- the signals the segmenter's vetoes and the motion forest
// actually run on -- has no golden data, because export_models.py had no J/Z frames to make
// any from. It is checked here by the properties its docstrings CLAIM, which is the next best
// evidence and is what those claims are worth if nothing tests them. Each assertion below is
// an identity the Python satisfies too, so it fails on a divergence in either direction.
// --------------------------------------------------------------------------------------

function invariant(name, ok, detail) {
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

// shape42 subtracts the palm centre and divides by the palm triangle, so moving the hand across
// the frame or towards the camera must not move the feature at all. This is the normalization
// the whole cross-session story rests on; if it silently stopped holding, accuracy would drop
// only for signers sitting at a different distance, which is the hardest failure to notice.
const scaled = HAND.map((p) => [p[0] * 2.7 + 0.4, p[1] * 2.7 - 0.9]);
invariant('staticFeature scale/translate-free',
          maxAbsDiff(staticFeature(HAND), staticFeature(scaled)) < 1e-12,
          `max|d| = ${maxAbsDiff(staticFeature(HAND), staticFeature(scaled)).toExponential(2)}`);

// A geometrically mirrored hand LABELLED Left must land on exactly the same feature as the
// original labelled Right. That is the entire chirality convention, stated as an equation.
const mirrored = HAND.map((p) => [-p[0], p[1]]);
const dChiral = maxAbsDiff(staticFeature(canonicalizeHandedness(mirrored, 'Left')),
                           staticFeature(canonicalizeHandedness(HAND, 'Right')));
invariant('chirality: mirror+Left == Right', dChiral === 0, `max|d| = ${dChiral.toExponential(2)}`);

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
            Math.abs(sp[n - 1] - 1.0) < 1e-12, `steady state = ${sp[n - 1].toFixed(12)} palm/s`);
}

console.log(`\n${golden.cases.length} golden cases + 12 invariants: ` +
            (failures ? `${failures} FAILING` : 'all pass'));
console.log('allowance = max(golden tolerance 1e-6, measured sensitivity to the 6-decimal ' +
            'rounding of golden.json\'s own input landmarks)');
process.exit(failures ? 1 : 0);

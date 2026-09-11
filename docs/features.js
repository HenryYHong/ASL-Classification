/* Shared feature construction for the 26-letter pipeline, ported from temporal/features.py.
 *
 * Pure functions, no I/O, no dependencies. Names, ordering and constants match the Python line
 * for line so the two can be diffed by eye -- that is the whole point. Approach A in this
 * repository failed because training and inference preprocessed differently; a browser port is
 * exactly that failure mode again, one language further apart, so this file deliberately does
 * nothing cleverer than the Python does.
 *
 * The Python operates on batched arrays (...,21,2). Everything here operates on ONE frame:
 * P is 21 pairs [u, v]. The sequence functions take arrays of frames explicitly.
 *
 * Three coordinate corrections, in this order, all load-bearing:
 *
 * 1. ISOTROPY.  MediaPipe normalizes x by frame width and y by frame height. On a 16:9 frame
 *    that squashes x by 1.78x relative to y, so a circle traced in the air becomes an ellipse
 *    and every diagonal is skewed. Z is *defined* by two diagonals. Correct with u = x * (W/H).
 *
 * 2. CHIRALITY. A left hand is the mirror of a right hand. Negate u for left hands. This depends
 *    on MediaPipe's handedness label, which inverts if you feed it a mirrored frame -- so never
 *    mirror the pixels going INTO MediaPipe, only the copy you draw on the canvas.
 *
 * 3. SCALE. Divide by the palm triangle S = mean(|p0-p5|, |p0-p17|, |p5-p17|). The palm triangle
 *    rather than the hand bounding box, because the bbox depends on the handshape -- dividing by
 *    it would cancel the very extent signal that separates A from B.
 *
 * Landmark indices (MediaPipe Hands):
 *   0 wrist | 4 thumb tip | 8 index tip | 12 middle tip | 16 ring tip | 20 pinky tip
 *   5 index MCP | 9 middle MCP | 13 ring MCP | 17 pinky MCP
 */

export const WRIST = 0;
export const TIPS = { thumb: 4, index: 8, mid: 12, ring: 16, pinky: 20 };
export const MCPS = [5, 9, 13, 17];
export const PALM = [0, 5, 9, 13, 17];
export const TIP_FOR_ARM = { J: 20, Z: 8 };  // J is signed with the pinky, Z with the index

export const SHAPE_DIM = 42;
export const EVENT_DIM = 79;
export const K_RESAMPLE = 16;
export const STATIC_DIM = 101;

//: Landmarks whose pairwise distances carry the handshape distinctions trees struggle to
//: express from raw coordinates: the five fingertips, the five MCP knuckles (2 is the thumb's)
//: and the wrist. ORDER IS PART OF THE FEATURE -- see pairDistances.
export const KEY_POINTS = [4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 0];

//: MCP, PIP, DIP, TIP for each finger -- the joint chain used by fingerStraightness.
export const FINGER_CHAIN = {
  index: [5, 6, 7, 8], mid: [9, 10, 11, 12],
  ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20],
};

// ---------------------------------------------------------------- numeric helpers
// These exist only to reproduce numpy semantics exactly. Each one is a place where an
// "obviously equivalent" JS idiom differs from numpy in a way no test of mine would catch
// until it showed up as a letter that stopped being recognized.

function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

/** numpy.median: the AVERAGE of the two middle values on even length, not the upper one. */
function median(a) {
  const b = Float64Array.from(a);
  b.sort();
  const n = b.length;
  const h = n >> 1;
  return (n & 1) ? b[h] : (b[h - 1] + b[h]) / 2.0;
}

/** numpy.median(X, axis=0) for X a list of equal-length vectors. */
function medianAxis0(rows) {
  const d = rows[0].length;
  const out = new Array(d);
  const col = new Float64Array(rows.length);
  for (let k = 0; k < d; k++) {
    for (let i = 0; i < rows.length; i++) col[i] = rows[i][k];
    const b = col.slice();
    b.sort();
    const n = b.length;
    const h = n >> 1;
    out[k] = (n & 1) ? b[h] : (b[h - 1] + b[h]) / 2.0;
  }
  return out;
}

/** numpy.std with the default ddof=0 (population, not sample). */
function std(a) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
  return Math.sqrt(s / a.length);
}

/** numpy.searchsorted(side='left'): the first index i with a[i] >= v. */
function searchsortedLeft(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** numpy.linspace(0, stop, k). numpy assigns the endpoint exactly rather than k*step. */
function linspace(stop, k) {
  const out = new Array(k);
  const step = stop / (k - 1);
  for (let i = 0; i < k; i++) out[i] = i * step;
  out[k - 1] = stop;
  return out;
}

/** numpy.interp for a single x, including its handling of DUPLICATE knots.
 *
 * numpy locates j = (last index with xp[j] <= x), clamps outside the range, and -- crucially --
 * returns fp[j] verbatim when xp[j] == x. With a zero-length path segment two cumulative
 * arc lengths are equal, and "the last knot at this x" is a different point from "the first".
 * A naive lower-bound search picks the other one and silently shifts a resampled path point.
 */
function interp(x, xp, fp) {
  const n = xp.length;
  // j = searchsorted(xp, x, side='right') - 1
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid + 1; else hi = mid;
  }
  const j = lo - 1;
  if (j < 0) return fp[0];
  if (j >= n - 1) return fp[n - 1];
  if (xp[j] === x) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  return slope * (x - xp[j]) + fp[j];
}

function dist(p, q) {
  const dx = p[0] - q[0], dy = p[1] - q[1];
  return Math.hypot(dx, dy);
}

// ---------------------------------------------------------------- per-frame geometry

/** (21,>=2) MediaPipe-normalized landmarks -> (21,2) square-aspect coordinates. */
export function toIsotropic(lm, width, height) {
  const a = Number(width) / Number(height);
  const out = new Array(lm.length);
  for (let i = 0; i < lm.length; i++) out[i] = [lm[i][0] * a, lm[i][1]];
  return out;
}

/** Mirror left hands onto the right-hand convention.
 *
 * `handedness` must be a SCALAR label ("Left" / "Right" / "Unknown" / "None" / null), not a
 * per-frame array. The recorder stores one label per frame, so passing that array straight in
 * is an easy mistake -- and in Python it used to fail SILENTLY, because str(ndarray) begins
 * with "[", which matches neither branch and leaves a genuinely mirrored clip unflipped. A
 * chirality error is invisible downstream and poisons everything, so this throws instead.
 */
export function canonicalizeHandedness(P, handedness) {
  if (handedness === null || handedness === undefined) return P;
  if (Array.isArray(handedness) || ArrayBuffer.isView(handedness)) {
    throw new TypeError(
      `canonicalizeHandedness expects one label, got a sequence of ${handedness.length}. ` +
      "Reduce per-frame labels to the modal real label first (ignoring the 'None'/'Unknown' " +
      'markers written on undetected frames).');
  }
  const label = String(handedness).trim().toLowerCase();
  if (label.startsWith('l')) {
    // A copy with only u negated, exactly as the Python's `P = P.copy(); P[...,0] = -P[...,0]`.
    // Copying every column rather than rebuilding a [u, v] pair matters because it keeps the
    // Left and Right paths the SAME shape: rebuilding silently dropped a third component that
    // a Right hand kept. In this pipeline toIsotropic has already reduced P to (21,2) so it
    // never bit, but an asymmetry between the two chirality branches is the exact place a
    // chirality bug hides, and chirality errors are invisible downstream.
    return P.map((p) => { const q = p.slice(); q[0] = -q[0]; return q; });
  }
  if (!(label.startsWith('r') || label === 'unknown' || label === 'none' || label === '')) {
    throw new Error(`unrecognized handedness label ${JSON.stringify(handedness)}`);
  }
  return P;
}

/** Palm triangle size, in the same units as P. */
export function palmScale(P) {
  const a = dist(P[0], P[5]);
  const b = dist(P[0], P[17]);
  const c = dist(P[5], P[17]);
  return (a + b + c) / 3.0;
}

/** Mean of wrist and the four MCPs. Stable under finger motion, unlike min(x)/min(y), which
 * hops between landmarks when fingers cross and injects frame-to-frame noise. */
export function palmCentre(P) {
  let sx = 0, sy = 0;
  for (const i of PALM) { sx += P[i][0]; sy += P[i][1]; }
  return [sx / PALM.length, sy / PALM.length];
}

/** (21,2) -> (42,) palm-centered, palm-scaled landmark shape.
 *
 * Deliberately NOT rotation-normalized: orientation is what separates P from K and H from U.
 */
export function shape42(P) {
  const S = palmScale(P);
  const m = palmCentre(P);
  const out = new Array(SHAPE_DIM);
  for (let i = 0; i < 21; i++) {
    out[2 * i] = (P[i][0] - m[0]) / S;
    out[2 * i + 1] = (P[i][1] - m[1]) / S;
  }
  return out;
}

/** All pairwise distances between KEY_POINTS, in palm units, plus per-finger straightness.
 *
 * A forest splits on one coordinate at a time, so "are the index and middle fingertips far
 * apart" costs it a deep chain of axis-aligned splits on four separate coordinates -- and that
 * single quantity is the whole difference between U and V, as thumb-to-fist distance is between
 * M and A. Handing over the distances directly turns those into one split each.
 *
 * The iteration order reproduces itertools.combinations(KEY_POINTS, 2): i ascending, j > i, in
 * the order KEY_POINTS is written, NOT sorted by landmark index. The forest indexes features
 * positionally, so a permuted vector produces confident nonsense rather than an error.
 */
export function pairDistances(P) {
  const S = palmScale(P);
  const out = [];
  for (let i = 0; i < KEY_POINTS.length; i++) {
    for (let j = i + 1; j < KEY_POINTS.length; j++) {
      out.push(dist(P[KEY_POINTS[i]], P[KEY_POINTS[j]]) / S);
    }
  }
  for (const n of ['index', 'mid', 'ring', 'pinky']) out.push(fingerStraightness(P, n));
  return out;
}

/** What the 24-class static classifier consumes: palm-normalized shape + distances (101-D).
 *
 * Deliberately NOT including absolute extent (the Python's legacy42). Adding it raises the
 * in-session benchmark 0.968 -> 0.983 and HALVES cross-session accuracy 0.520 -> 0.320, because
 * extent encodes how this signer happened to be sitting.
 */
export function staticFeature(P) {
  return shape42(P).concat(pairDistances(P));
}

/** Fingertip distance from the wrist, in palm units. */
export function extensionRatios(P) {
  const S = palmScale(P);
  const w = P[WRIST];
  const out = {};
  for (const k of Object.keys(TIPS)) out[k] = dist(P[TIPS[k]], w) / S;
  return out;
}

/** |tip - MCP| / (sum of the three bone lengths). 1.0 is a perfectly straight finger.
 *
 * This exists because wrist-to-tip DISTANCE is not orientation-invariant. MediaPipe gives 2-D
 * projected coordinates, so a finger pointing toward the camera foreshortens and its measured
 * extension collapses -- a live Z read ext_index 1.87 against a threshold of 1.80, failing 43%
 * of frames purely because the finger was not held flat to the lens.
 *
 * A ratio survives that: foreshortening shrinks numerator and denominator together, so a
 * straight finger reads ~1.0 whichever way it points.
 */
export function fingerStraightness(P, name) {
  const [a, b, c, d] = FINGER_CHAIN[name];
  const tip = dist(P[d], P[a]);
  const seg = dist(P[b], P[a]) + dist(P[c], P[b]) + dist(P[d], P[c]);
  return tip / Math.max(seg, 1e-9);
}

/** Thumb tip to pinky MCP distance, palm units.
 *
 * This replaces a thumb *extension* threshold in the J gate. Measured on the archive, no
 * threshold on thumb extension separates a held I from a Y (I's p95 = 1.74 sits above Y's
 * median 1.65), so a thumb-extension gate silently rejects a quarter of real I frames.
 */
export function thumbPinkyMcp(P) {
  return dist(P[4], P[17]) / palmScale(P);
}

/** Is this frame in a J launch pose (the 'I' handshape: pinky out, others curled)? */
export function jGate(P) {
  const e = extensionRatios(P);
  const others = Math.max(e.index, e.mid, e.ring);
  // 1.20, not the 1.15 the archive alone suggested. On a real 30-gesture take the signer's
  // thumb drifted from 1.10 early to 1.18 late as the hand tired, and the tighter threshold
  // silently dropped the gate from 70% to 27% -- half the recording lost, with no error
  // anywhere. At 1.20 the archive keeps I recall at 100% and admits one frame of Y in 100,
  // which cannot itself produce a false J: arming is only the first of the rising edge, the
  // vetoes and the classifier.
  return (e.pinky > 1.50) && (others < 1.30) && (thumbPinkyMcp(P) < 1.20);
}

/** Is this frame in a Z launch pose (index extended, others curled)?
 *
 * The index test is straightness, not wrist-to-tip distance, so the gate does not silently
 * require the finger to be held flat to the camera. The curled-finger and thumb tests stay on
 * extension ratios: straightness is unreliable for a folded finger, whose joints MediaPipe
 * often cannot see.
 */
export function zGate(P) {
  const e = extensionRatios(P);
  const others = Math.max(e.mid, e.ring, e.pinky);
  return (fingerStraightness(P, 'index') > 0.90) && (others < 1.20) && (e.thumb < 1.45);
}

/** Unit vector wrist -> middle MCP. */
export function handOrientation(P) {
  const v = [P[9][0] - P[0][0], P[9][1] - P[0][1]];
  const n = Math.max(Math.hypot(v[0], v[1]), 1e-9);
  return [v[0] / n, v[1] / n];
}

// ---------------------------------------------------------------- temporal signals

/** Shape deviation from the median shape over the TRAILING `windowS` seconds.
 *
 * shapes: array of 42-vectors from shape42. times: seconds. Returns one value per frame, in
 * palm units.
 *
 * This is the signal that separates "hand travelling, shape fixed" (a real J: rigid finger,
 * the arm moves it) from "shape changing" (an inter-letter transition). Speed alone cannot make
 * that distinction and the rigidity veto rests entirely on it.
 *
 * The trailing window is NOT interchangeable with a whole-clip reference. Over the archive,
 * deviation from a whole-clip median gives p95 = 0.381 because slow drift across a 6.6 s hold
 * accumulates; against the trailing 0.4 s it gives p95 = 0.120. A threshold calibrated on one
 * and applied to the other is wrong by a factor of three.
 */
export function rollingShapeSigma(shapes, times, windowS = 0.4) {
  const out = new Array(shapes.length);
  for (let i = 0; i < shapes.length; i++) {
    const j = searchsortedLeft(times, times[i] - windowS);
    const ref = medianAxis0(shapes.slice(j, i + 1));
    let s = 0;
    for (let k = 0; k < 21; k++) {
      s += Math.hypot(shapes[i][2 * k] - ref[2 * k], shapes[i][2 * k + 1] - ref[2 * k + 1]);
    }
    out[i] = s / 21;
  }
  return out;
}

/** Shape deviation AFTER removing the best-fit rotation. Palm units.
 *
 * rollingShapeSigma answers "did anything about this hand change", which is the right question
 * for "is it parked". It is the wrong question for "is this a rigid gesture", because shape42
 * is deliberately not rotation-normalized: a J hook rotates the wrist, so a perfectly rigid
 * hand registers a large deviation purely from turning. Calibrating a rigidity veto on that
 * forced it up to 1.35 palm, high enough that it stopped rejecting inter-letter transitions.
 *
 * THE 2-D KABSCH, IN CLOSED FORM. The Python calls numpy's SVD; there is no SVD here and
 * pulling in a linear-algebra library for a 2x2 would be absurd, so derive it instead.
 *
 *   Let c_k be the current shape's landmarks and r_k the reference's; both are already centered
 *   (shape42 subtracts the palm centre) and palm-scaled, so only a rotation is left to fit.
 *   Minimizing sum_k |R c_k - r_k|^2 over rotations R means maximizing sum_k r_k . (R c_k),
 *   which is sum_k tr(R c_k r_k^T) = tr(R H) with H = sum_k c_k r_k^T = cur^T @ ref -- the same
 *   cross-covariance the Python builds.
 *
 *   Write a proper rotation as R = [[cos t, -sin t], [sin t, cos t]]. Then
 *       tr(R H) = cos t * (H00 + H11) + sin t * (H01 - H10),
 *   a single sinusoid in t, maximized when (cos t, sin t) points along (H00+H11, H01-H10).
 *   So cos t and sin t are that vector normalized -- no iteration, no decomposition.
 *
 *   This is exactly what numpy's route returns: the det-correction R = V diag(1,d) U^T with
 *   d = sign(det(V U^T)) is precisely the constraint det R = +1 that the [[c,-s],[s,c]] form
 *   builds in. The one case they differ is a vanishing (H00+H11, H01-H10), where the optimum is
 *   degenerate and every rotation scores the same; identity is taken here, and the SVD would
 *   return some arbitrary rotation. It cannot occur for real hands -- it needs the reference
 *   and current shapes to be exactly orthogonal in this inner product.
 */
export function rollingShapeSigmaAligned(shapes, times, windowS = 0.4) {
  const out = new Array(shapes.length);
  for (let i = 0; i < shapes.length; i++) {
    const j = searchsortedLeft(times, times[i] - windowS);
    const ref = medianAxis0(shapes.slice(j, i + 1));
    const cur = shapes[i];
    let h00 = 0, h01 = 0, h10 = 0, h11 = 0;
    for (let k = 0; k < 21; k++) {
      const cx = cur[2 * k], cy = cur[2 * k + 1];
      const rx = ref[2 * k], ry = ref[2 * k + 1];
      h00 += cx * rx; h01 += cx * ry;
      h10 += cy * rx; h11 += cy * ry;
    }
    const a = h00 + h11, b = h01 - h10;
    const n = Math.hypot(a, b);
    const cs = n > 1e-12 ? a / n : 1.0;
    const sn = n > 1e-12 ? b / n : 0.0;
    let s = 0;
    for (let k = 0; k < 21; k++) {
      const cx = cur[2 * k], cy = cur[2 * k + 1];
      s += Math.hypot(cs * cx - sn * cy - ref[2 * k], sn * cx + cs * cy - ref[2 * k + 1]);
    }
    out[i] = s / 21;
  }
  return out;
}

/** Causal moving average over a time WINDOW, for variable frame rates.
 *
 * Frame index is not a clock: webcam frame rate is not constant, so every smoothing and
 * threshold in this pipeline is expressed in seconds, never in frames.
 */
export function movingAverageTime(values, times, windowS) {
  const out = new Array(values.length);
  let j = 0;
  for (let i = 0; i < values.length; i++) {
    while (times[i] - times[j] > windowS) j += 1;
    let s = 0;
    for (let k = j; k <= i; k++) s += values[k];
    out[i] = s / (i - j + 1);
  }
  return out;
}

/** Palm-centre speed in palm-widths per second, smoothed over `windowS` SECONDS.
 *
 * The window is in seconds, not samples, because a sample count is a different amount of
 * smoothing at every frame rate. A 5-sample average is 0.33 s at 15 fps and 0.17 s at 30 fps;
 * the under-smoothed signal crosses the motion threshold constantly without ever holding it, so
 * the rising-edge trigger never fires and no gesture is ever detected. That silently breaks the
 * detector on any camera faster than the one the thresholds were calibrated on -- and a browser
 * runs at whatever frame rate the machine happens to give, which is exactly the hazard.
 */
export function palmSpeed(centres, scales, times, windowS = 0.33) {
  const n = centres.length;
  if (n < 2) return new Array(n).fill(0.0);
  const v = new Array(n);
  v[0] = 0.0;
  for (let i = 1; i < n; i++) {
    const dt = Math.max(times[i] - times[i - 1], 1e-6);
    v[i] = dist(centres[i], centres[i - 1]) / (scales[i] * dt);
  }
  return movingAverageTime(v, times, windowS);
}

/** Resample a 2-D path to k points equally spaced by ARC LENGTH, not by time.
 *
 * Arc-length resampling removes absolute speed from the path shape (duration, mean speed and
 * peak speed are added back as explicit features so nothing is lost silently).
 */
export function resampleArclength(path, k = K_RESAMPLE) {
  if (path.length < 2) {
    const p = path.length ? path[0] : [0.0, 0.0];
    return Array.from({ length: k }, () => [p[0], p[1]]);
  }
  const cum = new Array(path.length);
  cum[0] = 0.0;
  for (let i = 1; i < path.length; i++) cum[i] = cum[i - 1] + dist(path[i], path[i - 1]);
  const total = cum[path.length - 1];
  if (total < 1e-9) {
    return Array.from({ length: k }, () => [path[0][0], path[0][1]]);
  }
  const xs = path.map((p) => p[0]);
  const ys = path.map((p) => p[1]);
  return linspace(total, k).map((t) => [interp(t, cum, xs), interp(t, cum, ys)]);
}

/** Signed turning angle at each interior point of a polyline. pts.length - 2 values. */
export function turningAngles(pts) {
  const out = [];
  for (let i = 0; i + 2 < pts.length; i++) {
    const ax = pts[i + 1][0] - pts[i][0], ay = pts[i + 1][1] - pts[i][1];
    const bx = pts[i + 2][0] - pts[i + 1][0], by = pts[i + 2][1] - pts[i + 1][1];
    out.push(Math.atan2(ax * by - ay * bx, ax * bx + ay * by));
  }
  return out;
}

export function pathLength(path) {
  if (path.length < 2) return 0.0;
  let s = 0;
  for (let i = 1; i < path.length; i++) s += dist(path[i], path[i - 1]);
  return s;
}

// ---------------------------------------------------------------- event feature

/** Build the 79-D descriptor of one segmented motion event.
 *
 * times : (T,) seconds
 * Pseq  : (T,21,2) isotropic, handedness-canonicalized landmarks (NOT yet palm-scaled)
 * arm   : "J" or "Z" -- which gate armed the track, selecting the writing fingertip
 *
 * `handedness` is accepted and unused, exactly as in the Python: the canonicalization has
 * already happened by the time a track is built.
 */
export function eventFeatures(times, Pseq, arm, handedness = null) {  // eslint-disable-line no-unused-vars
  const T = Pseq.length;
  if (T < 3) throw new Error('event needs at least 3 frames');

  const S_t = Pseq.map(palmScale);
  const S_evt = median(S_t);
  const tipIdx = TIP_FOR_ARM[arm];

  // Smooth the writing tip over 0.15 s before measuring anything from it.
  const tipX = movingAverageTime(Pseq.map((P) => P[tipIdx][0]), times, 0.15);
  const tipY = movingAverageTime(Pseq.map((P) => P[tipIdx][1]), times, 0.15);
  const tip_s = tipX.map((x, i) => [x, tipY[i]]);

  const duration = times[T - 1] - times[0];
  const L = pathLength(tip_s) / S_evt;
  const net = [(tip_s[T - 1][0] - tip_s[0][0]) / S_evt, (tip_s[T - 1][1] - tip_s[0][1]) / S_evt];
  const net_mag = Math.hypot(net[0], net[1]);
  const straightness = L > 1e-9 ? net_mag / L : 0.0;

  // [0:32] path shape: resample by arc length, centre on its own centroid, scale by S_evt.
  // Centring across TIME is the same trick shape42 applies across LANDMARKS: a J traced
  // top-left and the same J traced bottom-right give identical numbers. Absolute frame
  // position never enters any model.
  // The guard is written as the Python writes it -- L*S_evt vs 0.05*S_evt, i.e. L < 0.05 --
  // and is dead weight in practice because L_MIN rejects such a track upstream.
  const rs = (L * S_evt < 0.05 * S_evt)
    ? Array.from({ length: K_RESAMPLE }, () => [tip_s[0][0], tip_s[0][1]])
    : resampleArclength(tip_s, K_RESAMPLE);
  const rsMeanX = mean(rs.map((p) => p[0]));
  const rsMeanY = mean(rs.map((p) => p[1]));
  const g = [];
  for (const p of rs) { g.push((p[0] - rsMeanX) / S_evt, (p[1] - rsMeanY) / S_evt); }

  const theta = turningAngles(rs);                        // [32:46], 14 values
  const e = Pseq.map(extensionRatios);
  const ext_med = ['thumb', 'index', 'mid', 'ring', 'pinky']
    .map((k) => median(e.map((d) => d[k])));
  const ori = Pseq.map(handOrientation);
  const orient = [median(ori.map((o) => o[0])), median(ori.map((o) => o[1]))];

  // Same trailing-window definition the segmenter's rigidity veto uses, so [59]/[60] are
  // directly comparable to RIGID_VETO rather than being a differently-scaled quantity.
  const shapes = Pseq.map(shape42);
  const sig = rollingShapeSigma(shapes, times);

  const third = Math.max(1, Math.floor(tip_s.length / 3));
  const thirds = [];
  for (let i = 0; i < 3; i++) {
    const lo = i * third;
    const hi = i === 2 ? tip_s.length : Math.min(tip_s.length, (i + 1) * third);
    thirds.push((tip_s[hi - 1][0] - tip_s[lo][0]) / S_evt,
                (tip_s[hi - 1][1] - tip_s[lo][1]) / S_evt);
  }

  const speeds = [];
  for (let i = 1; i < T; i++) {
    const dt = Math.max(times[i] - times[i - 1], 1e-6);
    speeds.push((dist(tip_s[i], tip_s[i - 1]) / S_evt) / dt);
  }

  const s_first = median(S_t.slice(0, third));
  const s_last = median(S_t.slice(S_t.length - third));
  const scale_ratio = s_first > 1e-9 ? (s_last / s_first - 1.0) : 0.0;

  // [70:78] articulation: the tip measured RELATIVE to the palm centre. In a genuine J or Z the
  // finger is rigid and the arm carries it, so these are near zero. A wave or a finger wiggle
  // moves the tip relative to the palm and lights this block up.
  const rel = Pseq.map((P, i) => {
    const m = palmCentre(P);
    return [(P[tipIdx][0] - m[0]) / S_t[i], (P[tipIdx][1] - m[1]) / S_t[i]];
  });
  const rel_L = pathLength(rel);
  const rel_net = [rel[T - 1][0] - rel[0][0], rel[T - 1][1] - rel[0][1]];
  const rel_net_mag = Math.hypot(rel_net[0], rel_net[1]);
  const rel_dirDen = Math.max(rel_net_mag, 1e-9);
  const rel_dir = [rel_net[0] / rel_dirDen, rel_net[1] / rel_dirDen];
  const rel_turn = rel.length >= 3
    ? turningAngles(rel).reduce((s, v) => s + Math.abs(v), 0.0) : 0.0;

  const absTheta = theta.map(Math.abs);
  const f = [].concat(
    g,                                                                  // [0:32]
    theta,                                                              // [32:46]
    ext_med,                                                            // [46:51]
    [orient[0], orient[1]],                                             // [51:53]
    [L],                                                                // [53]
    [duration],                                                         // [54]
    [straightness],                                                     // [55]
    [absTheta.filter((v) => v > 0.6).length],                           // [56]
    [Math.max(...absTheta)],                                            // [57]
    [theta.reduce((s, v) => s + v, 0.0)],                               // [58] signed net turn
    [Math.max(...sig)],                                                 // [59]
    [median(sig)],                                                      // [60]
    [scale_ratio],                                                      // [61]
    thirds,                                                             // [62:68]
    [mean(speeds)],                                                     // [68]
    [Math.max(...speeds)],                                              // [69]
    // The Python writes `rel_L and rel_net_mag / rel_L or 0.0`: a zero path length yields 0.0,
    // and so does a zero ratio. Both collapse to this conditional.
    [rel_L, rel_net_mag, rel_L !== 0 ? rel_net_mag / rel_L : 0.0,       // [70:73]
      rel_dir[0], rel_dir[1], rel_turn,                                 // [73:76]
      std(rel.map((p) => p[0])), std(rel.map((p) => p[1]))],            // [76:78]
    [arm === 'J' ? 0.0 : 1.0],                                          // [78]
  );

  if (f.length !== EVENT_DIM) {
    throw new Error(`expected ${EVENT_DIM}-D, built ${f.length}-D`);
  }
  return f;
}

/* The page: camera in, letters out.
 *
 * This is the browser twin of temporal/live_demo.py and it is deliberately thin. Grab a frame,
 * run one hoisted HandLandmarker over the *unflipped* frame, hand (t, landmarks, handedness,
 * W, H) to Segmenter.step, append whatever it emits. All the substance lives in features.js,
 * forest.js and segmenter.js, which are ports of the Python of the same names; everything here
 * is plumbing and the overlay.
 *
 * Three things in here are not plumbing:
 *
 * MIRRORING. The frame given to MediaPipe is never flipped. MediaPipe assigns its handedness
 * label in image space, so a flipped frame reports a right hand as "Left", and
 * canonicalizeHandedness then mirrors exactly the hands it should have left alone -- a
 * chirality error that is invisible downstream and makes every feature wrong. The display copy
 * is mirrored because signing into a non-mirrored image is disorienting, and that mirror
 * happens in exactly one function, toDisplayPx, plus the one drawImage that uses the same flip.
 *
 * SECONDS. Every threshold in thresholds.py is in seconds and palm widths, never milliseconds
 * or frames. MediaPipe's detectForVideo wants milliseconds. The two clocks meet at exactly one
 * line in onFrame, and the segmenter never sees a millisecond.
 *
 * THE SELF-CHECK. web/golden.json is run through the loaded forest before the camera is ever
 * touched, and the result is printed in the debug panel. Every expensive bug in this project
 * was a silent divergence between two implementations that looked equivalent, and from the
 * emitted letters alone "the tree walk is subtly wrong" is indistinguishable from "this
 * visitor's hands are not the signer's".
 */
import {
  TIP_FOR_ARM, pathLength, toIsotropic, canonicalizeHandedness, staticFeature,
} from './features.js';
import { loadModels, predictProba } from './forest.js';
import { Segmenter, NO_HAND, SETTLING, HOLD, TRACKING } from './segmenter.js';

const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
  + 'hand_landmarker/float16/1/hand_landmarker.task';

//: The display copy is drawn at most this wide, so the overlay has one layout instead of one
//: per camera. Nothing measured passes through it: the segmenter is fed MediaPipe's normalized
//: landmarks and the capture frame's true W and H.
const DISPLAY_W = 960;
const FPS_WINDOW = 30;

//: The feature transform each branch of features.js builds. live_demo.py refuses to start when
//: a pickled model's tag is not the one the code constructs, because train/serve skew is this
//: project's documented failure mode and it is completely silent -- a model trained on a
//: different transform still returns confident probabilities. Same check here.
const STATIC_FEATURE_TAG = 'static/v3';
const MOTION_FEATURE_TAG = 'event/v1';

//: mp.solutions.hands.HAND_CONNECTIONS, written out. The Tasks API carries the same list as a
//: static on HandLandmarker, but the skeleton has to be drawable before that dynamic import
//: resolves, and this is twenty numbers.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

//: live_demo.py's STATE_COLOR, converted from OpenCV's BGR to the CSS variables in index.html.
const STATE_COLOR = {
  [NO_HAND]: 'var(--no-hand)',
  [SETTLING]: 'var(--settling)',
  [HOLD]: 'var(--hold)',
  [TRACKING]: 'var(--tracking)',
};

// ------------------------------------------------------------------ pure helpers

/** Normalized landmark -> display pixel. x is mirrored HERE and nowhere else (live_demo.px). */
export function toDisplayPx(xy, W, H) {
  return [(1.0 - xy[0]) * W, xy[1] * H];
}

/** Frame rate over a window of timestamps in seconds. live_demo.py's estimator. */
export function fpsFrom(times) {
  if (times.length < 2) return 0.0;
  const span = times[times.length - 1] - times[0];
  return span > 1e-6 ? (times.length - 1) / span : 0.0;
}

/** The overlay's gate line: what is armed, or what could arm. Follows draw_overlay() exactly,
 *  including the tie rule -- J wins a tie because _stepSettling arms "J" when jf >= zf. */
export function gateStatus(jf, zf, arm, th) {
  if (arm) return { tag: `ARMED ${arm}`, color: 'var(--tracking)' };
  const ready = (jf >= th.GATE_ARM_FRAC && jf >= zf) ? 'J'
    : (zf >= th.GATE_ARM_FRAC ? 'Z' : null);
  if (ready) return { tag: `gate ready ${ready}`, color: 'var(--hold)' };
  return { tag: 'gate none', color: 'var(--dim)' };
}

/** Run golden.json's cases through the loaded model. Returns {n, worst, worstEnd, wrong, tol, ok}.
 *
 * TWO legs, because a single number cannot say which half broke:
 *
 *   worst     golden.json's own 101-D vector -> the forest. A failure here is the tree walk.
 *   worstEnd  the case's RAW landmarks -> toIsotropic -> canonicalizeHandedness ->
 *             staticFeature -> the forest: the entire path the camera frames take. A failure
 *             here with `worst` clean is the feature transform.
 *
 * The second leg is not redundant and it is the more important one. The feature transform is
 * where this project's expensive bugs actually lived -- a permuted pair-distance block, a
 * chirality flip -- and none of them is visible to the forest leg, which never calls
 * features.js at all. Measured: permuting the distance block turns case 0 from A p=1.00 into
 * M p=0.16, and the forest leg still reports a perfect 0.0.
 *
 * Both legs compare PROBABILITIES rather than the feature vectors, even though golden.json
 * carries shape42 and static_feature. Its landmarks are rounded to 6 decimals while its outputs
 * were computed from the full-precision originals, so a faithful recomputation differs from the
 * stored vectors by up to 1.9e-5 -- above the file's 1e-6 feature tolerance, and not a port
 * error. The probabilities are unaffected by that rounding (measured worst 0.0 over all 15
 * cases), so they are the field that can be checked exactly. test_features.mjs makes the
 * feature-level comparison with the rounding allowance stated explicitly.
 */
export function checkGolden(model, golden) {
  const tol = (golden.tolerance && golden.tolerance.probabilities) || 1e-4;
  let worst = 0.0;
  let worstEnd = 0.0;
  let wrong = 0;
  let n = 0;
  for (const c of golden.cases) {
    const p = predictProba(model, c.static_feature);
    if (p.length !== c.static_probs.length) {
      throw new Error(`golden case ${n}: forest returned ${p.length} probabilities, `
        + `expected ${c.static_probs.length}`);
    }
    for (let i = 0; i < p.length; i++) worst = Math.max(worst, Math.abs(p[i] - c.static_probs[i]));

    // The same case again, entered the way a frame enters: raw MediaPipe landmarks, the
    // handedness label as reported on an unflipped frame, and the capture size.
    const P = canonicalizeHandedness(toIsotropic(c.landmarks, c.width, c.height), c.handedness);
    const q = predictProba(model, staticFeature(P));
    let bi = 0;
    for (let i = 0; i < q.length; i++) {
      worstEnd = Math.max(worstEnd, Math.abs(q[i] - c.static_probs[i]));
      if (q[i] > q[bi]) bi = i;
    }
    if (model.classes[bi] !== c.predicted) wrong += 1;
    n += 1;
  }
  return { n, worst, worstEnd, wrong, tol, ok: worst <= tol && worstEnd <= tol && wrong === 0 };
}

/** The path length of the live track, in palm units: the number the L_MIN/L_MAX veto reads.
 *
 * Taken from the segmenter's own buffer rather than from the drawing history, because the two
 * are in different coordinate systems -- the buffer is isotropic and chirality-canonicalized,
 * the drawing history is raw normalized pixels -- and only the first one is what the veto
 * measures. An overlay that reports a different number from the one the machine is deciding on
 * is worse than no overlay. */
export function trackPathLength(frames, arm) {
  if (!frames || frames.length < 2) return null;
  const tip = TIP_FOR_ARM[arm];
  if (tip === undefined) return null;
  const path = frames.map((f) => f.P[tip]);
  const scales = frames.map((f) => f.S).sort((a, b) => a - b);
  // np.median over an even count averages the middle two; _score uses the median scale, so the
  // overlay does too.
  const k = scales.length;
  const S = k % 2 ? scales[(k - 1) / 2] : (scales[k / 2 - 1] + scales[k / 2]) / 2;
  return pathLength(path) / Math.max(S, 1e-9);
}

// ------------------------------------------------------------------ page

function boot() {
  window.__aslBooted = true;

  const el = (id) => document.getElementById(id);
  const ui = {
    start: el('start'), stop: el('stop'), loadstate: el('loadstate'),
    stage: el('stage'), video: el('cam'), canvas: el('view'),
    letters: el('letters'), clear: el('clear'),
    state: el('state'), armtag: el('armtag'), fps: el('fps'),
    vval: el('vval'), vmeter: el('vmeter'), vticks: el('vticks'),
    sval: el('sval'), smeter: el('smeter'), sticks: el('sticks'),
    srigid: el('srigid'), vetoval: el('vetoval'), vsmooth: el('vsmooth'),
    curtain: el('curtain'),
    jf: el('jf'), zf: el('zf'), armfrac: el('armfrac'), track: el('track'),
    lastem: el('lastem'), selfcheck: el('selfcheck'), emlog: el('emlog'),
    fatal: el('fatal'), fatalmsg: el('fatalmsg'),
  };
  const ctx = ui.canvas.getContext('2d');

  let th = null;
  let seg = null;
// Localhost only: post diagnostics to devserver.py, the browser twin of live_demo.py --log.
// A deployed visitor posts nothing -- there is no endpoint and the guard short-circuits first.
const LOGGING = ['localhost', '127.0.0.1'].includes(location.hostname);
function postLog(obj) {
  if (!LOGGING) return;
  try {
    fetch('/log', { method: 'POST', headers: { 'content-type': 'application/json' },
                    body: JSON.stringify(obj) + '\n', keepalive: true }).catch(() => {});
  } catch (e) { /* diagnostics must never break the demo */ }
}

const trackLog = [];
let nTracks = 0;
  let landmarker = null;
  let stream = null;
  let running = false;
  let letters = [];
  const frameTimes = [];
  // Normalized landmarks with their timestamps, kept here rather than read back out of the
  // segmenter: its buffer holds isotropic, chirality-canonicalized coordinates, which cannot be
  // drawn on a frame.
  let pxHist = [];
  let lastVideoTime = -1;
  let lastTs = -1;
  let note = '';

  //: What the problem box is currently showing, so that starting the camera can clear the
  //: camera's own problem and nothing else. Clearing it unconditionally erased the self-check
  //: warning at exactly the moment it starts mattering -- the user is now signing at a
  //: recognizer that has already said it does not reproduce Python's answers.
  let fatalKind = null;

  /** Say what went wrong, naming the step that failed. `blocking` means nothing usable is
   *  left; a non-blocking problem is still shown, because a page that quietly drops half of
   *  itself is exactly how "J is unreachable" hides. */
  function problem(msg, blocking, kind = 'other') {
    ui.fatal.hidden = false;
    ui.fatalmsg.textContent = msg;
    fatalKind = kind;
    if (blocking) {
      ui.start.disabled = true;
      ui.loadstate.textContent = 'stopped: see the problem above';
    }
  }

  // ---------------------------------------------------------------- loading

  async function load() {
    let models;
    try {
      ui.loadstate.textContent = 'fetching models.json (about 8 MB, cached after the first time)';
      // forest.js owns the fetch, including the gzip sniffing GitHub Pages needs. A second
      // loader here would be a second implementation of the thing this project keeps being
      // burned by having two of.
      models = await loadModels('./models.json');
    } catch (err) {
      problem(`The model file failed to load. ${err.message} `
        + `If you opened index.html from the filesystem, serve the folder over http instead: `
        + `fetch and ES modules do not work from file://.`, true);
      return;
    }
    // A model exported from a different feature transform is the one failure that produces no
    // symptom at all: the vector is the right length, the forest is confident, and every letter
    // is wrong. live_demo.py exits on this; so does the page. A tag of null is tolerated, as in
    // the Python, because the oldest pickles carry none.
    const tagged = [[models.static, STATIC_FEATURE_TAG, 'static'],
      [models.motion, MOTION_FEATURE_TAG, 'motion']];
    const mismatched = tagged.filter(([m, want]) => m.feature != null && m.feature !== want);
    if (mismatched.length) {
      problem(mismatched.map(([m, want, which]) => `The ${which} model in models.json was `
        + `trained on feature "${m.feature}", but this page builds "${want}".`).join(' ')
        + ' Re-export models.json from the current temporal/ code; running it anyway would'
        + ' produce confident nonsense rather than an error.', true);
      return;
    }

    th = models.thresholds;
    ui.armfrac.textContent = th.GATE_ARM_FRAC.toFixed(2);
    ui.vetoval.textContent = th.RIGID_VETO.toFixed(2);
    ui.vsmooth.textContent = th.V_SMOOTH_WINDOW.toFixed(2);
    drawTicks();

    // Before the camera, so a broken tree walk is reported as a broken tree walk instead of as
    // a page that mysteriously reads every letter as D.
    try {
      const res = await fetch('./golden.json');
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      const g = checkGolden(models.static, await res.json());
      ui.selfcheck.textContent = g.ok
        ? `${g.n}/${g.n} cases, forest ${g.worst.toExponential(1)}, `
          + `whole path ${g.worstEnd.toExponential(1)}`
        : `FAILED: forest ${g.worst.toExponential(1)}, whole path `
          + `${g.worstEnd.toExponential(1)}, ${g.wrong} letters wrong (tol ${g.tol})`;
      ui.selfcheck.className = g.ok ? 'ok' : 'err';
      if (!g.ok) {
        // Name the leg that failed: the two have completely different causes and only one of
        // them is in this port's own arithmetic.
        const where = g.worst > g.tol
          ? 'the tree walk disagrees with sklearn'
          : 'the tree walk is exact, so the landmark-to-feature transform is what diverges';
        problem(`The classifier does not reproduce the outputs Python produced for the same `
          + `inputs -- ${where}. Worst probability error: ${g.worst.toExponential(2)} from the `
          + `stored feature vectors, ${g.worstEnd.toExponential(2)} from the raw landmarks, `
          + `against a tolerance of ${g.tol}; ${g.wrong} of ${g.n} letters come out wrong. `
          + `The page still runs, but do not trust the letters.`, false, 'selfcheck');
      }
    } catch (err) {
      ui.selfcheck.textContent = 'not run';
      note = `self-check skipped: ${err.message}`;
    }

    try {
      // Classes are left to default: forest.js keeps models.json's class list on the prepared
      // model and segmenter.js reads it from there, so there is one list, not two.
      // Track outcomes are otherwise invisible: a gesture that never arms, or one killed by a
      // veto, leaves no trace on screen at all. Every hard bug in the Python was found by
      // logging exactly this, so the browser gets the same instrument.
      seg = new Segmenter(th, {
        staticModel: models.static,
        motionModel: models.motion,
        onHold: (h) => postLog({ kind: 'hold', ...h }),
        onEvent: (ev) => {
          postLog({
            kind: 'track', reason: ev.reason || 'scored', arm: ev.arm,
            duration: ev.duration, t: ev.t_end,
            // ev.P is (frames x 21 x 2). Map over FRAMES, then over the landmarks inside each
            // one -- mapping a frame straight to [p[0], p[1]] keeps two landmarks, not two
            // coordinates, and silently logs a 2-point hand.
            times: Array.from(ev.times),
            P: ev.P.map((frame) => Array.from(frame, (pt) => [pt[0], pt[1]])),
          });
          trackLog.unshift({
            dur: ev.duration, arm: ev.arm,
            reason: ev.reason || 'scored', t: ev.t_end,
          });
          trackLog.length = Math.min(trackLog.length, 6);
          nTracks += 1;
        },
      });
    } catch (err) {
      problem(`The segmenter rejected the thresholds in models.json: ${err.message}`, true);
      return;
    }

    let vision;
    try {
      ui.loadstate.textContent = 'loading MediaPipe from the CDN (about 9 MB of WASM)';
      vision = await import(`${MP_CDN}/vision_bundle.mjs`);
    } catch (err) {
      problem(`MediaPipe could not be loaded from ${MP_CDN} (${err.message}). Without it there `
        + `is nothing to find the hand in the frame.`, true);
      return;
    }
    try {
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
      landmarker = await createLandmarker(vision, fileset);
    } catch (err) {
      problem(`MediaPipe loaded but the hand landmarker would not start (${err.message}). `
        + `The model file it fetches is hosted separately from the CDN, so this can also mean `
        + `that download was blocked.`, true);
      return;
    }

    ui.loadstate.textContent = `ready: ${models.static.classes.length} static letters from `
      + `${models.static.trees.length} trees, plus J and Z from the motion branch's `
      + `${models.motion.trees.length}.${note ? ` ${note}` : ''}`;
    ui.start.disabled = false;
    window.__aslReady = true;
  }

  async function createLandmarker(vision, fileset) {
    const opts = {
      baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      // One hand. The state machine assumes a single continuous hand; a second hand entering
      // frame would otherwise swap the tracked one mid-gesture.
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      // 0.3, as in live_demo.py: a fast J or Z motion-blurs badly, and a stricter tracking
      // threshold drops the hand mid-stroke, which the segmenter sees as a detection gap.
      minTrackingConfidence: 0.3,
    };
    try {
      return await vision.HandLandmarker.createFromOptions(fileset, opts);
    } catch (err) {
      // A machine with no usable WebGL context is common enough that failing here would be a
      // pointless dead end. The CPU delegate is slower and produces the same landmarks -- and
      // the frame rate it produces is on the overlay, where a slow one is visible rather than
      // merely felt.
      opts.baseOptions.delegate = 'CPU';
      const cpu = await vision.HandLandmarker.createFromOptions(fileset, opts);
      note = 'no GPU delegate; MediaPipe is running on the CPU';
      return cpu;
    }
  }

  // ---------------------------------------------------------------- camera

  async function startCamera() {
    if (!window.isSecureContext || !navigator.mediaDevices) {
      problem('Browsers only expose the camera to pages served over https, or from localhost. '
        + 'This page is not, so there is no camera to ask for.', true, 'camera');
      return;
    }
    ui.start.disabled = true;
    ui.loadstate.textContent = 'waiting for you to allow the camera';
    try {
      // 640x480 for live_demo.py's reason: a shorter exposure means less motion blur
      // mid-gesture, and landmarks are normalized, so no detail is lost by asking small.
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
    } catch (err) {
      ui.start.disabled = false;
      const why = {
        NotAllowedError: 'Camera access was refused, so there is no input. Nothing else on this '
          + 'page can substitute for it. If you change your mind, allow the camera from the '
          + 'address bar and press the button again.',
        NotFoundError: 'No camera was found on this device.',
        OverconstrainedError: 'No camera here can supply a usable video stream.',
        NotReadableError: 'The camera exists but another application is holding it.',
        SecurityError: 'The browser blocked camera access for this page.',
      }[err.name];
      problem(why || `The camera could not be started (${err.name}: ${err.message}).`,
        false, 'camera');
      ui.loadstate.textContent = 'camera not started';
      return;
    }
    // Only a camera problem is answered by the camera starting. A self-check failure is not,
    // and hiding it here would retract "do not trust the letters" the instant it applies.
    if (fatalKind === 'camera') {
      ui.fatal.hidden = true;
      ui.fatalmsg.textContent = '';
      fatalKind = null;
    }
    ui.video.srcObject = stream;
    await ui.video.play();
    ui.stage.hidden = false;
    if (ui.curtain) ui.curtain.hidden = true;
    ui.stop.hidden = false;
    ui.loadstate.textContent = 'running. Nothing is uploaded; stop the camera or close the tab '
      + 'to end it.';
    running = true;
    lastVideoTime = -1;
    schedule();
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    // Drop the drawing history: a trail from the previous session would otherwise be drawn over
    // the first frames of the next one, and the fps figure would average across the gap.
    pxHist = [];
    frameTimes.length = 0;
    ui.video.srcObject = null;
    ui.stage.hidden = true;
    if (ui.curtain) ui.curtain.hidden = false;
    ui.stop.hidden = true;
    ui.start.disabled = false;
    ui.loadstate.textContent = 'camera stopped';
  }

  function schedule() {
    if (!running) return;
    // requestVideoFrameCallback fires once per decoded frame, so no frame is measured twice and
    // none late; rAF is the fallback, filtered on currentTime for the same reason.
    if (ui.video.requestVideoFrameCallback) ui.video.requestVideoFrameCallback(() => onFrame());
    else requestAnimationFrame(() => onFrame());
  }

  // ---------------------------------------------------------------- frame loop

  function onFrame() {
    if (!running) return;
    const video = ui.video;
    if (video.readyState < 2 || !video.videoWidth) { schedule(); return; }
    if (video.currentTime === lastVideoTime) { schedule(); return; }
    lastVideoTime = video.currentTime;

    // The only place the two clocks meet. MediaPipe wants milliseconds and rejects a repeated
    // timestamp; the segmenter wants seconds, because every threshold it owns is in seconds.
    const nowMs = Math.max(performance.now(), lastTs + 1);
    lastTs = nowMs;
    const t = nowMs / 1000.0;

    let res;
    try {
      // The video element itself, unflipped. Never a mirrored copy: see the file header.
      res = landmarker.detectForVideo(video, nowMs);
    } catch (err) {
      // Release the camera as well as the loop. Recognition is dead either way, and a page that
      // has stopped reading the stream while the camera light stays on contradicts what the
      // consent card promises about stopping.
      stopCamera();
      problem(`MediaPipe stopped processing frames (${err.message}). The camera has been `
        + `switched off; reload the page.`, false);
      return;
    }

    let lm = null;
    let handed = null;
    if (res && res.landmarks && res.landmarks.length) {
      // 21 x [x, y, z], the array shape features.js indexes positionally. Handing it
      // MediaPipe's {x, y, z} objects instead would be read as a stream of non-detections.
      lm = res.landmarks[0].map((p) => [p.x, p.y, p.z]);
      // 0.10.x names it `handednesses`; other builds name it `handedness`. Either way the label
      // is assigned in image space on the frame as given -- which is unflipped here, exactly as
      // it was unflipped in training.
      const hs = res.handednesses || res.handedness;
      if (hs && hs.length && hs[0].length) {
        // SWAP the label. The two MediaPipe APIs disagree about what they are labeling: the
        // legacy `solutions` API that produced every training landmark reports handedness as if
        // the image were mirrored (the selfie convention), while the Tasks API reports it for
        // the frame exactly as given. Same hand, same unflipped frame, opposite word.
        //
        // Left unswapped, canonicalizeHandedness declines to mirror a hand that Python DID
        // mirror, so every x coordinate reaches the model negated. Measured on real browser
        // gestures: orient.x came out +0.574 where training averages -0.575 (z = +8.8) and
        // every path*.x had its sign flipped, which turned J into MOVE at 0.41 and left Z
        // abstaining at 0.45. Negating x on those same spans recovers EMIT J p=0.97 and
        // EMIT Z p=0.93.
        const raw = hs[0][0].categoryName;
        handed = raw === 'Left' ? 'Right' : raw === 'Right' ? 'Left' : raw;
      }
      pxHist.push({ t, lm });
      // The same span of history the segmenter keeps, read from the thresholds rather than
      // repeated as a number here: the longest thing ever drawn is one track, and a trail
      // shorter than the buffer would draw a J that the machine is still scoring in full.
      while (pxHist.length && t - pxHist[0].t > th.BUFFER) pxHist.shift();
    }

    let em = null;
    try {
      // The capture frame's true dimensions: the isotropy correction is u = x * (W/H), so the
      // ratio has to be the camera's and not the canvas's.
      em = seg.step(t, lm, handed, video.videoWidth, video.videoHeight);
    } catch (err) {
      // Thirty exceptions a second would scroll the console and leave the page looking frozen.
      stopCamera();
      problem(`The segmenter threw on a frame (${err.message}). Recognition has stopped and the `
        + `camera has been switched off; the stack is in the browser console.`, false);
      throw err;
    }
    if (em) onEmission(em, t);

    frameTimes.push(t);
    if (frameTimes.length > FPS_WINDOW) frameTimes.shift();

    draw(lm);
    updatePanel(t);
    schedule();
  }

  function onEmission(em, t) {
    letters.push(em.letter);
    ui.letters.textContent = letters.join('');
    ui.lastem.textContent = `${em.letter} (${em.kind})`;
    const line = document.createElement('div');
    const conf = Number.isFinite(em.confidence) ? em.confidence.toFixed(2) : '?';
    line.textContent = `${t.toFixed(2)}  ${em.letter}  ${em.kind.padEnd(6)} p=${conf}`;
    ui.emlog.prepend(line);
    while (ui.emlog.childElementCount > 40) ui.emlog.lastElementChild.remove();
  }

  // ---------------------------------------------------------------- drawing

  function draw(lm) {
    const video = ui.video;
    const W = Math.min(DISPLAY_W, video.videoWidth);
    const H = Math.round((W * video.videoHeight) / video.videoWidth);
    if (ui.canvas.width !== W || ui.canvas.height !== H) {
      ui.canvas.width = W;
      ui.canvas.height = H;
    }

    // The mirror, and the only one on the image. The frame handed to MediaPipe above was the
    // video element, not this canvas.
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    if (lm) {
      const pts = lm.map((p) => toDisplayPx(p, W, H));
      ctx.strokeStyle = 'rgba(230,232,234,0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
      }
      ctx.stroke();
      ctx.fillStyle = '#ffb43c';
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    drawTrack(W, H);
  }

  /** The tracked path, drawn while it is being accumulated.
   *
   * This is the part of the overlay that diagnosed the gesture thresholds. A J rejected for
   * being too short, or a track that starts late because the smoothed speed only confirms the
   * rise once the stroke is under way, is obvious as a drawn path and invisible in any number. */
  function drawTrack(W, H) {
    if (!seg || seg.state !== TRACKING || seg._trackFrom == null || !seg._arm) return;
    const tip = TIP_FOR_ARM[seg._arm];
    const trail = pxHist.filter((h) => h.t >= seg._trackFrom)
      .map((h) => toDisplayPx(h.lm[tip], W, H));
    if (trail.length < 2) return;
    ctx.strokeStyle = '#ff8000';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(trail[0][0], trail[0][1]);
    for (let i = 1; i < trail.length; i++) ctx.lineTo(trail[i][0], trail[i][1]);
    ctx.stroke();
    ctx.fillStyle = '#ff8000';
    ctx.beginPath();
    ctx.arc(trail[trail.length - 1][0], trail[trail.length - 1][1], 6, 0, Math.PI * 2);
    ctx.fill();
  }

  function vMax() { return Math.max(th.V_MOVE_UNARMED * 1.5, 3.0); }
  function sMax() { return Math.max(th.RIGID_VETO * 1.5, 0.3); }

  function drawTicks() {
    // Each threshold is drawn on its meter as a tick, because a number alone does not show that
    // v_bar is sitting just under V_MOVE_ARMED. The whole point of tuning is seeing how close a
    // signal comes to a line it never crosses.
    tick(ui.vmeter, th.V_STILL / vMax(), 'var(--hold)');
    tick(ui.vmeter, th.V_MOVE_ARMED / vMax(), 'var(--settling)');
    tick(ui.vmeter, th.V_MOVE_UNARMED / vMax(), 'var(--err)');
    tick(ui.smeter, th.SHAPE_STABLE / sMax(), 'var(--hold)');
    tick(ui.smeter, th.RIGID_VETO / sMax(), 'var(--err)');
    ui.vticks.textContent = `still<${th.V_STILL.toFixed(2)}  armed>${th.V_MOVE_ARMED.toFixed(2)}`
      + `  free>${th.V_MOVE_UNARMED.toFixed(2)}`;
    ui.sticks.textContent = `stable<${th.SHAPE_STABLE.toFixed(2)}`
      + `  veto>${th.RIGID_VETO.toFixed(2)}`;
  }

  function tick(meter, frac, color) {
    const d = document.createElement('div');
    d.className = 'tick';
    d.style.left = `${Math.min(Math.max(frac, 0), 1) * 100}%`;
    d.style.background = color;
    meter.appendChild(d);
  }

  function fill(meter, value, max) {
    meter.firstElementChild.style.width = `${Math.min(Math.max(value, 0) / max, 1) * 100}%`;
  }

  // ---------------------------------------------------------------- debug panel

  function updatePanel(t) {
    ui.state.textContent = seg.state;
    ui.state.style.color = STATE_COLOR[seg.state] || 'var(--ink)';
    ui.vval.textContent = seg.vBar.toFixed(2);
    ui.sval.textContent = seg.sigma.toFixed(3);
    // sigmaRigid is shown as a number rather than on the meter, because it is a DIFFERENT
    // signal from the one plotted: the meter draws the plain sigma, which drives SHAPE_STABLE,
    // while RIGID_VETO reads the rotation-aligned deviation. live_demo.py draws the veto tick on
    // the plain meter, which is convenient and slightly misleading -- the veto that aborts a
    // track is this number, so the panel prints it.
    ui.srigid.textContent = seg.sigmaRigid.toFixed(3);
    ui.srigid.className = seg.sigmaRigid > th.RIGID_VETO ? 'err' : '';
    fill(ui.vmeter, seg.vBar, vMax());
    fill(ui.smeter, seg.sigma, sMax());
    ui.fps.textContent = `${fpsFrom(frameTimes).toFixed(1)} fps`;
    if (ui.vetoval) {
      ui.vetoval.textContent = nTracks === 0
        ? 'no gesture tracks yet - park in the launch pose, then move'
        : `${nTracks} tracks | ` + trackLog.slice(0, 3)
            .map((x) => `${x.arm}:${x.dur.toFixed(2)}s ${x.reason}`).join('  |  ');
    }

    // The gate fractions come from the segmenter's own method over its own buffer. A
    // reimplementation here could drift from the state machine and make the overlay lie about
    // the one thing it exists to report.
    const jf = seg.buf.length ? seg._gateFraction(t, 'jGate') : 0.0;
    const zf = seg.buf.length ? seg._gateFraction(t, 'zGate') : 0.0;
    ui.jf.textContent = jf.toFixed(2);
    ui.zf.textContent = zf.toFixed(2);
    const arm = seg.state === TRACKING ? seg._arm : null;
    const g = gateStatus(jf, zf, arm, th);
    ui.armtag.textContent = g.tag;
    ui.armtag.style.color = g.color;

    if (arm && seg._trackFrom != null) {
      const L = trackPathLength(seg._since(seg._trackFrom), arm);
      ui.track.textContent = `${(t - seg._trackFrom).toFixed(2)}s / ${th.T_MAX.toFixed(2)}`
        + (L === null ? ''
          : `  path ${L.toFixed(2)} palm (need ${th.L_MIN.toFixed(1)}-${th.L_MAX.toFixed(0)})`);
    } else {
      ui.track.textContent = note || '-';
    }
  }

  // ---------------------------------------------------------------- wiring

  ui.start.addEventListener('click', () => { startCamera(); });
  ui.stop.addEventListener('click', stopCamera);
  ui.clear.addEventListener('click', () => {
    letters = [];
    ui.letters.textContent = '';
    ui.emlog.replaceChildren();
    ui.lastem.textContent = '-';
    // The segmenter is deliberately NOT reset: clearing the transcript is an edit of the text,
    // not a claim about the hand, and dropping its history would also drop the duplicate
    // suppression that stops one held letter emitting twice.
  });

  load();
}

// Importable from Node for the tests below the fold; boots only in a browser.
if (typeof document !== 'undefined' && document.getElementById('start')) boot();

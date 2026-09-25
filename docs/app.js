/* The page: camera in, letters out.
 *
 * This is the browser twin of temporal/live_demo.py and it is deliberately thin. Grab a frame,
 * run one hoisted HandLandmarker over the *unflipped* frame, hand (t, landmarks, handedness,
 * W, H) to Segmenter.step, append whatever it emits. All the substance lives in features.js,
 * forest.js and segmenter.js, which are ports of the Python of the same names; everything here
 * is plumbing and the overlay.
 *
 * Four things in here are not plumbing:
 *
 * MIRRORING. The frame given to MediaPipe is never flipped. MediaPipe assigns its handedness
 * label in image space, so a flipped frame reports a right hand as "Left", and
 * canonicalizeHandedness then mirrors exactly the hands it should have left alone -- a
 * chirality error that is invisible downstream and makes every feature wrong. The display copy
 * is mirrored because signing into a non-mirrored image is disorienting, and that mirror
 * happens in exactly one function, toDisplayPx, plus the one drawImage that uses the same flip.
 *
 * THE LABEL SWAP. The Tasks API this page runs and the legacy `solutions` API that produced every
 * training landmark disagree about what "Left" means on the same unflipped frame. The swap lives
 * in one exported function, swapTasksHandedness, and golden.json's Tasks cases go through it.
 *
 * SECONDS. Every threshold in thresholds.py is in seconds and palm widths, never milliseconds
 * or frames. MediaPipe's detectForVideo wants milliseconds. The two clocks meet at exactly one
 * line in onFrame, and the segmenter never sees a millisecond.
 *
 * THE SELF-CHECK. docs/golden.json is run through the loaded forests before the camera is ever
 * touched, and the result is printed in the debug panel. Every expensive bug in this project
 * was a silent divergence between two implementations that looked equivalent, and from the
 * emitted letters alone "the tree walk is subtly wrong" is indistinguishable from "this
 * visitor's hands are not the signer's".
 */
import {
  TIP_FOR_ARM, pathLength, toIsotropic, canonicalizeHandedness, staticFeatureFor,
  STATIC_FEATURES, DEFAULT_STATIC_TAG,
} from './features.js';
import { loadModels, predictProba } from './forest.js';
import { buildIndex, closestWord, posteriorsFor } from './words.js';
import { Segmenter, NO_HAND, SETTLING, HOLD, TRACKING } from './segmenter.js';
import { buildChart } from './reference.js';

const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
  + 'hand_landmarker/float16/1/hand_landmarker.task';

//: The display copy is drawn at most this wide, so the overlay has one layout instead of one
//: per camera. Nothing measured passes through it: the segmenter is fed MediaPipe's normalized
//: landmarks and the capture frame's true W and H.
const DISPLAY_W = 960;
const FPS_WINDOW = 30;

//: The feature transform each forest was trained on, by the tag the model export carries for it.
//: features.js can build every tag in STATIC_FEATURES and the Segmenter picks the function by
//: the tag, so a registered tag can never produce train/serve skew; an UNREGISTERED tag is the
//: one failure that produces no symptom at all -- the vector is the right length, the forest is
//: confident, every letter is wrong -- and live_demo.py refuses to start on it. Same refusal
//: here. Both forests are expected on static/v4 (112-D): the digit forest was trained and
//: measured on it too (train_digits.py), so one feature path serves both modes; another
//: registered tag is accepted with a note.
export const LETTERS_TAG = 'static/v4';
export const DIGITS_TAG = 'static/v4';
const MOTION_FEATURE_TAG = 'event/v1';

//: Seconds a single download may produce nothing before the load line says so. A blocked CDN
//: and a slow connection look the same for the first few seconds; after this long they do not.
const LOAD_WATCHDOG_S = 20;
//: Chrome rejects a keepalive fetch whose body (together with every other in-flight keepalive
//: body) exceeds 64 KiB, before anything reaches the server. Records above this go without it.
const KEEPALIVE_MAX_BYTES = 60 * 1024;

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
 *  including the tie rule -- J wins a tie because _stepSettling arms "J" when jf >= zf. With
 *  arming off (numbers mode) it says so, because a gate that reads "ready" but can never arm
 *  would send a visitor looking for a J bug in a mode that has no J. */
export function gateStatus(jf, zf, arm, th, armGates = true) {
  if (!armGates) return { tag: 'gates off (numbers)', color: 'var(--dim)' };
  if (arm) return { tag: `ARMED ${arm}`, color: 'var(--tracking)' };
  const ready = (jf >= th.GATE_ARM_FRAC && jf >= zf) ? 'J'
    : (zf >= th.GATE_ARM_FRAC ? 'Z' : null);
  if (ready) return { tag: `gate ready ${ready}`, color: 'var(--hold)' };
  return { tag: 'gate none', color: 'var(--dim)' };
}

/** The handedness label the training pipeline would have reported for this frame, from the
 * label the Tasks API reports.
 *
 * SWAP the label. The two MediaPipe APIs disagree about what they are labeling: the legacy
 * `solutions` API that produced every training landmark reports handedness as if the image
 * were mirrored (the selfie convention), while the Tasks API reports it for the frame exactly
 * as given. Same hand, same unflipped frame, opposite word.
 *
 * Left unswapped, canonicalizeHandedness declines to mirror a hand that Python DID mirror, so
 * every x coordinate reaches the model negated. Measured on real browser gestures: orient.x
 * came out +0.574 where training averages -0.575 (z = +8.8) and every path*.x had its sign
 * flipped, which turned J into MOVE at 0.41 and left Z abstaining at 0.45. Negating x on those
 * same spans recovers EMIT J p=0.97 and EMIT Z p=0.93. On the archive re-extracted with the
 * Tasks API, the cross-day static model scores 0.775 with the swap and 0.730 without it.
 *
 * Every Tasks label on this page goes through here: the frame loop, and golden.json's cases
 * tagged api "tasks", which store the RAW Tasks label so this function is what the self-check
 * exercises. Anything other than the two words passes through unchanged.
 */
export function swapTasksHandedness(raw) {
  return raw === 'Left' ? 'Right' : raw === 'Right' ? 'Left' : raw;
}

/** Whether a static forest's feature tag is one features.js can build, and whether it is the
 * tag this page expects for that forest: {tag, known, expected, asExpected}. A null tag is the
 * oldest export format and means static/v3, as in the Python. */
export function staticTagStatus(model, expected) {
  const tag = model.feature == null ? DEFAULT_STATIC_TAG : String(model.feature);
  const known = Object.prototype.hasOwnProperty.call(STATIC_FEATURES, tag);
  return { tag, known, expected, asExpected: known && tag === expected };
}

/** Run golden.json's cases through the loaded forests. Returns {n, worst, worstEnd, wrong,
 * tol, ok, counted}.
 *
 * `models` is {static, digits}: each case names its forest in `model` ("static" when absent),
 * and the feature function for a case is the one that forest's tag selects through
 * STATIC_FEATURES -- both shipped forests on static/v4 (112-D). A case whose forest
 * the export does not carry, or whose stored vector width or class count disagrees with that
 * forest, THROWS: those are exactly the export-side mismatches the self-check exists to catch,
 * and the caller reports them as a problem, not as "not run".
 *
 * TWO legs, because a single number cannot say which half broke:
 *
 *   worst     golden.json's own feature vector -> the forest. A failure here is the tree walk.
 *   worstEnd  the case's RAW landmarks -> toIsotropic -> canonicalizeHandedness ->
 *             the feature function -> the forest: the entire path the camera frames take. A
 *             failure here with `worst` clean is the feature transform.
 *
 * The second leg is not redundant and it is the more important one. The feature transform is
 * where this project's expensive bugs actually lived -- a permuted pair-distance block, a
 * chirality flip -- and none of them is visible to the forest leg, which never calls
 * features.js at all. Measured: permuting the distance block turns case 0 from A p=1.00 into
 * M p=0.16, and the forest leg still reports a perfect 0.0.
 *
 * Cases tagged api "tasks" store the label the Tasks API reported and are routed through
 * swapTasksHandedness before canonicalizeHandedness, exactly as the frame loop routes a live
 * label; fed raw, the same cases fail this leg by probability (test_app.mjs checks that).
 *
 * Both legs compare PROBABILITIES because they are what the page acts on and they exercise the
 * forest walk. golden.json's landmarks are rounded to 6 decimals BEFORE every stored field is
 * computed (export_models.make_case), so a correct port reproduces its features to their own
 * 6-decimal output rounding (5e-7 measured over the 41 cases, against the file's 1e-6) and
 * test_features.mjs holds that feature-level comparison to the file's 1e-6 with no allowance.
 * Probabilities differ from sklearn's by up to ~2e-6 on these cases because leaves are rounded
 * to 4 decimals on export (bound 5e-5), well inside the 1e-4 tolerance.
 */
export function checkGolden(models, golden) {
  const tol = (golden.tolerance && golden.tolerance.probabilities) || 1e-4;
  let worst = 0.0;
  let worstEnd = 0.0;
  let wrong = 0;
  let n = 0;
  const counted = {};
  for (const c of golden.cases) {
    const which = c.model || 'static';
    const model = models[which];
    if (!model) {
      throw new Error(`golden case ${n} needs the ${which} forest, which the export does not carry`);
    }
    const [featureFn, dim] = staticFeatureFor(model.feature == null ? null : model.feature);
    if (c.static_feature.length !== dim) {
      throw new Error(`golden case ${n}: stored a ${c.static_feature.length}-D feature, but the `
        + `${which} forest (${model.feature}) takes ${dim}-D`);
    }
    if (c.static_probs.length !== model.classes.length) {
      throw new Error(`golden case ${n}: stores ${c.static_probs.length} probabilities, the `
        + `${which} forest has ${model.classes.length} classes`);
    }
    const p = predictProba(model, c.static_feature);
    for (let i = 0; i < p.length; i++) worst = Math.max(worst, Math.abs(p[i] - c.static_probs[i]));

    // The same case again, entered the way a frame enters: raw MediaPipe landmarks, the
    // handedness label as the training API would have reported it on an unflipped frame, and
    // the capture size.
    const label = c.api === 'tasks' ? swapTasksHandedness(c.reported_handedness) : c.handedness;
    const P = canonicalizeHandedness(toIsotropic(c.landmarks, c.width, c.height), label);
    const q = predictProba(model, featureFn(P));
    let bi = 0;
    for (let i = 0; i < q.length; i++) {
      worstEnd = Math.max(worstEnd, Math.abs(q[i] - c.static_probs[i]));
      if (q[i] > q[bi]) bi = i;
    }
    if (model.classes[bi] !== c.predicted) wrong += 1;
    counted[which] = (counted[which] || 0) + 1;
    n += 1;
  }
  return { n, worst, worstEnd, wrong, tol, counted,
           ok: worst <= tol && worstEnd <= tol && wrong === 0 };
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
    letters: el('letters'), clear: el('clear'), mode: el('mode'),
    state: el('state'), armtag: el('armtag'), fps: el('fps'),
    vval: el('vval'), vmeter: el('vmeter'), vticks: el('vticks'),
    sval: el('sval'), smeter: el('smeter'), sticks: el('sticks'),
    srigid: el('srigid'), vetoval: el('vetoval'), vsmooth: el('vsmooth'),
    curtain: el('curtain'),
    jf: el('jf'), zf: el('zf'), armfrac: el('armfrac'), track: el('track'),
    lastem: el('lastem'), selfcheck: el('selfcheck'), emlog: el('emlog'),
    word: el('word'),
    fatal: el('fatal'), fatalmsg: el('fatalmsg'),
  };
  const ctx = ui.canvas.getContext('2d');
  // Nothing can be started until the forests and MediaPipe are here: a click before that ran
  // the frame loop into a landmarker that did not exist yet and reported a false "MediaPipe
  // stopped processing frames". index.html ships the button disabled for the same reason;
  // this is the guard against a markup edit removing it. The Numbers toggle gets the same
  // treatment: until the letters segmenter is built there is nothing to switch, and on a page
  // blocked before the thresholds were validated a click used to reach buildSegmenter with
  // `th` unset and throw out of the click handler.
  if (ui.mode) ui.mode.disabled = true;
  ui.start.disabled = true;

  let th = null;
  let seg = null;
  let models = null;
  let chart = null;
  // Letters on every load, never persisted: a visitor returning to a page silently left in
  // numbers mode would see spurious zeros from a relaxed hand with no cue why.
  let mode = 'letters';

  // Localhost only, and only when devserver.py is the server: post diagnostics to it, the
  // browser twin of live_demo.py --log. A deployed visitor posts nothing -- there is no
  // endpoint and the guard short-circuits first. The endpoint is probed ONCE, because
  // `python -m http.server` (which the README also suggests) answers every POST with a 501,
  // and a hold that abstains posts on every frame: thirty red lines a second in the console
  // for nothing. devserver.py answers OPTIONS /log with 204; the stdlib server does not.
  let LOGGING = false;
  if (['localhost', '127.0.0.1'].includes(location.hostname)) {
    fetch('/log', { method: 'OPTIONS' })
      .then((r) => { LOGGING = r.status === 204; })
      .catch(() => { LOGGING = false; });
  }
  function postLog(obj) {
    if (!LOGGING) return;
    try {
      const body = JSON.stringify(obj) + '\n';
      const opts = { method: 'POST', headers: { 'content-type': 'application/json' }, body };
      // keepalive lets a record survive the tab closing, but the browser caps keepalive bodies
      // at 64 KiB (all in-flight ones together) and REJECTS a larger one before sending it. A
      // long track at 30-60 fps (frames x 21 x 2 doubles, ~900 B per frame) is that large,
      // and those are exactly the records the instrument exists for, so big bodies go
      // without keepalive. A record that still fails is said so in the console, once per
      // record, rather than swallowed.
      if (body.length <= KEEPALIVE_MAX_BYTES) opts.keepalive = true;
      fetch('/log', opts).catch((e) => console.warn(`postLog dropped a ${body.length} B record: ${e.message}`));
    } catch (e) { /* diagnostics must never break the demo */ }
  }

  const trackLog = [];
  let nTracks = 0;
  let landmarker = null;
  let stream = null;
  let running = false;
  let letters = [];
  // One entry per letter of the word being spelled, each carrying the vote that produced it, so
  // the word layer can score candidate spellings when the word ends. Cleared at every break.
  let wordBuf = [];
  let wordIndex = null;
  // The class list of the forest the running segmenter votes with. It follows the mode
  // (24 letters or 10 digits) and is what onEmission/posteriorsFor read; reaching for
  // models.static.classes from the emission handler would name a letter for a digit.
  let staticClasses = null;
  // Timestamp of the last frame a hand was actually detected in. A word break is measured from
  // here rather than from the last emission, because the pause after the final letter of a word
  // is time spent with the hand down, not time spent holding a letter.
  let lastHandT = null;
  const frameTimes = [];
  // Normalized landmarks with their timestamps, kept here rather than read back out of the
  // segmenter: its buffer holds isotropic, chirality-canonicalized coordinates, which cannot be
  // drawn on a frame.
  let pxHist = [];
  let lastVideoTime = -1;
  let lastTs = -1;
  let note = '';

  //: The problem box holds one message PER KIND, not one message. Starting the camera retracts
  //: the camera's own problem and nothing else; the mode toggle retracts its own. With a
  //: single slot a camera refusal replaced the self-check warning and the next successful
  //: Start hid the box -- "do not trust the letters" gone at exactly the moment it starts
  //: mattering, the user now signing at a recognizer that has already said it does not
  //: reproduce Python's answers. Every open problem is shown; the box hides only when none is.
  const problems = new Map();
  function paintProblems() {
    ui.fatal.hidden = problems.size === 0;
    ui.fatalmsg.textContent = [...problems.values()].join(' ');
  }

  /** Say what went wrong, naming the step that failed. `blocking` means nothing usable is
   *  left; a non-blocking problem is still shown, because a page that quietly drops half of
   *  itself is exactly how "J is unreachable" hides. */
  function problem(msg, blocking, kind = 'other') {
    problems.set(kind, msg);
    paintProblems();
    if (blocking) {
      ui.start.disabled = true;
      if (ui.mode) ui.mode.disabled = true;
      ui.loadstate.textContent = 'stopped: see the problem above';
    }
  }

  /** Retract one kind of problem and nothing else. */
  function clearProblem(kind) {
    if (problems.delete(kind)) paintProblems();
  }

  /** Report a download that has produced nothing for LOAD_WATCHDOG_S seconds. The promise is
   *  left to settle or fail on its own; only the load line changes, from "loading" to a
   *  sentence that says the download is probably blocked rather than slow. */
  function watchdog(promise, what) {
    const timer = setTimeout(() => {
      ui.loadstate.textContent = `still ${what} after ${LOAD_WATCHDOG_S} s. A slow connection `
        + 'does not take this long; the download is probably blocked (a firewall, an ad '
        + 'blocker, an offline machine). Reload the page to try again.';
    }, LOAD_WATCHDOG_S * 1000);
    return promise.finally(() => clearTimeout(timer));
  }

  // ---------------------------------------------------------------- loading

  /** The Segmenter for the current mode, built fresh. Rebuilt rather than re-pointed on a mode
   *  change and on Stop, so a parked D cannot be delivered as a digit, a lastEmitted 'O' cannot
   *  suppress a '0', and a restart with the hand already up starts from NO_HAND rather than
   *  from a consumed hold. Classes are left to default: forest.js keeps the export's class
   *  list on the prepared model and segmenter.js reads it from there, so there is one list,
   *  not two. The feature function likewise comes from the model's own tag. */
  function buildSegmenter() {
    const digits = mode === 'digits';
    // Numbers mode: the same machine with the two vote constants DIGITS_OVERRIDES raises
    // (VOTE_MARGIN_CLEAR, VOTE_PROB_FLOOR), the digit forest, no motion branch and no
    // arming -- zGate passes most '1' frames, and a track would park the machine in
    // TRACKING where nothing static is voted.
    const thr = digits ? { ...th, ...(models.digitsThresholds || {}) } : th;
    seg = new Segmenter(thr, {
      staticModel: digits ? models.digits : models.static,
      motionModel: digits ? null : models.motion,
      armGates: !digits,
      onHold: (h) => postLog({ kind: 'hold', mode, ...h }),
      // Track outcomes are otherwise invisible: a gesture that never arms, or one killed by a
      // veto, leaves no trace on screen at all. Every hard bug in the Python was found by
      // logging exactly this, so the browser gets the same instrument.
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
    staticClasses = seg.staticClasses;
  }

  function paintMode() {
    if (!ui.mode) return;
    const digits = mode === 'digits';
    ui.mode.setAttribute('aria-pressed', String(digits));
    ui.mode.title = digits ? 'Switch back to letters' : 'Switch to numbers 0-9 (experimental)';
  }

  /** Switch between letters and numbers. The transcript is left alone; the word in progress
   *  is dropped (its letters and digits would not spell anything together), and the
   *  segmenter is rebuilt for the reasons buildSegmenter gives. */
  function setMode(next) {
    // `seg` is non-null exactly when buildSegmenter has succeeded once, i.e. the model export
    // arrived and the Segmenter accepted its thresholds; the button is disabled until then
    // (and again on a blocking problem), and this is the guard against a click that gets
    // through anyway.
    if (next === mode || !seg) return;
    if (next === 'digits' && !models.digits) {
      problem('This model export carries no numbers forest, so numbers mode is not available. '
        + 'Re-export it with temporal/model_digits.p present to enable it.',
      false, 'mode');
      return;
    }
    mode = next;
    wordBuf = [];
    ui.word.textContent = '';
    ui.word.className = '';
    clearProblem('mode');
    buildSegmenter();
    paintMode();
  }

  async function loadChart() {
    const host = document.getElementById('refchart');
    const toggle = document.getElementById('reftoggle');
    if (!host || !toggle) return;
    toggle.addEventListener('click', () => {
      const hide = !host.hidden;
      host.hidden = hide;
      toggle.textContent = hide ? 'Show' : 'Hide';
      toggle.setAttribute('aria-expanded', String(!hide));
    });
    try {
      const res = await fetch('./reference.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      chart = buildChart(host, data);
    } catch (err) {
      // Deliberately not problem(): the chart is a convenience and the page works without it.
      // Hiding the control is better than leaving a Hide button over an empty grid.
      document.getElementById('refwrap')?.setAttribute('hidden', '');
      console.warn('alphabet chart unavailable:', err.message);
    }
  }

  async function load() {
    try {
      ui.loadstate.textContent = 'fetching models.bin.gz and models.meta.json (about 1.18 MB, '
        + 'already compressed, cached after the first time)';
      // forest.js owns the fetch, including the gzip sniffing GitHub Pages needs. A second
      // loader here would be a second implementation of the thing this project keeps being
      // burned by having two of. The binary pair is 1,184,427 B on the wire against models.json's
      // 3,703,083 B, and loadModels returns the identical object either way -- docs/models.json
      // stays committed as the readable reference and still loads if this line is pointed at it.
      models = await watchdog(loadModels('./models.bin.gz', './models.meta.json'),
        'fetching models.bin.gz and models.meta.json');
    } catch (err) {
      problem(`The model file failed to load. ${err.message} `
        + `If you opened index.html from the filesystem, serve the folder over http instead: `
        + `fetch and ES modules do not work from file://.`, true);
      return;
    }
    // A model exported from a feature transform this page cannot build is the one failure that
    // produces no symptom at all: the vector is the right length, the forest is confident, and
    // every letter is wrong. live_demo.py exits on this; so does the page. A registered tag
    // other than the expected one is accepted with a note, because the Segmenter builds the
    // vector the tag names and no skew is possible; the note is there because the numbers
    // quoted on this page were measured on the expected tags.
    const roles = [['letters', models.static, LETTERS_TAG]];
    if (models.digits) roles.push(['numbers', models.digits, DIGITS_TAG]);
    const bad = [];
    const odd = [];
    for (const [which, m, want] of roles) {
      const s = staticTagStatus(m, want);
      if (!s.known) {
        bad.push(`The ${which} model in this export was trained on feature "${s.tag}", which `
          + `this page cannot build (it knows ${Object.keys(STATIC_FEATURES).join(', ')}).`);
      } else if (!s.asExpected) {
        odd.push(`${which} model on ${s.tag}, expected ${want}`);
      }
    }
    if (models.motion.feature != null && models.motion.feature !== MOTION_FEATURE_TAG) {
      bad.push(`The motion model in this export was trained on feature "${models.motion.feature}", `
        + `but this page builds "${MOTION_FEATURE_TAG}".`);
    }
    if (bad.length) {
      problem(`${bad.join(' ')} Re-export the models from the current temporal/ code; running `
        + 'it anyway would produce confident nonsense rather than an error.', true);
      return;
    }
    if (odd.length) note = `feature tags: ${odd.join('; ')}`;

    th = models.thresholds;
    // The Segmenter is what validates the thresholds (segmenter.js REQUIRED_THRESHOLDS): it is
    // built BEFORE any field is read here, so an export missing one is reported as the
    // readable "thresholds missing [...]" message and not as a TypeError on the first
    // toFixed. The word constants are checked by words.js at every verdict instead.
    try {
      buildSegmenter();
    } catch (err) {
      problem(`The segmenter rejected the thresholds in this export: ${err.message}`, true);
      return;
    }
    paintMode();
    // The letters segmenter exists and validated the thresholds: the toggle can switch now.
    // MediaPipe is not needed to change mode.
    if (ui.mode) ui.mode.disabled = false;
    ui.armfrac.textContent = th.GATE_ARM_FRAC.toFixed(2);
    ui.vetoval.textContent = th.RIGID_VETO.toFixed(2);
    ui.vsmooth.textContent = th.V_SMOOTH_WINDOW.toFixed(2);
    drawTicks();

    // Before the camera, so a broken tree walk is reported as a broken tree walk instead of as
    // a page that mysteriously reads every letter as D.
    let golden = null;
    try {
      ui.loadstate.textContent = 'fetching golden.json (the self-check cases)';
      // fetch() resolves on the headers and res.json() reads the body, so both halves are
      // under the watchdog: a connection that drops after the headers stalls the second.
      const res = await watchdog(fetch('./golden.json'), 'fetching golden.json');
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      golden = await watchdog(res.json(), 'fetching golden.json');
    } catch (err) {
      // Only a golden.json that cannot be fetched or parsed is "not run". Everything the check
      // itself throws is a mismatch, handled below.
      ui.selfcheck.textContent = 'not run';
      note = `self-check skipped: golden.json ${err.message}`;
    }
    if (golden) {
      try {
        const g = checkGolden({ static: models.static, digits: models.digits }, golden);
        const per = Object.keys(g.counted).map((k) => `${g.counted[k]} ${k}`).join(' + ');
        ui.selfcheck.textContent = g.ok
          ? `${g.n}/${g.n} cases (${per}), forest ${g.worst.toExponential(1)}, `
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
        // A throw here is structural: a case for a forest the export does not carry, or a
        // class count or feature width that disagrees with it. That is the export-side
        // mismatch the self-check exists to catch, so it is a problem in its own right --
        // it used to be downgraded to "not run" with Start enabled.
        ui.selfcheck.textContent = 'FAILED: could not run';
        ui.selfcheck.className = 'err';
        problem(`The self-check could not be run against this model export: ${err.message}. `
          + 'golden.json and the forests were not exported together. The page still runs, '
          + 'but do not trust the letters.', false, 'selfcheck');
      }
    }

    // The word list is a hint, so it loads in the background and its failure is silent: a page
    // that refused to recognize letters because a dictionary 404'd would be trading the thing
    // that works for the thing that decorates it.
    fetch('./words.txt')
      .then((res) => (res.ok ? res.text() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((text) => { wordIndex = buildIndex(text); })
      .catch(() => { wordIndex = null; });

    let vision;
    try {
      ui.loadstate.textContent = 'loading MediaPipe from the CDN (about 9 MB of WASM)';
      vision = await watchdog(import(`${MP_CDN}/vision_bundle.mjs`), 'loading MediaPipe from the CDN');
    } catch (err) {
      problem(`MediaPipe could not be loaded from ${MP_CDN} (${err.message}). Without it there `
        + `is nothing to find the hand in the frame.`, true);
      return;
    }
    try {
      // forVisionTasks does no I/O, so it gets no watchdog: in tasks-vision@0.10.18's bundle it
      // awaits an in-memory WebAssembly.instantiate of a SIMD probe and returns two path strings
      // ({wasmLoaderPath, wasmBinaryPath}). Every MediaPipe download -- the loader <script> it
      // injects (which pulls the 9 MB .wasm), both from the CDN, then fetch(modelAssetPath) for
      // the .task on storage.googleapis.com -- happens inside createFromOptions, and one
      // promise cannot say which of the three stalled, so the line names both hosts.
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
      landmarker = await watchdog(createLandmarker(vision, fileset),
                                  'fetching the MediaPipe WASM from cdn.jsdelivr.net or the hand '
                                  + 'landmarker model from storage.googleapis.com');
    } catch (err) {
      problem(`MediaPipe loaded but the hand landmarker would not start (${err.message}). `
        + `The model file it fetches is hosted separately from the CDN, so this can also mean `
        + `that download was blocked.`, true);
      return;
    }

    ui.loadstate.textContent = `ready: ${models.static.classes.length} static letters from `
      + `${models.static.trees.length} trees, plus J and Z from the motion branch's `
      + `${models.motion.trees.length}`
      + (models.digits
        ? `, and ${models.digits.classes.length} digits from the numbers forest's `
          + `${models.digits.trees.length}`
        : '')
      + `.${note ? ` ${note}` : ''}`;
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
      note = `${note ? `${note}; ` : ''}no GPU delegate; MediaPipe is running on the CPU`;
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
    clearProblem('camera');
    // The stream can end without anyone pressing Stop: the device is unplugged, another
    // application takes it, the permission is revoked from the address bar. The frame loop
    // would then run on a frozen last frame with the state line still reading as if live.
    const track = stream.getVideoTracks()[0];
    if (track) {
      track.addEventListener('ended', () => {
        if (!running) return;
        stopCamera();
        problem('The camera stream ended on its own: the device was unplugged, taken by '
          + 'another application, or its permission was revoked. Press Start to try again.',
        false, 'camera');
      });
    }
    ui.video.srcObject = stream;
    try {
      await ui.video.play();
    } catch (err) {
      // Rare (the click's activation normally carries through the getUserMedia await), but if
      // playback is refused the camera light is on and nothing is reading it: release it and
      // say so, rather than leave a dead Start button and a live camera.
      stopCamera();
      problem(`The camera started but the video element would not play (${err.name}: `
        + `${err.message}). The camera has been released; press Start to try again.`,
      false, 'camera');
      return;
    }
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
    // Close the word being spelled, so its verdict is shown rather than lost, and start the
    // next session from NO_HAND: a segmenter kept across Stop/Start resumed in a consumed
    // HOLD when the hand was already up (nothing emitted until it moved) and could not
    // re-emit the last letter; the old wall-clock in lastHandT would have written a word
    // break into the middle of the first word of the next session.
    closeWord();
    if (models) buildSegmenter();
    lastHandT = null;
    // Drop the drawing history: a trail from the previous session would otherwise be drawn over
    // the first frames of the next one, and the fps figure would average across the gap.
    pxHist = [];
    frameTimes.length = 0;
    ui.video.srcObject = null;
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
      // it was unflipped in training -- and then swapped into the training API's convention.
      const hs = res.handednesses || res.handedness;
      if (hs && hs.length && hs[0].length) {
        handed = swapTasksHandedness(hs[0][0].categoryName);
      }
      pxHist.push({ t, lm });
      // The same span of history the segmenter keeps, read from the thresholds rather than
      // repeated as a number here: the longest thing ever drawn is one track, and a trail
      // shorter than the buffer would draw a J that the machine is still scoring in full.
      while (pxHist.length && t - pxHist[0].t > th.BUFFER) pxHist.shift();
    }

    if (lm) {
      lastHandT = t;
    } else if (lastHandT !== null && t - lastHandT > th.SPACE_GAP) {
      closeWord();
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
    wordBuf.push({ letter: em.letter, post: posteriorsFor(em, staticClasses) });
    ui.word.textContent = '';
    ui.letters.textContent = letters.join('');
    ui.lastem.textContent = `${em.letter} (${em.kind})`;
    // The chart is on the same screen as the camera so it can answer "was that what I meant?".
    // chart is null until reference.json lands, and a digit has no cell, so highlight() no-ops.
    if (chart) chart.highlight(em.letter);
    const line = document.createElement('div');
    const conf = Number.isFinite(em.confidence) ? em.confidence.toFixed(2) : '?';
    line.textContent = `${t.toFixed(2)}  ${em.letter}  ${em.kind.padEnd(6)} p=${conf}`;
    // index.html seeds the log with a placeholder text node ("-"); childElementCount and
    // lastElementChild never see a text node, so it used to survive as a stray line under the
    // log. The first emission replaces it.
    if (ui.emlog.childElementCount === 0) ui.emlog.textContent = '';
    ui.emlog.prepend(line);
    while (ui.emlog.childElementCount > 40) ui.emlog.lastElementChild.remove();
  }

  /**
   * End the word being spelled: write the break, and say what it most looks like.
   *
   * Called from the frame loop rather than on a timer, so it cannot fire while the page is in a
   * background tab with no frames arriving and silently split a word in half. In numbers mode
   * the break is still written (a space between numbers is useful) but the dictionary is not
   * consulted: posteriorsFor maps classes to a..z and a digit has no place there, so a hint
   * would be scored against nothing.
   */
  function closeWord() {
    if (!wordBuf.length) return;
    if (mode === 'digits') {
      ui.word.textContent = '';
      ui.word.className = '';
    } else {
      const reading = wordBuf.map((x) => x.letter).join('');
      const verdict = closestWord(reading, wordBuf.map((x) => x.post), wordIndex, th);
      // Only two of the eight verdicts are worth a visitor's attention. "unlikely", "ambiguous"
      // and "too-short" are the layer working correctly and declining to guess, and the other
      // three ("no-list", "too-long", "no-thresholds") mean it could not look; announcing any of
      // them would be noise.
      if (verdict.reason === 'hint') {
        ui.word.textContent = `closest word: ${verdict.word.toUpperCase()}`;
        ui.word.className = 'hint';
      } else if (verdict.reason === 'exact') {
        ui.word.textContent = `${reading} is a word`;
        ui.word.className = 'exact';
      } else {
        ui.word.textContent = '';
        ui.word.className = '';
      }
    }
    wordBuf = [];
    letters.push(' ');
    ui.letters.textContent = letters.join('');
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
    // signal comes to a line it never crosses. V_MOVE_UNARMED is drawn although the state
    // machine never reads it: it is the calibrated ceiling of held-sign speed, and the meter
    // is where a signal is compared against it by eye.
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
    // while RIGID_VETO reads the rotation-aligned deviation. The veto that aborts a track is
    // this number, so the panel prints it.
    ui.srigid.textContent = seg.sigmaRigid.toFixed(3);
    ui.srigid.className = seg.sigmaRigid > th.RIGID_VETO ? 'err' : '';
    fill(ui.vmeter, seg.vBar, vMax());
    fill(ui.smeter, seg.sigma, sMax());
    ui.fps.textContent = `${fpsFrom(frameTimes).toFixed(1)} fps`;
    if (ui.vetoval) {
      ui.vetoval.textContent = nTracks === 0
        ? (seg.armGates ? 'no gesture tracks yet - park in the launch pose, then move'
          : 'no gesture tracks in numbers mode')
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
    const g = gateStatus(jf, zf, arm, th, seg.armGates);
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
  if (ui.mode) {
    ui.mode.addEventListener('click', () => { setMode(mode === 'digits' ? 'letters' : 'digits'); });
  }
  ui.clear.addEventListener('click', () => {
    letters = [];
    wordBuf = [];
    ui.letters.textContent = '';
    ui.word.textContent = '';
    ui.word.className = '';
    ui.emlog.replaceChildren();
    ui.lastem.textContent = '-';
    // The segmenter is deliberately NOT reset: clearing the transcript is an edit of the text,
    // not a claim about the hand, and dropping its history would also drop the duplicate
    // suppression that stops one held letter emitting twice.
  });

  loadChart();
  load();
}

// Importable from Node for the tests below the fold; boots only in a browser.
if (typeof document !== 'undefined' && document.getElementById('start')) boot();

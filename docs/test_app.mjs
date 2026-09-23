// Tests for docs/app.js -- the page's own logic, plus the contracts it depends on.
//
// app.js is mostly plumbing, and plumbing is where this project's expensive bugs lived: a
// mirrored frame that inverted the handedness label, a vote window that completed at 30 fps and
// never at 15, a Tasks-API label that means the opposite of the training API's. So the things
// tested here are exactly those: the mirror, the label swap, the clock, the self-check's
// refusal to be skipped, and the numbers the overlay claims to be reporting. Reading the code
// is not evidence.
//
// Some checks need the regenerated export (a static/v4 letter forest, Tasks-API and digit
// golden cases); until it lands they print a skip line with the reason rather than fail.
//
// Run:  node docs/test_app.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  toDisplayPx, fpsFrom, gateStatus, checkGolden, trackPathLength, swapTasksHandedness,
  staticTagStatus, LETTERS_TAG, DIGITS_TAG,
} from './app.js';
import { loadModels, prepareModel, predictProba } from './forest.js';
import { Segmenter, HOLD } from './segmenter.js';
import { TIP_FOR_ARM, STATIC_FEATURES } from './features.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const models = JSON.parse(readFileSync(join(HERE, 'models.json'), 'utf8'));
const golden = JSON.parse(readFileSync(join(HERE, 'golden.json'), 'utf8'));
const th = models.thresholds;
const prepared = {
  static: prepareModel(models.static),
  digits: models.digits ? prepareModel(models.digits) : null,
};
const skip = (what, why) => console.log(`skip  ${what}  (${why})`);

let failures = 0;
function check(name, ok, detail) {
  if (!ok) failures += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? `  ${detail}` : ''}`);
}
function close(a, b, tol) { return Math.abs(a - b) <= tol; }

// -------------------------------------------------------------- the mirror

// x=0 is the left edge of the frame MediaPipe saw and must land on the RIGHT edge of the
// display, or the signer's hand moves the wrong way on screen. The frame given to MediaPipe is
// untouched: nothing in app.js flips it, which is what keeps the handedness label -- and so the
// chirality correction in features.js -- pointing the right way.
{
  const [x0, y0] = toDisplayPx([0.0, 0.25], 960, 540);
  const [x1] = toDisplayPx([1.0, 0.5], 960, 540);
  const [xm] = toDisplayPx([0.5, 0.5], 960, 540);
  check('toDisplayPx mirrors x and leaves y alone', x0 === 960 && x1 === 0 && xm === 480 && y0 === 135,
    `x(0)=${x0} x(1)=${x1} x(0.5)=${xm} y(0.25)=${y0}`);
}
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  const html = readFileSync(join(HERE, 'index.html'), 'utf8');
  // The video element is what is handed to detectForVideo, so a CSS mirror on it would flip the
  // frame MediaPipe sees. The canvas is where the flip belongs.
  const cssMirrorsVideo = /#cam\s*\{[^}]*scaleX?\(/.test(html) || /#cam\s*\{[^}]*transform:/.test(html);
  check('index.html does not CSS-mirror the video element', !cssMirrorsVideo);
  check('detectForVideo is given the video element itself',
    /detectForVideo\(video, nowMs\)/.test(src));
  const flips = src.match(/ctx\.scale\(-1, 1\)/g) || [];
  check('exactly one canvas flip', flips.length === 1, `found ${flips.length}`);
}

// -------------------------------------------------------------- the label swap

// The Tasks API and the training API disagree about what "Left" means on the same unflipped
// frame. One function swaps, every Tasks label on the page goes through it, and it must pass
// anything else through untouched (an "Unknown" must not become a mirror).
{
  check('swapTasksHandedness swaps the two words',
    swapTasksHandedness('Left') === 'Right' && swapTasksHandedness('Right') === 'Left');
  check('...and passes anything else through',
    swapTasksHandedness('Unknown') === 'Unknown' && swapTasksHandedness(null) === null
    && swapTasksHandedness('left') === 'left');
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('the frame loop routes the live label through the swap',
    /handed = swapTasksHandedness\(hs\[0\]\[0\]\.categoryName\)/.test(src));
  check('...and no other swap is written out by hand',
    !/=== 'Left' \? 'Right'/.test(src.replace(/export function swapTasksHandedness[\s\S]*?\n}/, '')));
}

// -------------------------------------------------------------- the clock

// Every threshold in thresholds.py is in seconds. MediaPipe's detectForVideo is in
// milliseconds. app.js converts at one line; if that line ever divides the wrong way, the
// segmenter sees 1000x time and every window closes instantly.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('seconds are handed to the segmenter, milliseconds to MediaPipe',
    /const t = nowMs \/ 1000\.0;/.test(src) && /seg\.step\(t,/.test(src));
}
{
  const times = [10.0, 10.1, 10.2, 10.3];          // 3 gaps of 0.1 s over 4 samples
  check('fpsFrom is (n-1)/span', close(fpsFrom(times), 10.0, 1e-12), `${fpsFrom(times)}`);
  check('fpsFrom of one sample is 0', fpsFrom([1.0]) === 0);
  check('fpsFrom of a zero span is 0', fpsFrom([1.0, 1.0]) === 0);
}

// -------------------------------------------------------------- the overlay's claims

// draw_overlay(): ready = "J" if jf >= GATE_ARM_FRAC and jf >= zf else "Z" if zf >= ... The tie
// goes to J, matching _stepSettling's `arm = jf >= zf ? "J" : "Z"`. An overlay that named the
// other one would send you looking for a Z bug during a J failure.
{
  check('gate none below the arming fraction', gateStatus(0.49, 0.49, null, th).tag === 'gate none');
  check('gate ready J', gateStatus(0.60, 0.10, null, th).tag === 'gate ready J');
  check('gate ready Z', gateStatus(0.10, 0.60, null, th).tag === 'gate ready Z');
  check('a tie arms J, as the segmenter does', gateStatus(0.6, 0.6, null, th).tag === 'gate ready J');
  check('ARMED wins over ready', gateStatus(0.9, 0.0, 'Z', th).tag === 'ARMED Z');
}

// trackPathLength must report the number the L_MIN/L_MAX veto actually reads: the tip path in
// the segmenter's own isotropic coordinates, over the median palm scale.
{
  const frame = (x, y, S) => {
    const P = Array.from({ length: 21 }, () => [0, 0]);
    P[TIP_FOR_ARM.Z] = [x, y];
    return { P, S };
  };
  // Three steps of 1.0 -> path 3.0. Palm scales 1, 2, 3, 3: an even count, so the median is
  // the mean of the middle two, 2.5, exactly as np.median gives it to _score. 3.0 / 2.5 = 1.2.
  const frames = [frame(0, 0, 1), frame(1, 0, 2), frame(2, 0, 3), frame(3, 0, 3)];
  const L = trackPathLength(frames, 'Z');
  check('trackPathLength divides the path by the median palm scale', close(L, 1.2, 1e-12), `${L}`);
  check('trackPathLength of one frame is null', trackPathLength([frame(0, 0, 1)], 'Z') === null);
}

// -------------------------------------------------------------- the self-check the page runs

{
  const g = checkGolden(prepared, golden);
  const per = Object.keys(g.counted).map((k) => `${g.counted[k]} ${k}`).join(' + ');
  check(`checkGolden passes all ${g.n} golden cases (${per})`, g.ok && g.n === golden.cases.length,
    `worst ${g.worst.toExponential(2)} against tol ${g.tol}`);

  // ...and it must FAIL when the forest is wrong, or it is checking nothing. Corrupting a
  // single leaf is not enough of a test: 15 cases need not visit it, and a lone tree moves the
  // mean by at most 1/n_trees. Corrupt every leaf of one tree, which every case does visit.
  const broken = prepareModel(models.static);
  for (let i = 0; i < broken.trees[0].leaf.length; i += broken.nClasses) {
    broken.trees[0].leaf[i] += 1.0;
  }
  const b = checkGolden({ ...prepared, static: broken }, golden);
  check('checkGolden fails on a corrupted forest', !b.ok, `worst ${b.worst.toExponential(2)}`);

  // The second leg: raw landmarks in, through features.js, not golden's stored vector. Without
  // it the self-check is blind to the half of the pipeline where this project's bugs actually
  // lived -- a permuted distance block or an inverted chirality leaves the tree walk perfect and
  // every letter wrong. It must reproduce Python exactly, and it must name the right letter.
  check('checkGolden runs the whole path from the raw landmarks',
    g.worstEnd <= g.tol && g.wrong === 0,
    `whole path ${g.worstEnd.toExponential(2)}, ${g.wrong} letters wrong`);

  // A model whose feature columns are permuted is what a broken transform looks like from the
  // forest's side: same length, same confidence, different answer. Both legs must fail.
  const permuted = prepareModel(models.static);
  for (const tree of permuted.trees) {
    for (let i = 0; i < tree.f.length; i += 1) {
      if (tree.f[i] >= 42) tree.f[i] = 42 + (100 - tree.f[i]);   // reverse the distance block
    }
  }
  const p = checkGolden({ ...prepared, static: permuted }, golden);
  check('checkGolden fails when the feature columns are permuted', !p.ok && p.wrong > 0,
    `whole path ${p.worstEnd.toExponential(2)}, ${p.wrong} of ${p.n} letters wrong`);

  // A Tasks-API case stores the RAW Tasks label. Fed to canonicalizeHandedness without the
  // swap, the hand is mirrored the wrong way and the whole-path leg must fail BY PROBABILITY --
  // the argmax may or may not survive (a mirrored A is still an A-ish fist), which is why the
  // check compares distributions and not letters.
  const tasks = golden.cases.filter((c) => c.api === 'tasks');
  if (!tasks.length) {
    skip('the swap-routing test', 'golden.json carries no Tasks-API cases yet; export_models.py adds them from static_sequences_tasks.npz');
  } else {
    const swapped = checkGolden(prepared, { ...golden, cases: tasks });
    check(`the ${tasks.length} Tasks cases pass through the swap`, swapped.ok,
      `whole path ${swapped.worstEnd.toExponential(2)}, ${swapped.wrong} wrong`);
    const raw = tasks.map((c) => ({ ...c, api: 'solutions', handedness: c.reported_handedness }));
    const unswapped = checkGolden(prepared, { ...golden, cases: raw });
    check('...and fail by probability when fed their raw label without it',
      !unswapped.ok && unswapped.worstEnd > unswapped.tol && unswapped.worst <= unswapped.tol,
      `whole path ${unswapped.worstEnd.toExponential(2)} (forest leg ${unswapped.worst.toExponential(2)}), ` +
      `${unswapped.wrong} of ${unswapped.n} argmax wrong`);
    check('the Tasks cases carry the raw label, not the swapped one',
      tasks.every((c) => ['Left', 'Right'].includes(c.reported_handedness)));
  }

  // A golden/model mismatch is a PROBLEM, not a skipped check. checkGolden throws on a case
  // whose forest models.json does not carry, on a class count that disagrees, and on a stored
  // vector of the wrong width; load() turns that throw into problem(..., 'selfcheck'). It used
  // to be caught by the same catch as a 404 and shown as "not run" with Start enabled.
  const throwsOn = (label, fn) => {
    let err = null;
    try { fn(); } catch (e) { err = e; }
    check(`checkGolden throws on ${label}`, err !== null, err ? err.message : 'returned normally');
  };
  const one = golden.cases.find((c) => (c.model || 'static') === 'static');
  throwsOn('a digits case with no digits forest', () => checkGolden(
    { static: prepared.static, digits: null }, { ...golden, cases: [{ ...one, model: 'digits' }] }));
  throwsOn('a class-count mismatch', () => checkGolden(prepared, { ...golden,
    cases: [{ ...one, static_probs: one.static_probs.slice(0, -1) }] }));
  throwsOn('a feature-width mismatch', () => checkGolden(prepared, { ...golden,
    cases: [{ ...one, static_feature: one.static_feature.slice(0, -1) }] }));
  throwsOn('a forest tag the registry does not know', () => checkGolden(
    { static: { ...prepared.static, feature: 'static/v9' } }, { ...golden, cases: [one] }));
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('load() reports a checkGolden throw as a self-check problem',
    /catch \(err\) \{[^}]*could not be run against this model export[\s\S]*?false, 'selfcheck'\)/.test(src));
  check('...and only a golden.json fetch failure is "not run"',
    /golden = await watchdog\(res\.json\(\), 'fetching golden\.json'\);\s*\} catch \(err\) \{\s*\/\/[^\n]*\n(\s*\/\/[^\n]*\n)*\s*ui\.selfcheck\.textContent = 'not run'/.test(src));
}

// -------------------------------------------------------------- the feature-version guard

// live_demo.py refuses to start when a model's feature tag is not one features.py can build,
// because train/serve skew is silent: the vector is the right length and the forest is
// confident. The page resolves every static forest's tag through STATIC_FEATURES (so the
// Segmenter builds the vector the forest was trained on, whichever registered tag it is) and
// refuses an unregistered one; both shipped forests are expected on static/v4, and any other
// registered tag is accepted with a note.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('app.js expects static/v4 letters, static/v4 digits, event/v1 motion',
    LETTERS_TAG === 'static/v4' && DIGITS_TAG === 'static/v4' && src.includes("'event/v1'"));
  check('both expected tags are in the registry',
    LETTERS_TAG in STATIC_FEATURES && DIGITS_TAG in STATIC_FEATURES);
  check('staticTagStatus: the expected tag is known and as expected',
    (() => { const s = staticTagStatus({ feature: 'static/v4' }, LETTERS_TAG); return s.known && s.asExpected; })());
  check('staticTagStatus: another registered tag is known but noted',
    (() => { const s = staticTagStatus({ feature: 'static/v3' }, LETTERS_TAG); return s.known && !s.asExpected; })());
  check('staticTagStatus: a null tag is the oldest export, static/v3, and is noted against a v4 expectation',
    (() => { const s = staticTagStatus({ feature: null }, DIGITS_TAG); return s.known && s.tag === 'static/v3' && !s.asExpected; })());
  check('staticTagStatus: an unregistered tag is refused',
    !staticTagStatus({ feature: 'static/v9' }, LETTERS_TAG).known);
  check('load() blocks on an unregistered tag and notes an unexpected one',
    /if \(!s\.known\) \{[\s\S]*?bad\.push/.test(src) && /else if \(!s\.asExpected\) \{[\s\S]*?odd\.push/.test(src)
    && /if \(bad\.length\) \{\s*problem\(/.test(src));
  check('no hard-coded STATIC_FEATURE_TAG remains', !src.includes('STATIC_FEATURE_TAG'));
  if (models.static.feature !== 'static/v4') {
    skip('the shipped models.json is static/v4', `it still carries ${models.static.feature}; regenerate with export_models.py after train_static.py`);
  } else {
    check('the shipped models.json carries the expected tags',
      models.static.feature === 'static/v4' && models.motion.feature === 'event/v1'
      && (!models.digits || models.digits.feature === 'static/v4'),
      `${models.static.feature} / ${models.motion.feature} / ${models.digits ? models.digits.feature : 'no digits'}`);
  }
  if (!models.digits) skip('the digits forest tag', 'models.json carries no digits block yet');
}

// -------------------------------------------------------------- numbers mode

// Letters on every load, never remembered; the gate line says the gates are off; the toggle is
// refused with a problem when the export carries no digits forest; the word layer is skipped.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  const html = readFileSync(join(HERE, 'index.html'), 'utf8');
  check('mode starts as letters on every load', /let mode = 'letters';/.test(src));
  check('...and is never persisted', !/localStorage[^\n]*mode/i.test(src) && !/mode[^\n]*localStorage/i.test(src));
  check('gateStatus says the gates are off in numbers mode',
    gateStatus(0.9, 0.9, null, th, false).tag === 'gates off (numbers)'
    && gateStatus(0.9, 0.0, null, th, true).tag === 'gate ready J');
  check('the toggle rebuilds the segmenter with the digit forest, no motion, gates off',
    /staticModel: digits \? models\.digits : models\.static/.test(src)
    && /motionModel: digits \? null : models\.motion/.test(src)
    && /armGates: !digits/.test(src)
    && /\{ \.\.\.th, \.\.\.\(models\.digitsThresholds \|\| \{\}\) \}/.test(src));
  check('the toggle is refused with a problem when there is no digits forest',
    /if \(next === 'digits' && !models\.digits\) \{\s*problem\(/.test(src));
  check('staticClasses follows the mode', /staticClasses = seg\.staticClasses;/.test(src));
  check('closeWord skips the dictionary in numbers mode but still writes the break',
    /if \(mode === 'digits'\) \{\s*ui\.word\.textContent = '';/.test(src)
    && /wordBuf = \[\];\s*letters\.push\(' '\);/.test(src));
  check('the ready line names the digit forest',
    /digits from the numbers forest's/.test(src));
  check('index.html has the mode control with aria-pressed',
    /<button id="mode"[^>]*aria-pressed="false"/.test(html));
  // Whitespace collapses when the page renders, so the source may wrap the sentence. The note
  // is the block-quoted paragraph of SPEC_digits_mode.md section 4, verbatim -- it carries the
  // only dataset name and license the served page shows -- with "(experimental)" after
  // "Numbers mode" and the shipped forest's idle rate appended as its own sentence: 0.162
  // single-frame over 2,890 logged hold records at the 0.40/0.60 gate, 57% '0' / 40% '1'
  // (temporal/thresholds.py, DIGITS_OVERRIDES; the static/v3 candidate's 'about a quarter'
  // never shipped).
  const text = html.replace(/&ndash;/g, '-').replace(/\s+/g, ' ');
  check('index.html carries the numbers-mode provenance, verbatim',
    text.includes('Numbers mode (experimental) reads the ASL digits 0-9 with a separate forest trained on '
      + '218 signers from a public dataset (Sign Language Digits Dataset, Ankara Ayranci Anadolu High '
      + 'School, Apache-2.0), not on the author. It has not been verified on the author\'s hand or on '
      + 'live video: 0.986 leave-signer-out on that dataset\'s photos, and 96% of the author\'s O, V, '
      + 'W, F and B frames read as 0, 2, 6, 9 and 4. J and Z are off in this mode, and a relaxed hand '
      + 'can read as 0 or 1.'));
  check('...and the shipped digit forest\'s idle rate, one in six',
    text.includes('a relaxed hand reads as a digit (mostly 0 or 1) about one time in six in this mode')
    && !/about a quarter/.test(text));
  check('the Numbers button ships disabled, is disabled at boot and on a blocking problem, and is '
        + 'enabled once the letters segmenter is built',
    /<button id="mode" type="button" disabled aria-pressed="false"/.test(html)
    && /if \(ui\.mode\) ui\.mode\.disabled = true;\s*\n\s*ui\.start\.disabled = true;/.test(src)
    && /if \(blocking\) \{\s*\n\s*ui\.start\.disabled = true;\s*\n\s*if \(ui\.mode\) ui\.mode\.disabled = true;/.test(src)
    && /paintMode\(\);\s*\n(\s*\/\/[^\n]*\n)*\s*if \(ui\.mode\) ui\.mode\.disabled = false;/.test(src)
    && /if \(next === mode \|\| !seg\) return;/.test(src));
}

// -------------------------------------------------------------- the audit's plumbing fixes

{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  const html = readFileSync(join(HERE, 'index.html'), 'utf8');
  check('Start ships disabled and is disabled again at boot',
    /<button id="start" disabled>/.test(html) && /ui\.start\.disabled = true;\s*\n\s*let th = null;/.test(src));
  check('Start is enabled only when load() completes', /ui\.start\.disabled = false;\s*window\.__aslReady = true;/.test(src));
  check('Stop closes the word and rebuilds the segmenter and the word clock',
    /function stopCamera\(\) \{[\s\S]*?closeWord\(\);[\s\S]*?buildSegmenter\(\);[\s\S]*?lastHandT = null;[\s\S]*?ui\.loadstate\.textContent = 'camera stopped';/.test(src));
  check('a camera track that ends on its own stops the page with a message',
    /addEventListener\('ended', \(\) => \{[\s\S]*?stopCamera\(\);[\s\S]*?problem\(/.test(src));
  // Named, not counted: models.json, golden.json (both halves -- fetch() resolves on the
  // headers and res.json() reads the body), the MediaPipe bundle, and the landmarker start,
  // which is where every MediaPipe download happens (the loader .js and the .wasm from the
  // CDN, the .task model from storage.googleapis.com); forVisionTasks does no I/O and is
  // deliberately not wrapped, so its old 'loading the MediaPipe WASM' label must be gone.
  check('a 20 s watchdog wraps every download, each labeled with what it fetches',
    /const LOAD_WATCHDOG_S = 20;/.test(src)
    && /await watchdog\(loadModels\('\.\/models\.bin\.gz', '\.\/models\.meta\.json'\),\s*\n\s*'fetching models\.bin\.gz and models\.meta\.json'\)/.test(src)
    && /await watchdog\(fetch\('\.\/golden\.json'\), 'fetching golden\.json'\)/.test(src)
    && /await watchdog\(res\.json\(\), 'fetching golden\.json'\)/.test(src)
    && /await watchdog\(import\(`\$\{MP_CDN\}\/vision_bundle\.mjs`\), 'loading MediaPipe from the CDN'\)/.test(src)
    && /await watchdog\(createLandmarker\(vision, fileset\),\s*\n\s*'fetching the MediaPipe WASM from cdn\.jsdelivr\.net or the hand '\s*\n\s*\+ 'landmarker model from storage\.googleapis\.com'\)/.test(src)
    && !/watchdog\(vision\.FilesetResolver/.test(src) && !/loading the MediaPipe WASM/.test(src)
    && (src.match(/await watchdog\(/g) || []).length === 5);
  // 1,183,438 B of models.bin.gz + 982 B of models.meta.json under Pages' own gzip = 1,184,420 B
  // on the wire, measured on the committed export; models.json would be 3,703,069 B at gzip -5.
  check('the load line states the real size of what the page fetches',
    /fetching models\.bin\.gz and models\.meta\.json \(about 1\.18 MB, '\s*\n\s*\+ 'already compressed, cached after the first time\)/.test(src)
    && /fetching golden\.json \(the self-check cases\)/.test(src));
  check('postLog drops keepalive above ~60 KB',
    /const KEEPALIVE_MAX_BYTES = 60 \* 1024;/.test(src)
    && /if \(body\.length <= KEEPALIVE_MAX_BYTES\) opts\.keepalive = true;/.test(src));
  check('logging is on only when devserver answers the probe',
    /let LOGGING = false;/.test(src) && /fetch\('\/log', \{ method: 'OPTIONS' \}\)/.test(src)
    && /LOGGING = r\.status === 204;/.test(src));
  check('the emissions placeholder is removed on the first emission',
    /if \(ui\.emlog\.childElementCount === 0\) ui\.emlog\.textContent = '';/.test(src));
  check('thresholds are read only after the Segmenter validated them',
    src.indexOf('buildSegmenter();\n    } catch (err) {') < src.indexOf('th.GATE_ARM_FRAC.toFixed(2)'));
  const css = html.replace(/\/\*[\s\S]*?\*\//g, '');           // the rules, not their comments
  check('index.html no longer crops the stream to 16:9',
    !/object-fit:\s*cover/.test(css) && !/aspect-ratio:\s*16\s*\/\s*9/.test(css)
    && /#view \{[^}]*height: auto/.test(css));
  check('video.play() rejection is handled', /await ui\.video\.play\(\);\s*\} catch \(err\) \{/.test(src));
}

// -------------------------------------------------------------- the warning that must not vanish

// A failed self-check says "do not trust the letters", and the moment the camera starts is
// exactly when that applies. Clearing the problem box on every camera start retracted it.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  // The box holds one message per kind (a Map), so a later camera or mode problem cannot
  // replace the self-check's; the only place that hides it is the paint that runs when the
  // map is empty, and the two retractions -- the camera starting, the mode toggle succeeding
  // -- each delete only their own kind, never the self-check's.
  const hides = src.match(/ui\.fatal\.hidden = [^;]*;/g) || [];
  check('the problem box hides only when no problem of any kind is left',
    hides.length === 1 && hides[0] === 'ui.fatal.hidden = problems.size === 0;', hides.join(' | '));
  check('problem() files its message under its kind and paints every open one',
    /function problem\(msg, blocking, kind = 'other'\) \{\s*\n\s*problems\.set\(kind, msg\);\s*\n\s*paintProblems\(\);/.test(src)
    && /ui\.fatalmsg\.textContent = \[\.\.\.problems\.values\(\)\]\.join\(' '\);/.test(src));
  check('...and only the camera and the mode toggle retract, each its own kind',
    /clearProblem\('camera'\);/.test(src) && /clearProblem\('mode'\);/.test(src)
    && !/clearProblem\('selfcheck'\)/.test(src)
    && (src.match(/clearProblem\('/g) || []).length === 2);
  check('the self-check failure is tagged so that guard can tell it apart',
    src.includes("false, 'selfcheck'"));
}

// -------------------------------------------------------------- end to end, at two frame rates

// What app.js actually does per frame, with the camera replaced by one still golden hand: a
// letter must be emitted, and it must be the letter Python predicts for that hand. Run at both
// 15 and 30 fps because the vote window is the exact place a frame-rate assumption hid before
// (VOTE_MIN in thresholds.py). The case is a solutions-API one, so its "Left" label is passed
// through as that API would report it; the Segmenter picks the feature by the model's own tag.
const firstStatic = golden.cases.findIndex((c) => (c.model || 'static') === 'static' && c.api !== 'tasks');
async function replay(caseIdx, fps) {
  const motion = prepareModel(models.motion);
  const seg = new Segmenter(th, { staticModel: prepared.static, motionModel: motion });
  const c = golden.cases[caseIdx];
  const dt = 1.0 / fps;
  const out = [];
  let state_seen = false;
  for (let i = 0; i < Math.round(3.0 * fps); i++) {
    // The same conversion app.js performs: a millisecond clock divided once into seconds.
    const t = (1000.0 * i * dt) / 1000.0;
    const em = seg.step(t, c.landmarks, c.handedness, c.width, c.height);
    if (seg.state === HOLD) state_seen = true;
    if (em) out.push(em.letter);
  }
  return { out, state_seen, expected: c.predicted };
}
for (const fps of [15, 30]) {
  const r = await replay(firstStatic, fps);
  check(`a still hand reaches HOLD at ${fps} fps`, r.state_seen);
  check(`one letter is emitted at ${fps} fps, and it is ${r.expected}`,
    r.out.length === 1 && r.out[0] === r.expected, `emitted [${r.out.join('')}]`);
}

// A hand that vanishes must clear the machine rather than leave a stale letter parked: this is
// the GAP_RESET path app.js exercises every time the hand leaves frame.
{
  const seg = new Segmenter(th, { staticModel: prepared.static });
  const c = golden.cases[firstStatic];
  const emitted = [];
  for (let i = 0; i < 90; i++) {
    const em = seg.step(i / 30, c.landmarks, c.handedness, c.width, c.height);
    if (em) emitted.push(em.letter);
  }
  for (let i = 90; i < 120; i++) seg.step(i / 30, null, null, c.width, c.height);
  for (let i = 120; i < 210; i++) {
    const em = seg.step(i / 30, c.landmarks, c.handedness, c.width, c.height);
    if (em) emitted.push(em.letter);
  }
  check('the same letter signed twice, with the hand dropped between, emits twice',
    emitted.length === 2 && emitted[0] === emitted[1], `emitted [${emitted.join('')}]`);
}

// -------------------------------------------------------------- page contract

// Every element app.js reaches for must exist, or the page throws on the first frame and the
// overlay -- the only instrument this system has -- is gone.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  const html = readFileSync(join(HERE, 'index.html'), 'utf8');
  const ids = [...src.matchAll(/el\('([a-zA-Z]+)'\)/g)].map((m) => m[1]);
  const missing = ids.filter((id) => !new RegExp(`id="${id}"`).test(html));
  check(`all ${ids.length} element ids exist in index.html`, missing.length === 0,
    missing.length ? `missing ${missing.join(', ')}` : '');
  // The accuracy numbers are the ones crossval_static.py / crossval_signers.py /
  // crossval_strangers.py print, with the hold-level interval, and the BY-SIGNER figure named
  // as the one a visitor should plan around rather than the author's own fold. A page that
  // rounds them up is the one thing the brief forbids outright, so every figure here is a
  // number one of those harnesses printed: pooled 0.926 (4,515/4,878) and cross-day 0.895 at
  // seed 0, bootstrap interval [0.877, 0.965] over the 57 session x letter bursts.
  check('index.html states the measured accuracies with the interval',
    /0\.926 of 4,878 held-out frames/.test(html) && /0\.877(&ndash;|-)0\.965/.test(html)
    && /0\.895 on the one fold recorded on a different day/.test(html)
    && /other people's hands from\s+three public/.test(html));
  // The three-seed spread beside the published figure: crossval_static.py seeds 0-2 on the
  // shipped recipe give 0.8949 / 0.8848 / 0.8894 on the cross-day fold, and the J/Z figure
  // counts events, not gestures: 102 events = 74 J/Z gestures + 28 movements over 80 items.
  check('...with the three-seed spread beside the cross-day figure and the J/Z events itemized',
    /\(0\.8848(&ndash;|-)0\.8949 over three seeds\)/.test(html)
    && /J and Z: 0\.95 over 102 events \(74 J\/Z gestures and 28 movements\) in 80 prompted items/.test(html));
  // What a visitor should plan around is now MEASURED BY SIGNER, not bounded: crossval_signers.py
  // holds out one of ten named signers at a time (0.9406 over three seeds, 0.9379-0.9453), and
  // crossval_strangers.py holds out a whole second set (0.8292 over three seeds, 0.8212-0.8362)
  // and scores the permanent never-train holdout (0.889). The page must lead with those and say
  // plainly that the author's own folds are not a visitor's number.
  check('index.html leads with the by-signer numbers, not the author\'s own fold',
    /If you are not the author, the numbers to plan around are\s+the ones measured by holding out other people/.test(html)
    && /ten named signers: 0\.9406 across three seeds \(0\.9379(&ndash;|-)0\.9453\)/.test(html)
    && /1,874\s+frames from multiple participants captured with this page's own landmarker/.test(html)
    && /0\.8292\s+across three seeds \(0\.8212(&ndash;|-)0\.8362\)/.test(html)
    && /five signers nothing here is ever\s+trained on, reads 0\.889/.test(html)
    && /The author's own folds are higher and they are not yours/.test(html));
  // The runtime row, from replay_strangers.py over temporal/openhands_replay.json: 266 clips,
  // 96 exact, 170 silent, ZERO wrong letters. The zero is the claim worth pinning.
  check('...and states the end-to-end video result with its zero wrong letters',
    /266 video clips of other people fingerspelling gave the right letter on 96, the wrong\s+letter on none, and silence on the other 170/.test(html));
  // The weak letters are the ones crossval_signers.py (U 0.671 as R, R 0.703 as U, C 0.816,
  // S 0.814) and crossval_static.py (cross-day fold: M 0.12) print.
  check('index.html names the weak letters on both folds',
    /the author's M is read on about one frame in eight/.test(html)
    && /on other people's hands U and R are read as each\s+other, with C, S and G next least reliable/.test(html));
  check('...labels the previous release\'s numbers as previous',
    /previous\s+release measured 0\.910 and 0\.870 on the author's folds and had no by-signer number at all/.test(html)
    && /replaces rather than improves on the earlier 0\.864/.test(html));
  check('index.html says fingerspelling only', /no ASL word signs, no grammar/i.test(html));
  check('index.html explains the word break, common words first',
    /drop your hand for a second/i.test(html) && /common words first, from a frequency list/.test(html));
  check('the lede claims numbers mode only as experimental',
    /26 letters, plus an experimental numbers mode/.test(html));
  check('American spelling', !/centre|colour|recognise|behaviour|analyse/.test(html));
}

// The loaded-through-fetch path forest.js provides is what the page calls; confirm the shape it
// returns is the shape app.js consumes (thresholds object, prepared models with classes/trees).
{
  const p = predictProba(prepared.static, golden.cases[firstStatic].static_feature);
  const appSrc = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('predictProba returns one probability per class', p.length === models.static.classes.length);
  // The page fetches the binary pair, not the 20,361,559 B models.json that this file still
  // reads off disk as the readable reference. forest.js decides by the bytes, not the name, and
  // returns the identical object either way -- test_forest.mjs is what proves that, node by node.
  check('loadModels fetches the binary pair, and is the only model fetch in app.js',
    /loadModels\('\.\/models\.bin\.gz', '\.\/models\.meta\.json'\)/.test(appSrc)
    && !/loadModels\('\.\/models\.json'\)/.test(appSrc)
    && (appSrc.match(/loadModels\(/g) || []).length === 1);
  check('thresholds carry the constants the overlay draws',
    [th.V_STILL, th.V_MOVE_ARMED, th.V_MOVE_UNARMED, th.SHAPE_STABLE, th.RIGID_VETO,
      th.GATE_ARM_FRAC, th.T_MAX, th.L_MIN, th.L_MAX].every(Number.isFinite));
}

// -------------------------------------------------------------- the booted page, no camera
//
// The problem-box and Numbers-button contracts above are regex over the source. This boots
// app.js itself against a DOM just wide enough for boot() -- no browser, no camera:
// getUserMedia is a stub each scenario steers, and the MediaPipe CDN import is routed to a
// stub through an in-process module hook -- and drives the two sequences the source checks
// promise. With one message slot the first sequence lost "do not trust the letters" the
// moment the camera came on; with the button live before load, the second threw out of the
// click handler on every second click.
{
  const { register } = await import('node:module');
  const stubUrl = 'data:text/javascript,' + encodeURIComponent(
    'export const FilesetResolver = { forVisionTasks: async () => ({}) };'
    + 'export const HandLandmarker = { createFromOptions: async () => ({'
    + '  detectForVideo: () => ({ landmarks: [], handednesses: [] }) }) };');
  register('data:text/javascript,' + encodeURIComponent(
    `const STUB = ${JSON.stringify(stubUrl)};`
    + 'export async function resolve(spec, ctx, next) {'
    + "  if (spec.startsWith('https://cdn.jsdelivr.net/')) return { url: STUB, shortCircuit: true };"
    + '  return next(spec, ctx); }'));

  class El {
    constructor(id = '') {
      this.id = id; this.children = []; this.attrs = {}; this.listeners = {}; this.style = {};
      this.className = ''; this.hidden = false; this.disabled = false; this.title = '';
    }
    get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
    set textContent(v) { this.children = v === '' ? [] : [String(v)]; }
    get childElementCount() { return this.children.filter((c) => c instanceof El).length; }
    get firstElementChild() { return this.children.find((c) => c instanceof El) || null; }
    get lastElementChild() { const e = this.children.filter((c) => c instanceof El); return e.length ? e[e.length - 1] : null; }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
    addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
    click() { for (const fn of this.listeners.click || []) fn({ type: 'click' }); }
    appendChild(el) { el.parent = this; this.children.push(el); }
    prepend(el) { el.parent = this; this.children.unshift(el); }
    remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }
    replaceChildren() { this.children = []; }
    focus() {}
    getContext() { return new Proxy({}, { get: () => () => {} }); }
  }
  const IDS = [...readFileSync(join(HERE, 'app.js'), 'utf8').matchAll(/el\('([a-zA-Z]+)'\)/g)].map((m) => m[1]);
  const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));
  let boots = 0;
  /** Install the globals, import a fresh app.js (a query string defeats the module cache) and
   *  wait for the page to reach ready or stop on a blocking problem. */
  async function bootPage({ golden: mapGolden = null, models: mapModels = null } = {}) {
    const els = {};
    for (const id of IDS) els[id] = new El(id);
    els.start.disabled = true; els.mode.disabled = true; els.fatal.hidden = true; els.stop.hidden = true;
    els.mode.setAttribute('aria-pressed', 'false');
    els.vmeter.appendChild(new El()); els.smeter.appendChild(new El());
    Object.assign(els.cam, { readyState: 0, videoWidth: 0, videoHeight: 0, currentTime: 0, srcObject: null,
      play: async () => { els.cam.readyState = 4; els.cam.videoWidth = 640; els.cam.videoHeight = 480; } });
    globalThis.document = { getElementById: (id) => els[id] || null, createElement: () => new El() };
    globalThis.window = { isSecureContext: true };
    globalThis.location = { hostname: 'example.test' };   // not localhost: no /log probe
    globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(performance.now()), 5);
    globalThis.fetch = async (url) => {
      const name = String(url).replace(/^\.\//, '');
      let buf = readFileSync(join(HERE, name));
      // The page loads models.bin.gz + models.meta.json, so a scenario that bends the export
      // bends the META file: the class lists, feature tags and thresholds all live there now,
      // and the .bin beside it is only the trees. readFileSync hands back the gzip bytes and
      // forest.js gunzips them through the platform's DecompressionStream, exactly as a browser
      // does with a Pages-served .gz.
      if (name === 'models.meta.json' && mapModels) buf = Buffer.from(JSON.stringify(mapModels(JSON.parse(buf.toString()))));
      if (name === 'golden.json' && mapGolden) buf = Buffer.from(JSON.stringify(mapGolden(JSON.parse(buf.toString()))));
      return { ok: true, status: 200, statusText: 'OK',
        arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
        json: async () => JSON.parse(buf.toString()), text: async () => buf.toString() };
    };
    const cam = { mode: 'ok' };
    globalThis.navigator = { mediaDevices: { getUserMedia: async () => {
      if (cam.mode !== 'ok') { const e = new Error('stub'); e.name = cam.mode; throw e; }
      const track = new El(); track.stop = () => {};
      return { getVideoTracks: () => [track], getTracks: () => [track] };
    } } };
    boots += 1;
    await import(`./app.js?boot=${boots}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 10000 && !globalThis.window.__aslReady
           && !els.loadstate.textContent.startsWith('stopped')) await tick();
    return { els, cam };
  }
  const box = (els) => `${els.fatal.hidden ? 'hidden' : 'shown'}: ${els.fatalmsg.textContent.slice(0, 60)}`;
  const warning = (els) => !els.fatal.hidden && els.fatalmsg.textContent.includes('do not trust the letters');

  // (a) a failed self-check (one stored probability moved by 0.5), then a camera refusal, then
  // a camera grant: the warning must survive both.
  {
    const wrongProb = (g) => ({ ...g, cases: g.cases.map((c, i) => (i === 0
      ? { ...c, static_probs: c.static_probs.map((p, j) => (j === 0 ? p + 0.5 : p)) } : c)) });
    const { els, cam } = await bootPage({ golden: wrongProb });
    check('booted: a golden mismatch is reported and Start is enabled',
      globalThis.window.__aslReady && !els.start.disabled && els.selfcheck.textContent.startsWith('FAILED')
      && warning(els), box(els));
    check('booted: the Numbers button is enabled once the page is ready', !els.mode.disabled);
    cam.mode = 'NotAllowedError';
    els.start.click(); await tick(60);
    check('booted: a camera refusal is shown WITH the self-check warning, not instead of it',
      els.loadstate.textContent === 'camera not started' && warning(els)
      && els.fatalmsg.textContent.includes('Camera access was refused'), box(els));
    cam.mode = 'ok';
    els.start.click(); await tick(60);
    check('booted: the camera starting retracts its own problem and keeps the warning',
      els.loadstate.textContent.startsWith('running') && warning(els)
      && !els.fatalmsg.textContent.includes('Camera access was refused'), box(els));
    els.stop.click(); await tick(30);
  }

  // (b) a page blocked before the thresholds are validated (an unregistered letter tag): the
  // Numbers button is disabled, two clicks throw nothing and change nothing, and the blocking
  // diagnostic is not overwritten.
  {
    const badTag = (m) => ({ ...m, static: { ...m.static, feature: 'static/v9' } });
    const { els } = await bootPage({ models: badTag });
    const before = els.fatalmsg.textContent;
    check('booted: an unregistered tag blocks the page with Start and Numbers disabled',
      els.loadstate.textContent.startsWith('stopped') && els.start.disabled && els.mode.disabled
      && before.includes('static/v9'), box(els));
    let threw = null;
    try { els.mode.click(); els.mode.click(); } catch (e) { threw = e; }
    check('booted: two Numbers clicks on a blocked page throw nothing and leave mode alone',
      threw === null && els.mode.getAttribute('aria-pressed') === 'false' && els.fatalmsg.textContent === before,
      threw ? threw.message : box(els));
  }

  // (c) an export without a digits forest: the refusal is filed under its own kind and a
  // later camera start does not clear it.
  {
    const noDigits = (m) => { const { digits, ...rest } = m; return rest; };
    const { els, cam } = await bootPage({ models: noDigits });
    els.mode.click(); await tick(30);
    check('booted: the toggle is refused on an export without a digits forest',
      els.mode.getAttribute('aria-pressed') === 'false'
      && !els.fatal.hidden && els.fatalmsg.textContent.includes('no numbers forest'), box(els));
    cam.mode = 'ok';
    els.start.click(); await tick(60);
    check('booted: the camera starting leaves the mode refusal in place',
      els.loadstate.textContent.startsWith('running') && els.fatalmsg.textContent.includes('no numbers forest'), box(els));
    els.stop.click(); await tick(30);
  }
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);

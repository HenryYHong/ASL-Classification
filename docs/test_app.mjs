// Tests for web/app.js -- the page's own logic, plus the contracts it depends on.
//
// app.js is mostly plumbing, and plumbing is where this project's expensive bugs lived: a
// mirrored frame that inverted the handedness label, a vote window that completed at 30 fps and
// never at 15. So the things tested here are exactly those: the mirror, the clock, and the
// numbers the overlay claims to be reporting. Reading the code is not evidence.
//
// Run:  /usr/local/bin/node web/test_app.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { toDisplayPx, fpsFrom, gateStatus, checkGolden, trackPathLength } from './app.js';
import { loadModels, prepareModel, predictProba } from './forest.js';
import { Segmenter, HOLD } from './segmenter.js';
import { TIP_FOR_ARM } from './features.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const models = JSON.parse(readFileSync(join(HERE, 'models.json'), 'utf8'));
const golden = JSON.parse(readFileSync(join(HERE, 'golden.json'), 'utf8'));
const th = models.thresholds;

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
  const model = prepareModel(models.static);
  const g = checkGolden(model, golden);
  check(`checkGolden passes all ${g.n} golden cases`, g.ok && g.n === golden.cases.length,
    `worst ${g.worst.toExponential(2)} against tol ${g.tol}`);

  // ...and it must FAIL when the forest is wrong, or it is checking nothing. Corrupting a
  // single leaf is not enough of a test: 15 cases need not visit it, and a lone tree moves the
  // mean by at most 1/400. Corrupt every leaf of one tree, which every case does visit.
  const broken = prepareModel(models.static);
  for (let i = 0; i < broken.trees[0].leaf.length; i += broken.nClasses) {
    broken.trees[0].leaf[i] += 1.0;
  }
  const b = checkGolden(broken, golden);
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
  const p = checkGolden(permuted, golden);
  check('checkGolden fails when the feature columns are permuted', !p.ok && p.wrong > 0,
    `whole path ${p.worstEnd.toExponential(2)}, ${p.wrong} of ${p.n} letters wrong`);
}

// -------------------------------------------------------------- the feature-version guard

// live_demo.py refuses to start when a model's feature tag is not the transform the code
// builds, because train/serve skew is silent: the vector is the right length and the forest is
// confident. The page must carry the same refusal, and must name both tags.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  check('app.js knows which feature transform features.js builds',
    src.includes("'static/v3'") && src.includes("'event/v1'"));
  check('...and refuses a models.json exported from a different one',
    /m\.feature != null && m\.feature !== want/.test(src));
  check('the shipped models.json is the transform this page builds',
    models.static.feature === 'static/v3' && models.motion.feature === 'event/v1',
    `${models.static.feature} / ${models.motion.feature}`);
}

// -------------------------------------------------------------- the warning that must not vanish

// A failed self-check says "do not trust the letters", and the moment the camera starts is
// exactly when that applies. Clearing the problem box on every camera start retracted it.
{
  const src = readFileSync(join(HERE, 'app.js'), 'utf8');
  const hides = src.match(/ui\.fatal\.hidden = true/g) || [];
  check('exactly one place hides the problem box', hides.length === 1, `${hides.length}`);
  check('...and only for a problem the camera itself raised',
    /if \(fatalKind === 'camera'\) \{\s*\n\s*ui\.fatal\.hidden = true;/.test(src));
  check('the self-check failure is tagged so that guard can tell it apart',
    src.includes("false, 'selfcheck'"));
}

// -------------------------------------------------------------- end to end, at two frame rates

// What app.js actually does per frame, with the camera replaced by one still golden hand: a
// letter must be emitted, and it must be the letter Python predicts for that hand. Run at both
// 15 and 30 fps because the vote window is the exact place a frame-rate assumption hid before
// (VOTE_MIN in thresholds.py). Nothing here mirrors anything, so the "Left" label in the golden
// case is passed through as MediaPipe would report it.
async function replay(caseIdx, fps) {
  const prepared = { static: prepareModel(models.static), motion: prepareModel(models.motion) };
  const seg = new Segmenter(th, { staticModel: prepared.static, motionModel: prepared.motion });
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
  const r = await replay(0, fps);
  check(`a still hand reaches HOLD at ${fps} fps`, r.state_seen);
  check(`one letter is emitted at ${fps} fps, and it is ${r.expected}`,
    r.out.length === 1 && r.out[0] === r.expected, `emitted [${r.out.join('')}]`);
}

// A hand that vanishes must clear the machine rather than leave a stale letter parked: this is
// the GAP_RESET path app.js exercises every time the hand leaves frame.
{
  const prepared = prepareModel(models.static);
  const seg = new Segmenter(th, { staticModel: prepared });
  const c = golden.cases[0];
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
  // The accuracy numbers are the ones the README measured. A page that rounds them up is the
  // one thing the brief forbids outright.
  check('index.html states the measured accuracies',
    html.includes('0.769') && html.includes('0.889') && /one signer/i.test(html));
  check('index.html says fingerspelling only', /no words, no grammar/i.test(html));
}

// The loaded-through-fetch path forest.js provides is what the page calls; confirm the shape it
// returns is the shape app.js consumes (thresholds object, prepared models with classes/trees).
{
  const prepared = prepareModel(models.static);
  const p = predictProba(prepared, golden.cases[0].static_feature);
  check('predictProba returns one probability per class', p.length === models.static.classes.length);
  check('loadModels is the only fetch of models.json in app.js',
    /loadModels\('\.\/models\.json'\)/.test(readFileSync(join(HERE, 'app.js'), 'utf8')));
  check('thresholds carry the constants the overlay draws',
    [th.V_STILL, th.V_MOVE_ARMED, th.V_MOVE_UNARMED, th.SHAPE_STABLE, th.RIGID_VETO,
      th.GATE_ARM_FRAC, th.T_MAX, th.L_MIN, th.L_MAX].every(Number.isFinite));
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);

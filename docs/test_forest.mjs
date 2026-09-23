// Golden-vector check for forest.js.  Run:  node docs/test_forest.mjs
//
// Reading the port next to the Python is not evidence. Every case in golden.json carries the
// exact numbers Python produced, so this compares against those and prints the worst error it
// saw -- a port that is merely close is a port that is wrong somewhere. Each case names its
// forest (`model`: "static" by default, "digits" for the numbers forest); the letter forest is
// expected at 112-D static/v4 with 24 classes, the digit forest at 112-D static/v4 with '0'..'9'.

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

import {
  prepareModel, prepareModelBin, prepareModelsBin, loadModels, predictProba, argmaxWithMargin,
} from './forest.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// The probe set docs/parity_probes.py builds and docs/README.md quotes, pinned on this side so
// that a forest which changes without a re-measure fails here instead of inheriting a figure
// that is no longer about it. 26 golden letter features + all 325 of their pairwise midpoints
// + 864 noisy copies at the four sigmas = 1,215.
const PROBE_TOTAL = 1215;
const PROBE_BASES = 26;
const PROBE_MIDPOINTS = 325;
const PROBE_NOISY = 864;
const PROBE_SIGMAS = [1e-4, 1e-3, 1e-2, 5e-2];

// How far each export's probabilities may sit from sklearn's own predict_proba over that set.
// models.json rounds every leaf to 4 decimals, which bounds it at 5e-5 by the rounding alone
// (6.25e-6 letters / 5.07e-6 digits measured, 0 argmax changes). models.bin stores the leaf's
// exact integer class counts, so nothing is quantized but the Float32Array the page has always
// held the table in, and it measures three orders better: 4.05e-9 letters / 4.75e-9 digits.
// The 1e-8 ceiling is what says that gap is real rather than remembered.
const PARITY_JSON_TOL = 5e-5;
const PARITY_BIN_TOL = 1e-8;

const payload = JSON.parse(fs.readFileSync(path.join(HERE, 'models.json'), 'utf8'));
const golden = JSON.parse(fs.readFileSync(path.join(HERE, 'golden.json'), 'utf8'));
const TOL = golden.tolerance.probabilities;

const staticModel = prepareModel(payload.static);
const digitsModel = payload.digits ? prepareModel(payload.digits) : null;
console.log(`static forest: ${staticModel.trees.length} trees, dim ${staticModel.dim}, ` +
            `${staticModel.nClasses} classes, feature ${payload.static.feature}`);
console.log(digitsModel
  ? `digits forest: ${digitsModel.trees.length} trees, dim ${digitsModel.dim}, ` +
    `${digitsModel.nClasses} classes, feature ${payload.digits.feature}`
  : 'digits forest: none in models.json');
console.log(`tolerance (probabilities): ${TOL}\n`);

let failures = 0;
let worstErr = 0;
let worstCase = '';

// --- the shape of what shipped ---------------------------------------------------------------
//
// The letter forest ships on static/v4: 112 features, 24 classes (A-Y without J and Z). Until
// the export is regenerated from the v4 pickle the file is the previous 101-D forest, and that
// is reported as a skip rather than a failure, because everything below still checks the walk.
if (payload.static.feature === 'static/v4') {
  const ok = staticModel.dim === 112 && staticModel.nClasses === 24
    && staticModel.classes.join('') === 'ABCDEFGHIKLMNOPQRSTUVWXY';
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  letter forest is static/v4: dim ${staticModel.dim} (112), ` +
              `${staticModel.nClasses} classes (24)`);
} else {
  console.log(`skip  letter forest dim 112 / 24 classes: models.json still carries ` +
              `${payload.static.feature} (${staticModel.dim}-D); regenerate with export_models.py`);
}
if (digitsModel) {
  const dth = payload.digits.thresholds || {};
  const ok = digitsModel.dim === 112 && payload.digits.feature === 'static/v4'
    && digitsModel.classes.join('') === '0123456789';
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  digit forest is static/v4: dim ${digitsModel.dim} (112), ` +
              `classes ${digitsModel.classes.join('')}`);
  // Numbers mode runs its own measured gate (thresholds.py DIGITS_OVERRIDES): a wider margin
  // and its own floor and confident route. The letters' floor has since moved above the
  // digits' (0.75 against 0.60, idle_gate.py), so this pins the digits block's own values
  // rather than an ordering between the two modes.
  const thOk = dth.VOTE_MARGIN_CLEAR === 0.40 && dth.VOTE_PROB_FLOOR === 0.60 && dth.VOTE_PROB === 0.70
    && dth.VOTE_MARGIN_CLEAR > payload.thresholds.VOTE_MARGIN_CLEAR
    && Object.keys(dth).length === Object.keys(payload.thresholds).length;
  if (!thOk) failures++;
  console.log(`${thOk ? 'pass' : 'FAIL'}  digits thresholds carry their measured gate: ` +
              `VOTE_MARGIN_CLEAR ${dth.VOTE_MARGIN_CLEAR}, VOTE_PROB_FLOOR ${dth.VOTE_PROB_FLOOR}, VOTE_PROB ${dth.VOTE_PROB} ` +
              `(letters ${payload.thresholds.VOTE_MARGIN_CLEAR} / ${payload.thresholds.VOTE_PROB_FLOOR} / ${payload.thresholds.VOTE_PROB})`);
}
console.log('');

for (let i = 0; i < golden.cases.length; i++) {
  const c = golden.cases[i];
  const which = c.model || 'static';
  const model = which === 'digits' ? digitsModel : staticModel;
  if (!model) {
    failures++;
    console.log(`FAIL  case ${String(i).padStart(2)}  needs the ${which} forest, absent from models.json`);
    continue;
  }
  const probs = predictProba(model, Float64Array.from(c.static_feature));

  let err = 0;
  for (let j = 0; j < probs.length; j++) {
    err = Math.max(err, Math.abs(probs[j] - c.static_probs[j]));
  }
  if (err >= worstErr) { worstErr = err; worstCase = `case ${i} (${c.predicted})`; }

  const { index, p, margin } = argmaxWithMargin(probs);
  const letter = model.classes[index];

  const probsOk = probs.length === c.static_probs.length && err <= TOL;
  const letterOk = letter === c.predicted;
  const ok = probsOk && letterOk;
  if (!ok) failures++;

  console.log(`${ok ? 'pass' : 'FAIL'}  case ${String(i).padStart(2)}  ${which.padEnd(6)} ` +
              `expect ${c.predicted}  got ${letter}  ` +
              `p=${p.toFixed(4)}  margin=${margin.toFixed(4)}  ` +
              `max|dp|=${err.toExponential(2)}`);
  if (!probsOk) console.log(`      probabilities differ by ${err.toExponential(3)} > ${TOL}`);
  if (!letterOk) console.log(`      argmax letter ${letter} != golden ${c.predicted}`);
}

console.log(`\nworst probability error: ${worstErr.toExponential(3)} at ${worstCase}`);

// --- averaging ---------------------------------------------------------------------------
//
// sklearn averages the leaf *distributions*; hard voting on per-tree argmaxes agrees with it on
// unanimous inputs and disagrees on contested ones. The golden set now carries contested cases
// on purpose (the Tasks-API A frame reads at p 0.43, three P frames at 0.54-0.60), and a
// hard-vote mutant misses golden.json's static_probs by up to 0.068 on 17 of the 41 cases, so
// the 1e-4 golden comparison above pins the averaging rule; the two-tree fixture under the
// guards below pins it again without any data file. (The single-tree threshold fixture cannot:
// the mean of one leaf is its own argmax.)

// --- guards on the pieces the golden cases cannot exercise ---------------------------------

// A wrong-length feature must throw rather than answer plausible nonsense.
try {
  predictProba(staticModel, new Float64Array(staticModel.dim - 1));
  console.log('FAIL  short feature vector was accepted');
  failures++;
} catch (e) {
  console.log(`pass  short feature vector rejected: ${e.message}`);
}

// So must a non-finite one: `NaN <= t` is false at every split, so a NaN feature walks right
// all the way down and the forest answers a confident letter from no information. sklearn
// refuses NaN; the walk must too, and it used to accept it silently.
for (const bad of [NaN, Infinity, -Infinity]) {
  const x = Float64Array.from(golden.cases[0].static_feature);
  if (x.length !== staticModel.dim) break;             // a case from another export; skip
  x[7] = bad;
  try {
    predictProba(staticModel, x);
    console.log(`FAIL  a feature vector with ${bad} was accepted`);
    failures++;
  } catch (e) {
    console.log(`pass  ${String(bad).padEnd(9)} feature rejected: ${e.message}`);
  }
}

// np.argmax tie-breaking: lowest index wins, and the margin is then 0.
{
  const { index, margin } = argmaxWithMargin(Float64Array.from([0.25, 0.5, 0.5, 0]));
  const ok = index === 1 && margin === 0;
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  tie goes to the lower index (index=${index}, ` +
              `margin=${margin})`);
}

// The boundary of sklearn's rule: a value sitting EXACTLY on a split threshold goes LEFT.
// Neither golden.json nor the ambiguous blends land on a threshold, so `<` and `<=` are
// indistinguishable there and the half of the rule that decides ties went untested. A synthetic
// two-leaf tree pins it without needing a fixture. (Checked against the real forests too: feeding
// models.json's own thresholds back in as feature values, a `<` walk moves 18 of 120 such inputs
// to a different leaf, by up to 0.081 probability.)
{
  const spec = {
    classes: ['left', 'right'], feature: 'synthetic', dim: 1,
    trees: [{ f: [0, -2, -2], t: [0.5, -2, -2], l: [1, -1, -1], r: [2, -1, -1],
              v: { 1: [1, 0], 2: [0, 1] } }],
  };
  const m = prepareModel(spec);
  const onIt = m.classes[argmaxWithMargin(predictProba(m, [0.5])).index];
  const above = m.classes[argmaxWithMargin(predictProba(m, [0.5 + 1e-12])).index];
  const ok = onIt === 'left' && above === 'right';
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  x == threshold goes left (x=t -> ${onIt}, ` +
              `x=t+eps -> ${above})`);
}

// Averaging without a data file: two trees whose leaves disagree at the probed input. Tree 1
// says [0.6, 0.4], tree 2 says [0.3, 0.7]; sklearn's mean is [0.45, 0.55] -> 'right', while a
// hard vote counts one argmax each, [0.5, 0.5], and the lowest-index tie-break says 'left'. The
// averaged VECTOR is asserted, not only the winner, so a mutant that votes cannot pass.
{
  const tree = (leftLeaf) => ({ f: [0, -2, -2], t: [0.5, -2, -2], l: [1, -1, -1], r: [2, -1, -1],
                                v: { 1: leftLeaf, 2: [0, 1] } });
  const m = prepareModel({ classes: ['left', 'right'], feature: 'synthetic', dim: 1,
                           trees: [tree([0.6, 0.4]), tree([0.3, 0.7])] });
  const probs = predictProba(m, [0]);
  const { index } = argmaxWithMargin(probs);
  const ok = Math.abs(probs[0] - 0.45) < 1e-6 && Math.abs(probs[1] - 0.55) < 1e-6
    && m.classes[index] === 'right';
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  forest averages leaf distributions, not per-tree argmaxes ` +
              `([${Array.from(probs).map((v) => v.toFixed(3)).join(', ')}] -> ${m.classes[index]}, ` +
              'expected [0.450, 0.550] -> right)');
}

// loadModels over HTTP, with the .gz served the way GitHub Pages serves it: as an opaque body,
// no Content-Encoding, so the gzip bytes reach us undecoded and DecompressionStream must run.
// models.json.gz is gitignored (export_models.py writes it beside models.json), so a fresh
// clone has none: say so and move on rather than crash before the summary line.
{
  const gzPath = path.join(HERE, 'models.json.gz');
  if (!fs.existsSync(gzPath)) {
    console.log('skip  models.json.gz absent (gitignored; export_models.py writes it), raw-gzip ' +
                'loadModels check not run');
  } else {
    const gz = fs.readFileSync(gzPath);
    const server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      res.end(gz);
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    const url = `http://127.0.0.1:${server.address().port}/models.json.gz`;
    try {
      const models = await loadModels(url);
      const c = golden.cases.find((k) => (k.model || 'static') === 'static');
      const probs = predictProba(models.static, Float64Array.from(c.static_feature));
      const letter = models.static.classes[argmaxWithMargin(probs).index];
      // The digits block is optional in the payload and must come through as null when absent
      // and as a prepared model plus its override thresholds when present.
      const digitsOk = payload.digits
        ? (models.digits !== null && models.digits.dim === payload.digits.dim
           && models.digitsThresholds !== null && typeof models.digitsThresholds === 'object')
        : (models.digits === null && models.digitsThresholds === null);
      const ok = letter === c.predicted && models.motion.classes.join(',') === 'J,MOVE,Z' &&
                 typeof models.thresholds.VOTE_WINDOW === 'number' && digitsOk;
      if (!ok) failures++;
      console.log(`${ok ? 'pass' : 'FAIL'}  loadModels decompressed a raw .gz body ` +
                  `(-> ${letter}, motion classes ${models.motion.classes.join('/')}, ` +
                  `VOTE_WINDOW=${models.thresholds.VOTE_WINDOW}, digits ` +
                  `${models.digits ? `${models.digits.nClasses} classes` : 'null'})`);
    } finally {
      server.close();
    }
  }
}

// loadModels over HTTP with the PLAIN models.json, which nothing on the page fetches any more:
// app.js asks for models.bin.gz + models.meta.json, and models.json stays committed as the
// readable reference. One walker, two containers -- and the container no page exercises is the
// one that rots, so it is exercised here, unconditionally (models.json is committed, unlike its
// .gz above). The same forests must come back through it: same letter, same motion classes,
// same thresholds object, same digits block.
{
  const raw = fs.readFileSync(path.join(HERE, 'models.json'));
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(raw);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  try {
    const models = await loadModels(`http://127.0.0.1:${server.address().port}/models.json`);
    const c = golden.cases.find((k) => (k.model || 'static') === 'static');
    const probs = predictProba(models.static, Float64Array.from(c.static_feature));
    const letter = models.static.classes[argmaxWithMargin(probs).index];
    const digitsOk = payload.digits
      ? (models.digits !== null && models.digits.dim === payload.digits.dim)
      : models.digits === null;
    const ok = letter === c.predicted && models.motion.classes.join(',') === 'J,MOVE,Z'
               && typeof models.thresholds.VOTE_WINDOW === 'number' && digitsOk
               && models.static.trees.length === payload.static.trees.length;
    if (!ok) failures++;
    console.log(`${ok ? 'pass' : 'FAIL'}  loadModels still reads the plain models.json the page ` +
                `no longer fetches (-> ${letter}, ${models.static.trees.length} letter trees, ` +
                `motion ${models.motion.classes.join('/')}, digits ` +
                `${models.digits ? `${models.digits.nClasses} classes` : 'null'})`);
  } finally {
    server.close();
  }
}

// --- the binary sidecar ----------------------------------------------------------------------
//
// models.bin carries the same three forests as models.json. The whole point of it is that the
// page cannot tell which one it loaded, so the first check below is not a tolerance at all: the
// decoded f, t, l and r must be equal element by element on all 278,750 nodes. Only the leaf
// table is allowed to differ, and only in the direction of being MORE exact -- models.json
// rounds a leaf probability to 4 decimals, models.bin stores the leaf's integer class counts
// and the probability is recovered by dividing.
{
  const gzPath = path.join(HERE, 'models.bin.gz');
  const metaPath = path.join(HERE, 'models.meta.json');
  if (!fs.existsSync(gzPath) || !fs.existsSync(metaPath)) {
    failures++;
    console.log('FAIL  models.bin.gz / models.meta.json missing; both are committed and '
                + 'docs/export_models.py writes them. Re-export');
  } else {
    const nb = zlib.gunzipSync(fs.readFileSync(gzPath));
    const binBuf = nb.buffer.slice(nb.byteOffset, nb.byteOffset + nb.byteLength);
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));

    // 1. Routing: the arrays that decide WHERE a walk lands, node by node.
    let nodes = 0, routingBad = 0, worstLeafGap = 0, leafCells = 0;
    for (const which of ['static', 'motion', 'digits']) {
      if (!payload[which] || !meta[which]) continue;
      const A = prepareModel(payload[which]);
      const B = prepareModelBin(meta[which], binBuf);
      const shapeOk = A.trees.length === B.trees.length && A.dim === B.dim
        && A.nClasses === B.nClasses && A.classes.join(',') === B.classes.join(',')
        && A.feature === B.feature;
      if (!shapeOk) { routingBad++; continue; }
      for (let k = 0; k < A.trees.length; k++) {
        const a = A.trees[k], b = B.trees[k];
        if (a.f.length !== b.f.length || a.leaf.length !== b.leaf.length) { routingBad++; continue; }
        nodes += a.f.length;
        for (let i = 0; i < a.f.length; i++) {
          // Object.is rather than !==, so a NaN threshold would compare equal to a NaN
          // threshold and a -0 would not compare equal to a 0. Neither occurs in these
          // forests; a NaN threshold is exactly the value that would ruin a walk in silence.
          if (a.f[i] !== b.f[i] || !Object.is(a.t[i], b.t[i]) || a.l[i] !== b.l[i]
              || a.r[i] !== b.r[i] || a.leafOff[i] !== b.leafOff[i]) routingBad++;
        }
        leafCells += a.leaf.length;
        for (let i = 0; i < a.leaf.length; i++) {
          worstLeafGap = Math.max(worstLeafGap, Math.abs(a.leaf[i] - b.leaf[i]));
        }
      }
    }
    if (routingBad !== 0) failures++;
    console.log(`${routingBad === 0 ? 'pass' : 'FAIL'}  models.bin decodes to the same walk as `
                + `models.json: ${routingBad} f/t/l/r/leafOff differences over ${nodes} nodes`);

    // 1b. The constants ride in models.meta.json on the path the page now takes, and the
    // segmenter reads them off whichever container loaded. They must be the same float64s:
    // a threshold that arrives rounded is a page that behaves differently from live_demo.py
    // and says nothing about it.
    const overrides = (m) => JSON.stringify(m && m.thresholds ? m.thresholds : null);
    const thOk = JSON.stringify(meta.thresholds) === JSON.stringify(payload.thresholds)
      && overrides(meta.digits) === overrides(payload.digits);
    if (!thOk) failures++;
    console.log(`${thOk ? 'pass' : 'FAIL'}  models.meta.json carries the same thresholds as `
                + `models.json (${Object.keys(meta.thresholds).length} fields, and the same `
                + `${Object.keys(JSON.parse(overrides(meta.digits)) || {}).length} in the digits `
                + 'block numbers mode swaps in)');
    // 5e-5 is the 4-decimal leaf rounding in models.json, which is the whole gap: anything
    // larger would mean the two files hold different forests, not different precisions.
    const gapOk = worstLeafGap <= 5e-5;
    if (!gapOk) failures++;
    console.log(`${gapOk ? 'pass' : 'FAIL'}  the two leaf tables differ only by models.json's `
                + `4-decimal rounding: worst ${worstLeafGap.toExponential(2)} over ${leafCells} `
                + 'cells (bound 5e-5)');

    // 2. The golden cases again, through the binary forests. Same tolerance, and the binary
    //    path must not be the worse of the two -- exact counts are the reason it exists.
    const binModels = { static: prepareModelBin(meta.static, binBuf),
                        digits: meta.digits ? prepareModelBin(meta.digits, binBuf) : null };
    let binWorst = 0, binWrong = 0, counted = 0;
    for (const c of golden.cases) {
      const m = binModels[c.model || 'static'];
      if (!m) continue;
      const probs = predictProba(m, Float64Array.from(c.static_feature));
      for (let j = 0; j < probs.length; j++) {
        binWorst = Math.max(binWorst, Math.abs(probs[j] - c.static_probs[j]));
      }
      if (m.classes[argmaxWithMargin(probs).index] !== c.predicted) binWrong++;
      counted++;
    }
    const goldOk = counted === golden.cases.length && binWrong === 0 && binWorst <= TOL
      && binWorst <= worstErr;
    if (!goldOk) failures++;
    console.log(`${goldOk ? 'pass' : 'FAIL'}  ${counted} golden cases through models.bin: `
                + `worst |dp| ${binWorst.toExponential(2)} against ${worstErr.toExponential(2)} `
                + `through models.json (tolerance ${TOL}), ${binWrong} letters wrong`);

    // 3. The header refusals. Every one of these fails SILENTLY if it is not checked: a forest
    //    read at the wrong layout, or paired with another export's class lists, still answers.
    const tamper = (i, v) => {
      const copy = binBuf.slice(0);
      new Uint8Array(copy)[i] = v;
      return copy;
    };
    const refuses = (label, run) => {
      try {
        run();
        failures++;
        console.log(`FAIL  ${label} was accepted`);
      } catch (e) {
        console.log(`pass  ${label} refused: ${e.message.slice(0, 96)}`);
      }
    };
    refuses('a models.bin with a bumped format version',
      () => prepareModelsBin(tamper(8, 9), meta));
    refuses('a models.bin written big-endian', () => prepareModelsBin(tamper(12, 9), meta));
    refuses('a models.meta.json from another export',
      () => prepareModelsBin(binBuf, { ...meta, build: '0'.repeat(32) }));
    refuses('a section that runs past the end of the file',
      () => prepareModelBin({ ...meta.static,
        sections: { ...meta.static.sections, t: [meta.static.sections.t[0], 1e9] } }, binBuf));

    // 4. The load the page actually performs, over HTTP, with the .gz served the way GitHub
    //    Pages serves it: an opaque application/octet-stream body, no Content-Encoding, so the
    //    gzip bytes arrive undecoded and DecompressionStream has to run. Pages does not
    //    compress octet-stream, which is why the .gz is the committed file.
    const files = {
      '/models.bin.gz': [fs.readFileSync(gzPath), 'application/octet-stream'],
      '/models.meta.json': [fs.readFileSync(metaPath), 'application/json'],
      '/models.json': [fs.readFileSync(path.join(HERE, 'models.json')), 'application/json'],
    };
    const server = http.createServer((req, res) => {
      const hit = files[req.url];
      if (!hit) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': hit[1] });
      res.end(hit[0]);
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    const root = `http://127.0.0.1:${server.address().port}`;
    try {
      const models = await loadModels(`${root}/models.bin.gz`, `${root}/models.meta.json`);
      const c = golden.cases.find((k) => (k.model || 'static') === 'static');
      const probs = predictProba(models.static, Float64Array.from(c.static_feature));
      const letter = models.static.classes[argmaxWithMargin(probs).index];
      const digitsOk = payload.digits
        ? (models.digits !== null && models.digits.dim === payload.digits.dim
           && models.digitsThresholds !== null && typeof models.digitsThresholds === 'object')
        : (models.digits === null && models.digitsThresholds === null);
      // The thresholds ride in models.meta.json for the same reason they ride in models.json:
      // a constant the page compares a probability against must be the one the Python measured
      // with, so it travels with the forest instead of living in the page.
      const thOk = JSON.stringify(models.thresholds) === JSON.stringify(payload.thresholds);
      const ok = letter === c.predicted && models.motion.classes.join(',') === 'J,MOVE,Z'
        && thOk && digitsOk;
      if (!ok) failures++;
      console.log(`${ok ? 'pass' : 'FAIL'}  loadModels read models.bin.gz + models.meta.json off `
                  + `HTTP (-> ${letter}, motion ${models.motion.classes.join('/')}, `
                  + `VOTE_WINDOW=${models.thresholds.VOTE_WINDOW}, digits `
                  + `${models.digits ? `${models.digits.nClasses} classes` : 'null'})`);
      // The same entry point must still read the old format, and decide which it is from the
      // bytes rather than the file name: one switch, no second walker.
      const fromJson = await loadModels(`${root}/models.json`);
      const j = fromJson.static.classes[
        argmaxWithMargin(predictProba(fromJson.static, Float64Array.from(c.static_feature))).index];
      const jsonOk = j === c.predicted && fromJson.motion.classes.join(',') === 'J,MOVE,Z';
      if (!jsonOk) failures++;
      console.log(`${jsonOk ? 'pass' : 'FAIL'}  the same loadModels still reads models.json `
                  + `served from the same root (-> ${j})`);
    } finally {
      server.close();
    }
  }
}

// --- the 1,215-probe parity claim --------------------------------------------------------------
//
// docs/README.md has quoted this measurement for a while and the generator behind it was never
// committed, so the figure could only be remembered. docs/parity_probes.py rebuilds the set and
// writes docs/parity.json; this reads it back and refuses to let it rot. The build id is the
// part that matters: parity.json records which models.bin it was measured against, so a new
// forest makes this FAIL rather than inherit a number that is no longer about it.
{
  const parityPath = path.join(HERE, 'parity.json');
  const metaPath = path.join(HERE, 'models.meta.json');
  if (!fs.existsSync(parityPath)) {
    failures++;
    console.log('FAIL  docs/parity.json is missing; run ./.venv/bin/python docs/parity_probes.py');
  } else {
    const par = JSON.parse(fs.readFileSync(parityPath, 'utf8'));
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const freshOk = par.build === meta.build;
    if (!freshOk) failures++;
    console.log(`${freshOk ? 'pass' : 'FAIL'}  parity.json was measured on THIS forest `
                + `(build ${String(par.build).slice(0, 16)} vs models.bin `
                + `${String(meta.build).slice(0, 16)})`
                + (freshOk ? '' : ' -- run ./.venv/bin/python docs/parity_probes.py'));
    const s = par.forests && par.forests.static;
    const shapeOk = !!s && par.probe_total === PROBE_TOTAL && s.probes.total === PROBE_TOTAL
      && s.probes.bases === PROBE_BASES && s.probes.midpoints === PROBE_MIDPOINTS
      && s.probes.noisy === PROBE_NOISY
      && JSON.stringify(par.sigmas) === JSON.stringify(PROBE_SIGMAS);
    if (!shapeOk) failures++;
    console.log(`${shapeOk ? 'pass' : 'FAIL'}  the set is the one the README describes: `
                + `${s ? s.probes.bases : '?'} golden + ${s ? s.probes.midpoints : '?'} midpoints`
                + ` + ${s ? s.probes.noisy : '?'} noisy at sigma `
                + `${par.sigmas ? par.sigmas.join('/') : '?'} = ${par.probe_total} probes`);
    for (const which of ['static', 'digits']) {
      const f = par.forests && par.forests[which];
      if (!f || !f.probes) continue;
      const ok = f.leaf_disagreements === 0 && f.json.argmax_changes === 0
        && f.bin.argmax_changes === 0 && f.bin.max_abs_dp <= PARITY_BIN_TOL
        && f.json.max_abs_dp <= PARITY_JSON_TOL && f.bin.max_abs_dp <= f.json.max_abs_dp;
      if (!ok) failures++;
      console.log(`${ok ? 'pass' : 'FAIL'}  ${which.padEnd(6)} over ${f.probes.total} probes: `
                  + `${f.leaf_disagreements}/${f.leaf_visits} leaf disagreements, max|dp| vs `
                  + `sklearn ${f.json.max_abs_dp.toExponential(2)} through models.json (tol `
                  + `${PARITY_JSON_TOL}) and ${f.bin.max_abs_dp.toExponential(2)} through `
                  + `models.bin (tol ${PARITY_BIN_TOL}), `
                  + `${f.json.argmax_changes + f.bin.argmax_changes} argmax changes`);
    }
  }
}

console.log(failures === 0
  ? `\nOK  ${golden.cases.length}/${golden.cases.length} golden cases match`
  : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

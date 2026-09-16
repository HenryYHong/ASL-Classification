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

import { prepareModel, loadModels, predictProba, argmaxWithMargin } from './forest.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

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
  const thOk = Number.isFinite(dth.VOTE_MARGIN_CLEAR) && Number.isFinite(dth.VOTE_PROB_FLOOR)
    && dth.VOTE_MARGIN_CLEAR > payload.thresholds.VOTE_MARGIN_CLEAR
    && dth.VOTE_PROB_FLOOR > payload.thresholds.VOTE_PROB_FLOOR;
  if (!thOk) failures++;
  console.log(`${thOk ? 'pass' : 'FAIL'}  digits thresholds raise the vote gate: ` +
              `VOTE_MARGIN_CLEAR ${dth.VOTE_MARGIN_CLEAR}, VOTE_PROB_FLOOR ${dth.VOTE_PROB_FLOOR} ` +
              `(letters ${payload.thresholds.VOTE_MARGIN_CLEAR} / ${payload.thresholds.VOTE_PROB_FLOOR})`);
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

console.log(failures === 0
  ? `\nOK  ${golden.cases.length}/${golden.cases.length} golden cases match`
  : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

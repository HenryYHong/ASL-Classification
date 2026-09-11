// Golden-vector check for forest.js.  Run:  /usr/local/bin/node web/test_forest.mjs
//
// Reading the port next to the Python is not evidence. Every case in golden.json carries the
// exact numbers Python produced, so this compares against those and prints the worst error it
// saw -- a port that is merely close is a port that is wrong somewhere.

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
console.log(`static forest: ${staticModel.trees.length} trees, dim ${staticModel.dim}, ` +
            `${staticModel.nClasses} classes`);
console.log(`tolerance (probabilities): ${TOL}\n`);

let failures = 0;
let worstErr = 0;
let worstCase = '';

for (let i = 0; i < golden.cases.length; i++) {
  const c = golden.cases[i];
  const probs = predictProba(staticModel, Float64Array.from(c.static_feature));

  let err = 0;
  for (let j = 0; j < probs.length; j++) {
    err = Math.max(err, Math.abs(probs[j] - c.static_probs[j]));
  }
  if (err >= worstErr) { worstErr = err; worstCase = `case ${i} (${c.predicted})`; }

  const { index, p, margin } = argmaxWithMargin(probs);
  const letter = staticModel.classes[index];

  const probsOk = probs.length === c.static_probs.length && err <= TOL;
  const letterOk = letter === c.predicted;
  const ok = probsOk && letterOk;
  if (!ok) failures++;

  console.log(`${ok ? 'pass' : 'FAIL'}  case ${String(i).padStart(2)}  ` +
              `expect ${c.predicted}  got ${letter}  ` +
              `p=${p.toFixed(4)}  margin=${margin.toFixed(4)}  ` +
              `max|dp|=${err.toExponential(2)}`);
  if (!probsOk) console.log(`      probabilities differ by ${err.toExponential(3)} > ${TOL}`);
  if (!letterOk) console.log(`      argmax letter ${letter} != golden ${c.predicted}`);
}

console.log(`\nworst probability error: ${worstErr.toExponential(3)} at ${worstCase}`);

// --- averaging, which golden.json cannot test ----------------------------------------------
//
// Every golden case comes back at p ~ 1.0, so hard voting on per-tree argmaxes would reproduce
// all 15 of them; the distinction that matters -- sklearn averages the leaf *distributions* --
// is invisible there. golden_ambiguous.json is sklearn's answer for blends of golden features
// that sit between classes (top-1 around 0.18-0.24). Optional: skip if it was not generated.
{
  const ambPath = path.join(HERE, 'golden_ambiguous.json');
  if (!fs.existsSync(ambPath)) {
    console.log('\nskip  golden_ambiguous.json absent (web/export_ambiguous.py regenerates it)');
  } else {
    const amb = JSON.parse(fs.readFileSync(ambPath, 'utf8'));
    let ambWorst = 0, ambFail = 0, ambTop = 0;
    for (const c of amb.cases) {
      const probs = predictProba(staticModel, Float64Array.from(c.feature));
      let err = 0;
      for (let j = 0; j < probs.length; j++) err = Math.max(err, Math.abs(probs[j] - c.probs[j]));
      ambWorst = Math.max(ambWorst, err);
      const { index, p } = argmaxWithMargin(probs);
      ambTop = Math.max(ambTop, p);
      if (err > amb.tolerance.probabilities || staticModel.classes[index] !== c.predicted) {
        ambFail++;
      }
    }
    failures += ambFail;
    console.log(`\n${ambFail === 0 ? 'pass' : 'FAIL'}  ${amb.cases.length - ambFail}/` +
                `${amb.cases.length} ambiguous cases match  ` +
                `(worst |dp| ${ambWorst.toExponential(2)}, highest top-1 ${ambTop.toFixed(3)})`);
  }
}

// --- guards on the pieces the golden cases cannot exercise ---------------------------------

// A wrong-length feature must throw rather than answer plausible nonsense.
try {
  predictProba(staticModel, new Float64Array(staticModel.dim - 1));
  console.log('FAIL  short feature vector was accepted');
  failures++;
} catch (e) {
  console.log(`pass  short feature vector rejected: ${e.message}`);
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

// loadModels over HTTP, with the .gz served the way GitHub Pages serves it: as an opaque body,
// no Content-Encoding, so the gzip bytes reach us undecoded and DecompressionStream must run.
{
  const gz = fs.readFileSync(path.join(HERE, 'models.json.gz'));
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
    res.end(gz);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${server.address().port}/models.json.gz`;
  try {
    const models = await loadModels(url);
    const c = golden.cases[0];
    const probs = predictProba(models.static, Float64Array.from(c.static_feature));
    const letter = models.static.classes[argmaxWithMargin(probs).index];
    const ok = letter === c.predicted && models.motion.classes.join(',') === 'J,MOVE,Z' &&
               typeof models.thresholds.VOTE_WINDOW === 'number';
    if (!ok) failures++;
    console.log(`${ok ? 'pass' : 'FAIL'}  loadModels decompressed a raw .gz body ` +
                `(case 0 -> ${letter}, motion classes ${models.motion.classes.join('/')}, ` +
                `VOTE_WINDOW=${models.thresholds.VOTE_WINDOW})`);
  } finally {
    server.close();
  }
}

console.log(failures === 0
  ? `\nOK  ${golden.cases.length}/${golden.cases.length} golden cases match`
  : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

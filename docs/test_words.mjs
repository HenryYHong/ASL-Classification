// Tests for words.js, the word layer.  Run:  node docs/test_words.mjs
//
// What is worth testing here is not that the arithmetic adds up but that the layer SHUTS UP when
// it should. A hint that fires on a close field is worse than no hint at all: it would dress a
// coin flip as a reading, and every accuracy number this project publishes exists to avoid
// exactly that. So most of these cases assert silence.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildIndex, closestWord, scoreWord, posteriorsFor } from './words.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TH = JSON.parse(fs.readFileSync(path.join(HERE, 'models.json'), 'utf8')).thresholds;

let failures = 0;
function check(name, ok, detail = '') {
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '    ' + detail : ''}`);
}

// A distribution over a..z: `spec` maps a letter to its probability, the rest share the leftover.
function dist(spec) {
  const out = new Float64Array(26);
  let named = 0;
  for (const p of Object.values(spec)) named += p;
  const rest = Math.max(0, 1 - named) / (26 - Object.keys(spec).length);
  out.fill(rest);
  for (const [letter, p] of Object.entries(spec)) out[letter.charCodeAt(0) - 97] = p;
  return out;
}
const certain = (word) => [...word].map((c) => dist({ [c]: 0.99 }));

const index = buildIndex('cat\ncot\nhenry\nzebra\njazz\nab\n');
check('buildIndex counts every entry', index.size === 6, `size=${index.size}`);
check('buildIndex buckets by length', index.byLength(3).join(',') === 'cat,cot');
check('buildIndex has()', index.has('henry') && !index.has('henrx'));

// --- scoring ------------------------------------------------------------------------------
const p = certain('cat');
check('scoreWord prefers the read letters', scoreWord('cat', p) > scoreWord('cot', p));
check('an unreachable letter costs a floor, not -Infinity',
      Number.isFinite(scoreWord('jazz', certain('jazz'))));

// --- the four verdicts --------------------------------------------------------------------
check('a reading that is already a word says so',
      closestWord('HENRY', certain('henry'), index, TH).reason === 'exact');

check('no list, no opinion',
      closestWord('HENRX', certain('henrx'), null, TH).reason === 'no-list');

check('nothing of that length, no opinion',
      closestWord('CATTLEPROD', certain('cattleprod'), index, TH).reason === 'too-long');

// Read "cqt", which is not a word, and the middle letter was a near-coin-flip between o and a.
// Both of those spellings are real words, so the layer must not pick one.
const ambiguous = [dist({ c: 0.99 }), dist({ q: 0.46, o: 0.26, a: 0.25 }), dist({ t: 0.99 })];
const amb = closestWord('CQT', ambiguous, index, TH);
check('two real words fit equally -> abstain', amb.reason === 'ambiguous' && amb.word === null,
      `reason=${amb.reason}`);

// Read "cbt", which is not a word. "cat" is the only same-length word the frames support, and
// the runner-up "cot" is far behind, so this is the case the hint exists for.
const one = [dist({ c: 0.99 }), dist({ b: 0.5, a: 0.4, o: 0.001 }), dist({ t: 0.99 })];
const hint = closestWord('CBT', one, index, TH);
check('one real word dominates -> hint it', hint.reason === 'hint' && hint.word === 'cat',
      `reason=${hint.reason} word=${hint.word}`);

// Same shape, but the classifier was sure about the b. No real word is close, so stay quiet.
const sure = [dist({ c: 0.99 }), dist({ b: 0.999 }), dist({ t: 0.99 })];
const far = closestWord('CBT', sure, index, TH);
check('a confident non-word is left alone', far.reason === 'unlikely' && far.word === null,
      `reason=${far.reason} word=${far.word}`);

// --- posteriors off an emission -------------------------------------------------------------
const classes = ['A', 'B', 'C'];
const staticEm = { letter: 'B', confidence: 0.6, detail: { probs: [0.3, 0.6, 0.1] } };
const sp = posteriorsFor(staticEm, classes);
check('a static emission carries its whole vote',
      Math.abs(sp[0] - 0.3) < 1e-12 && Math.abs(sp[1] - 0.6) < 1e-12);

// A motion letter reports one number. The other 25 letters get the remainder, not zero: the
// motion branch abstains on about one genuine gesture in six and the word layer should see that
// uncertainty rather than a fabricated certainty.
const motionEm = { letter: 'J', confidence: 0.8, detail: { arm: 'J' } };
const mp = posteriorsFor(motionEm, classes);
const spread = (1 - 0.8) / 25;
check('a motion emission spreads its remainder',
      Math.abs(mp['j'.charCodeAt(0) - 97] - 0.8) < 1e-12 && Math.abs(mp[0] - spread) < 1e-12);
check('posteriors sum to one',
      Math.abs([...mp].reduce((a, b) => a + b, 0) - 1) < 1e-12);

// --- the shipped list -------------------------------------------------------------------
const wordsPath = path.join(HERE, 'words.txt');
if (fs.existsSync(wordsPath)) {
  const real = buildIndex(fs.readFileSync(wordsPath, 'utf8'));
  check('shipped list is indexed', real.size > 100000, `${real.size} words`);
  check('a stale models.json disables the layer instead of firing it',
        closestWord('CBT', one, real, {}).reason === 'no-thresholds');
  check('the name in the demo is in it', real.has('henry'));
  check('every entry is a-z only', real.byLength(5).every((w) => /^[a-z]+$/.test(w)));
  // The demo spells A B C H E N R Y J Z, which is not a word in any list. Whatever the layer
  // does with it, it must not claim it is one.
  const demo = closestWord('ABCHENRYJZ', certain('abchenryjz'), real, TH);
  check('the demo string is not called a word', demo.reason !== 'exact', `reason=${demo.reason}`);
} else {
  console.log('skip  words.txt not built (run python3 docs/build_words.py)');
}

console.log(failures === 0 ? '\nall passed' : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

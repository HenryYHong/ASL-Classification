// Tests for words.js, the word layer.  Run:  node docs/test_words.mjs
//
// What is worth testing here is not that the arithmetic adds up but that the layer SHUTS UP when
// it should, and that the prior can only ever move a verdict within the window the frames allow.
// Expected outputs were produced by temporal/simulate_words.py's Python port of the same
// algorithm (crosscheck: 621 spelled readings, 0 mismatches).
//
// The last block reads models.json: the page takes WORD_MIN_RATIO, WORD_DOMINANCE and
// WORD_PRIOR from there, and an export that predates WORD_PRIOR silently disables the layer
// (verdict "no-thresholds" on every word). Until the export is regenerated that block prints
// a SKIP line rather than failing, so the algorithm tests above it still mean something.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildIndex, closestWord, scoreWord, posteriorsFor, collapse, collapseReading,
         HINT_MIN_LEN } from './words.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// The constants the page reads from models.json (thresholds.py: WORD_MIN_RATIO 0.10,
// WORD_DOMINANCE 10, WORD_PRIOR 2.5); pinned here so the tests are about the algorithm, not
// about whatever the export currently carries. The models.json block at the end checks that
// the export agrees with these.
const TH = { WORD_MIN_RATIO: 0.10, WORD_DOMINANCE: 10, WORD_PRIOR: 2.5 };
const SHIPPED = { WORD_MIN_RATIO: 0.10, WORD_DOMINANCE: 10, WORD_PRIOR: 2.5 };

let failures = 0;
function check(name, ok, detail = '') {
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '    ' + detail : ''}`);
}
function verdict(name, reading, posts, index, th, reason, word = null) {
  const v = closestWord(reading, posts, index, th);
  check(name, v.reason === reason && v.word === word,
        `got ${v.reason}/${v.word}, want ${reason}/${word}`);
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

// A frequency-ordered list: line number is rank. Line 19 is blank on purpose.
const LIST = ['the', 'of', 'and', 'to', 'a', 'i', 'been', 'good', 'too', 'god', 'off', 'cat',
              'cot', 'henry', 'ben', 'tho', 'hello', 'jazz', '', 'zebra'].join('\n') + '\n';
const index = buildIndex(LIST);

// --- the index ------------------------------------------------------------------------------
check('collapse folds a run to one letter', collapse('hello') === 'helo' && collapse('aaa') === 'a'
      && collapse('') === '');
check('buildIndex counts every entry once, blank lines excluded', index.size === 19, `size=${index.size}`);
check('buildIndex buckets by COLLAPSED length, in file order',
      index.byLength(2).map((e) => e.word).join(',') === 'of,to,too,off');
check('an entry carries its collapsed key', index.byLength(4)[0].key === 'helo'
      && index.byLength(4)[0].word === 'hello');
check('rank is the line number, blank lines included',
      Math.abs(index.byLength(5)[1].logRank - Math.log(20)) < 1e-12);
check('a duplicated word keeps its first rank',
      buildIndex('cat\ncat\n').size === 1 && buildIndex('cat\ncat\n').byLength(3)[0].logRank === 0);
check('buildIndex has() is on the raw spelling', index.has('hello') && !index.has('helo'));

// --- scoring ------------------------------------------------------------------------------
const p = certain('cat');
check('scoreWord prefers the read letters', scoreWord('cat', p) > scoreWord('cot', p));
check('an unreachable letter costs a floor, not -Infinity',
      Number.isFinite(scoreWord('jazz', certain('jazz'))));
const run = collapseReading('ll', [dist({ l: 0.8 }), dist({ l: 0.6 })]);
check('a doubled reading collapses to one position with the mean posterior',
      run.key === 'l' && Math.abs(run.posteriors[0]['l'.charCodeAt(0) - 97] - 0.7) < 1e-12);

// --- the verdicts ---------------------------------------------------------------------------
verdict('a reading that is already a word says so', 'HENRY', certain('henry'), index, TH, 'exact', 'henry');
verdict('no list, no opinion', 'HENRX', certain('henrx'), null, TH, 'no-list');
verdict('a models.json without WORD_PRIOR disables the layer', 'HENRX', certain('henrx'), index,
        { WORD_MIN_RATIO: 0.10, WORD_DOMINANCE: 10 }, 'no-thresholds');
verdict('nothing of that length, no opinion', 'CATTLEPROD', certain('cattleprod'), index, TH, 'too-long');

// Read "cbt", not a word; the middle letter was b 0.5 / a 0.4. "cat" is the only same-length
// word the frames support and "cot" is far behind: the case the hint exists for.
verdict('one real word dominates -> hint it', 'CBT',
        [dist({ c: 0.99 }), dist({ b: 0.5, a: 0.4, o: 0.001 }), dist({ t: 0.99 })], index, TH, 'hint', 'cat');
// Read "cqt"; the middle letter was a near coin-flip between o and a, and cat/cot are adjacent
// in rank, so the prior cannot break the tie: abstain.
verdict('two real words fit equally -> abstain', 'CQT',
        [dist({ c: 0.99 }), dist({ q: 0.46, o: 0.26, a: 0.25 }), dist({ t: 0.99 })], index, TH, 'ambiguous');
// The classifier was sure about the b. No real word is within WORD_MIN_RATIO: stay quiet.
verdict('a confident non-word is left alone', 'CBT',
        [dist({ c: 0.99 }), dist({ b: 0.999 }), dist({ t: 0.99 })], index, TH, 'unlikely');

// The prior at work. THO is a listed word (rank 16) and THE is rank 1: 2.5*ln(16) = 6.9 nats
// of prior. With e at 0.3 against o at 0.6 the frames only cost THE 0.7 nats, so THE wins by
// more than ln(10) and is hinted over the exact reading...
verdict('a rare exact reading yields to a common word the frames allow', 'THO',
        [dist({ t: 0.99 }), dist({ h: 0.99 }), dist({ o: 0.6, e: 0.3 })], index, TH, 'hint', 'the');
// ...but not when the frames rule THE out: e at 0.001 costs THE 6.9 nats, which cancels the
// 6.9-nat prior of THIS list, so THE is ahead by only 0.03 nats, far under rule R's ln(10)
// dominance bar, and THO keeps its exact. This holds on the synthetic list only; on the
// shipped list THO is rank 5,235 (21 nats of prior) and the same reading is silenced -- see
// the shipped-list block below.
verdict('the prior cannot override frames that disagree (synthetic ranks)', 'THO',
        [dist({ t: 0.99 }), dist({ h: 0.99 }), dist({ o: 0.99, e: 0.001 })], index, TH, 'exact', 'tho');

// Doubled letters. The segmenter emits HELLO as HELO; the index knows.
verdict('a collapsed reading of a doubled word is hinted to the word', 'HELO', certain('helo'), index, TH,
        'hint', 'hello');
verdict('a reading that spells the double out is exact', 'HELLO', certain('hello'), index, TH, 'exact', 'hello');
// TO (rank 4) and TOO (rank 9) score identically on the frames. Read TO, it is TO; read TOO with
// the hand dropped between the Os, the double is evidence and TO may not outrank it.
verdict('TO is TO', 'TO', certain('to'), index, TH, 'exact', 'to');
verdict('a deliberate double is not overridden by its collapsed neighbor', 'TOO', certain('too'), index, TH,
        'exact', 'too');
// GOD (rank 10) and GOOD (rank 8): 2.5*ln(10/8) = 0.56 nats, well under ln(10). The reading holds.
verdict('a same-key neighbor needs the dominance bar too', 'GOD', certain('god'), index, TH, 'exact', 'god');

// One-letter readings: A and I are exact; anything else is a stray, never corrected to A. A
// letter re-signed with a hand drop between the holds (AA, key a) is its own word and is
// hinted to it through rule R, which sits before the HINT_MIN_LEN gate: that hint repeats the
// reading, it does not overrule it. A doubled stray (NN) is still too short.
verdict('A is a word', 'A', certain('a'), index, TH, 'exact', 'a');
verdict('a stray letter is not hinted to A', 'N', [dist({ n: 0.6, a: 0.3 })], index, TH, 'too-short');
verdict('a re-signed A is hinted to its own word, not corrected', 'AA', certain('aa'), index, TH, 'hint', 'a');
verdict('a doubled stray is still too short', 'NN', [dist({ n: 0.6, a: 0.3 }), dist({ n: 0.6, a: 0.3 })],
        index, TH, 'too-short');
check('HINT_MIN_LEN is 2', HINT_MIN_LEN === 2);

// Motion letters are not duplicate-suppressed by the segmenter: JAZZ can arrive whole.
verdict('a motion double spelled out is exact', 'JAZZ', certain('jazz'), index, TH, 'exact', 'jazz');
verdict('a motion double collapsed is hinted', 'JAZ', certain('jaz'), index, TH, 'hint', 'jazz');

// --- posteriors off an emission -------------------------------------------------------------
const classes = ['A', 'B', 'C'];
const staticEm = { letter: 'B', confidence: 0.6, detail: { probs: [0.3, 0.6, 0.1] } };
const sp = posteriorsFor(staticEm, classes);
check('a static emission carries its whole vote',
      Math.abs(sp[0] - 0.3) < 1e-12 && Math.abs(sp[1] - 0.6) < 1e-12);
const motionEm = { letter: 'J', confidence: 0.8, detail: { arm: 'J' } };
const mp = posteriorsFor(motionEm, classes);
const spread = (1 - 0.8) / 25;
check('a motion emission spreads its remainder',
      Math.abs(mp['j'.charCodeAt(0) - 97] - 0.8) < 1e-12 && Math.abs(mp[0] - spread) < 1e-12);
check('posteriors sum to one', Math.abs([...mp].reduce((a, b) => a + b, 0) - 1) < 1e-12);

// --- the shipped list -------------------------------------------------------------------
const wordsPath = path.join(HERE, process.env.WORDS || 'words.txt');
if (fs.existsSync(wordsPath)) {
  const real = buildIndex(fs.readFileSync(wordsPath, 'utf8'));
  check('shipped list is indexed', real.size > 30000 && real.size < 50000, `${real.size} words`);
  // The file's ORDER is the prior. Sorting it alphabetically would leave every other check
  // here passing and silently rank AARDVARK above THE; this is the guard against that.
  check('the list starts with the most frequent word (order guard)',
        real.byLength(3)[0].word === 'the', `byLength(3)[0] = ${real.byLength(3)[0].word}`);
  check('a stale models.json disables the layer instead of firing it',
        closestWord('CBT', certain('cbt'), real, {}).reason === 'no-thresholds');
  check('the name in the demo is in it', real.has('henry'));
  check('A and I are in it and nothing else of length one',
        real.byLength(1).map((e) => e.word).join(',') === 'a,i');
  check('every entry is a-z only', real.byLength(5).every((e) => /^[a-z]+$/.test(e.word)));
  check('no entry is one letter repeated', !real.has('ii') && !real.has('mm') && !real.has('xxx'));
  check('the two-letter bucket is the curated set only', !real.has('de') && !real.has('et') && real.has('ok'));
  // The demo spells A B C H E N R Y J Z, which is not a word in any list. Whatever the layer
  // does with it, it must not claim it is one.
  const demo = closestWord('ABCHENRYJZ', certain('abchenryjz'), real, TH);
  check('the demo string is not called a word', demo.reason === 'unlikely' && demo.word === null,
        `reason=${demo.reason}`);
  // BEEN is rank 42 and BEN rank 1025: 2.5*ln(1025/42) = 8.0 nats. Read BEN, the page says BEEN.
  verdict('BEN is hinted to BEEN', 'BEN', certain('ben'), real, TH, 'hint', 'been');
  // GOOD (123) and GOD (163): 2.5*ln(163/123) = 0.7 nats, within the bar; GOD stays GOD.
  verdict('GOD stays GOD', 'GOD', certain('god'), real, TH, 'exact', 'god');
  // A misread of the demo name: y read as x with y close behind.
  verdict('HENRX is hinted to HENRY', 'HENRX',
          [...certain('henr'), dist({ x: 0.5, y: 0.4 })], real, TH, 'hint', 'henry');
  // The header's worked example is tied to the file it describes: THO sits at rank 5,235, so
  // THE outranks it by 2.5 * ln(5235) = 21.4 nats of prior. A rebuilt list moves the rank and
  // this says so before the comment goes stale.
  const tho = real.byLength(3).find((e) => e.word === 'tho');
  check('THO is at rank 5,235, as words.js says (header guard)',
        Boolean(tho) && Math.abs(Math.exp(tho.logRank) - 5235) < 0.5
        && Math.abs(2.5 * tho.logRank - 21.4) < 0.1,
        tho ? `rank ${Math.exp(tho.logRank).toFixed(0)}, prior ${(2.5 * tho.logRank).toFixed(2)} nats` : 'no tho');
  // The policy, pinned rather than accidental: a listed reading that a rank-1 neighbor
  // outranks by more than the frames can recover (one position at FLOOR is 9.2 nats; THE is
  // 21 nats ahead of THO, 22 ahead of TIE at rank 5,513) loses its exact, and the neighbor then
  // fails WORD_MIN_RATIO, so the page shows nothing rather than confirm a probable misread.
  // The reading is not restored on purpose: the same route is what keeps AND-read-as-AD from
  // being announced as a word (words.js header).
  verdict('a rare word a common neighbor outranks is silenced, not confirmed', 'TIE', certain('tie'), real, TH,
          'unlikely');
  verdict('...even with the neighbor\'s letter ruled out by the frames', 'THO',
          [dist({ t: 0.99 }), dist({ h: 0.99 }), dist({ o: 0.99, e: 0.001 })], real, TH, 'unlikely');
} else {
  console.log('skip  words.txt not built (run python3 docs/build_words.py)');
}

// --- the constants the page will actually read ------------------------------------------
//
// closestWord takes th from models.json, not from this file. An export made before WORD_PRIOR
// existed makes every verdict "no-thresholds", and a value that drifted from thresholds.py would
// make the page hint differently from what simulate_words.py measured.
{
  const modelsPath = path.join(HERE, 'models.json');
  let th = null;
  try {
    th = JSON.parse(fs.readFileSync(modelsPath, 'utf8')).thresholds;
  } catch (e) {
    console.log(`skip  models.json not readable (${e.message}); WORD_* export check not run`);
  }
  if (th && !Number.isFinite(th.WORD_PRIOR)) {
    console.log('skip  models.json carries no WORD_PRIOR: it predates the word layer\'s prior; '
                + 'regenerate it with export_models.py (the page hints nothing until then)');
  } else if (th) {
    const names = Object.keys(SHIPPED);
    check('models.json carries finite WORD_MIN_RATIO / WORD_DOMINANCE / WORD_PRIOR',
          names.every((k) => Number.isFinite(th[k])),
          names.map((k) => `${k}=${th[k]}`).join(' '));
    check('...equal to thresholds.py (0.10 / 10 / 2.5)',
          names.every((k) => Math.abs(th[k] - SHIPPED[k]) < 1e-12),
          names.map((k) => `${k}=${th[k]}`).join(' '));
    if (fs.existsSync(wordsPath)) {
      const real = buildIndex(fs.readFileSync(wordsPath, 'utf8'));
      verdict('the exported constants let the layer speak', 'HENRX',
              [...certain('henr'), dist({ x: 0.5, y: 0.4 })], real, th, 'hint', 'henry');
    }
  }
}

console.log(failures === 0 ? '\nall passed' : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);

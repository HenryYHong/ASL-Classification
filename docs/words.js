// The word layer: turn a run of emitted letters into a word, and say which real word the
// frames most look like.
//
// This sits ON TOP of recognition and never changes it. What the page displays as the reading is
// always the letters the model emitted; a dictionary match is shown beside them as a hint, and
// it abstains far more often than it speaks. That ordering is deliberate. A silent auto-correct
// would make the recognizer look better than it is, and it is exactly the kind of flattery this
// project keeps removing -- the same reason the demo is not sped up and the accuracy quoted is
// the cross-session one.
//
// The scoring is ordinary maximum likelihood under a uniform prior over the word list. Each
// emitted letter carries the whole 24-class vote that produced it, so a candidate spelling can
// be scored letter by letter against what the classifier actually saw:
//
//     log P(word) = sum_i log p_i(word_i)
//
// There is no frequency prior, because the list (Webster's Second, via /usr/share/dict) does not
// carry one. That absence is the reason for WORD_DOMINANCE: with 150k entries almost every
// letter string has a same-length neighbour, and without frequencies there is nothing to break
// a tie with, so a close field has to abstain instead of guessing.

const A_CODE = 'a'.charCodeAt(0);

// A letter the static branch cannot emit at all -- J and Z never appear among its 24 classes --
// still has to cost something finite, or a candidate containing one scores -Infinity and the
// word JAZZ could never be hinted at even when the motion branch supplied both letters.
const FLOOR = 1e-4;

/**
 * Index a newline-separated word list by length: {size, has(word), byLength(n)}.
 *
 * Candidates are always the same length as the reading, so length is the only bucket worth
 * building. Insertions and deletions are out of scope: a dropped letter is a recognition failure
 * the hint should not paper over, and admitting length changes multiplies the candidate set by
 * the alphabet without any evidence to constrain it.
 */
export function buildIndex(text) {
  const byLength = new Map();
  const all = new Set();
  for (const raw of text.split('\n')) {
    const word = raw.trim();
    if (!word) continue;
    all.add(word);
    let bucket = byLength.get(word.length);
    if (bucket === undefined) {
      bucket = [];
      byLength.set(word.length, bucket);
    }
    bucket.push(word);
  }
  return {
    size: all.size,
    has: (word) => all.has(word),
    byLength: (n) => byLength.get(n) || [],
  };
}

/** log P(word) under the per-position distributions. `posteriors[i]` is 26 long, a..z. */
export function scoreWord(word, posteriors) {
  let total = 0;
  for (let i = 0; i < word.length; i++) {
    const c = word.charCodeAt(i) - A_CODE;
    const p = (c >= 0 && c < 26) ? posteriors[i][c] : 0;
    total += Math.log(Math.max(p, FLOOR));
  }
  return total;
}

/**
 * What word the frames most look like: {reading, word, reason}.
 *
 * `reason` is the useful part and is always set:
 *   "exact"      the letters already spell a word in the list; nothing to add
 *   "hint"       one candidate is both plausible and unrivalled -- `word` holds it
 *   "unlikely"   no real word comes close enough to what was read
 *   "ambiguous"  several real words fit about equally, so the page says nothing
 *   "no-list"    the word list is not loaded
 *   "too-long"   nothing in the list is this length
 *
 * `word` is null unless the reason is "exact" or "hint".
 */
export function closestWord(reading, posteriors, index, th) {
  const word = reading.toLowerCase();
  if (!index) return { reading, word: null, reason: 'no-list' };
  // A models.json exported before this layer existed has neither constant, and `Math.log(
  // undefined)` is NaN, which loses every comparison below and turns both guards off -- the
  // layer would hint on absolutely everything, most confidently when it was least entitled to.
  // Caught by the tests exactly once, which is the argument for checking it here rather than
  // trusting the file.
  if (!Number.isFinite(th && th.WORD_MIN_RATIO) || !Number.isFinite(th && th.WORD_DOMINANCE)) {
    return { reading, word: null, reason: 'no-thresholds' };
  }
  if (index.has(word)) return { reading, word, reason: 'exact' };

  const candidates = index.byLength(word.length);
  if (!candidates.length) return { reading, word: null, reason: 'too-long' };

  // The reading is the per-position argmax, so nothing can outscore it; the question is only how
  // far behind the best real word falls.
  const readScore = scoreWord(word, posteriors);
  let best = null, bestScore = -Infinity, secondScore = -Infinity;
  for (const cand of candidates) {
    const s = scoreWord(cand, posteriors);
    if (s > bestScore) {
      secondScore = bestScore;
      bestScore = s;
      best = cand;
    } else if (s > secondScore) {
      secondScore = s;
    }
  }

  if (best === null) return { reading, word: null, reason: 'unlikely' };
  if (bestScore - readScore < Math.log(th.WORD_MIN_RATIO)) {
    return { reading, word: null, reason: 'unlikely' };
  }
  if (bestScore - secondScore < Math.log(th.WORD_DOMINANCE)) {
    return { reading, word: null, reason: 'ambiguous' };
  }
  return { reading, word: best, reason: 'hint' };
}

/**
 * The 26-long distribution behind one emission.
 *
 * A static letter carries its whole vote. A motion letter (J or Z) carries one probability and
 * nothing about the other twenty-five, so the remainder is spread evenly rather than asserting a
 * certainty the motion branch never claimed -- it abstains on roughly one genuine gesture in
 * six, and a one-hot vector here would hide that from the word layer entirely.
 */
export function posteriorsFor(emission, staticClasses) {
  const out = new Float64Array(26);
  const probs = emission.detail && emission.detail.probs;
  if (probs && staticClasses && probs.length === staticClasses.length) {
    for (let i = 0; i < probs.length; i++) {
      const c = staticClasses[i].toLowerCase().charCodeAt(0) - A_CODE;
      if (c >= 0 && c < 26) out[c] = probs[i];
    }
    return out;
  }
  const p = Number.isFinite(emission.confidence)
    ? Math.min(Math.max(emission.confidence, 0), 1)
    : 1;
  out.fill((1 - p) / 25);
  const c = emission.letter.toLowerCase().charCodeAt(0) - A_CODE;
  if (c >= 0 && c < 26) out[c] = p;
  return out;
}

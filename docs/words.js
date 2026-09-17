// The word layer: turn a run of emitted letters into a word, and say which real word the
// frames most look like.
//
// This sits ON TOP of recognition and never changes it. What the page displays as the reading is
// always the letters the model emitted; a dictionary match is shown beside them as a hint, and
// it abstains when the field is close. That ordering is deliberate. A silent auto-correct
// would make the recognizer look better than it is, and it is exactly the kind of flattery this
// project keeps removing -- the same reason the demo is not sped up and the accuracy quoted is
// the cross-session one.
//
// The scoring is maximum a posteriori under a rank prior. Each emitted letter carries the whole
// 24-class vote that produced it, so a candidate spelling can be scored letter by letter against
// what the classifier actually saw, and words.txt is written in frequency order so its line
// number is the prior:
//
//     score(word) = sum_i log p_i(word_i) - WORD_PRIOR * ln(rank(word))
//
// The prior is what breaks ties. Without it (the previous layer, a 150k-entry Webster list with
// no frequencies) almost every letter string had a same-length neighbor and a close field had
// to abstain; with it, THE (rank 1) outranks THO (rank 5,235 in the shipped list) by
// WORD_PRIOR * ln(5235) = 3.0 * ln(5235), about 26 nats, before a single frame is looked at.
// WORD_MIN_RATIO bounds how far the prior may go toward a HINT: a hint can never be a word the
// frames make less than WORD_MIN_RATIO as likely as the letters actually read, whatever its
// rank. It does not restore an exact reading the prior has outranked: with FLOOR 1e-4 one
// position can charge a rival at most ln(1/FLOOR) = 9.2 nats, so a listed reading that a far
// commoner same-length neighbor outranks by more than the frames can ever recover loses its
// "exact" and, the neighbor then failing WORD_MIN_RATIO, is reported "unlikely" -- silence,
// not a confirmation. About 4% of the shipped list is in that position at any confidence
// (1,379 of 34,702 with the reading's letters at 0.99: THO, TIE, ODD, PEN, ANN ...). That is
// the policy the simulator's numbers were measured under (test_words.mjs pins it): the same
// route is what keeps a dropped-letter misread that happens to be listed (AND read as AD) from
// being announced as a word.
//
// Two facts about the segmenter shape the index. It never emits the same letter twice in a
// row (a held sign re-enters HOLD several times and only the first vote counts), so HELLO
// arrives as HELO: entries are indexed by their run-length-collapsed spelling and a reading
// is collapsed the same way before it is scored. And a word break with one letter in it is
// either A or I or a stray, so a one-letter reading can be "exact" but is never corrected to
// A or I on the strength of a single posterior (HINT_MIN_LEN below).

const A_CODE = 'a'.charCodeAt(0);

// A letter the static branch cannot emit at all -- J and Z never appear among its 24 classes --
// still has to cost something finite, or a candidate containing one scores -Infinity and the
// word JAZZ could never be hinted at even when the motion branch supplied both letters.
export const FLOOR = 1e-4;

// A hint that CHANGES a letter needs at least this many collapsed positions. The one-letter
// bucket is {a, i}; a stray letter read as N or T would otherwise be "corrected" to A on the
// strength of a single posterior. The gate sits after the own-candidate return (rule R, step 3
// of closestWord), so a reading whose own collapsed spelling is listed -- AA, the letter
// re-signed with a hand drop between the holds, collapses to key a -- is still hinted at one
// position, the way DOOG is hinted to dog: that hint repeats what was read, it does not
// overrule it. NN and TT stay "too-short".
export const HINT_MIN_LEN = 2;

/** Run-length collapse: 'hello' -> 'helo'. What the segmenter emits for a doubled letter. */
export function collapse(word) {
  let out = '';
  for (let i = 0; i < word.length; i++) {
    if (i === 0 || word[i] !== word[i - 1]) out += word[i];
  }
  return out;
}

/**
 * Index a newline-separated, frequency-ordered word list: {size, has(word), byLength(n)}.
 *
 * Line i (1-based) is rank i. Each entry is {word, key, logRank}: `key` is the collapsed
 * spelling, `logRank` is ln(rank). Buckets are keyed by the COLLAPSED length and keep file
 * order, so the first entry in a bucket is the most frequent and ties break toward it.
 * A word repeated in the file keeps its first (lowest) rank. Blank lines are skipped but still
 * count toward rank, so the file's line numbers stay meaningful.
 *
 * Insertions and deletions are out of scope: a dropped letter is a recognition failure the hint
 * should not paper over, and admitting length changes multiplies the candidate set by the
 * alphabet without any evidence to constrain it.
 */
export function buildIndex(text) {
  const byLength = new Map();
  const all = new Set();
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const word = lines[i].trim();
    if (!word || all.has(word)) continue;
    all.add(word);
    const key = collapse(word);
    let bucket = byLength.get(key.length);
    if (bucket === undefined) {
      bucket = [];
      byLength.set(key.length, bucket);
    }
    bucket.push({ word, key, logRank: Math.log(i + 1) });
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
 * Collapse a reading the way the index collapses a word: runs of one letter become one
 * position whose posterior is the element-wise mean of the run. The segmenter only produces a
 * run when the hand left the frame for GAP_RESET between two holds of the same letter.
 */
export function collapseReading(reading, posteriors) {
  let key = '';
  const posts = [];
  for (let i = 0; i < reading.length; i++) {
    if (i > 0 && reading[i] === reading[i - 1]) {
      posts[posts.length - 1].push(posteriors[i]);
    } else {
      key += reading[i];
      posts.push([posteriors[i]]);
    }
  }
  const merged = posts.map((run) => {
    if (run.length === 1) return run[0];
    const out = new Float64Array(26);
    for (const p of run) for (let c = 0; c < 26; c++) out[c] += p[c];
    for (let c = 0; c < 26; c++) out[c] /= run.length;
    return out;
  });
  return { key, posteriors: merged };
}

/**
 * What word the frames most look like: {reading, word, reason}.
 *
 * `reason` is the useful part and is always set:
 *   "exact"      the letters as emitted are a listed word and nothing outranks them -- `word`
 *                is the reading itself
 *   "hint"       one candidate is both plausible and unrivalled -- `word` holds it. A word
 *                whose collapsed spelling IS the reading (HELO -> hello) arrives this way too
 *   "unlikely"   no real word the frames allow is worth showing: either nothing comes close to
 *                what was read, or the reading is itself a listed word that a far commoner
 *                neighbor outranks on the prior while that neighbor is too unlikely to hint;
 *                the page stays silent rather than confirm a probable misread (header)
 *   "ambiguous"  several real words fit about equally, so the page says nothing
 *   "too-short"  a one-letter reading that is not A or I
 *   "no-list"    the word list is not loaded
 *   "too-long"   nothing in the list is this length
 *   "no-thresholds"  models.json predates the layer (see below)
 *
 * `word` is null unless the reason is "exact" or "hint".
 *
 * Verdict order, after collapsing the reading to `key` with n = key.length:
 *   1. bucket = byLength(n); empty -> "too-long"
 *   2. own = the entry whose word === reading if the bucket has one, else the first entry whose
 *      key === reading key, if any (the reading's own candidate; its lik equals readScore, the
 *      maximum, because the reading is the per-position argmax)
 *      for every entry: lik = scoreWord(key, posts); score = lik - WORD_PRIOR * logRank
 *      best   = highest score, ties to the EARLIER entry (lower rank)
 *      second = highest score among the other entries (an equal score counts)
 *      Skipped from best/second: when the reading carries a double (reading !== key) and is
 *      itself the listed word, every OTHER entry with the same key. The frames cannot tell TOO
 *      from TO; the hand drop between the two Os can, and it said TOO.
 *   3. if own exists and (best === own or score(best) - score(own) < ln WORD_DOMINANCE):
 *        the reading holds its ground -> "exact" if own.word === reading, else "hint" own.word
 *   4. n < HINT_MIN_LEN -> "too-short"
 *   5. lik(best) - readScore < ln WORD_MIN_RATIO -> "unlikely"
 *   6. score(best) - second < ln WORD_DOMINANCE -> "ambiguous"
 *   7. "hint" best.word
 */
export function closestWord(reading, posteriors, index, th) {
  const lower = reading.toLowerCase();
  if (!index) return { reading, word: null, reason: 'no-list' };
  // A models.json exported before this layer existed has none of these constants, and
  // `Math.log(undefined)` is NaN, which loses every comparison below and turns every guard
  // off -- the layer would hint on absolutely everything, most confidently when it was least
  // entitled to. Caught by the tests exactly once, which is the argument for checking it here
  // rather than trusting the file.
  if (!Number.isFinite(th && th.WORD_MIN_RATIO) || !Number.isFinite(th && th.WORD_DOMINANCE)
      || !Number.isFinite(th && th.WORD_PRIOR)) {
    return { reading, word: null, reason: 'no-thresholds' };
  }
  const { key, posteriors: posts } = collapseReading(lower, posteriors);
  if (!key.length) return { reading, word: null, reason: 'too-long' };

  const candidates = index.byLength(key.length);
  if (!candidates.length) return { reading, word: null, reason: 'too-long' };

  const lnDominance = Math.log(th.WORD_DOMINANCE);
  // The reading is the per-position argmax, so nothing can outscore it on likelihood; the
  // question is only how far behind the best real word falls, and how far ahead its rank puts it.
  const readScore = scoreWord(key, posts);
  let own = null;
  for (const cand of candidates) {
    if (cand.word === lower) { own = cand; break; }
    if (own === null && cand.key === key) own = cand;
  }
  // A doubled letter the segmenter actually emitted (the hand dropped between two holds) is
  // evidence the posteriors do not carry: TOO and TO score identically on the frames.
  const doubled = own !== null && own.word === lower && lower !== key;
  let best = null, bestLik = -Infinity, bestScore = -Infinity, secondScore = -Infinity;
  let ownScore = -Infinity;
  for (const cand of candidates) {
    const lik = scoreWord(cand.key, posts);
    const s = lik - th.WORD_PRIOR * cand.logRank;
    if (cand === own) ownScore = s;
    else if (doubled && cand.key === key) continue;
    if (s > bestScore) {
      secondScore = bestScore;
      bestScore = s;
      bestLik = lik;
      best = cand;
    } else if (s > secondScore) {
      secondScore = s;
    }
  }

  if (own !== null && (best === own || bestScore - ownScore < lnDominance)) {
    return own.word === lower
      ? { reading, word: own.word, reason: 'exact' }
      : { reading, word: own.word, reason: 'hint' };
  }
  if (key.length < HINT_MIN_LEN) return { reading, word: null, reason: 'too-short' };
  if (bestLik - readScore < Math.log(th.WORD_MIN_RATIO)) {
    return { reading, word: null, reason: 'unlikely' };
  }
  if (bestScore - secondScore < lnDominance) {
    return { reading, word: null, reason: 'ambiguous' };
  }
  return { reading, word: best.word, reason: 'hint' };
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

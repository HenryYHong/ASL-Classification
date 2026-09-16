// Random-forest inference for the browser port of temporal/.
//
// The trees come from docs/export_models.py, which flattens each sklearn estimator to the four
// arrays a tree walk actually needs (f, t, l, r) plus the leaf class distributions. The walk here
// must land on sklearn's leaf in every tree (verified at 0.00e+0 against tree_.apply on 1,215
// probes, export_models.py); the probabilities then match predict_proba to the exported 4-dp
// leaf precision (bound 5e-5, golden tolerance 1e-4), so a disagreement with docs/golden.json
// larger than that is a bug on this side, not in the export.
//
// Two things in this file are load-bearing and easy to get subtly wrong:
//   - the comparison direction (see predictProba),
//   - averaging the per-tree *probability* vectors rather than voting on per-tree argmaxes.
//     sklearn's forest averages distributions; hard voting agrees with it most of the time, which
//     is exactly what makes the difference expensive to find.

/**
 * Turn one exported forest ({classes, feature, dim, trees}) into typed arrays.
 *
 * Done once at load so the per-frame path never touches a JS object or a string key. The leaf
 * vectors arrive as {"<node index>": [...]} and are packed into one flat array with a per-node
 * offset table; a dense nodes x classes table would be mostly holes, since only half the nodes
 * are leaves.
 */
export function prepareModel(spec) {
  const nClasses = spec.classes.length;
  const trees = new Array(spec.trees.length);

  for (let k = 0; k < spec.trees.length; k++) {
    const src = spec.trees[k];
    const n = src.f.length;
    const leafOff = new Int32Array(n).fill(-1);

    const leafKeys = Object.keys(src.v);
    const leaf = new Float32Array(leafKeys.length * nClasses);
    let off = 0;
    for (let i = 0; i < n; i++) {
      // Leaves are the nodes with f < 0 (sklearn writes TREE_UNDEFINED = -2 there). Walking `f`
      // rather than the key set keeps the packed order equal to node order, which is only a
      // tidiness point, but it also catches a leaf whose vector the export dropped.
      if (src.f[i] >= 0) continue;
      const v = src.v[String(i)];
      if (v === undefined) throw new Error(`tree ${k}: leaf node ${i} has no class vector`);
      if (v.length !== nClasses) {
        throw new Error(`tree ${k}: leaf ${i} has ${v.length} classes, expected ${nClasses}`);
      }
      leafOff[i] = off;
      for (let c = 0; c < nClasses; c++) leaf[off + c] = v[c];
      off += nClasses;
    }

    trees[k] = {
      f: Int32Array.from(src.f),
      t: Float64Array.from(src.t),
      l: Int32Array.from(src.l),
      r: Int32Array.from(src.r),
      leafOff,
      // Float32, not Float64: it halves the resident size of a 100-tree, 134k-node forest,
      // which matters more on a phone than the bits do. It is NOT lossless -- a 4-decimal
      // probability like 0.0833 has no exact Float32 form, and with min_samples_leaf 5 about
      // 65% of the letter forest's leaves are mixed (43,412 of 66,890 at export; 61% of the
      // digit forest's) -- but it is not what limits parity either: export_models.py rounds
      // every leaf to 4 decimals, which bounds the page's probabilities within 5e-5 of
      // sklearn's predict_proba by that rounding alone (5.1e-6 measured over 1,215 probes,
      // 2.0e-6 on the golden cases, 0 argmax changes; half of golden.json's 1e-4 tolerance).
      // Float32 adds ~2e-9 on top (1.7e-9 letters / 2.2e-9 digits, measured on the shipped
      // forests against a Float64 walk over the same JSON with this Float64 accumulator),
      // three orders under the rounding. What must not change is the Float64 accumulator.
      leaf,
    };
  }

  return {
    classes: spec.classes.slice(),
    feature: spec.feature,
    dim: spec.dim,
    nClasses,
    trees,
  };
}

/**
 * Fetch and parse docs/models.json, returning {static, motion, thresholds, digits,
 * digitsThresholds}.
 *
 * `digits` is the numbers-mode forest, prepared when the export carried one and null
 * otherwise (the letter export never depends on the digit pickle); `digitsThresholds` is the
 * override block that mode applies on top of `thresholds` (thresholds.DIGITS_OVERRIDES:
 * VOTE_MARGIN_CLEAR and VOTE_PROB_FLOOR raised), or null with it.
 *
 * `url` may point at models.json or models.json.gz. When a server sends the gzipped file with
 * Content-Encoding: gzip the browser has already decompressed it by the time we see the bytes;
 * GitHub Pages instead serves a .gz file as an opaque body, so the bytes are still gzip. Both
 * cases are handled by sniffing the magic number and handing the body to the platform's
 * DecompressionStream. Nothing here implements inflate -- a hand-rolled one would be a second
 * implementation of something the platform already has, which is how this project got burned.
 */
export async function loadModels(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`loadModels: ${url} -> HTTP ${res.status} ${res.statusText}`);

  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (head.length === 2 && head[0] === 0x1f && head[1] === 0x8b) {
    if (typeof DecompressionStream !== 'function') {
      throw new Error('loadModels: body is gzip and this browser has no DecompressionStream; ' +
                      'serve models.json uncompressed, or with Content-Encoding: gzip');
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = await new Response(stream).arrayBuffer();
  }

  const payload = JSON.parse(new TextDecoder().decode(buf));
  return {
    static: prepareModel(payload.static),
    motion: prepareModel(payload.motion),
    thresholds: payload.thresholds,
    digits: payload.digits ? prepareModel(payload.digits) : null,
    digitsThresholds: payload.digits && payload.digits.thresholds ? payload.digits.thresholds : null,
  };
}

/**
 * Mean of the per-tree leaf distributions: sklearn's RandomForestClassifier.predict_proba.
 *
 * Runs once per frame on the static branch, so the walk reads the flat arrays directly and the
 * only allocation is the accumulator that gets returned.
 */
export function predictProba(model, x) {
  if (x.length !== model.dim) {
    // The classic silent divergence in this project is a feature vector of the wrong length,
    // which a tree walk will happily consume (reading undefined -> NaN, or just the wrong
    // column) and answer with plausible nonsense. Fail loudly instead.
    throw new Error(`predictProba: feature length ${x.length}, model expects ${model.dim}`);
  }
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) {
      // `NaN <= t` is false, so a NaN feature walks RIGHT at every split it meets and the
      // forest answers with a confident letter from a vector that carries no information at
      // all. sklearn refuses NaN input; so does this.
      throw new Error(`predictProba: feature ${i} is ${x[i]}; the forest takes finite numbers only`);
    }
  }

  const trees = model.trees;
  const nTrees = trees.length;
  const nClasses = model.nClasses;
  const out = new Float64Array(nClasses);

  for (let k = 0; k < nTrees; k++) {
    const tree = trees[k];
    const f = tree.f, t = tree.t, l = tree.l, r = tree.r;
    let i = 0;
    // Internal nodes carry f >= 0; leaves carry sklearn's TREE_UNDEFINED (-2).
    while (f[i] >= 0) {
      // sklearn's rule, exactly: `x[feature] <= threshold` goes LEFT, everything else RIGHT.
      // Flipping this (or using < instead of <=) still classifies most inputs correctly, because
      // most samples are nowhere near a split point -- and quietly ruins the ones that are.
      i = x[f[i]] <= t[i] ? l[i] : r[i];
    }
    const off = tree.leafOff[i];
    const leaf = tree.leaf;
    for (let c = 0; c < nClasses; c++) out[c] += leaf[off + c];
  }

  for (let c = 0; c < nClasses; c++) out[c] /= nTrees;
  return out;
}

/**
 * Winner plus its lead over the runner-up: {index, p, margin}.
 *
 * The runtime emits on EITHER a high probability OR a clear margin (segmenter.py: `confident` is
 * meanp >= VOTE_PROB, `decisive` is margin >= VOTE_MARGIN_CLEAR with meanp above a floor), so the
 * margin is a first-class output, not a diagnostic.
 *
 * Ties go to the lowest index, matching np.argmax -- which is what the static branch's per-frame
 * votes use. segmenter.py's motion branch instead takes its winner from np.argsort(proba)[::-1],
 * which on an exact tie lands on the HIGHER index. The two cannot disagree about an emission: a
 * tie makes the margin 0, and 0 < MARGIN, so the motion branch abstains whichever index it names.
 * The margin itself is the same quantity either way (`proba[order[0]] - proba[order[1]]`).
 *
 * Note the one place this does NOT match the Python: segmenter.py's static vote takes its winner
 * by majority of per-frame argmaxes and only then measures that winner's mean probability against
 * the second-largest mean. When the majority winner is not the argmax of the mean, its margin can
 * even be negative. That is the vote's business; this function is the per-frame/motion-branch
 * form (segmenter.py `_score_track`).
 */
export function argmaxWithMargin(probs) {
  let bi = -1, best = -Infinity, second = -Infinity;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > best) {            // strict >, so an exact tie keeps the earlier index
      second = best;
      best = p;
      bi = i;
    } else if (p > second) {
      second = p;
    }
  }
  if (bi < 0) throw new Error('argmaxWithMargin: empty probability vector');
  // Only reachable for a one-class forest, which neither of ours is. 0 is the static path's
  // `runner = 0.0` fallback, and it also reproduces the motion path's separate `else 1.0`
  // fallback, because a one-class forest always returns p = 1.0 and 1.0 - 0 = 1.0.
  if (second === -Infinity) second = 0;
  return { index: bi, p: best, margin: best - second };
}

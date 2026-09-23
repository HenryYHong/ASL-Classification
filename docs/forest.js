// Random-forest inference for the browser port of temporal/.
//
// The trees come from docs/export_models.py, which flattens each sklearn estimator to the four
// arrays a tree walk actually needs (f, t, l, r) plus the leaf class distributions. The walk here
// must land on sklearn's leaf in every tree (verified at 0 leaf disagreements out of 72,900 tree
// visits on 1,215 probes, docs/parity_probes.py); the probabilities then match predict_proba to
// the exported leaf precision, so a disagreement with docs/golden.json larger than that is a bug
// on this side, not in the export.
//
// The same forests arrive in either of two formats and leave as the same typed arrays:
// models.json, which is readable and 20,361,559 B, and models.bin + models.meta.json, which is
// 2,555,774 + 2,726 B and is what the page fetches -- as models.bin.gz + models.meta.json,
// 1,184,420 B on the wire against models.json's 3,703,069 B. prepareModel reads the first,
// prepareModelBin the second, loadModels picks by looking at the bytes, and everything below
// this point is shared.
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
      // Float32, not Float64: it halves the resident size of the 60-tree, 245,064-node letter
      // forest, which matters more on a phone than the bits do. It is NOT lossless -- a
      // 4-decimal probability like 0.0833 has no exact Float32 form, and with min_samples_leaf 5
      // about 68% of the letter forest's leaves are mixed (83,223 of 122,562 at export; 61% of
      // the digit forest's) -- but it is not what limits parity either: export_models.py rounds
      // every leaf to 4 decimals, which bounds the page's probabilities within 5e-5 of
      // sklearn's predict_proba by that rounding alone (6.25e-6 measured over 1,215 probes,
      // 3.33e-6 on the golden cases, 0 argmax changes; a fifteenth of golden.json's 1e-4
      // tolerance). Float32 is the only lossy step left on the binary path, which carries exact
      // integer leaf counts instead of rounded probabilities, so what models.bin scores against
      // sklearn IS the Float32 cost measured end to end: 4.05e-9 letters / 4.75e-9 digits over
      // the same probes, three orders under the rounding. The Float64 accumulator must not
      // change -- summing 60 Float32 rows in Float32 is what would put this back in play.
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

// ---- models.bin ------------------------------------------------------------------------------
//
// The binary sidecar export_models.py writes beside models.json. Same forests, same walk: the
// only thing that changes is how the trees arrive, and one thing that gets BETTER, which is the
// leaf table (exact integer class counts instead of 4-decimal probabilities).
//
// These four constants mirror export_models.py's BIN_MAGIC / BIN_VERSION / BIN_HEADER_BYTES /
// BIN_ENDIAN_PROBE and must move with them.
const BIN_MAGIC = [0x41, 0x53, 0x4c, 0x46, 0x52, 0x53, 0x54, 0x00];   // "ASLFRST\0"
const BIN_VERSION = 1;
const BIN_HEADER_BYTES = 40;
const BIN_ENDIAN_PROBE = 0x04030201;

/** True when `buf` opens with models.bin's magic. This is the whole format switch. */
function isModelsBin(buf) {
  if (buf.byteLength < BIN_HEADER_BYTES) return false;
  const b = new Uint8Array(buf, 0, BIN_MAGIC.length);
  for (let i = 0; i < BIN_MAGIC.length; i++) if (b[i] !== BIN_MAGIC[i]) return false;
  return true;
}

/**
 * Read models.bin's 40-byte fixed header and refuse anything this build cannot decode.
 *
 * Every check here exists because its failure is silent rather than loud. A future VERSION
 * decoded at this layout is not a crash: it is a forest walking to whatever nodes the old
 * offsets happen to point at, answering letters with full confidence. So is a meta file from a
 * different export, which is why both carry a build id. And so is a big-endian reader, which
 * would byte-swap all 139,145 thresholds in the three shipped forests without noticing --
 * every view below is platform-endian, so the probe is the only thing standing there.
 */
function readBinHeader(buf, meta) {
  if (!isModelsBin(buf)) {
    throw new Error('models.bin does not start with its magic number; this is not a models.bin, '
                    + 'or the body was truncated or decompressed wrong');
  }
  const dv = new DataView(buf);
  const version = dv.getUint32(8, true);
  if (version !== BIN_VERSION) {
    throw new Error(`models.bin is format version ${version}; this build reads version `
                    + `${BIN_VERSION}. Re-export with docs/export_models.py, or load models.json`);
  }
  const probe = new Uint32Array(buf, 12, 1)[0];
  if (probe !== BIN_ENDIAN_PROBE) {
    throw new Error('models.bin is little-endian and this machine is not; every typed-array '
                    + 'view over it would be byte-swapped. Load models.json instead');
  }
  const headerBytes = dv.getUint32(16, true);
  if (headerBytes !== BIN_HEADER_BYTES) {
    throw new Error(`models.bin declares a ${headerBytes}-byte header; this build expects `
                    + `${BIN_HEADER_BYTES}`);
  }
  let build = '';
  for (let i = 24; i < 40; i++) build += dv.getUint8(i).toString(16).padStart(2, '0');
  if (meta.version !== BIN_VERSION || meta.build !== build) {
    throw new Error(`models.meta.json describes build ${meta.build} (format version `
                    + `${meta.version}) but models.bin is build ${build} (version ${version}); `
                    + 'the two were not exported together. Deploy both files from one run of '
                    + 'docs/export_models.py');
  }
  if (typeof meta.bytes === 'number' && meta.bytes !== buf.byteLength) {
    throw new Error(`models.bin is ${buf.byteLength} bytes, ${meta.bytes} expected; it is `
                    + 'truncated or was decompressed wrong');
  }
}

/**
 * Turn one forest's meta block plus the shared models.bin buffer into exactly what
 * prepareModel returns: {classes, feature, dim, nClasses, trees[{f, t, l, r, leafOff, leaf}]}.
 *
 * Nothing downstream can tell which loader ran, and that is checked rather than asserted:
 * test_forest.mjs compares f, t, l and r element by element against the models.json path for
 * all three forests (278,750 nodes; identical) before it trusts any of the numbers above.
 *
 * Two things are rebuilt here rather than stored, because storing them is pure waste:
 *
 *   l[i]  The left child of an internal node is always i + 1 -- sklearn's depth-first builder
 *         emits the left subtree immediately after its parent. export_models.py asserts that
 *         on every internal node of every tree and refuses to write the file otherwise, so the
 *         section simply is not there to disagree with.
 *
 *   leaf  The file stores each leaf's nonzero class COUNTS (vk how many, vi which, vc the
 *         integers), and the probability is recovered here by dividing by their sum. That is
 *         the same rational number sklearn's predict_proba averages, so the page's leaves stop
 *         being a quantization: models.json rounds them to 4 decimals, which costs up to 5e-5
 *         per leaf, while a count divided by a total is exact up to the Float32Array the table
 *         has always been. Measured on docs/golden.json's 41 cases, worst error against
 *         sklearn's own probabilities: 3.33e-6 through models.json, 4.98e-7 through models.bin;
 *         over the 1,215 parity probes it is 6.25e-6 against 4.05e-9 (docs/parity.json).
 *         Sparse because it pays: 2.08 of the letter forest's 24 classes are nonzero in the
 *         average leaf (254,521 nonzero counts over 122,562 leaves), so a dense table would be
 *         91% zeros.
 */
export function prepareModelBin(spec, buf) {
  const sections = spec.sections || {};
  const view = (name, Ctor) => {
    const s = sections[name];
    if (!s) throw new Error(`models.bin: meta block has no "${name}" section`);
    const [off, len] = s;
    const width = Ctor.BYTES_PER_ELEMENT;
    if (off % width !== 0 || off < BIN_HEADER_BYTES || off + len * width > buf.byteLength) {
      throw new Error(`models.bin: section "${name}" at ${off} for ${len} x ${width} B does not `
                      + `fit a ${buf.byteLength}-byte file, or is not ${width}-byte aligned`);
    }
    return new Ctor(buf, off, len);
  };
  const nPer = view('n', Uint32Array);      // nodes per tree
  const bits = view('bits', Uint8Array);    // leaf bitmap, MSB first, byte-aligned per tree
  const F = view('f', Int8Array);           // split feature, internal nodes only
  const T = view('t', Float64Array);        // split threshold, full float64
  const R = view('r', Uint16Array);         // right child as an offset from the node
  const VK = view('vk', Uint8Array);        // nonzero classes in this leaf
  const VI = view('vi', Uint8Array);        // their class indices
  const VC = view('vc', Uint16Array);       // their exact integer counts

  const nClasses = spec.classes.length;
  const trees = new Array(nPer.length);
  let bitOff = 0, intOff = 0, leafIdx = 0, nzOff = 0;

  for (let k = 0; k < nPer.length; k++) {
    const n = nPer[k];
    // Bounds first, per tree. A section that is one element short does not throw on its own:
    // reading past a typed array yields undefined, `undefined >> 3` is 0, and the tree decodes
    // into something plausible and wrong. Checking costs eight comparisons per tree.
    const nBytes = (n + 7) >> 3;
    if (bitOff + nBytes > bits.length) {
      throw new Error(`models.bin: tree ${k} needs ${nBytes} bitmap bytes at ${bitOff}, but the `
                      + `"bits" section holds ${bits.length}`);
    }
    const f = new Int32Array(n);
    const t = new Float64Array(n);
    const l = new Int32Array(n);
    const r = new Int32Array(n);
    const leafOff = new Int32Array(n).fill(-1);

    // Count this tree's leaves first so the leaf table can be sized exactly, the way the
    // models.json path sizes it from the key set. One extra pass over a bitmap is nothing.
    let nLeaves = 0;
    for (let i = 0; i < n; i++) nLeaves += (bits[bitOff + (i >> 3)] >> (7 - (i & 7))) & 1;
    const nInternal = n - nLeaves;
    let nnz = 0;
    if (leafIdx + nLeaves > VK.length) {
      throw new Error(`models.bin: tree ${k} has ${nLeaves} leaves but the "vk" section ends `
                      + `after ${VK.length - leafIdx} more`);
    }
    for (let j = 0; j < nLeaves; j++) nnz += VK[leafIdx + j];
    if (intOff + nInternal > F.length || intOff + nInternal > T.length
        || intOff + nInternal > R.length || nzOff + nnz > VI.length
        || nzOff + nnz > VC.length) {
      throw new Error(`models.bin: tree ${k} wants ${nInternal} internal nodes and ${nnz} `
                      + 'nonzero leaf counts, more than the f/t/r/vi/vc sections hold');
    }
    const leaf = new Float32Array(nLeaves * nClasses);

    let li = 0, ii = 0, cell = 0, p = nzOff;
    for (let i = 0; i < n; i++) {
      if ((bits[bitOff + (i >> 3)] >> (7 - (i & 7))) & 1) {
        // sklearn's own constants at a leaf: TREE_UNDEFINED in f and t, TREE_LEAF in both
        // children. export_models.py checks the pickle carries exactly these before it drops
        // them, so writing them back reproduces models.json's arrays rather than approximating
        // them -- and predictProba's `while (f[i] >= 0)` depends on the -2.
        f[i] = -2; t[i] = -2; l[i] = -1; r[i] = -1;
        leafOff[i] = cell;
        const k2 = VK[leafIdx + li];
        let total = 0;
        for (let c = 0; c < k2; c++) total += VC[p + c];
        // A leaf sklearn grew holds at least min_samples_leaf samples, so a zero total means
        // the file is corrupt -- and dividing by it would put Infinity or NaN in the table and
        // let the forest answer anyway, which is the one outcome worth a branch per leaf.
        if (total <= 0) {
          throw new Error(`models.bin: tree ${k} leaf ${i} holds no samples; its class counts `
                          + 'cannot be turned into probabilities');
        }
        for (let c = 0; c < k2; c++) leaf[cell + VI[p + c]] = VC[p + c] / total;
        p += k2; cell += nClasses; li++;
      } else {
        f[i] = F[intOff + ii];
        t[i] = T[intOff + ii];
        l[i] = i + 1;
        r[i] = i + R[intOff + ii];
        ii++;
      }
    }
    // The bookkeeping is the only thing standing between a section read half a tree out of step
    // and a forest that answers anyway, so it is checked per tree rather than at the end.
    if (li !== nLeaves || ii + nLeaves !== n) {
      throw new Error(`models.bin: tree ${k} has ${n} nodes but decoded ${ii} internal + ${li} `
                      + 'leaves; the sections are out of step');
    }
    trees[k] = { f, t, l, r, leafOff, leaf };
    bitOff += nBytes; intOff += ii; leafIdx += nLeaves; nzOff = p;
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
 * Fetch one file and hand back its bytes, gunzipped when they arrive gzipped.
 *
 * `url` may point at the plain file or at its .gz. When a server sends the gzipped file with
 * Content-Encoding: gzip the browser has already decompressed it by the time we see the bytes;
 * GitHub Pages instead serves a .gz file as an opaque body, so the bytes are still gzip. Both
 * cases are handled by sniffing the magic number and handing the body to the platform's
 * DecompressionStream. Nothing here implements inflate -- a hand-rolled one would be a second
 * implementation of something the platform already has, which is how this project got burned.
 *
 * models.bin.gz is the committed one of the pair, because GitHub Pages compresses text media
 * types on the fly and leaves application/octet-stream alone: served raw, the binary would
 * cross the wire at 2,555,774 B instead of 1,183,438 B.
 */
async function fetchBody(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`loadModels: ${url} -> HTTP ${res.status} ${res.statusText}`);
  const buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (head.length !== 2 || head[0] !== 0x1f || head[1] !== 0x8b) return buf;
  if (typeof DecompressionStream !== 'function') {
    throw new Error(`loadModels: ${url} is gzip and this browser has no DecompressionStream; `
                    + 'serve it uncompressed, or with Content-Encoding: gzip');
  }
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

/**
 * models.bin's bytes plus its models.meta.json, as the same {static, motion, thresholds,
 * digits, digitsThresholds} loadModels returns. Split out from the fetch so the header
 * refusals can be exercised without a server: docs/test_forest.mjs calls this with a bumped
 * version byte, a flipped endian probe and another export's build id, because every one of
 * those failures is a forest that answers rather than a forest that stops.
 */
export function prepareModelsBin(buf, meta) {
  readBinHeader(buf, meta);
  return {
    static: prepareModelBin(meta.static, buf),
    motion: prepareModelBin(meta.motion, buf),
    thresholds: meta.thresholds,
    digits: meta.digits ? prepareModelBin(meta.digits, buf) : null,
    digitsThresholds: meta.digits && meta.digits.thresholds ? meta.digits.thresholds : null,
  };
}

/**
 * Load the forests, returning {static, motion, thresholds, digits, digitsThresholds}.
 *
 * `url` is models.bin / models.bin.gz (then `metaUrl` is models.meta.json, which carries the
 * class lists, the feature tags and the thresholds) or models.json / models.json.gz, which
 * carries all of it in one file. WHICH one is decided by the bytes, not by the file name: a
 * body that opens with models.bin's magic is binary and everything else is JSON. One switch,
 * one prepared model, one walk -- prepareModel and prepareModelBin return the same structure
 * and predictProba below cannot tell them apart.
 *
 * Both fetches start together when `metaUrl` is given. The meta file is 2,726 B (982 B
 * compressed), so on a slow connection the second request costs a round trip and nothing else,
 * and serializing it behind a 1.18 MB body would be the one avoidable stall in the load.
 *
 * `digits` is the numbers-mode forest, prepared when the export carried one and null otherwise
 * (the letter export never depends on the digit pickle); `digitsThresholds` is the override
 * block that mode applies on top of `thresholds` (thresholds.DIGITS_OVERRIDES: VOTE_MARGIN_CLEAR
 * and VOTE_PROB_FLOOR raised), or null with it.
 */
export async function loadModels(url, metaUrl = null) {
  // Kick both off before awaiting either. The .catch keeps a meta failure from surfacing as an
  // unhandled rejection while the body is still in flight; it is rethrown below, and only if
  // the body turns out to be binary, so `loadModels('./models.json', './models.meta.json')`
  // still works with no meta file present.
  const metaPending = metaUrl ? fetchBody(metaUrl).catch((err) => err) : null;
  const buf = await fetchBody(url);

  if (!isModelsBin(buf)) {
    const payload = JSON.parse(new TextDecoder().decode(buf));
    return {
      static: prepareModel(payload.static),
      motion: prepareModel(payload.motion),
      thresholds: payload.thresholds,
      digits: payload.digits ? prepareModel(payload.digits) : null,
      digitsThresholds: payload.digits && payload.digits.thresholds
        ? payload.digits.thresholds : null,
    };
  }

  if (!metaPending) {
    throw new Error(`loadModels: ${url} is a models.bin, which carries the trees only. Pass the `
                    + 'models.meta.json beside it as the second argument');
  }
  const metaBuf = await metaPending;
  if (metaBuf instanceof Error) throw metaBuf;
  return prepareModelsBin(buf, JSON.parse(new TextDecoder().decode(metaBuf)));
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

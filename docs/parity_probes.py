"""The 1,215 probe vectors the precision claims in docs/README.md are measured on.

Run:  ./.venv/bin/python docs/parity_probes.py     (after docs/export_models.py)

Why this file exists: the README has claimed for a while that the exported forests reproduce
sklearn "measured over 1,215 probe vectors -- the real golden features, convex blends of them,
and noisy copies at four scales", and the generator that produced that number was never
committed. A measurement nobody can re-run is a memory, and a memory about a forest outlives
the forest it was taken on. This rebuilds the set to that description, re-measures it, and
writes docs/parity.json -- which carries the models.bin build id it was measured against, so
docs/test_forest.mjs can fail rather than quote a stale figure when the forest moves.

The set, for each forest, is:

  26   the golden static features themselves (the letter cases of docs/golden.json; 15 for the
       digit forest). Real hands, but unanimous ones: every tree agrees, and a probe on which
       every tree agrees cannot detect a split point that moved, because moving one tree's vote
       does not move the answer. They are here because they are what the page actually sees.
  325  every pairwise midpoint of those 26 (105 for the digit forest). Halfway between two
       letters is where the forest is contested and where a nudged threshold shows up.
  864  noisy copies at sigma 1e-4, 1e-3, 1e-2 and 5e-2 over the 351 bases, drawn from
       numpy's default_rng(0) so the set is the same on every run (1,095 for the digit forest,
       which has fewer bases; both forests end at exactly 1,215 probes).

The two export precisions are computed here the way each exporter writes them -- models.json
rounds a leaf to 4 decimals, models.bin stores the leaf's exact integer class counts -- and
then the SHIPPED files are read back and checked to carry exactly those values, so this is a
measurement of what is on disk rather than of what the code meant to put there.

What is NOT re-implemented here is the page's tree walk. Landing on sklearn's leaf is checked
with the reference walk below, which reads sklearn's own tree arrays rather than an exported
file; that forest.js lands where this does is docs/test_forest.mjs's job, and it checks it by
decoding models.bin and models.json and comparing f, t, l and r node by node.
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPORAL = os.path.join(HERE, "..", "temporal")
sys.path.insert(0, os.path.abspath(TEMPORAL))

#: Probe count. This is the number docs/README.md quotes, so it is pinned here and again in
#: docs/test_forest.mjs; the two must move together or the claim stops meaning anything.
PROBE_TOTAL = 1215

#: Noise scales, in feature units. static/v4 is built from palm-normalized coordinates and
#: pairwise distances of order 1, so 1e-4 is below the 6-decimal rounding golden.json stores
#: its landmarks at, and 5e-2 is a visibly different hand shape. Four scales rather than one
#: because a threshold that moved shows up at whichever scale straddles it.
SIGMAS = (1e-4, 1e-3, 1e-2, 5e-2)

#: Seed for the noisy copies. Fixed so re-running this file re-measures the same set: a
#: precision figure that wanders between runs cannot be compared with the one in the README.
SEED = 0


def build_probes(features, total=PROBE_TOTAL, seed=SEED):
    """(X, counts) for one forest's golden feature vectors: bases, midpoints, then noise."""
    base = np.asarray(features, np.float64)
    mids = np.array([0.5 * (base[i] + base[j])
                     for i in range(len(base)) for j in range(i + 1, len(base))])
    bank = np.vstack([base, mids])
    rng = np.random.default_rng(seed)
    need = total - len(bank)
    if need < 0:
        raise SystemExit(f"{len(bank)} bases already exceed the {total}-probe set")
    per, extra = divmod(need, len(SIGMAS))
    noisy = []
    for s_i, s in enumerate(SIGMAS):
        for k in range(per + (1 if s_i < extra else 0)):
            # Walk the bank with a stride coprime to nothing in particular; the point is only
            # that the four scales do not all land on the same handful of bases.
            v = bank[(k * 7 + int(s * 1e5)) % len(bank)]
            noisy.append(v + rng.normal(0, s, v.shape))
    X = np.vstack([bank, np.array(noisy)])
    if len(X) != total:
        raise SystemExit(f"built {len(X)} probes, expected {total}")
    return X, {"bases": len(base), "midpoints": len(mids), "noisy": len(noisy), "total": len(X)}


def walk_float64(tree, X):
    """Leaf index per row, walking sklearn's own arrays in float64.

    sklearn's tree_.apply casts X to float32 first; the page never narrows its feature vector,
    so this is the walk the page performs and the difference between the two is a real one --
    it is what "the walk lands on sklearn's leaf" is actually asserting. `x <= t` goes left, as
    in forest.js and in sklearn.
    """
    feat = tree.feature.astype(np.int64)
    thr = tree.threshold.astype(np.float64)
    left = tree.children_left.astype(np.int64)
    right = tree.children_right.astype(np.int64)
    isleaf = feat < 0
    idx = np.zeros(len(X), np.int64)
    for _ in range(tree.max_depth + 2):
        live = np.flatnonzero(~isleaf[idx])
        if not len(live):
            return idx
        cur = idx[live]
        go_left = X[live, feat[cur]] <= thr[cur]
        idx[live] = np.where(go_left, left[cur], right[cur])
    raise SystemExit("walk_float64 did not terminate; the tree arrays are not a tree")


def leaf_tables(model):
    """(json_f32, bin_f32, counts, json_f64) per tree: the leaf values each export puts on the
    wire, and the ones the page ends up holding.

    json_f64 is what models.json's text carries, a probability rounded to 4 decimals. counts is
    what models.bin carries, the leaf's exact integer class counts. The two _f32 tables are
    those same values after the narrowing forest.js does when it builds its Float32Array leaf
    table, which is the precision the page actually predicts at in either format -- the binary
    one being a count divided by its total, i.e. the rational number sklearn's predict_proba
    averages, limited by nothing but that Float32Array.
    """
    out = []
    for est in model.estimators_:
        t = est.tree_
        val = t.value[:, 0, :].astype(np.float64)
        s = val.sum(axis=1, keepdims=True)
        p = np.divide(val, s, out=np.zeros_like(val), where=s > 0)
        counts = np.rint(val * t.weighted_n_node_samples.astype(np.float64)[:, None]).astype(np.int64)
        tot = counts.sum(axis=1, keepdims=True)
        exact = np.divide(counts, tot, out=np.zeros_like(p), where=tot > 0)
        r4 = np.round(p, 4)
        out.append((r4.astype(np.float32).astype(np.float64),
                    exact.astype(np.float32).astype(np.float64),
                    counts, r4))
    return out


def forest_probs(model, leaves, tables, which):
    """Mean of the per-tree leaf vectors at the given export precision, in float64."""
    acc = None
    for k in range(len(model.estimators_)):
        v = tables[k][which][leaves[:, k]]
        acc = v.copy() if acc is None else acc + v
    return acc / len(model.estimators_)


def check_models_json(payload_block, model, tables, name):
    """The 4-decimal leaf vectors models.json carries must be the ones measured above."""
    worst, cells = 0.0, 0
    for k, tree in enumerate(payload_block["trees"]):
        want = tables[k][3]
        for key, vec in tree["v"].items():
            i = int(key)
            cells += len(vec)
            worst = max(worst, float(np.abs(np.asarray(vec, np.float64) - want[i]).max()))
    if worst > 0:
        raise SystemExit(f"{name}: models.json's leaf vectors differ from the 4-decimal rounding "
                         f"measured here by up to {worst:.3e}; the export changed")
    return cells


BIN_WIDTH = {"n": 4, "bits": 1, "f": 1, "t": 8, "r": 2, "vk": 1, "vi": 1, "vc": 2}
BIN_DTYPE = {"n": "<u4", "bits": "u1", "f": "i1", "t": "<f8", "r": "<u2",
             "vk": "u1", "vi": "u1", "vc": "<u2"}


def check_models_bin(meta_block, raw, model, tables, name):
    """The counts and thresholds models.bin carries must be the pickle's, exactly.

    This reads the sections directly rather than rebuilding a tree walk: what is in question is
    whether the file says what the pickle says, and that is an array comparison. Whether
    forest.js then walks it the same way is checked in docs/test_forest.mjs, which decodes both
    formats and compares f, t, l and r node by node.
    """
    sec = {}
    for nm, (off, ln) in meta_block["sections"].items():
        sec[nm] = np.frombuffer(raw, dtype=BIN_DTYPE[nm], count=ln, offset=off)
    i_off = leaf_i = nz_off = bit_off = 0
    nnz = 0
    for k, est in enumerate(model.estimators_):
        t = est.tree_
        n = int(t.node_count)
        feat = t.feature.astype(np.int64)
        isleaf = feat < 0
        internal = np.flatnonzero(~isleaf)
        leaves = np.flatnonzero(isleaf)
        bits = np.unpackbits(sec["bits"][bit_off:bit_off + (n + 7) // 8])[:n].astype(bool)
        if not np.array_equal(bits, isleaf):
            raise SystemExit(f"{name} tree {k}: models.bin's leaf bitmap is not the pickle's")
        if not np.array_equal(sec["f"][i_off:i_off + len(internal)].astype(np.int64), feat[internal]):
            raise SystemExit(f"{name} tree {k}: models.bin's split features are not the pickle's")
        if not np.array_equal(sec["t"][i_off:i_off + len(internal)],
                              t.threshold.astype(np.float64)[internal]):
            raise SystemExit(f"{name} tree {k}: models.bin's thresholds are not bit-exact")
        rr = sec["r"][i_off:i_off + len(internal)].astype(np.int64) + internal
        if not np.array_equal(rr, t.children_right.astype(np.int64)[internal]):
            raise SystemExit(f"{name} tree {k}: models.bin's right children are not the pickle's")
        counts = tables[k][2][leaves]
        vk = sec["vk"][leaf_i:leaf_i + len(leaves)].astype(np.int64)
        if not np.array_equal(vk, (counts > 0).sum(axis=1)):
            raise SystemExit(f"{name} tree {k}: models.bin's nonzero-class counts are wrong")
        k_nnz = int(vk.sum())
        got = np.zeros_like(counts)
        rows = np.repeat(np.arange(len(leaves)), vk)
        got[rows, sec["vi"][nz_off:nz_off + k_nnz].astype(np.int64)] = \
            sec["vc"][nz_off:nz_off + k_nnz].astype(np.int64)
        if not np.array_equal(got, counts):
            raise SystemExit(f"{name} tree {k}: models.bin's leaf class counts are not the "
                             "pickle's integer counts")
        nnz += k_nnz
        i_off += len(internal); leaf_i += len(leaves); nz_off += k_nnz
        bit_off += (n + 7) // 8
    return nnz


def measure(name, pkl, cases, payload, meta, raw):
    model = pickle.load(open(os.path.join(TEMPORAL, pkl), "rb"))["model"]
    model.n_jobs = 1
    X, counts = build_probes([c["static_feature"] for c in cases])

    ref = model.predict_proba(X)                                    # full precision, sklearn
    ref_leaf = np.stack([e.tree_.apply(X.astype(np.float32)) for e in model.estimators_], axis=1)
    got_leaf = np.stack([walk_float64(e.tree_, X) for e in model.estimators_], axis=1)
    bad_leaf = int((got_leaf != ref_leaf).sum())

    tables = leaf_tables(model)
    cells = check_models_json(payload[name], model, tables, name)
    nnz = check_models_bin(meta[name], raw, model, tables, name)

    row = {"probes": counts, "trees": len(model.estimators_),
           "leaf_visits": int(ref_leaf.size), "leaf_disagreements": bad_leaf,
           "json_leaf_cells": cells, "bin_leaf_nonzeros": nnz}
    for tag, which in (("json", 0), ("bin", 1)):
        got = forest_probs(model, got_leaf, tables, which)
        row[tag] = {"max_abs_dp": float(np.abs(got - ref).max()),
                    "argmax_changes": int((got.argmax(1) != ref.argmax(1)).sum())}
    print(f"{name:8s} {counts['total']} probes ({counts['bases']} golden + "
          f"{counts['midpoints']} midpoints + {counts['noisy']} noisy), "
          f"{row['trees']} trees")
    print(f"         leaf disagreements vs sklearn's own apply(): {bad_leaf}/{ref_leaf.size}")
    for tag in ("json", "bin"):
        print(f"         {tag:4s} max|dp| {row[tag]['max_abs_dp']:.3e}   argmax changes "
              f"{row[tag]['argmax_changes']}/{counts['total']}")
    return row


def main():
    golden = json.load(open(os.path.join(HERE, "golden.json")))
    payload = json.load(open(os.path.join(HERE, "models.json")))
    meta = json.load(open(os.path.join(HERE, "models.meta.json")))
    raw = open(os.path.join(HERE, "models.bin"), "rb").read()
    if meta["build"] != hashlib.sha256(raw[40:]).digest()[:16].hex():
        raise SystemExit("models.meta.json and models.bin are from different exports; re-run "
                         "docs/export_models.py before measuring")

    by_model = {}
    for c in golden["cases"]:
        by_model.setdefault(c.get("model", "static"), []).append(c)

    out = {"probe_total": PROBE_TOTAL, "sigmas": list(SIGMAS), "seed": SEED,
           "build": meta["build"], "forests": {}}
    out["forests"]["static"] = measure("static", "model_static.p", by_model["static"],
                                       payload, meta, raw)
    if "digits" in meta and "digits" in by_model:
        out["forests"]["digits"] = measure("digits", "model_digits.p", by_model["digits"],
                                           payload, meta, raw)
    # The motion forest has no golden cases -- its inputs are 79-D event features read off a
    # whole track, not a single frame -- so there is no probe to build for it here. Its binary
    # encoding is still checked against its own pickle, section by section, and that forest.js
    # walks it the same way the models.json path does is docs/test_forest.mjs's job.
    motion = pickle.load(open(os.path.join(TEMPORAL, "model_motion.p"), "rb"))["model"]
    m_nnz = check_models_bin(meta["motion"], raw, motion, leaf_tables(motion), "motion")
    out["forests"]["motion"] = {"probes": None, "trees": len(motion.estimators_),
                                "bin_leaf_nonzeros": m_nnz}
    print(f"motion   no probes (no golden cases for 79-D event features); models.bin's "
          f"{len(motion.estimators_)} trees match the pickle section by section, "
          f"{m_nnz} nonzero leaf counts")
    path = os.path.join(HERE, "parity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(path, os.path.dirname(HERE))} for build {out['build'][:16]} "
          f"({os.path.getsize(path)} B). docs/test_forest.mjs fails if this build id stops "
          "matching docs/models.meta.json.")


if __name__ == "__main__":
    main()

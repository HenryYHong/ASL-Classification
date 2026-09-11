"""Export the trained forests and a set of golden vectors for the browser port.

Two outputs:

  models.json    both forests as flat typed arrays, gzip-friendly. A sklearn pickle cannot run
                 in a browser -- it is a Python object graph -- so the trees are flattened to
                 the four arrays a tree walk actually needs.

  golden.json    inputs paired with the exact outputs Python produces for them. The browser port
                 is checked against these rather than against a reading of the code, because the
                 expensive failures in this project have all been silent divergences between two
                 implementations that looked equivalent: training mirrored while inference did
                 not, a vote window that completed at one frame rate and never at another.

Run:  ../.venv/bin/python web/export_models.py
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPORAL = os.path.join(HERE, "..", "temporal")
sys.path.insert(0, TEMPORAL)

import features as F            # noqa: E402
from thresholds import DEFAULT  # noqa: E402


def flatten(forest):
    """A forest as the four arrays a tree walk needs, plus leaf class distributions.

    Internal node i: feature[i], threshold[i], left[i], right[i]. Leaves carry -1 for feature
    and an index into `leaves`. sklearn's rule is `x[feature] <= threshold -> left`, which the
    port must match exactly; an inverted comparison is the kind of error that still classifies
    most inputs correctly and quietly ruins the rest.
    """
    trees = []
    for est in forest.estimators_:
        t = est.tree_
        feat = t.feature.astype(int).tolist()
        # Thresholds are NOT rounded. Rounding to 5 decimals moved a split point past a
        # feature value on roughly one probe vector in a thousand, so the exported forest
        # stopped reproducing sklearn on arbitrary input while still matching every golden
        # case -- the unanimous cases are exactly the ones a nudged threshold cannot flip.
        # repr() round-trips a float64 exactly and costs about 40% more bytes before gzip.
        thr = [float(v) for v in t.threshold]
        left = t.children_left.astype(int).tolist()
        right = t.children_right.astype(int).tolist()
        leaves = {}
        for i, f in enumerate(feat):
            if f < 0:
                v = t.value[i][0]
                s = v.sum()
                leaves[i] = [round(float(x / s), 4) for x in v] if s > 0 else [0.0] * len(v)
        trees.append({"f": feat, "t": thr, "l": left, "r": right,
                      "v": {str(k): v for k, v in leaves.items()}})
    return trees


def main():
    static = pickle.load(open(os.path.join(TEMPORAL, "model_static.p"), "rb"))
    motion = pickle.load(open(os.path.join(TEMPORAL, "model_motion.p"), "rb"))

    payload = {
        "static": {"classes": list(static["classes"]), "feature": static["feature"],
                   "dim": int(static["model"].n_features_in_),
                   "trees": flatten(static["model"])},
        "motion": {"classes": list(motion["classes"]), "feature": motion["feature"],
                   "dim": int(motion["model"].n_features_in_),
                   "trees": flatten(motion["model"])},
        "thresholds": {k: v for k, v in vars(DEFAULT).items()} if vars(DEFAULT) else {},
    }
    # dataclass instances expose fields via __dataclass_fields__, not __dict__ on the class
    from dataclasses import asdict
    payload["thresholds"] = asdict(DEFAULT)

    out = os.path.join(HERE, "models.json")
    raw = json.dumps(payload, separators=(",", ":"))
    open(out, "w").write(raw)
    with gzip.open(out + ".gz", "wt") as fh:
        fh.write(raw)
    print(f"models.json      {len(raw)/1e6:6.2f} MB")
    print(f"models.json.gz   {os.path.getsize(out + '.gz')/1e6:6.2f} MB")

    # ---- golden vectors -------------------------------------------------------------
    rng = np.random.default_rng(0)
    seq = np.load(os.path.join(TEMPORAL, "static_sequences.npz"))
    lm, found = seq["lm"], seq["found"]
    cases = []
    for c in (0, 3, 8, 12, 20):                     # A, D, I, N, V
        idx = np.where(found[c])[0][:3]
        for i in idx:
            # Round FIRST, then compute. Storing rounded landmarks beside outputs derived
            # from the unrounded originals makes the file's own stated tolerance unreachable:
            # re-running Python on golden.json's landmarks missed its stored values by 1.9e-5
            # against a claimed 1e-6, which reads as a port bug in any language.
            raw_lm = np.round(lm[c, i], 6)
            P = F.to_isotropic(raw_lm[None, :, :2], 1920, 1080)
            P = F.canonicalize_handedness(P, "Left")
            feat = F.static_feature(P)[0]
            probs = static["model"].predict_proba(feat.reshape(1, -1))[0]
            cases.append({
                "landmarks": [[round(float(v), 6) for v in p] for p in raw_lm[:, :3]],
                "handedness": "Left", "width": 1920, "height": 1080,
                "shape42": [round(float(v), 6) for v in F.shape42(P)[0]],
                "static_feature": [round(float(v), 6) for v in feat],
                "palm_scale": round(float(F.palm_scale(P)[0]), 6),
                "j_gate": bool(F.j_gate(P)[0]), "z_gate": bool(F.z_gate(P)[0]),
                "static_probs": [round(float(v), 6) for v in probs],
                "predicted": static["classes"][int(probs.argmax())],
            })
    golden = {"note": "browser port must reproduce every field to the stated tolerance",
              "tolerance": {"features": 1e-6, "probabilities": 1e-4},
              "cases": cases}
    gout = os.path.join(HERE, "golden.json")
    json.dump(golden, open(gout, "w"), indent=1)
    print(f"golden.json      {os.path.getsize(gout)/1e3:6.1f} KB, {len(cases)} cases")


if __name__ == "__main__":
    main()

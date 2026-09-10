"""Retrain the 24-class static-letter forest on the palm-normalized feature.

Also runs the ablation that justifies the change. The original 42-D feature subtracts
min(x)/min(y) but never divides by hand size, so it encodes how close the hand was to the
camera. That inflates accuracy on a single-session benchmark -- where camera distance is a
free label -- and degrades in use. Dividing by the palm triangle removes it.

Read the ablation table before trusting either number: the palm-normalized feature scores
LOWER on the leaky split, because removing scale removes a cue the old forest was using to
re-identify frames from one capture burst. That is the fix working, not the fix failing.
"""
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "static_sequences.npz")
OUT = os.path.join(HERE, "model_static.p")
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
W, H = 1920, 1080


def legacy_feature(P):
    """The feature the committed model.p uses: translation-normalized, NOT scale-normalized."""
    x, y = P[..., 0], P[..., 1]
    out = np.empty((*P.shape[:-2], 42))
    out[..., 0::2] = x - x.min(axis=-1, keepdims=True)
    out[..., 1::2] = y - y.min(axis=-1, keepdims=True)
    return out


def load(aspect=True):
    d = np.load(SEQ)
    lm, found = d["lm"], d["found"]
    per_class = []
    for c in range(24):
        idx = np.where(found[c])[0]          # already in numeric frame order
        P = lm[c][idx][:, :, :2]
        P = F.to_isotropic(P, W, H) if aspect else np.asarray(P, dtype=np.float64)
        per_class.append(P)
    return per_class


def contiguous_split(per_class, frac=0.8):
    """Per class, the first `frac` of the burst trains and the tail tests.

    This is still a within-session split, so it does NOT measure generalization to a new
    signer, room or day -- consecutive frames of one held sign are near-duplicates either way.
    It is reported because it is comparable to the number the notebooks reported, not because
    it means what an accuracy usually means.
    """
    tr_i, te_i = [], []
    for c, P in enumerate(per_class):
        k = int(len(P) * frac)
        tr_i.append((c, P[:k]))
        te_i.append((c, P[k:]))
    return tr_i, te_i


def build(split, featfn, rescale=1.0):
    X, y = [], []
    for c, P in split:
        if rescale != 1.0:
            centre = P.mean(axis=(0, 1), keepdims=True)
            P = centre + (P - centre) * rescale
        f = featfn(P)
        X.append(f)
        y.append(np.full(len(f), c))
    return np.concatenate(X), np.concatenate(y)


def main():
    per_class = load()
    tr, te = contiguous_split(per_class)

    print("=== scale-robustness ablation (contiguous per-class split) ===")
    print("test landmarks rescaled about the hand centroid; training never sees the rescale\n")
    print(f"{'k':>6}  {'legacy 42-D':>12}  {'palm-normalized':>16}")
    rows = {}
    for name, fn in (("legacy", legacy_feature), ("palm", F.shape42)):
        Xtr, ytr = build(tr, fn)
        model = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
        rows[name] = []
        for k in (0.7, 0.85, 1.0, 1.2, 1.5):
            Xte, yte = build(te, fn, rescale=k)
            rows[name].append(accuracy_score(yte, model.predict(Xte)))
    for i, k in enumerate((0.7, 0.85, 1.0, 1.2, 1.5)):
        print(f"{k:>6.2f}  {rows['legacy'][i]:>12.3f}  {rows['palm'][i]:>16.3f}")
    print(f"\nlegacy spread across k: {max(rows['legacy']) - min(rows['legacy']):.3f}"
          f"   palm-normalized spread: {max(rows['palm']) - min(rows['palm']):.3f}")

    # Ship a model trained on everything; the split above exists to characterize, not to select.
    Xall, yall = build([(c, P) for c, P in enumerate(per_class)], F.shape42)
    model = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xall, yall)
    with open(OUT, "wb") as fh:
        pickle.dump({"model": model, "classes": LETTERS, "feature": "shape42/v1",
                     "aspect": [W, H], "n_train": len(Xall)}, fh)
    print(f"\nwrote {OUT}: {len(Xall)} frames, {len(LETTERS)} classes, feature shape42/v1")


if __name__ == "__main__":
    main()

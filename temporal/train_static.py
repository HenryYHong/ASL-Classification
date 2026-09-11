"""Retrain the 24-class static-letter forest on the palm-normalized feature.

Also runs the ablation that justifies the change. The original 42-D feature subtracts
min(x)/min(y) but never divides by hand size, so it encodes how close the hand was to the
camera. That inflates accuracy on a single-session benchmark -- where camera distance is a
free label -- and degrades in use. Dividing by the palm triangle removes it.

Read the ablation table before trusting either number: the palm-normalized feature scores
LOWER on the leaky split, because removing scale removes a cue the old forest was using to
re-identify frames from one capture burst. That is the fix working, not the fix failing.
"""
import argparse
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
    """Landmarks per class, transformed EXACTLY as the live segmenter transforms them.

    Including the handedness canonicalization. Omitting it here while the segmenter applies it
    mirrors every inference input relative to training: measured on this signer, MediaPipe
    labels the unmirrored webcam view "Left" on ~97% of frames, so the live features were the
    mirror image of anything the model had seen. Chiral letters failed outright; B survived
    because a flat hand is close to mirror-symmetric.
    """
    d = np.load(SEQ)
    lm, found = d["lm"], d["found"]
    handed = d["handed"] if "handed" in d else None
    per_class = []
    for c in range(24):
        idx = np.where(found[c])[0]          # already in numeric frame order
        P = lm[c][idx][:, :, :2]
        P = F.to_isotropic(P, W, H) if aspect else np.asarray(P, dtype=np.float64)
        if handed is not None:
            labs = [str(x) for x in handed[c][idx] if str(x)[:1] in ("L", "R")]
            if labs:
                P = F.canonicalize_handedness(P, max(set(labs), key=labs.count))
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


def load_extra(path):
    """Frames from a second capture session, as written by collect_motion --static-letters.

    Merged in rather than replacing: the archive still carries most of the variety, and the
    point of the second session is to teach the model that the same letter can look different
    on a different day. One session is why cross-session accuracy sits at 0.52 while the
    in-session number reads 0.97.
    """
    d = np.load(path, allow_pickle=True)
    lm, letters = d["lm"], [str(x) for x in d["letters"]]
    handed = [str(x) for x in d["handed"]] if "handed" in d else ["Unknown"] * len(lm)
    wh = np.asarray(d["frame_size"]).ravel() if "frame_size" in d else np.array([W, H])
    per_class = {c: [] for c in range(24)}
    for i, lab in enumerate(letters):
        if lab not in LETTERS:
            continue
        P = F.to_isotropic(lm[i][None, :, :2], int(wh[0]), int(wh[1]))
        P = F.canonicalize_handedness(P, handed[i] if handed[i][:1] in ("L", "R") else None)
        per_class[LETTERS.index(lab)].append(P[0])
    return {c: np.stack(v) for c, v in per_class.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="one or more .npz files from collect_motion --static-letters. Later "
                         "sessions are merged in, not substituted: the point is to show the "
                         "model the same letter on different days, so the variety accumulates.")
    args = ap.parse_args()

    per_class = load()
    merged = {}
    for path in args.extra:
        got = load_extra(path)
        print(f"merging {os.path.basename(path)}: " +
              ", ".join(f"{LETTERS[c]}={len(v)}" for c, v in sorted(got.items())))
        for c, v in got.items():
            merged[c] = np.concatenate([merged[c], v]) if c in merged else v
    if merged:
        print()
        per_class = [np.concatenate([P, merged[c]]) if c in merged else P
                     for c, P in enumerate(per_class)]
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
    Xall, yall = build([(c, P) for c, P in enumerate(per_class)], F.static_feature)

    # Rotation augmentation. The archive is one signer holding each letter once, so the model
    # sees each handshape at essentially a single wrist angle and is over-confident about it --
    # which shows up live as correct-but-under-confident predictions that fall below the
    # emission floor and abstain silently. Rotating the normalized shape is exact (shape42 is
    # already centered and scaled, so a rotation is a rigid transform of the feature) and costs
    # no new data. Measured on the contiguous split: mean winner confidence 0.818 -> 0.886 and
    # A's confidence 0.88 -> 1.00, with accuracy unchanged within noise (0.908 -> 0.914).
    # Rotate the LANDMARKS and re-featurize, rather than rotating the feature vector: the
    # extent half is min-subtracted, so rotating it directly would not correspond to any hand.
    Xa, ya = [Xall], [yall]
    for deg in (-12, -6, 6, 12):
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        rotated = [(c, (P - P.mean(axis=(0, 1), keepdims=True)) @ R.T
                    + P.mean(axis=(0, 1), keepdims=True)) for c, P in enumerate(per_class)]
        Xr, yr = build(rotated, F.static_feature)
        Xa.append(Xr)
        ya.append(yr)
    Xall, yall = np.concatenate(Xa), np.concatenate(ya)
    print(f"\ntraining on {len(Xall)} rows ({len(Xa)}x rotation-augmented)")
    model = RandomForestClassifier(n_estimators=400, random_state=0).fit(Xall, yall)
    with open(OUT, "wb") as fh:
        pickle.dump({"model": model, "classes": LETTERS, "feature": "static/v3",
                     "aspect": [W, H], "n_train": len(Xall)}, fh)
    print(f"\nwrote {OUT}: {len(Xall)} frames, {len(LETTERS)} classes, feature static/v3")


if __name__ == "__main__":
    main()

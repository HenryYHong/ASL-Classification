"""Leave-one-session-out accuracy for the static branch -- the only number worth publishing.

train_static.py characterizes the feature and ships a model; it never measures generalization,
because every split it takes is inside one capture burst. This file is the measurement, and it
exists as a committed script rather than a remembered figure so the README's number can be
re-derived after the next recording session.

Protocol: hold out one session, train on the rest, test on every frame of the held-out session.
Rotation augmentation is applied to TRAINING folds only -- augmenting the test fold would score
the model on copies of frames it is already being scored on.

Read the per-fold table, not just the mean. The four sessions do not cover the same letters: S1
is the full 24-letter archive, S2 is 23 letters, and S3 and S4 are targeted re-recordings of 6
and 4 letters. A fold that tests four well-separated letters scores near 1.00 and says nothing
about the alphabet, so an unweighted mean over folds flatters the model -- the same error this
repository's README criticizes in the original 100.00% (a letter with no test samples cannot be
got wrong). The pooled figure -- every held-out frame across all folds, counted once -- is the
headline, and the fold that holds out S1 is the only one that tests all 24 letters.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import RandomForestClassifier

import features as F
import train_static as T

EXTRA = (("S2", "static_s2.npz"), ("S3", "static_s3.npz"), ("S4", "static_s4.npz"))
HERE = os.path.dirname(os.path.abspath(__file__))
NC = len(T.LETTERS)
EMPTY = np.zeros((0, 21, 2))


def sessions():
    """{tag: [per-class landmark arrays]} for every capture session on disk."""
    out = {"S1": T.load()}
    for tag, name in EXTRA:
        got = T.load_extra(os.path.join(HERE, name))
        out[tag] = [got.get(c, EMPTY) for c in range(NC)]
    return out


def rows(per_class, augment):
    pairs = [(c, P) for c, P in enumerate(per_class) if len(P)]
    X, y = T.build(pairs, F.static_feature)
    Xs, ys = [X], [y]
    if augment:
        # Same four angles train_static.py ships with: the archive holds each letter at one
        # wrist angle, and the model is over-confident about it without them.
        for deg in (-12, -6, 6, 12):
            a = np.radians(deg)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            mid = [(c, (P - P.mean(axis=(0, 1), keepdims=True)) @ R.T
                    + P.mean(axis=(0, 1), keepdims=True)) for c, P in pairs]
            Xr, yr = T.build(mid, F.static_feature)
            Xs.append(Xr)
            ys.append(yr)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    src = sessions()
    print(f"{'held out':>9}  {'letters':>7}  {'frames':>6}  {'accuracy':>8}")
    correct = total = 0
    per_fold = []
    for held in src:
        train = [np.concatenate([src[s][c] for s in src if s != held and len(src[s][c])])
                 if any(len(src[s][c]) for s in src if s != held) else EMPTY
                 for c in range(NC)]
        Xtr, ytr = rows(train, True)
        Xte, yte = rows(src[held], False)
        if not len(Xte):
            continue
        model = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        hit = int((model.predict(Xte) == yte).sum())
        correct += hit
        total += len(yte)
        per_fold.append(hit / len(yte))
        print(f"{held:>9}  {len(set(yte)):>7}  {len(yte):>6}  {hit / len(yte):>8.3f}")

    print(f"\npooled over every held-out frame: {correct / total:.3f}  ({correct}/{total})")
    print(f"unweighted mean over folds:       {np.mean(per_fold):.3f}  <- flattered by the "
          f"small folds; do not publish this one")


if __name__ == "__main__":
    main()

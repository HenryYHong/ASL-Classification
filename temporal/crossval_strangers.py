"""The cross-signer numbers: other people's hands, each set scored with itself held out.

Everything in crossval_static.py holds out one of the author's sessions. This script holds
out the strangers (strangers.py) instead, and prints the three numbers the README quotes for
"a visitor is not the author":

  1. ASLNow held out    train on the author's four sessions + the 218-signer O/V/W/F photos,
                        test on the 1,874 ASLNow letter records (24 letters, multiple
                        participants, the browser's own landmarker). Nothing in the training
                        set comes from that dataset. This is the honest "a dataset the forest
                        never saw" figure, and the per-letter table beside it is where the
                        one-signer forest's habits show (V read as U, X as D, S as A).
  2. Ankara held out    train on the author's sessions + ASLNow, test on the 218 signers'
                        O/V/W/F photos. Four letters only, but 218 different hands.
  3. ASLNow fifths      train on the author + Ankara + four fifths of ASLNow, test on the
                        fifth, five times. ASLNow carries no participant id, so a record's
                        signer may be in the training fifths: this is "a new capture of
                        possibly the same people", an upper bound, and it is labeled so.

The shipped forest trains on all of it, so none of the three is a measurement of the shipped
forest on a dataset it never saw; they are the same recipe with each stranger set held out in
turn, which is the closest honest thing. A vote gate line is printed too: single-frame
emission rate and precision under thresholds.py's DEFAULT, so the effect of the floor on
strangers is visible next to the accuracy.

Also printed: the same three numbers for the author-only recipe (--henry-only), so the gain
from training on strangers is measured, not asserted.

    ./.venv/bin/python temporal/crossval_strangers.py            # seed 0
    ./.venv/bin/python temporal/crossval_strangers.py --seeds 3
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossval_static as CV  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

LETTERS = T.LETTERS
NC = len(LETTERS)


def fit(per_class, seed, ruleset=None, n_jobs=4):
    X, y, _ = T.training_rows(per_class, ruleset=T.RULESET if ruleset is None else ruleset, seed=seed)
    return T.make_forest(seed, n_jobs=n_jobs).fit(X, y)


def proba(model, X):
    out = np.zeros((len(X), NC))
    out[:, model.classes_.astype(int)] = model.predict_proba(X)
    return out


def gate_line(pr, y, th=DEFAULT):
    """Single-frame emission under the static vote rule (a 4-frame vote of identical frames)."""
    srt = np.sort(pr, axis=1)
    meanp, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    emit = (margin >= th.VOTE_MARGIN) & ((meanp >= th.VOTE_PROB) |
                                         ((margin >= th.VOTE_MARGIN_CLEAR) & (meanp >= th.VOTE_PROB_FLOOR)))
    ok = pr.argmax(1) == y
    return emit.mean(), ok[emit].mean() if emit.any() else float("nan")


def run(seed=0, henry_only=False, n_jobs=4, verbose=True):
    src = CV.sessions()
    henry = ST.merge(*[src[s][0] for s in src])
    ank_per, ank_P, ank_L, _ = ST.load_ankara_letters()
    asl_per, asl_P, asl_L, _ = ST.load_aslnow()
    asl_y = np.array([LETTERS.index(L) for L in asl_L])
    ank_y = np.array([LETTERS.index(L) for L in ank_L])
    res = {"seed": seed}
    t0 = time.time()

    # 1. ASLNow held out
    train = henry if henry_only else ST.merge(henry, ank_per)
    m = fit(train, seed, n_jobs=n_jobs)
    pr = proba(m, T.FEATFN(asl_P))
    pred = pr.argmax(1)
    res["aslnow"] = (pred == asl_y).mean()
    res["aslnow_per_letter"] = {L: (pred[asl_y == c] == c).mean() for c, L in enumerate(LETTERS)}
    res["aslnow_confusions"] = {}
    for c, L in enumerate(LETTERS):
        wrong = pred[(asl_y == c) & (pred != c)]
        if len(wrong):
            top = np.bincount(wrong, minlength=NC).argmax()
            res["aslnow_confusions"][L] = (LETTERS[top], int((wrong == top).sum()))
    res["aslnow_gate"] = gate_line(pr, asl_y)

    # 2. Ankara held out
    train = henry if henry_only else ST.merge(henry, asl_per)
    m = fit(train, seed, n_jobs=n_jobs)
    pred = proba(m, T.FEATFN(ank_P)).argmax(1)
    res["ankara"] = (pred == ank_y).mean()
    res["ankara_per_letter"] = {L: (pred[ank_y == LETTERS.index(L)] == LETTERS.index(L)).mean()
                                for L in sorted(set(ank_L))}

    # 3. ASLNow fifths (participants may repeat across fifths: an upper bound)
    if henry_only:
        res["fifths"] = float("nan")
    else:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(asl_P))
        folds = np.array_split(idx, 5)
        hits = 0
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            per_tr = [asl_P[tr][asl_y[tr] == c] if (asl_y[tr] == c).any() else ST.EMPTY for c in range(NC)]
            m = fit(ST.merge(henry, ank_per, per_tr), seed, n_jobs=n_jobs)
            hits += int((proba(m, T.FEATFN(asl_P[te])).argmax(1) == asl_y[te]).sum())
        res["fifths"] = hits / len(asl_P)
    res["time"] = time.time() - t0
    if verbose:
        report(res, henry_only)
    return res


def report(res, henry_only):
    tag = "author only" if henry_only else "author + the other stranger set"
    print(f"training side: {tag}; seed {res['seed']}")
    print(f"  1. ASLNow held out (1,874 records, 24 letters, browser landmarker): {res['aslnow']:.3f}")
    worst = sorted(res["aslnow_per_letter"].items(), key=lambda kv: kv[1])[:8]
    print("     weakest letters: " + ", ".join(
        f"{L} {v:.2f}" + (f" (->{res['aslnow_confusions'][L][0]} {res['aslnow_confusions'][L][1]})"
                          if L in res["aslnow_confusions"] else "") for L, v in worst))
    e, p = res["aslnow_gate"]
    print(f"     single-frame vote gate at DEFAULT: emits on {e:.2f} of records, right on {p:.3f} of those")
    print(f"  2. Ankara O/V/W/F held out (218 signers): {res['ankara']:.3f}  "
          + " ".join(f"{L} {v:.2f}" for L, v in res["ankara_per_letter"].items()))
    if not henry_only:
        print(f"  3. ASLNow fifths (a new capture of possibly the same people; upper bound): {res['fifths']:.3f}")
    print(f"  ({res['time']:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-henry-only", action="store_true", help="omit the author-only comparison rows")
    args = ap.parse_args()
    print(f"recipe: {T.FEATURE_TAG}, filter {T.RULESET!r}, jitter {T.JITTER_SIGMA} x{T.JITTER_COPIES}, "
          f"RF({T.FOREST['n_estimators']}, min_samples_leaf={T.FOREST['min_samples_leaf']})\n")
    runs = []
    for seed in range(args.seeds):
        runs.append(run(seed, n_jobs=args.n_jobs))
        print()
    if not args.skip_henry_only:
        print("for comparison, the previous release's training set (the author only):")
        run(0, henry_only=True, n_jobs=args.n_jobs)
        print()
    if args.seeds > 1:
        for key, name in (("aslnow", "ASLNow held out"), ("ankara", "Ankara held out"), ("fifths", "ASLNow fifths")):
            v = [r[key] for r in runs]
            print(f"over {args.seeds} seeds: {name} {np.mean(v):.3f} +- {np.std(v):.3f} (range {min(v):.3f}-{max(v):.3f})")


if __name__ == "__main__":
    main()

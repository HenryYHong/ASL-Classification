"""The cross-signer numbers: other people's hands, each set scored with itself held out.

Everything in crossval_static.py holds out one of the author's sessions. This script holds
out the strangers (strangers.py) instead, and prints the three numbers the README quotes for
"a visitor is not the author":

  1. ASLNow held out    train on the author's capped sessions + the 218-signer O/V/W/F photos
                        + ASL-HG at the shipped cap, test on the 1,874 ASLNow letter records
                        (24 letters, multiple participants, the browser's own landmarker).
                        Nothing in the training set comes from that dataset. This is the honest
                        "a dataset the forest never saw" figure, and the per-letter table
                        beside it is where the one-signer forest's habits show (V read as U, X
                        as D, S as A).
  2. Ankara held out    train on the author's sessions + ASLNow + ASL-HG, test on the 218
                        signers' O/V/W/F photos. Four letters only, but 218 different hands.
  3. ASLNow fifths      train on the author + Ankara + ASL-HG + four fifths of ASLNow, test on
                        the fifth, five times. ASLNow carries no participant id, so a record's
                        signer may be in the training fifths: this is "a new capture of
                        possibly the same people", an upper bound, and it is labeled so.
  4. ayuraj             the COMMITTED forest scored on the one set nothing trains on, ever
                        (strangers.NEVER_TRAIN). Not a re-fit and not a fold: the pickle that
                        ships, on 1,111 letter frames from 5 people it has never seen. It is
                        the only row here that measures the shipped model rather than a
                        stand-in for it, which is the whole reason that set is kept out. The
                        previous release's pickle reads 0.797 on it and this one 0.889, the
                        same 1,111 frames scored the same way (ayuraj_report on each).

ASL-HG is not held out here as a set: it carries real signer ids, so it gets the stronger
protocol of its own in crossval_signers.py -- one of its ten people held out at a time, all
2,400 of that person's frames scored, none of them anywhere in the training side. And ayuraj
is not scored here at all, because it is never trained on by anything (strangers.NEVER_TRAIN):
it is the one set whose number is a measurement of the shipped forest and not of a re-fit.

The shipped forest trains on all of it, so none of the three below is a measurement of the
shipped forest on a dataset it never saw; they are the same recipe with each stranger set held
out in turn, which is the closest honest thing. A vote gate line is printed too: single-frame
emission rate and precision under thresholds.py's DEFAULT, so the effect of the floor on
strangers is visible next to the accuracy.

Both caps are applied to the training side of every arm and to no test side: the held-out set
is always scored whole (1,874 ASLNow records, 687 Ankara letter frames), so two rows of the
table share a denominator.

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
import static_aug as A  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

LETTERS = T.LETTERS
NC = len(LETTERS)


def fit(per_class, seed, ruleset=None, n_jobs=4):
    X, y, _ = T.training_rows(per_class, ruleset=T.RULESET if ruleset is None else ruleset, seed=seed)
    return T.fit_letters(X, y, seed, n_jobs)


def proba(model, X):
    out = np.zeros((len(X), NC))
    out[:, model.classes_.astype(int)] = model.predict_proba(X)
    return out


def per_letter_floors(th=DEFAULT):
    """(prob, floor) arrays indexed by class, so a vectorized gate can read the per-letter
    thresholds (thresholds.VOTE_PROB_LETTER) the segmenter reads one vote at a time."""
    prob = np.array([th.vote_prob_for(L) for L in LETTERS])
    floor = np.array([th.vote_floor_for(L) for L in LETTERS])
    return prob, floor


def gate_line(pr, y, th=DEFAULT):
    """Single-frame emission under the static vote rule (a 4-frame vote of identical frames)."""
    srt = np.sort(pr, axis=1)
    meanp, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    prob, floor = per_letter_floors(th)
    win = pr.argmax(1)
    emit = (margin >= th.VOTE_MARGIN) & ((meanp >= prob[win]) |
                                         ((margin >= th.VOTE_MARGIN_CLEAR) & (meanp >= floor[win])))
    ok = win == y
    return emit.mean(), ok[emit].mean() if emit.any() else float("nan")


def run(seed=0, henry_only=False, n_jobs=4, verbose=True, author_cap=None, aslhg_cap=None,
        fifths=True):
    src = CV.sessions()
    author_cap = T.AUTHOR_CAP if author_cap is None else author_cap
    aslhg_cap = ST.ASLHG_CAP if aslhg_cap is None else aslhg_cap
    henry = ST.merge(*[src[s][0] for s in src])
    # The shipped caps, on the training side of every arm and on no test side: the held-out
    # set is always scored whole (1,874 ASLNow records, 687 Ankara letter frames).
    if author_cap:
        henry = A.cap_per_letter(henry, author_cap)
    ank_per, ank_P, ank_L, _ = ST.load_ankara_letters()
    asl_per, asl_P, asl_L, _ = ST.load_aslnow()
    hg_per = [] if henry_only else [ST.load_aslhg(cap=aslhg_cap or None)[0]]
    asl_y = np.array([LETTERS.index(L) for L in asl_L])
    ank_y = np.array([LETTERS.index(L) for L in ank_L])
    res = {"seed": seed, "author_cap": author_cap, "aslhg_cap": aslhg_cap}
    t0 = time.time()

    # 1. ASLNow held out
    train = henry if henry_only else ST.merge(henry, ank_per, *hg_per)
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
    train = henry if henry_only else ST.merge(henry, asl_per, *hg_per)
    m = fit(train, seed, n_jobs=n_jobs)
    pred = proba(m, T.FEATFN(ank_P)).argmax(1)
    res["ankara"] = (pred == ank_y).mean()
    res["ankara_per_letter"] = {L: (pred[ank_y == LETTERS.index(L)] == LETTERS.index(L)).mean()
                                for L in sorted(set(ank_L))}

    # 3. ASLNow fifths (participants may repeat across fifths: an upper bound). Five more
    # fits; a candidate sweep that only needs the two held-out numbers passes fifths=False.
    if henry_only or not fifths:
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
            m = fit(ST.merge(henry, ank_per, *hg_per, per_tr), seed, n_jobs=n_jobs)
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
    if not henry_only and not np.isnan(res["fifths"]):
        print(f"  3. ASLNow fifths (a new capture of possibly the same people; upper bound): {res['fifths']:.3f}")
    print(f"  ({res['time']:.0f}s)")


def ayuraj_report(model_path=None, th=DEFAULT):
    """The permanent holdout, scored on the COMMITTED pickle -- no re-fit, by design.

    Every other number in this file is a fresh forest trained with one set held out, because
    the shipped forest has seen them all. ayuraj is the one set the shipped forest has not
    seen and never will (strangers.NEVER_TRAIN), so the honest thing is to score the pickle
    that ships rather than a stand-in for it, and the number stays comparable across releases.
    """
    import pickle
    import features as F
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_static.p") \
        if model_path is None else model_path
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    featfn, _ = F.static_feature_for(blob["feature"])
    per, P, L, S = ST.load_ayuraj()
    y = np.array([LETTERS.index(x) for x in L])
    pr = proba(blob["model"], featfn(P))
    pred = pr.argmax(1)
    e, r = gate_line(pr, y, th)
    print(f"  4. ayuraj held out FOREVER ({len(P)} letter frames, 5 signers, CC0), scored on the "
          f"committed {os.path.basename(path)}: {(pred == y).mean():.3f}")
    print("     per signer: " + "  ".join(
        f"{s} {(pred[S == s] == y[S == s]).mean():.3f} (n={int((S == s).sum())})" for s in sorted(set(S))))
    print(f"     single-frame vote gate at DEFAULT: emits on {e:.2f} of records, right on {r:.3f} of those")
    lo, hi = signer_ci(pred == y, S)
    print(f"     signer-clustered 95% CI [{lo:.3f}, {hi:.3f}] -- resampling the 5 SIGNERS, not the "
          f"frames, because 1,111 frames from 5 people are not 1,111 independent draws")
    print("     CAVEAT: these are crops around a hand and MediaPipe finds one in 72.33% of them, "
          "with the misses on the fists (T 8/65, S 14/70, M 15/70), so do not quote it per letter")
    return float((pred == y).mean())


def signer_ci(correct, signers, n_boot=10000, seed=0):
    """Percentile 95% CI over signers resampled with replacement.

    The frame-level interval on 1,111 frames is about +-0.019 and it is fiction: the frames come
    from five people, so the unit that varies between one holdout set and the next is the SIGNER.
    Resampling signers gives an interval roughly three times wider, and that is the one to quote.
    """
    sig = sorted(set(signers))
    idx = [np.flatnonzero(signers == s) for s in sig]
    rng = np.random.default_rng(seed)
    draws = [correct[np.concatenate([idx[i] for i in rng.choice(len(sig), len(sig), replace=True)])].mean()
             for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def ayuraj_delta(prev_path, model_path=None):
    """The same 1,111 frames through two pickles, with a signer-clustered CI on the DIFFERENCE.

    This is the only clean before-and-after in the repository. Every other set crossed onto the
    training side at some point, so its "held out" figure stopped being comparable across
    releases the moment it did. ayuraj never can (strangers.NEVER_TRAIN), so the two numbers are
    one measurement repeated, and the paired difference is tighter than either interval alone.
    """
    import pickle
    import features as F
    here = os.path.dirname(os.path.abspath(__file__))
    now_path = os.path.join(here, "model_static.p") if model_path is None else model_path
    _, P, L, S = ST.load_ayuraj()
    y = np.array([LETTERS.index(x) for x in L])
    hits = {}
    for tag, path in (("prev", prev_path), ("now", now_path)):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        featfn, _ = F.static_feature_for(blob["feature"])
        hits[tag] = proba(blob["model"], featfn(P)).argmax(1) == y
    d = hits["now"].astype(int) - hits["prev"].astype(int)
    lo, hi = signer_ci(d, S)
    print(f"  ayuraj, the same {len(P)} frames through both pickles:")
    for tag in ("prev", "now"):
        a, b = signer_ci(hits[tag], S)
        print(f"     {tag:4s} {hits[tag].mean():.4f}  signer-clustered 95% CI [{a:.3f}, {b:.3f}]")
    print(f"     difference {d.mean():+.4f}  signer-clustered 95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  ({int((d > 0).sum())} frames flipped right, {int((d < 0).sum())} flipped wrong)")
    print("     per signer: " + "  ".join(
        f"{s} {hits['now'][S == s].mean() - hits['prev'][S == s].mean():+.3f}" for s in sorted(set(S))))
    return float(d.mean()), (lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-henry-only", action="store_true", help="omit the author-only comparison rows")
    ap.add_argument("--compare-prev", metavar="PICKLE",
                    help="score ayuraj through this pickle as well as the committed one and print "
                         "the paired difference with a signer-clustered CI. Get a previous "
                         "release's forest with: git show <rev>:temporal/model_static.p > prev.p")
    ap.add_argument("--skip-ayuraj", action="store_true",
                    help="omit the permanent-holdout row (it scores the committed pickle, not a re-fit)")
    ap.add_argument("--author-cap", type=int, default=None,
                    help="frames per letter kept from the author's pooled sessions (0 disables)")
    ap.add_argument("--aslhg-cap", type=int, default=None,
                    help="ASL-HG frames per (signer, letter) on the training side (0 = all)")
    ap.add_argument("--trees", type=int, default=None, help="forest n_estimators (candidate sweep)")
    ap.add_argument("--leaf", type=int, default=None, help="forest min_samples_leaf (candidate sweep)")
    args = ap.parse_args()
    with T.forest_override(n_estimators=args.trees, min_samples_leaf=args.leaf) as forest:
        print(f"recipe: {T.FEATURE_TAG}, filter {T.RULESET!r}, jitter {T.JITTER_SIGMA} "
              f"x{T.JITTER_COPIES}, author cap "
              f"{(T.AUTHOR_CAP if args.author_cap is None else args.author_cap) or 'none'}/letter, "
              f"ASL-HG cap {(ST.ASLHG_CAP if args.aslhg_cap is None else args.aslhg_cap) or 'all'}"
              f"/signer/letter, RF({forest['n_estimators']}, "
              f"min_samples_leaf={forest['min_samples_leaf']})\n")
        runs = []
        for seed in range(args.seeds):
            runs.append(run(seed, n_jobs=args.n_jobs, author_cap=args.author_cap,
                            aslhg_cap=args.aslhg_cap))
            if seed == 0 and not args.skip_ayuraj:
                ayuraj_report()
                if args.compare_prev:
                    ayuraj_delta(args.compare_prev)
            print()
        if not args.skip_henry_only:
            print("for comparison, the previous release's training set (the author only):")
            run(0, henry_only=True, n_jobs=args.n_jobs, author_cap=args.author_cap)
            print()
    if args.seeds > 1:
        for key, name in (("aslnow", "ASLNow held out"), ("ankara", "Ankara held out"), ("fifths", "ASLNow fifths")):
            v = [r[key] for r in runs]
            print(f"over {args.seeds} seeds: {name} {np.mean(v):.4f} +- {np.std(v):.4f} (range {min(v):.4f}-{max(v):.4f})")


if __name__ == "__main__":
    main()

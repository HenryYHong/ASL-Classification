"""Leave-one-SIGNER-out over ASL-HG's ten volunteers: the first real cross-signer split here.

Every other measurement in this project holds out a DATASET or a SESSION. crossval_static.py
holds out one of the author's four sessions -- a new day, but the same hand. crossval_strangers.py
holds out a whole public set -- new hands, but also a new camera, a new room, a new landmarker
and a new labeling convention, so a fall there cannot be attributed to the hands. ASL-HG is the
first source here that carries real participant ids (ingest_aslhg.py recovers them from the
P<k>_ filename prefix), so it is the first one that can answer the question the hosted page
actually raises: the forest was trained on N people, how does it read the N+1th?

Protocol, one fold per signer:

    train   the author's four sessions, capped at train_static.AUTHOR_CAP per letter pooled
            + ASLNow + the Ankara O/V/W/F photos
            + the OTHER NINE ASL-HG signers AT THE SHIPPED CAP (strangers.ASLHG_CAP = 25 per
              signer per letter). Not uncapped: the point is to measure the recipe that ships,
              and at the cap nine signers contribute 5,400 frames rather than 21,600.
    test    every one of the held-out signer's 2,400 frames (P8: 2,384 -- MediaPipe misses 16
            of that signer's H). Nothing of that signer is anywhere in the training set.

The capped subset is drawn ONCE, by strangers.cap_per_signer_letter at strangers.CAP_SEED, and
the held-out signer is then removed from it. That matters: the nine signers each fold trains on
are backed by exactly the frames the shipped forest trains on, not by a fresh draw per fold.

READ THE SPREAD AS A PROPERTY OF THESE TEN PEOPLE. The per-signer accuracies are not ten
samples of "a visitor": they are ten volunteers photographed in one place over two months, and
the fold-to-fold spread is dominated by which ten people you have -- how far each one's U is
from his R, how each holds a C. A mean +- sd over ten such folds is a description of this
sample, not a confidence interval for the next person to open the page. The pooled figure
(every held-out frame, counted once) is the one to quote, with the range beside it.

    ./.venv/bin/python temporal/crossval_signers.py                    # seed 0
    ./.venv/bin/python temporal/crossval_signers.py --seeds 3
    ./.venv/bin/python temporal/crossval_signers.py --trees 60 --leaf 8   # a candidate
    ./.venv/bin/python temporal/crossval_signers.py --aslhg-cap 0          # nine signers uncapped
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


def signer_order(signer):
    """P1, P2, ... P10 -- numeric, not lexicographic (sorted() would read P10 after P1)."""
    return sorted(set(signer), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))


def base_training_set(author_cap=None):
    """The training side minus ASL-HG: the author's capped sessions + ASLNow + Ankara.

    -> (per_class, label). load_strangers is asked for its two non-ASL-HG sets here because
    ASL-HG has to be added per fold instead, with the held-out signer removed; the ASL-HG cap
    is applied in run(), where that removal happens.
    """
    src = CV.sessions()
    author_cap = T.AUTHOR_CAP if author_cap is None else author_cap
    author = ST.merge(*[src[s][0] for s in src])
    if author_cap:
        author = A.cap_per_letter(author, author_cap)
    strangers, sources = ST.load_strangers(aslhg=False)
    return ST.merge(author, strangers), f"author cap {author_cap or 'none'}/letter + {sources}"


def run(seed=0, n_jobs=4, author_cap=None, aslhg_cap=None, verbose=True, signers=None):
    """One leave-one-signer-out pass. -> dict with per-signer accuracy, pooled, per-letter recall."""
    author_cap = T.AUTHOR_CAP if author_cap is None else author_cap
    aslhg_cap = ST.ASLHG_CAP if aslhg_cap is None else aslhg_cap
    base, base_label = base_training_set(author_cap)
    n_base = sum(len(P) for P in base)

    # every frame, for testing; the capped draw, for training
    _, P_all, L_all, S_all = ST.load_aslhg(cap=None)
    _, P_cap, L_cap, S_cap = ST.load_aslhg(cap=aslhg_cap or None)
    y_all = np.array([LETTERS.index(L) for L in L_all])
    order = [s for s in signer_order(S_all) if signers is None or s in signers]

    res = {"seed": seed, "author_cap": author_cap, "aslhg_cap": aslhg_cap,
           "base": base_label, "n_base": n_base, "signers": {}, "n_train_rows": {}}
    pred_all = np.full(len(y_all), -1)
    t0 = time.time()
    if verbose:
        print(f"training side per fold: {n_base} frames ({base_label}) + the other "
              f"{len(signer_order(S_all)) - 1} ASL-HG signers at cap {aslhg_cap or 'none'}")
        print(f"{'held out':>9}  {'frames':>6}  {'accuracy':>8}  {'rows':>7}  {'gate emits':>10}  {'right':>6}")
    for s in order:
        te = S_all == s
        tr = S_cap != s                       # the shipped draw, minus this signer
        per_tr = ST.per_class_of(P_cap[tr], L_cap[tr])
        X, y, _ = T.training_rows(ST.merge(base, per_tr), seed=seed)
        model = T.fit_letters(X, y, seed, n_jobs)
        pr = np.zeros((int(te.sum()), NC))
        pr[:, model.classes_.astype(int)] = model.predict_proba(T.FEATFN(P_all[te]))
        pred = pr.argmax(1)
        pred_all[te] = pred
        acc = float((pred == y_all[te]).mean())
        emit, right = gate_line(pr, y_all[te])
        res["signers"][s] = {"acc": acc, "n": int(te.sum()), "emit": emit, "right": right}
        res["n_train_rows"][s] = int(len(X))
        if verbose:
            print(f"{s:>9}  {int(te.sum()):>6}  {acc:>8.4f}  {len(X):>7}  {emit:>10.3f}  {right:>6.4f}",
                  flush=True)
    done = pred_all >= 0
    res["pooled"] = float((pred_all[done] == y_all[done]).mean())
    res["n_test"] = int(done.sum())
    res["recall"] = {}
    res["confusion"] = {}
    for c, L in enumerate(LETTERS):
        m = done & (y_all == c)
        if not m.any():
            continue
        res["recall"][L] = float((pred_all[m] == c).mean())
        wrong = pred_all[m & (pred_all != y_all)]
        if len(wrong):
            top = int(np.bincount(wrong, minlength=NC).argmax())
            res["confusion"][L] = (LETTERS[top], int((wrong == top).sum()), int(m.sum()))
    res["time"] = time.time() - t0
    return res


def gate_line(pr, y, th=DEFAULT):
    """Single-frame emission under the static vote rule (a vote of identical frames), as
    crossval_strangers.py prints it: a photograph offers one window, a held sign offers many.
    Imported from there so the per-letter floors are read in one place."""
    import crossval_strangers as CS
    return CS.gate_line(pr, y, th)


def report(res):
    v = np.array([f["acc"] for f in res["signers"].values()])
    print(f"\npooled over every held-out frame: {res['pooled']:.4f} ({res['n_test']} frames, "
          f"{len(v)} signers)")
    print(f"mean +- sd over signers: {v.mean():.4f} +- {v.std():.4f} (range {v.min():.4f}-{v.max():.4f})"
          f"  <- a description of THESE ten volunteers, not an interval for the next visitor")
    print("per-letter recall (pooled over folds):")
    for L in LETTERS:
        if L not in res["recall"]:
            continue
        conf = res["confusion"].get(L)
        tail = f"  ->{conf[0]} {conf[1]}/{conf[2]}" if conf else ""
        print(f"  {L} {res['recall'][L]:.3f}{tail}")
    weak = sorted((v, k) for k, v in res["recall"].items() if v < 0.8)
    if weak:
        print("  below 0.80: " + ", ".join(f"{k} {v:.3f}" for v, k in weak))
    print(f"({res['time']:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1 and print the spread")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--author-cap", type=int, default=None,
                    help="frames per letter kept from the author's pooled sessions (0 disables)")
    ap.add_argument("--aslhg-cap", type=int, default=None,
                    help="ASL-HG frames per (signer, letter) on the TRAINING side (0 = all)")
    ap.add_argument("--trees", type=int, default=None, help="forest n_estimators (candidate sweep)")
    ap.add_argument("--leaf", type=int, default=None, help="forest min_samples_leaf (candidate sweep)")
    ap.add_argument("--signers", nargs="*", default=None, help="a subset of P1..P10 (a quick check)")
    args = ap.parse_args()
    runs = []
    with T.forest_override(n_estimators=args.trees, min_samples_leaf=args.leaf) as forest:
        print(f"recipe: {T.FEATURE_TAG} ({T.FEATURE_DIM}-D), filter {T.RULESET!r}, jitter "
              f"{T.JITTER_SIGMA} x{T.JITTER_COPIES} on the training side only, "
              f"RF({forest['n_estimators']}, min_samples_leaf={forest['min_samples_leaf']})\n")
        for seed in range(args.seeds):
            if args.seeds > 1:
                print(f"--- seed {seed} ---")
            res = run(seed=seed, n_jobs=args.n_jobs, author_cap=args.author_cap,
                      aslhg_cap=args.aslhg_cap, signers=args.signers)
            report(res)
            runs.append(res)
            print()
    if args.seeds > 1:
        p = [r["pooled"] for r in runs]
        print(f"over {args.seeds} seeds: pooled {np.mean(p):.4f} +- {np.std(p):.4f} "
              f"(range {min(p):.4f}-{max(p):.4f})")


if __name__ == "__main__":
    main()

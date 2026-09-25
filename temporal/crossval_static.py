"""Leave-one-session-out accuracy for the static branch -- the only number worth publishing.

train_static.py characterizes the feature and ships a model; it never measures generalization,
because every split it takes is inside one capture burst. This file is the measurement, and it
exists as a committed script rather than a remembered figure so the README's number can be
re-derived after the next recording session.

Protocol: hold out one of the author's sessions, train on the other three PLUS the strangers
(strangers.py: the ASLNow records, the 218-signer digit photos that are letters, and ASL-HG's
ten signers capped at 25 frames per signer-letter -- other people's hands, never held out
here because they are never the author's) with EXACTLY the shipped recipe
(train_static.training_rows: the author's three training sessions capped at
train_static.AUTHOR_CAP frames per letter POOLED, jitter sigma 0.12 x4 on the training
frames, no filter, static/v4, RF 60 trees / min leaf 5), test on every un-jittered frame of
the held-out session.

BOTH CAPS AND THE JITTER ARE TRAINING-SIDE ONLY. A test fold is scored whole, every frame of
it, at every recipe -- otherwise the denominator would move with the recipe and two rows of
the same table would not be comparable. Jittering a test fold would score copies of frames
already being scored; capping one would quietly drop the letters the author recorded most.

--henry-only leaves the strangers out (0.873 / 0.782 with this recipe at seed 0, and 0.872 /
0.787 with --author-cap 0 as well; the previous release's filtered 100-tree forest measured
0.861 / 0.763). --no-aslhg leaves only ASLNow and Ankara in (0.913 / 0.873), and with
--author-cap 0 --trees 80 --leaf 5 beside it that is the previous release's recipe exactly and
reproduces its pooled 0.9131 / cross-day 0.8726 / macro S1 0.8708 at seed 0. crossval_strangers.py holds the strangers
out instead, and crossval_signers.py holds out one of ten real people; that is where the
cross-signer numbers come from.

Read the per-fold table, not just the mean. The four sessions do not cover the same letters: S1
is the full 24-letter archive recorded months before the others, S2 is 23 letters, and S3 and S4
are targeted re-recordings of 6 and 4 letters made the same evening as S2, about two and a half
hours later. A fold that tests four well-separated letters scores near 1.00 and says nothing
about the alphabet, so an
unweighted mean over folds flatters the model -- the same error this repository's README
criticizes in the original 100.00% (a letter with no test samples cannot be got wrong). Two
numbers matter: the POOLED figure (every held-out frame across all folds, counted once) and the
S1 fold, the only cross-day fold and the only one that tests all 24 letters. It is printed
first and labeled as such.

Uncertainty: the hold-level bootstrap resamples the 57 (session x letter) bursts, not frames --
consecutive frames of one held sign are near-duplicates, and a frame-level interval would be
several times too narrow. Seeds move the pooled number by about +-0.005 and the S1 fold by
+-0.01 (--seeds 3 prints the spread); two implementations of the same recipe differ by 0.01
from the jitter RNG alone. Quote the seed range and the CI, not the third decimal of one run.

    ./.venv/bin/python temporal/crossval_static.py                 # seed 0
    ./.venv/bin/python temporal/crossval_static.py --seeds 3       # seed spread
    ./.venv/bin/python temporal/crossval_static.py --oof oof.npz   # out-of-fold posteriors
    ./.venv/bin/python temporal/crossval_static.py --henry-only    # the one-signer forest
    ./.venv/bin/python temporal/crossval_static.py --legacy        # the retired recipe

--legacy runs the previous release's recipe (static/v3, four-angle rotation augmentation, no
filter, RF 400, S2-S4 mirrored by per-frame handedness labels) and must still print pooled
0.759 / S1 0.659; it is the regression guard for this harness, not a number to publish. Under
the shipped loader (hold-modal handedness, which re-mirrors 2 frames of the S2 M hold) the
same recipe prints 0.756 / 0.658: that is what 2 frames do to a 400-tree forest's bootstrap
draws, and why no third decimal here means anything. --oof writes the per-frame out-of-fold posteriors in the
schema simulate_words.py reads (probs, y, session, hold, order, letters); make_oof.py is the
one-line wrapper for that.
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import train_static as T
import strangers as ST
import static_aug as A

EXTRA = (("S2", "static_s2.npz"), ("S3", "static_s3.npz"), ("S4", "static_s4.npz"))
HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = T.LETTERS
NC = len(LETTERS)
EMPTY = np.zeros((0, 21, 2))
CROSS_DAY = "S1"
ROT_DEGREES = (-12, -6, 6, 12)      # the retired augmentation, kept for --legacy only


def sessions(legacy=False):
    """{tag: ([per-class landmark arrays], [per-class hold ids])} for every session on disk.

    S1 is one hold per letter (the archive holds each letter once); S2-S4 holds split at
    HOLD_GAP within a letter. Hold ids are unique across sessions. legacy=True loads S2-S4
    with the previous release's per-frame handedness rule (2 frames differ), which the
    --legacy guard needs to reproduce its numbers exactly.
    """
    out = {}
    S1 = T.load()
    out["S1"] = (S1, [np.full(len(P), c) for c, P in enumerate(S1)])
    base = NC
    for tag, name in EXTRA:
        got = T.load_extra(os.path.join(HERE, name), holds=True, per_frame_handedness=legacy)
        per = [got[c][0] if c in got else EMPTY for c in range(NC)]
        hid = [got[c][1] + base if c in got else np.zeros(0, int) for c in range(NC)]
        base += max((int(h.max()) + 1 for h in hid if len(h)), default=0)
        out[tag] = (per, hid)
    return out


def shipped_training_set(src=None, author_cap=None, aslhg_cap=None, strangers=True,
                         aslhg=True):
    """Exactly the per-class block the shipped forest is fitted on. -> (per_class, label)

    train_static.py builds this from its own --extra list; everything that needs "the shipped
    training set" without re-deriving it -- idle_gate.py --seeds, crossval_signers.py, a
    candidate sweep -- comes here instead, so there is one definition of it and not four.
    tests/test_strangers.py checks that this and train_static.py agree frame for frame.
    """
    src = sessions() if src is None else src
    author_cap = T.AUTHOR_CAP if author_cap is None else author_cap
    aslhg_cap = ST.ASLHG_CAP if aslhg_cap is None else aslhg_cap
    per = ST.merge(*[src[s][0] for s in src])
    label = f"author {sum(len(P) for P in per)}f"
    if author_cap:
        per = A.cap_per_letter(per, author_cap)
        label += f" capped to {sum(len(P) for P in per)} at {author_cap}/letter"
    if strangers:
        got, sources = ST.load_strangers(cap=aslhg_cap or None, aslhg=aslhg)
        per = ST.merge(per, got)
        label += f" + {sum(len(P) for P in got)}f from {sources}"
    return per, label


def legacy_rows(per_class):
    """The previous release's training rows: static/v3, originals then each rotation as a
    block. Row order matters (it changes the forest's bootstrap draws), so this reproduces the
    old crossval_static.rows() exactly."""
    pairs = [(c, P) for c, P in enumerate(per_class) if len(P)]
    X, y = T.build(pairs, F.static_feature)
    Xs, ys = [X], [y]
    for deg in ROT_DEGREES:
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        mid = [(c, (P - P.mean(axis=(0, 1), keepdims=True)) @ R.T
                + P.mean(axis=(0, 1), keepdims=True)) for c, P in pairs]
        Xr, yr = T.build(mid, F.static_feature)
        Xs.append(Xr)
        ys.append(yr)
    return np.concatenate(Xs), np.concatenate(ys), 0


def test_rows(per_class, featfn):
    pairs = [(c, P) for c, P in enumerate(per_class) if len(P)]
    return T.build(pairs, featfn)


def bootstrap_ci(bursts, n_boot=2000, seed=0):
    """95% interval on the pooled accuracy, resampling (session x letter) bursts with
    replacement. bursts: list of (correct, n)."""
    rng = np.random.default_rng(seed)
    b = np.asarray(bursts, dtype=np.float64)
    accs = []
    for _ in range(n_boot):
        s = b[rng.integers(0, len(b), len(b))]
        accs.append(s[:, 0].sum() / s[:, 1].sum())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def run(seed=0, legacy=False, src=None, n_jobs=4, verbose=True, strangers=None, ruleset=None,
        author_cap=None):
    """One leave-one-session-out pass. Returns a dict with per-fold accuracies, the pooled
    figure, the hold-level CI, S1 macro recall, and the out-of-fold arrays. `strangers` is a
    per-class list added to every TRAINING fold (None = none); `ruleset` overrides the filter;
    `author_cap` is train_static.AUTHOR_CAP applied to the three TRAINING sessions pooled
    (None = no cap). Both caps are training-side only: the held-out session is scored whole,
    every frame of it, so the number stays comparable across recipes."""
    src = sessions(legacy) if src is None else src
    ruleset = T.RULESET if ruleset is None else ruleset
    featfn = F.static_feature if legacy else T.FEATFN
    order = [CROSS_DAY] + [s for s in src if s != CROSS_DAY]
    if verbose:
        print(f"{'held out':>9}  {'letters':>7}  {'frames':>6}  {'accuracy':>8}  {'dropped':>7}")
    res = {"folds": {}, "seed": seed, "legacy": legacy, "confusion": np.zeros((NC, NC), int)}
    oof = {"probs": [], "y": [], "session": [], "hold": [], "order": []}
    bursts = []
    correct = total = 0
    t0 = time.time()
    for held in order:
        train = [np.concatenate([src[s][0][c] for s in src if s != held and len(src[s][0][c])])
                 if any(len(src[s][0][c]) for s in src if s != held) else EMPTY
                 for c in range(NC)]
        if legacy:
            Xtr, ytr, dropped = legacy_rows(train)
            model = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=n_jobs)
        else:
            if author_cap:
                train = A.cap_per_letter(train, author_cap)
            if strangers is not None:
                train = ST.merge(train, strangers)
            Xtr, ytr, dropped = T.training_rows(train, ruleset=ruleset, seed=seed)
            model = None            # fit_letters below: the REST class is part of the recipe
        Xte, yte = test_rows(src[held][0], featfn)
        if not len(Xte):
            continue
        # --legacy reproduces a forest from before any of this existed, so it gets no REST
        # class and no wrapper; that is what makes it a regression guard rather than a rerun.
        model = model.fit(Xtr, ytr) if legacy else T.fit_letters(Xtr, ytr, seed, n_jobs)
        proba = np.zeros((len(Xte), NC))
        proba[:, model.classes_.astype(int)] = model.predict_proba(Xte)
        pred = proba.argmax(axis=1)
        hit = int((pred == yte).sum())
        correct += hit
        total += len(yte)
        np.add.at(res["confusion"], (yte, pred), 1)
        for c in np.unique(yte):
            m = yte == c
            bursts.append((int((pred[m] == c).sum()), int(m.sum())))
        res["folds"][held] = {"acc": hit / len(yte), "n": int(len(yte)),
                              "letters": int(len(set(yte.tolist()))), "y": yte, "pred": pred,
                              "dropped": int(dropped)}
        hid = np.concatenate([h for h in src[held][1] if len(h)])
        assert len(hid) == len(yte)
        ordr = np.zeros(len(hid), int)
        for i in range(1, len(hid)):
            ordr[i] = ordr[i - 1] + 1 if hid[i] == hid[i - 1] else 0
        oof["probs"].append(proba); oof["y"].append(yte)
        oof["session"].append(np.full(len(yte), held)); oof["hold"].append(hid); oof["order"].append(ordr)
        if verbose:
            tag = "cross-day" if held == CROSS_DAY else ""
            print(f"{held:>9}  {len(set(yte.tolist())):>7}  {len(yte):>6}  {hit / len(yte):>8.3f}"
                  f"  {dropped:>7}  {tag}")
    res["pooled"] = correct / total
    res["correct"], res["total"] = correct, total
    res["ci"] = bootstrap_ci(bursts)
    res["n_bursts"] = len(bursts)
    cm1 = np.zeros((NC, NC), int)
    f1 = res["folds"].get(CROSS_DAY)
    if f1 is not None:
        np.add.at(cm1, (f1["y"], f1["pred"]), 1)
        rec = [cm1[c, c] / cm1[c].sum() for c in range(NC) if cm1[c].sum()]
        res["macro_s1"] = float(np.mean(rec))
        res["recall_s1"] = {LETTERS[c]: cm1[c, c] / cm1[c].sum() for c in range(NC) if cm1[c].sum()}
    res["oof"] = {k: np.concatenate(v) for k, v in oof.items()}
    res["time"] = time.time() - t0
    return res


def report(res):
    lo, hi = res["ci"]
    f1 = res["folds"].get(CROSS_DAY)
    print(f"\npooled over every held-out frame: {res['pooled']:.3f}  ({res['correct']}/{res['total']})"
          f"   hold-level bootstrap 95% CI [{lo:.3f}, {hi:.3f}] over {res['n_bursts']} session x letter bursts")
    if f1 is not None:
        print(f"cross-day fold ({CROSS_DAY} held out): {f1['acc']:.3f}  ({int(round(f1['acc'] * f1['n']))}/{f1['n']})"
              f"   macro-averaged per-letter recall {res['macro_s1']:.3f}")
        weak = sorted((v, k) for k, v in res["recall_s1"].items() if v < 0.6)
        if weak:
            print(f"  {CROSS_DAY} letters below 0.6 recall: " + ", ".join(f"{k} {v:.2f}" for v, k in weak))
    print(f"unweighted mean over folds:       {np.mean([f['acc'] for f in res['folds'].values()]):.3f}"
          f"  <- flattered by the small folds; do not publish this one")
    print(f"({res['time']:.0f}s)")


def write_oof(res, path):
    o = res["oof"]
    assert abs((o["probs"].argmax(1) == o["y"]).mean() - res["pooled"]) < 1e-12
    np.savez(path, probs=o["probs"], y=o["y"], session=o["session"], hold=o["hold"],
             order=o["order"], letters=np.array(LETTERS))
    print(f"wrote {path}: {o['probs'].shape[0]} frames x {o['probs'].shape[1]} classes, "
          f"{len(set(o['hold'].tolist()))} holds, seed {res['seed']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1,
                    help="run seeds 0..N-1 (forest random_state and jitter seed) and print the spread")
    ap.add_argument("--legacy", action="store_true",
                    help="the retired recipe (static/v3, rotation augmentation, RF 400); must print 0.759 / 0.659")
    ap.add_argument("--oof", default=None,
                    help="write seed 0's out-of-fold per-frame posteriors here (simulate_words.py's input)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--henry-only", action="store_true",
                    help="no strangers on the training side: the previous release's one-signer forest (0.861 / 0.763)")
    ap.add_argument("--ruleset", default=None, choices=sorted(A.RULESETS),
                    help="defining-geometry training filter (default: the shipped setting, none)")
    ap.add_argument("--author-cap", type=int, default=None,
                    help="frames per letter kept from the author's POOLED training sessions "
                         "(default: train_static.AUTHOR_CAP; 0 disables the cap)")
    ap.add_argument("--aslhg-cap", type=int, default=None,
                    help="ASL-HG frames per (signer, letter) on the training side "
                         "(default: strangers.ASLHG_CAP; 0 = all 23,984)")
    ap.add_argument("--no-aslhg", action="store_true",
                    help="leave ASL-HG out of the training side entirely (the previous release's data)")
    ap.add_argument("--trees", type=int, default=None, help="forest n_estimators (candidate sweep)")
    ap.add_argument("--leaf", type=int, default=None, help="forest min_samples_leaf (candidate sweep)")
    args = ap.parse_args()

    src = sessions(legacy=args.legacy)
    strangers = None
    # The caps are training-side recipe, so --legacy (which reproduces a recipe that predates
    # both of them) takes neither.
    author_cap = None if args.legacy else (T.AUTHOR_CAP if args.author_cap is None else args.author_cap)
    aslhg_cap = ST.ASLHG_CAP if args.aslhg_cap is None else args.aslhg_cap
    if args.legacy:
        print("LEGACY recipe: static/v3 + rotation (-12,-6,6,12) + RF400, no filter, per-frame "
              "handedness, author only -- the regression guard (expected pooled 0.759, S1 0.659), "
              "not the shipped model\n")
    else:
        ruleset = T.RULESET if args.ruleset is None else args.ruleset
        if not args.henry_only:
            strangers, sources = ST.load_strangers(cap=aslhg_cap or None, aslhg=not args.no_aslhg)
            n_str = sum(len(P) for P in strangers)
        print(f"recipe: {T.FEATURE_TAG} ({T.FEATURE_DIM}-D), filter {ruleset!r}, jitter sigma "
              f"{T.JITTER_SIGMA} x{T.JITTER_COPIES} on training folds only, author cap "
              f"{author_cap or 'none'}/letter pooled, "
              f"RF({args.trees or T.FOREST['n_estimators']}, "
              f"min_samples_leaf={args.leaf or T.FOREST['min_samples_leaf']}), "
              + (f"strangers on the training side: {n_str} frames from {sources}" if strangers is not None
                 else "author's sessions only (--henry-only)") + "\n")
    runs = []
    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f"--- seed {seed} ---")
        with T.forest_override(n_estimators=args.trees, min_samples_leaf=args.leaf):
            res = run(seed=seed, legacy=args.legacy, src=src, n_jobs=args.n_jobs,
                      strangers=strangers, ruleset=args.ruleset, author_cap=author_cap)
        report(res)
        runs.append(res)
        if seed == 0 and args.oof:
            write_oof(res, args.oof)
        if args.seeds > 1:
            print()
    if args.seeds > 1:
        pooled = [r["pooled"] for r in runs]
        s1 = [r["folds"][CROSS_DAY]["acc"] for r in runs]
        print(f"over {args.seeds} seeds: pooled {np.mean(pooled):.3f} +- {np.std(pooled):.3f} "
              f"(range {min(pooled):.3f}-{max(pooled):.3f}); cross-day {np.mean(s1):.3f} +- "
              f"{np.std(s1):.3f} (range {min(s1):.3f}-{max(s1):.3f}); macro S1 recall "
              f"{np.mean([r['macro_s1'] for r in runs]):.3f}")


if __name__ == "__main__":
    main()

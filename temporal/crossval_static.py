"""Leave-one-session-out accuracy for the static branch -- the only number worth publishing.

train_static.py characterizes the feature and ships a model; it never measures generalization,
because every split it takes is inside one capture burst. This file is the measurement, and it
exists as a committed script rather than a remembered figure so the README's number can be
re-derived after the next recording session.

Protocol: hold out one session, train on the rest with EXACTLY the shipped recipe
(train_static.training_rows: strong-rule filter, jitter sigma 0.12 x4 on the training frames,
static/v4, RF 100 trees / min leaf 5), test on every un-filtered, un-jittered frame of the
held-out session. Filter and jitter touch TRAINING folds only -- filtering the test fold would
score the model on the frames it finds easy, and jittering it would score copies of frames
already being scored.

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
from the jitter RNG alone. Quote the range 0.86-0.87 / 0.76-0.78 and the CI, not the third
decimal of one run.

    ./.venv/bin/python temporal/crossval_static.py                 # seed 0
    ./.venv/bin/python temporal/crossval_static.py --seeds 3       # seed spread
    ./.venv/bin/python temporal/crossval_static.py --oof oof.npz   # out-of-fold posteriors
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


def run(seed=0, legacy=False, src=None, n_jobs=4, verbose=True):
    """One leave-one-session-out pass. Returns a dict with per-fold accuracies, the pooled
    figure, the hold-level CI, S1 macro recall, and the out-of-fold arrays."""
    src = sessions(legacy) if src is None else src
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
            Xtr, ytr, dropped = T.training_rows(train, seed=seed)
            model = T.make_forest(seed, n_jobs=n_jobs)
        Xte, yte = test_rows(src[held][0], featfn)
        if not len(Xte):
            continue
        model.fit(Xtr, ytr)
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
    args = ap.parse_args()

    src = sessions(legacy=args.legacy)
    if args.legacy:
        print("LEGACY recipe: static/v3 + rotation (-12,-6,6,12) + RF400, no filter, per-frame "
              "handedness -- the regression guard (expected pooled 0.759, S1 0.659), not the "
              "shipped model\n")
    else:
        print(f"recipe: {T.FEATURE_TAG} ({T.FEATURE_DIM}-D), filter {T.RULESET!r}, jitter sigma "
              f"{T.JITTER_SIGMA} x{T.JITTER_COPIES} on training folds only, "
              f"RF({T.FOREST['n_estimators']}, min_samples_leaf={T.FOREST['min_samples_leaf']})\n")
    runs = []
    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f"--- seed {seed} ---")
        res = run(seed=seed, legacy=args.legacy, src=src, n_jobs=args.n_jobs)
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

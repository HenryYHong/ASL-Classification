"""Train the 10-digit static forest (model_digits.p) on the Ankara landmarks and measure it by
leave-signer-out.

    ./.venv/bin/python temporal/train_digits.py                       # CV table + ship
    ./.venv/bin/python temporal/train_digits.py --no-cv               # just ship
    ./.venv/bin/python temporal/train_digits.py --ship-feature static/v3

The data is temporal/digits_ankara.npz: MediaPipe landmarks of 1,805 photographs (of 2,062;
87.5% detected) from the Sign Language Digits Dataset (Ankara Ayranci Anadolu High School,
Apache-2.0), 218 students photographed once per digit, written by ingest_images.py with one
signer id per run of consecutive images (222 signers among the detected photos). NOT ONE
FRAME IS THE AUTHOR'S HAND: the forest reads a stranger's hand shapes, and everything
downstream of it (live_demo.py --mode digits, the page's numbers mode) says so.

Every image is its own hold, so handedness is canonicalized per image with the label MediaPipe
gave that image. Evaluation is GroupKFold(5) by SIGNER, never by image: one photo per digit
per signer means any split that is not by signer leaks the hand into the test fold (the
IMG//10 grouping the first audit used mixed two students in 187 of 219 groups and is not a
signer split). Augmentation is static_aug.jitter_frames (sigma 0.12 palm units, 4 copies) on
the training folds only; the held-out fold is always the un-jittered originals. No rule
filter: the digit rules were never written, and the set has no in-sample confusions to fix.

Feature: the tag is chosen by --ship-feature from features.STATIC_FEATURES, and static/v4
(the letter forest's 112-D vector) is the default. Leave-signer-out does not separate the
candidates -- static/v3 and static/v4 both score 0.986 (1,780 of 1,805; worst digit 6 at
0.965, read as 4), RF400 the same at 6.5x the nodes -- so the choice was made on the two
checks that involve THIS signer, neither of which is a digit recording: (1) the author's O,
V, W, F and B letter frames read as 0, 2, 6, 9 and 4 (the only proxy for the author's
digits) on 0.956 of frames with v4 vs 0.957 with v3, and a 4-frame vote at the digits-mode
gate (VOTE_MARGIN_CLEAR 0.40 / VOTE_PROB_FLOOR 0.60, thresholds.DIGITS_OVERRIDES) emits the
right digit on 0.92 of windows (v3 0.91) and a wrong one on 0.00; (2) on the 2,890 logged
browser holds of the author's hand in letters mode (docs/browser_log.jsonl), where every
digit emission is a false one, v4 emits on 0.162 of single frames at that gate against 0.267
for v3 (0.433 vs 0.537 at the 0.20/0.50 gate, the letters gate before VOTE_PROB_FLOOR was
raised to 0.55; not re-measured at 0.55). Same accuracy, 40% fewer spurious digits (at the
digits gate they split 57% '0' / 40% '1'), and one feature path in the browser: v4 ships.
Both checks are stand-ins; recording
the author's own digits is the first follow-up, and until then nothing here is verified on
the hand that will use it.
"""
import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import static_aug as A

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "digits_ankara.npz")
OUT = os.path.join(HERE, "model_digits.p")
DIGITS = [str(d) for d in range(10)]
SOURCE = ("Sign Language Digits Dataset (Ankara Ayranci Anadolu High School; ardamavi, "
          "Apache-2.0); landmarks only, 218 signers, none of them the author")

#: The shipped recipe: the smallest forest that ties every larger one on leave-signer-out.
SHIP_FEATURE = "static/v4"
JITTER_SIGMA, JITTER_COPIES = 0.12, 4
FOREST = dict(n_estimators=100, min_samples_leaf=5, max_features="sqrt")


# ---------------------------------------------------------------- data

def load_digits(path=NPZ):
    """-> P (N,21,2) isotropic + handedness-canonicalized, y (N,) int 0..9, signer (N,) str."""
    d = np.load(path, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    letters = [str(x) for x in d["letters"]]
    handed = [str(x) for x in d["handed"]] if "handed" in d else ["Unknown"] * len(lm)
    if "frame_size" not in d:
        raise SystemExit(f"{path} carries no frame_size; the aspect ratio is load-bearing")
    wh = np.asarray(d["frame_size"]).ravel()
    signer = [str(x) for x in d["signer"]] if "signer" in d else [str(i) for i in range(len(lm))]
    P, y, g = [], [], []
    for i, lab in enumerate(letters):
        if lab not in DIGITS:
            continue
        Q = F.to_isotropic(lm[i][None, :, :2], int(wh[0]), int(wh[1]))
        Q = F.canonicalize_handedness(Q, handed[i] if handed[i][:1] in ("L", "R") else None)
        P.append(Q[0]); y.append(DIGITS.index(lab)); g.append(signer[i])
    if not P:
        raise SystemExit(f"{path} has no '0'..'9' label")
    return np.stack(P), np.array(y), np.array(g)


def build(P, y, featfn, sigma=0.0, copies=0, rng=None):
    """Originals first, then each jitter copy as a block (train_static's row order)."""
    blocks = A.jitter_frames(P, sigma, copies, rng) if copies and sigma > 0 else [P]
    X = np.concatenate([featfn(B) for B in blocks])
    Y = np.concatenate([y] * len(blocks))
    return X, Y


# ---------------------------------------------------------------- models

def make_forest(seed=0, n_jobs=4):
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **FOREST)


def node_count(model):
    return int(sum(e.tree_.node_count for e in model.estimators_))


def cv(P, y, g, featfn, sigma, copies, seed=0, n_splits=5, n_jobs=4):
    """GroupKFold by signer. Returns accuracy, per-digit recall, confusion, mean nodes, OOF."""
    rng = np.random.default_rng(seed)
    cm = np.zeros((10, 10), int)
    oof = np.zeros((len(y), 10))
    node_counts, t0 = [], time.time()
    for tr, te in GroupKFold(n_splits=n_splits).split(P, y, g):
        assert not (set(g[tr]) & set(g[te])), "a signer on both sides of a fold"
        Xtr, ytr = build(P[tr], y[tr], featfn, sigma, copies, rng)
        Xte = featfn(P[te])
        m = make_forest(seed, n_jobs).fit(Xtr, ytr)
        full = np.zeros((len(te), 10)); full[:, m.classes_] = m.predict_proba(Xte)
        oof[te] = full
        np.add.at(cm, (y[te], full.argmax(1)), 1)
        node_counts.append(node_count(m))
    recall = [cm[c, c] / cm[c].sum() if cm[c].sum() else float("nan") for c in range(10)]
    return {"acc": float(np.trace(cm) / cm.sum()), "recall": recall, "confusion": cm,
            "nodes_mean": float(np.mean(node_counts)), "n_test": int(cm.sum()), "oof": oof,
            "time": time.time() - t0, "mean_top_p": float(oof.max(1).mean())}


def fmt(res, label):
    lines = [f"== {label}",
             f"  acc={res['acc']:.3f} ({int(np.trace(res['confusion']))}/{res['n_test']})  "
             f"mean nodes/fold={res['nodes_mean']:.0f}  mean top p={res['mean_top_p']:.2f}  {res['time']:.0f}s",
             "  per-digit recall: " + "  ".join(f"{d}={r:.3f}" for d, r in zip(DIGITS, res["recall"])),
             "  confusion (rows=true, cols=pred):"]
    for c in range(10):
        lines.append(f"    {c}: " + " ".join(f"{v:4d}" for v in res["confusion"][c]))
    off = [(DIGITS[i], DIGITS[j], int(res["confusion"][i, j])) for i in range(10) for j in range(10)
           if i != j and res["confusion"][i, j]]
    lines.append("  top confusions: " + ", ".join(f"{a}->{b}:{n}" for a, b, n in
                                                  sorted(off, key=lambda t: -t[2])[:8]))
    return "\n".join(lines)


def fit_all(P, y, feature_tag=SHIP_FEATURE, sigma=JITTER_SIGMA, copies=JITTER_COPIES, seed=0,
            n_jobs=4):
    """The shipped fit: every frame plus its jittered copies, one forest. -> (model, X)."""
    featfn, _ = F.static_feature_for(feature_tag)
    X, Y = build(P, y, featfn, sigma, copies, np.random.default_rng(seed))
    return make_forest(seed, n_jobs).fit(X, Y), X


def blob_for(model, feature_tag, n_train, n_frames, n_signers, sigma, copies, seed):
    """The pickle's keys. 'feature' is what the segmenter, live_demo.py and the export resolve
    the feature function from; 'trained_on_author' is read by nothing and written for the
    person who opens the pickle."""
    return {"model": model, "classes": DIGITS, "feature": feature_tag,
            "n_train": int(n_train), "n_frames": int(n_frames), "signers": int(n_signers),
            "aspect": [1, 1],
            "augment": {"kind": "jitter", "sigma_palm": sigma, "copies": copies,
                        "originals_kept": True, "seed": seed},
            "forest": dict(FOREST, random_state=seed),
            "source": SOURCE, "trained_on_author": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sigma", type=float, default=JITTER_SIGMA)
    ap.add_argument("--copies", type=int, default=JITTER_COPIES)
    ap.add_argument("--features", nargs="*", default=sorted(F.STATIC_FEATURES),
                    help="feature tags to cross-validate (default: every registered tag)")
    ap.add_argument("--ship-feature", default=SHIP_FEATURE, choices=sorted(F.STATIC_FEATURES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args(argv)

    P, y, g = load_digits(args.npz)
    print(f"frames={len(P)}  signers={len(set(g))}  per digit="
          f"{dict(sorted(collections.Counter(y.tolist()).items()))}")
    print(f"jitter sigma={args.sigma} x{args.copies} (train folds only); GroupKFold(5) by signer; "
          f"seed={args.seed}")

    if not args.no_cv:
        summary = {}
        for tag in args.features:
            featfn, _ = F.static_feature_for(tag)
            for sig, cop, aug in ((args.sigma, args.copies, f"jitter {args.sigma} x{args.copies}"),
                                  (0.0, 0, "no jitter")):
                label = f"{tag} | RF({FOREST['n_estimators']}, leaf {FOREST['min_samples_leaf']}) | {aug}"
                r = cv(P, y, g, featfn, sig, cop, seed=args.seed, n_jobs=args.n_jobs)
                print(fmt(r, label))
                summary[label] = r
        print("\n# summary (leave-signer-out)")
        for k, v in summary.items():
            print(f"RESULT {k}: acc {v['acc']:.3f}  nodes/fold {v['nodes_mean']:.0f}  "
                  f"top-p {v['mean_top_p']:.2f}  worst digit "
                  f"{DIGITS[int(np.nanargmin(v['recall']))]}={np.nanmin(v['recall']):.3f}")

    model, X = fit_all(P, y, args.ship_feature, args.sigma, args.copies, args.seed, args.n_jobs)
    assert list(model.classes_) == list(range(10)), "a digit has no training frame"
    blob = blob_for(model, args.ship_feature, len(X), len(P), len(set(g)), args.sigma,
                    args.copies, args.seed)
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"\nwrote {args.out}: RF({FOREST['n_estimators']}, leaf {FOREST['min_samples_leaf']}) on "
          f"{args.ship_feature}, {len(X)} rows ({len(P)} frames x {1 + args.copies}), "
          f"{len(set(g))} signers, {node_count(model)} nodes, "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB pickle (export_models.py prints the "
          "models.json share)")


if __name__ == "__main__":
    main()

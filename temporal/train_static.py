"""Train the 24-class static-letter forest that ships (model_static.p).

What ships, and why each piece is there. Every number is leave-one-session-out over the
author's four sessions (4,878 held-out frames, crossval_static.py), or one of the two
cross-signer tests in crossval_strangers.py -- the only protocols that measure generalization
here:

  feature    static/v4 (features.static_feature_v4, 112-D): the 101-D palm-normalized
             static/v3 vector plus an 11-value thumb block. The letters the 101-D vector
             confuses are the fists (A, E, M, N, S, T) and D/X, which differ in where the
             thumb sits.
  data       the author's four sessions PLUS other people's hands (strangers.py): the ASLNow
             records (all 24 letters, multiple participants, captured with the browser's own
             landmarker) and the 218-signer digit photos whose handshape is a letter (0/O,
             2/V, 6/W, 9/F). This is the largest gain in the project: on the author's own
             unseen day the cross-day fold goes 0.778 -> 0.873 and the pooled figure 0.867 ->
             0.913 (same recipe, strangers out vs in), because the strangers teach invariances
             that transfer; on strangers the forest goes from reading one person to reading
             people (crossval_strangers.py: held-out ASLNow 0.79, held-out 218-signer O/V/W/F
             0.93 where the one-signer forest read V at 0.17, ASLNow fifths 0.95).
             --no-strangers trains the one-signer version of this recipe (0.867 / 0.778); the
             previous release's forest, with its geometry filter and 100 trees, measured
             0.861 / 0.763.
  jitter     static_aug.jitter_frames(sigma 0.12 palm units, 4 copies + originals) on
             TRAINING frames only, re-featurized. Before the strangers this was the largest
             lever (+0.09 pooled / +0.14 cross-day over no augmentation on the author alone).
             The rotation augmentation it replaced lowered the pooled number (0.782 -> 0.759)
             and was only ever justified by within-session confidence.
  no filter  static_aug.filter_training(ruleset="strong") dropped 137 of the author's frames
             that violated their letter's defining geometry (a near-straight X, a G with an
             extended middle finger). Its thresholds were set on one hand: applied to the
             strangers it discards half their X, P and R frames, and with strangers in the
             training set it costs on every axis (ASLNow fifths 0.945 -> 0.903; three seeds).
             RULESET is "none"; --ruleset strong is kept for the ablation.
  forest     RandomForestClassifier(n_estimators=80, min_samples_leaf=5, max_features="sqrt",
             random_state=0): about 200k nodes, every one shipped to the browser inside
             models.json. 100 trees score the same at 250k nodes; 60 trees the same at 150k
             but leave one relaxed-hand hold above every vote floor (thresholds.py).

The idle-hand check in idle_gate.py and the hold replay in replay_static.py are the two
runtime measurements a retrained forest must pass before models.json is re-exported; the
vote floor in thresholds.py was raised to 0.75 for this forest because it reads a relaxed
hand as a loose G at 0.55-0.70.

The ablation printed first is older and narrower: it shows why the feature divides by palm
size at all. The original 42-D feature subtracted min(x)/min(y) but never divided by hand
size, so it encoded camera distance; on a within-burst split that reads as accuracy, in use
it reads as failure. Read that table before trusting either number: the palm-normalized
feature scores LOWER on the leaky split, because removing scale removes a cue the old forest
used to re-identify frames from one capture burst. That is the fix working.
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
import static_aug as A
import strangers as ST

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "static_sequences.npz")
OUT = os.path.join(HERE, "model_static.p")
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
W, H = 1920, 1080

#: The shipped recipe. crossval_static.py imports these so the measured protocol and the
#: shipped forest can never drift apart silently.
FEATURE_TAG = "static/v4"
FEATFN, FEATURE_DIM = F.static_feature_for(FEATURE_TAG)
RULESET = "none"
JITTER_SIGMA, JITTER_COPIES = 0.12, 4
FOREST = dict(n_estimators=80, min_samples_leaf=5, max_features="sqrt")

#: Frames of one letter more than this many seconds apart belong to different holds (the
#: signer dropped the hand and re-formed the letter). Used for the hold ids the OOF carries.
HOLD_GAP = 0.5


def legacy_feature(P):
    """The feature the notebooks' model.p used: translation-normalized, NOT scale-normalized."""
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
            P = F.canonicalize_handedness(P, modal_handedness(handed[c][idx]))
        per_class.append(P)
    return per_class


def modal_handedness(labels):
    """The commonest real label ("Left"/"Right") among per-frame labels, or None if there is
    none. MediaPipe flips its label on a few frames of a held sign (two frames of the S2 M
    hold read "Right" inside 59 "Left" ones while the coordinates barely move), and a frame
    canonicalized by its own flipped label enters training mirrored."""
    labs = [str(x) for x in labels if str(x)[:1] in ("L", "R")]
    return max(sorted(set(labs)), key=labs.count) if labs else None


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


def resolve_extra(path):
    """An --extra path as given, or the same name under temporal/ when that is where it is.

    The README's run order is written from the repository root while the session files live
    beside this script, so `--extra static_s2.npz` used to raise FileNotFoundError.
    """
    if os.path.exists(path):
        return path
    alt = os.path.join(HERE, path)
    if os.path.exists(alt):
        return alt
    raise SystemExit(f"--extra {path}: not found as given nor as {alt}")


def load_extra(path, holds=False, per_frame_handedness=False):
    """Frames from a later capture session, as written by collect_motion --static-letters.

    Merged in rather than replacing: the archive still carries most of the variety, and the
    point of a second session is to teach the model that the same letter can look different
    on a different day. One session is why cross-session accuracy sat at 0.52 while the
    in-session number read 0.97.

    Returns {class index: (n,21,2)}; with holds=True, {class index: ((n,21,2), (n,) hold id)}
    where a hold is a run of one letter whose consecutive stamps are within HOLD_GAP (the
    unit the out-of-fold posteriors and the word simulator count by).

    Handedness is canonicalized per contiguous run of one letter in the file, with that run's
    modal label -- the rule load() applies to the archive and the one
    features.canonicalize_handedness documents; per-frame labels flip on a few frames and
    would mirror those frames into training. frame_size is required: an npz without it would
    otherwise train at a guessed aspect, and a wrong aspect is invisible in every number this
    pipeline prints (label_events.py and train_motion.py refuse for the same reason).
    Labels outside LETTERS (digits, prompts) are skipped, and a file that contributes no frame
    at all is reported rather than merged silently.

    per_frame_handedness=True is the previous release's behavior (each frame mirrored by its
    own label); crossval_static.py --legacy uses it so the regression guard reproduces the
    shipped 0.759 / 0.659 exactly. The two rules differ on 2 of the 2,500 S2-S4 frames.
    """
    d = np.load(path, allow_pickle=True)
    if "frame_size" not in d:
        raise SystemExit(f"{path} carries no frame_size; the aspect ratio is load-bearing and "
                         "is not guessed. Re-record with collect_motion.py or add the (W,H) the "
                         "session was captured at.")
    lm, letters = d["lm"], [str(x) for x in d["letters"]]
    handed = [str(x) for x in d["handed"]] if "handed" in d else ["Unknown"] * len(lm)
    stamps = np.asarray(d["stamps"], dtype=np.float64) if "stamps" in d else np.arange(len(lm))
    wh = np.asarray(d["frame_size"]).ravel()
    per_class = {c: [] for c in range(24)}
    per_hold = {c: [] for c in range(24)}
    skipped = set()
    hold_no = -1
    i = 0
    while i < len(letters):
        j = i
        while j < len(letters) and letters[j] == letters[i]:
            j += 1
        lab = letters[i]
        if lab in LETTERS:
            c = LETTERS.index(lab)
            P = F.to_isotropic(lm[i:j][:, :, :2], int(wh[0]), int(wh[1]))
            if per_frame_handedness:
                P = np.stack([F.canonicalize_handedness(P[k], modal_handedness([handed[i + k]]))
                              for k in range(j - i)])
            else:
                P = F.canonicalize_handedness(P, modal_handedness(handed[i:j]))
            for k in range(i, j):
                if k == i or stamps[k] - stamps[k - 1] > HOLD_GAP:
                    hold_no += 1
                per_class[c].append(P[k - i])
                per_hold[c].append(hold_no)
        else:
            skipped.add(lab)
        i = j
    if not any(per_class.values()):
        print(f"WARNING: {os.path.basename(path)} contributed no frame -- its labels "
              f"{sorted(skipped)} are not static letters; a digit file belongs to "
              "train_digits.py, not here")
    if holds:
        return {c: (np.stack(v), np.array(per_hold[c], int)) for c, v in per_class.items() if v}
    return {c: np.stack(v) for c, v in per_class.items() if v}


def training_rows(per_class, featfn=None, ruleset=RULESET, sigma=JITTER_SIGMA,
                  copies=JITTER_COPIES, seed=0):
    """Feature rows for a TRAINING set: filter, jitter, featurize. -> (X, y, frames_dropped).

    Row order is every class's kept originals first, then each jittered copy as a block over
    all classes. Row order changes the forest's bootstrap draws, so it is fixed here and
    shared with crossval_static.py rather than left to each caller. The jitter RNG is seeded
    by the block's content (static_aug.content_rng), so the same post-filter class block
    always draws the same noise regardless of call order or which other classes are present.
    A fold's block (three sessions) and the final fit's block (four sessions) are different
    arrays for every letter the held-out session contains, so their draws are unrelated; only
    a letter absent from the held-out session shares its augmented rows with the shipped
    forest (S1 fold 0 of 24 letters, S2 1, S3 18, S4 20 -- and those 39 shared blocks jitter
    byte-identically).
    """
    featfn = FEATFN if featfn is None else featfn
    n_raw = sum(len(P) for P in per_class)
    kept = A.filter_training(per_class, ruleset)
    variants = [A.jitter_frames(P, sigma, copies, A.content_rng(seed, P)) if len(P) else [P]
                for P in kept]
    n_var = max(len(v) for v in variants) if variants else 1
    X, y = [], []
    for v in range(n_var):
        for c, vs in enumerate(variants):
            if v < len(vs) and len(vs[v]):
                f = featfn(vs[v])
                X.append(f)
                y.append(np.full(len(f), c))
    return np.concatenate(X), np.concatenate(y), n_raw - sum(len(P) for P in kept)


def make_forest(seed=0, n_jobs=4):
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **FOREST)


def node_count(model):
    return int(sum(e.tree_.node_count for e in model.estimators_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="one or more .npz files from collect_motion --static-letters. Later "
                         "sessions are merged in, not substituted: the point is to show the "
                         "model the same letter on different days, so the variety accumulates. "
                         "A bare file name is also looked up under temporal/.")
    ap.add_argument("--seed", type=int, default=0, help="forest random_state and jitter seed")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-strangers", action="store_true",
                    help="train on the author's sessions only (the previous release's data)")
    ap.add_argument("--ruleset", default=RULESET, choices=sorted(A.RULESETS),
                    help="defining-geometry training filter; the shipped forest uses none")
    args = ap.parse_args()

    per_class = load()
    sessions = ["S1"]
    merged = {}
    for path in args.extra:
        path = resolve_extra(path)
        got = load_extra(path)
        sessions.append(os.path.splitext(os.path.basename(path))[0])
        print(f"merging {os.path.basename(path)}: " +
              ", ".join(f"{LETTERS[c]}={len(v)}" for c, v in sorted(got.items())))
        for c, v in got.items():
            merged[c] = np.concatenate([merged[c], v]) if c in merged else v
    if merged:
        print()
        per_class = [np.concatenate([P, merged[c]]) if c in merged else P
                     for c, P in enumerate(per_class)]
    n_frames = sum(len(P) for P in per_class)
    tr, te = contiguous_split(per_class)
    stranger_sources, n_strangers = [], 0
    if not args.no_strangers:
        strangers, stranger_sources = ST.load_strangers()
        n_strangers = sum(len(P) for P in strangers)
        print(f"strangers: {n_strangers} frames from {stranger_sources} -- "
              + ", ".join(f"{LETTERS[c]}={len(P)}" for c, P in enumerate(strangers) if len(P)))

    print("=== scale-robustness ablation (contiguous per-class split) ===")
    print("test landmarks rescaled about the hand centroid; training never sees the rescale\n")
    print(f"{'k':>6}  {'legacy 42-D':>12}  {FEATURE_TAG:>16}")
    rows = {}
    for name, fn in (("legacy", legacy_feature), ("shipped", FEATFN)):
        Xtr, ytr = build(tr, fn)
        model = make_forest(args.seed).fit(Xtr, ytr)
        rows[name] = []
        for k in (0.7, 0.85, 1.0, 1.2, 1.5):
            Xte, yte = build(te, fn, rescale=k)
            rows[name].append(accuracy_score(yte, model.predict(Xte)))
    for i, k in enumerate((0.7, 0.85, 1.0, 1.2, 1.5)):
        print(f"{k:>6.2f}  {rows['legacy'][i]:>12.3f}  {rows['shipped'][i]:>16.3f}")
    print(f"\nlegacy spread across k: {max(rows['legacy']) - min(rows['legacy']):.3f}"
          f"   {FEATURE_TAG} spread: {max(rows['shipped']) - min(rows['shipped']):.3f}")

    # Ship a model trained on everything; the split above exists to characterize, not to select.
    train = ST.merge(per_class, strangers) if not args.no_strangers else per_class
    Xall, yall, dropped = training_rows(train, ruleset=args.ruleset, seed=args.seed)
    print(f"\ntraining on {len(Xall)} rows: {n_frames} frames from {len(sessions)} sessions"
          f" + {n_strangers} stranger frames, {dropped} dropped by the {args.ruleset!r} rules, "
          f"x{1 + JITTER_COPIES} with jitter sigma {JITTER_SIGMA} (originals kept), "
          f"{FEATURE_DIM}-D {FEATURE_TAG}")
    model = make_forest(args.seed).fit(Xall, yall)
    assert list(model.classes_) == list(range(len(LETTERS))), \
        "a letter has no training frame; the segmenter indexes classes by position"
    blob = {"model": model, "classes": LETTERS, "feature": FEATURE_TAG,
            "aspect": [W, H], "n_train": int(len(Xall)), "n_frames": int(n_frames),
            "frames_dropped": int(dropped), "filter": args.ruleset,
            "strangers": stranger_sources, "n_stranger_frames": int(n_strangers),
            "augment": {"kind": "jitter", "sigma_palm": JITTER_SIGMA, "copies": JITTER_COPIES,
                        "originals_kept": True, "rng": "static_aug.content_rng", "seed": args.seed},
            "forest": dict(FOREST, random_state=args.seed), "sessions": sessions}
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"\nwrote {args.out}: {len(Xall)} rows, {len(LETTERS)} classes, feature {FEATURE_TAG}, "
          f"{len(model.estimators_)} trees, {node_count(model)} nodes, "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB pickle")


if __name__ == "__main__":
    main()

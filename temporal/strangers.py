"""Other people's hands: the public landmark sets the static letters train and test on.

Everything in RandomForest/data and temporal/static_s*.npz is one signer. These sets are not,
and they are the only reason the hosted page can say anything measured about a visitor:

  ASLNow      temporal/aslnow.npz (ingest_aslnow.py): 1,874 static-letter records from multiple
              participants, captured with the Web Hand Landmarker the page runs. All 24 letters.
              No participant ids, so it cannot be split by signer.
  Ankara      temporal/digits_ankara.npz (ingest_images.py): 218 students photographed signing
              the digits 0-9. Four digits ARE letters -- 0/O, 2/V, 6/W, 9/F share the handshape
              exactly -- so those frames are letter frames from 218 more hands. (4 is a
              spread-finger B and is left out; 1 is close to D but tucks the thumb.)
  ASL-HG      temporal/aslhg.npz (ingest_aslhg.py): 23,984 photographs of all 24 letters from
              10 named volunteers, 100 per (signer, letter). The first source here with real
              signer ids, which is what crossval_signers.py's leave-one-SIGNER-out needs.
              TRAINING TAKES IT CAPPED at ASLHG_CAP = 25 per (signer, letter) = 6,000 frames.
              The cap is not a size compromise, it is the better model: on the shipped recipe
              at seed 0 the author's own pooled leave-one-session-out is 0.9256 at the cap and
              0.9147 / 0.9086 with every ASL-HG frame in instead (crossval_static.py
              --aslhg-cap 100 and --aslhg-cap 0 -- the same 23,984 frames both times, differing
              only in the row order the draw leaves them in, which is itself worth 0.006 here).
              Uncapped, 24,000 frames from ten hands outvote the 3,807 the author's own
              sessions contribute, and the page is meant to read him too.
  ayuraj      temporal/ayuraj.npz (ingest_ayuraj.py): 1,111 letter frames from 5 signers.
              NEVER TRAINED ON, permanently -- see NEVER_TRAIN below.

Both training sets and the two evaluation sets are loaded into the same canonical frame as the
author's sessions: isotropic coordinates (x scaled by the capture's width/height), mirrored
where needed onto the right-hand convention. Every consumer -- train_static.py,
crossval_static.py, crossval_strangers.py, crossval_signers.py -- goes through here so no set
can be canonicalized two ways.

Measured effect of training on the strangers -- crossval_static.py, leave-one-session-out over
the author's four sessions, strangers always on the training side, the shipped recipe
throughout, seed 0, pooled / cross-day:

    the author's four sessions alone        0.873 / 0.782   (--henry-only)
    + ASLNow + the Ankara O/V/W/F photos    0.913 / 0.873   (--no-aslhg)
    + ASL-HG at the cap                     0.926 / 0.895   (the shipped training set)

The strangers teach invariances that transfer back to the author's own unseen day: he gains
five points on a day the forest has never seen from photographs of people who are not him. Cross-signer evidence for the
shipped forest is in crossval_strangers.py (each set scored with itself held out) and
crossval_signers.py (one of ten real people held out).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
ASLNOW = os.path.join(HERE, "aslnow.npz")
ANKARA = os.path.join(HERE, "digits_ankara.npz")
ASLHG = os.path.join(HERE, "aslhg.npz")
AYURAJ = os.path.join(HERE, "ayuraj.npz")
#: Digits whose ASL handshape is a letter's, exactly.
DIGIT_LETTERS = {"0": "O", "2": "V", "6": "W", "9": "F"}
EMPTY = np.zeros((0, 21, 2))

#: Files that may NEVER reach a training set, whatever a future recipe wants.
#:
#: A set that is never trained on is the only one whose number cannot drift: it is measured
#: the same way against every forest that ships, so a fall in it is a regression and not a
#: change of protocol. Every other stranger set here has crossed over -- ASLNow did, the
#: Ankara digits did, ASL-HG does in this release -- and each crossing turned a "never seen"
#: figure into an in-sample one that had to be relabeled historical. ayuraj (ingest_ayuraj.py,
#: 5 signers, CC0) is kept on the other side permanently so one such number always exists.
#:
#: This is an assertion and not a convention: training_source() below is what every path
#: bound for a training set passes through, and tests/test_strangers.py builds the real
#: training set and checks that not one ayuraj frame is in it.
NEVER_TRAIN = (AYURAJ,)

#: ASL-HG frames per (signer, letter) on the TRAINING side, and the seed that picks them.
#: The cap is part of the recipe, so its draw is fixed rather than following the forest seed:
#: a seed sweep then measures the forest and the jitter, not which 25 photographs were kept.
ASLHG_CAP = 25
CAP_SEED = 0


def training_source(path):
    """Gate for every file that reaches a TRAINING set. Raises on anything in NEVER_TRAIN."""
    if os.path.abspath(path) in {os.path.abspath(p) for p in NEVER_TRAIN}:
        raise AssertionError(
            f"{os.path.basename(path)} is in strangers.NEVER_TRAIN and must never be trained "
            "on. It is this project's only permanently held-out cross-signer set; training on "
            "it would leave none. Use load_ayuraj() for evaluation and take training frames "
            "from somewhere else.")
    return path


def empty_per_class():
    return [np.zeros((0, 21, 2)) for _ in LETTERS]


def load_aslnow(path=ASLNOW):
    """-> (per_class [24 x (n,21,2)], P (N,21,2), letters (N,), names (N,)) for the static-letter
    records only; the J/Z stills are skipped (a still of a J is an I to a static classifier)."""
    d = np.load(path, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    letters, names, mirror = np.asarray(d["letters"]), np.asarray(d["names"]), np.asarray(d["mirror"])
    w, h = [int(v) for v in np.asarray(d["aspect"]).ravel()]
    P = F.to_isotropic(lm[..., :2], w, h)
    P[mirror, :, 0] = -P[mirror, :, 0]
    keep = np.isin(letters, LETTERS)
    P, letters, names = P[keep], letters[keep], names[keep]
    per = [P[letters == L] if (letters == L).any() else np.zeros((0, 21, 2)) for L in LETTERS]
    return per, P, letters, names


def load_ankara_letters(path=ANKARA, mapping=DIGIT_LETTERS):
    """-> (per_class, P, letters, signer) for the digit photos whose handshape is a letter."""
    d = np.load(path, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    labels = np.asarray([str(x) for x in d["letters"]])
    handed = np.asarray([str(x) for x in d["handed"]]) if "handed" in d else np.full(len(lm), "Unknown")
    if "frame_size" not in d:
        raise SystemExit(f"{path} carries no frame_size; the aspect ratio is load-bearing")
    aspect = F.frame_aspect(d["frame_size"], len(lm), where=path)
    signer = np.asarray([str(x) for x in d["signer"]]) if "signer" in d else np.arange(len(lm)).astype(str)
    keep = np.isin(labels, list(mapping))
    P = F.apply_aspect(lm[keep][..., :2], aspect if len(aspect) == 1 else aspect[keep])
    for i, lab in enumerate(handed[keep]):
        P[i:i + 1] = F.canonicalize_handedness(P[i:i + 1], lab if lab[:1] in ("L", "R") else None)
    letters = np.asarray([mapping[x] for x in labels[keep]])
    per = [P[letters == L] if (letters == L).any() else np.zeros((0, 21, 2)) for L in LETTERS]
    return per, P, letters, signer[keep]


def per_class_of(P, letters):
    """[24 x (n,21,2)] from a flat frame array and its letter labels."""
    return [P[letters == L] if (letters == L).any() else EMPTY.copy() for L in LETTERS]


def cap_per_signer_letter(letters, signer, k, seed=CAP_SEED):
    """Indices of at most `k` frames per (signer, letter), drawn once and deterministically.

    The draw is a permutation rather than the first k in file order: ASL-HG's filenames are
    P<k>_<CLASS>_<n>.jpg with n running over one continuous shoot, so the first 25 are 25
    consecutive shutter presses of one pose and the last 75 carry the variation the cap is
    supposed to sample from. Signers are walked in sorted() order and letters in LETTERS
    order, so the draw depends on nothing but `seed`.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for s in sorted(set(signer)):
        for L in LETTERS:
            idx = np.where((signer == s) & (letters == L))[0]
            if len(idx):
                keep.append(rng.permutation(idx)[:k])
    return np.concatenate(keep) if keep else np.zeros(0, int)


def load_aslhg(cap=None, path=ASLHG, seed=CAP_SEED):
    """-> (per_class, P (N,21,2), letters (N,), signer (N,)) for ASL-HG, canonicalized.

    cap=None returns all 23,984 frames (what crossval_signers.py holds out by signer);
    cap=k keeps at most k per (signer, letter) -- ASLHG_CAP is what ships on the training side.
    Handedness is canonicalized per image by MediaPipe's own label, as for the Ankara photos:
    ASL-HG reads 23,974 "Left" and 10 "Right", so it is an unmirrored right-hand convention
    and a per-image decision is the honest one.
    """
    d = np.load(path, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    labels = np.asarray([str(x) for x in d["letters"]])
    handed = np.asarray([str(x) for x in d["handed"]])
    signer = np.asarray([str(x) for x in d["signer"]])
    aspect = F.frame_aspect(d["frame_size"], len(lm), where=path)
    keep = np.isin(labels, LETTERS)
    P = F.apply_aspect(lm[keep][..., :2], aspect if len(aspect) == 1 else aspect[keep])
    for i, lab in enumerate(handed[keep]):
        P[i:i + 1] = F.canonicalize_handedness(P[i:i + 1], lab if lab[:1] in ("L", "R") else None)
    letters, signer = labels[keep], signer[keep]
    if cap is not None:
        sel = cap_per_signer_letter(letters, signer, cap, seed)
        P, letters, signer = P[sel], letters[sel], signer[sel]
    return per_class_of(P, letters), P, letters, signer


def load_ayuraj(path=AYURAJ):
    """-> (per_class, P, letters, signer) for the PERMANENT never-train holdout.

    For evaluation only; NEVER_TRAIN is what enforces that, and training_source() is what any
    training path has to pass. Its images are crops tight around a hand, so MediaPipe finds one
    in 72.33% of them and the misses concentrate on the fists (T 8 of 65, S 14 of 70, M 15 of
    70). 1,111 of the 1,819 detected frames are the 24 static letters; the fists' cells are
    single-digit n and should not be quoted per letter.
    """
    d = np.load(path, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    labels = np.asarray([str(x) for x in d["letters"]])
    handed = np.asarray([str(x) for x in d["handed"]])
    signer = np.asarray([str(x) for x in d["signer"]])
    aspect = F.frame_aspect(d["frame_size"], len(lm), where=path)
    keep = np.isin(labels, LETTERS)
    P = F.apply_aspect(lm[keep][..., :2], aspect if len(aspect) == 1 else aspect[keep])
    for i, lab in enumerate(handed[keep]):
        P[i:i + 1] = F.canonicalize_handedness(P[i:i + 1], lab if lab[:1] in ("L", "R") else None)
    letters, signer = labels[keep], signer[keep]
    return per_class_of(P, letters), P, letters, signer


def load_strangers(aslnow=True, ankara=True, aslhg=True, cap=ASLHG_CAP, seed=CAP_SEED):
    """Per-class TRAINING frames from the stranger sets, canonical. -> (per_class, sources)

    Order is fixed -- Ankara, then ASLNow, then ASL-HG -- because row order changes a forest's
    bootstrap draws, and every number quoted anywhere in this project was measured in this one.
    Every file opened here passes training_source() first, so ayuraj cannot arrive by accident
    or by a later edit.
    """
    per = empty_per_class()
    sources = []
    if ankara and os.path.isfile(training_source(ANKARA)):
        got, _, _, _ = load_ankara_letters()
        per = merge(per, got)
        sources.append("ankara:" + "".join(sorted(DIGIT_LETTERS.values())))
    if aslnow and os.path.isfile(training_source(ASLNOW)):
        got, _, _, _ = load_aslnow()
        per = merge(per, got)
        sources.append("aslnow")
    if aslhg and os.path.isfile(training_source(ASLHG)):
        got, _, _, sg = load_aslhg(cap=cap, seed=seed)
        per = merge(per, got)
        sources.append(f"aslhg:cap{cap}" if cap is not None else "aslhg:all")
    return per, sources


def merge(*per_lists):
    """Concatenate per-class lists; empty classes stay (0,21,2)."""
    out = []
    for c in range(len(LETTERS)):
        parts = [p[c] for p in per_lists if len(p[c])]
        out.append(np.concatenate(parts) if parts else np.zeros((0, 21, 2)))
    return out


if __name__ == "__main__":
    per, src = load_strangers()
    print("sources:", src)
    print("frames per letter:", ", ".join(f"{L}={len(P)}" for L, P in zip(LETTERS, per)))
    print("total:", sum(len(P) for P in per))
    _, _, _, sg = load_aslhg()
    print(f"ASL-HG uncapped: {len(sg)} frames, {len(set(sg))} signers "
          f"({', '.join(f'{s}={int((sg == s).sum())}' for s in sorted(set(sg), key=lambda x: int(x[1:])))})")
    _, Pa, La, sa = load_ayuraj()
    print(f"ayuraj (NEVER trained on): {len(Pa)} letter frames, {len(set(sa))} signers")

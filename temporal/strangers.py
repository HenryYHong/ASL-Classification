"""Other people's hands: the two public landmark sets the static letters train and test on.

Everything in RandomForest/data and temporal/static_s*.npz is one signer. These two sets are
not, and they are the only reason the hosted page can say anything measured about a visitor:

  ASLNow      temporal/aslnow.npz (ingest_aslnow.py): 1,874 static-letter records from multiple
              participants, captured with the Web Hand Landmarker the page runs. All 24 letters.
  Ankara      temporal/digits_ankara.npz (ingest_images.py): 218 students photographed signing
              the digits 0-9. Four digits ARE letters -- 0/O, 2/V, 6/W, 9/F share the handshape
              exactly -- so those frames are letter frames from 218 more hands. (4 is a
              spread-finger B and is left out; 1 is close to D but tucks the thumb.)

Both are loaded into the same canonical frame as the author's sessions: isotropic coordinates
(x scaled by the capture's width/height), mirrored where needed onto the right-hand convention.
Every consumer -- train_static.py, crossval_static.py, crossval_strangers.py -- goes through
here so the two sets cannot be canonicalized two ways.

Measured effect of training on them (crossval_static.py, leave-one-session-out over the
author's four sessions, strangers always on the training side, same recipe otherwise): pooled
0.867 -> 0.913 and the cross-day fold 0.778 -> 0.873, i.e. the strangers teach invariances that
transfer back to the author's own unseen day. Cross-signer evidence for the shipped forest is in
crossval_strangers.py, where each set is scored with the other one held out.
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
#: Digits whose ASL handshape is a letter's, exactly.
DIGIT_LETTERS = {"0": "O", "2": "V", "6": "W", "9": "F"}
EMPTY = np.zeros((0, 21, 2))


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
    w, h = [int(v) for v in np.asarray(d["frame_size"]).ravel()[:2]]
    signer = np.asarray([str(x) for x in d["signer"]]) if "signer" in d else np.arange(len(lm)).astype(str)
    keep = np.isin(labels, list(mapping))
    P = F.to_isotropic(lm[keep][..., :2], w, h)
    for i, lab in enumerate(handed[keep]):
        P[i:i + 1] = F.canonicalize_handedness(P[i:i + 1], lab if lab[:1] in ("L", "R") else None)
    letters = np.asarray([mapping[x] for x in labels[keep]])
    per = [P[letters == L] if (letters == L).any() else np.zeros((0, 21, 2)) for L in LETTERS]
    return per, P, letters, signer[keep]


def load_strangers(aslnow=True, ankara=True):
    """Per-class training frames from both sets, in the canonical frame. -> (per_class, sources)"""
    per = empty_per_class()
    sources = []
    if ankara and os.path.isfile(ANKARA):
        got, _, _, _ = load_ankara_letters()
        per = merge(per, got)
        sources.append("ankara:" + "".join(sorted(DIGIT_LETTERS.values())))
    if aslnow and os.path.isfile(ASLNOW):
        got, _, _, _ = load_aslnow()
        per = merge(per, got)
        sources.append("aslnow")
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

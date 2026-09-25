"""Write docs/reference.json: one real hand per letter, for the chart on the page.

The page asks people to sign letters it has never explained. A reference chart is the missing
half of that, and every obvious way to get one is worse than this one: photographs of hands are
somebody's copyright, and a drawing made by hand would show what I think the letter looks like
rather than what the classifier was actually taught.

So the chart is drawn from the training data. For each letter, this picks the MEDOID of that
letter's palm-normalized frames -- the real frame whose shape is closest to every other frame of
the same letter -- and writes its 21 landmarks. Not the mean: averaging hands at different
rotations produces a shape no hand can make, with fingers shrunk toward the palm. The medoid is
a frame that actually happened.

Source is the author's own sessions, deliberately. ASL-HG would give a cleaner medoid (ten
people, one protocol) but it is CC BY 4.0, and a redistributed chart derived from it would carry
an attribution requirement onto every page that embeds it. The author's own frames carry none.
The cost is that the chart shows one person's hands, which is stated on the page.

J and Z have no static shape. Their entry carries the launch pose -- the shape the hand starts
in, which is what the motion branch is watching for -- plus a `motion` string the page renders
as a note rather than pretending a still frame is the letter.

    ../.venv/bin/python docs/make_reference.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "temporal"))
import features as F  # noqa: E402
import train_static as T  # noqa: E402

OUT = os.path.join(HERE, "reference.json")
#: J starts from I and Z from a pointing index; the motion branch keys on exactly these.
MOTION = {"J": ("I", "little finger traces a J: down, curve left, and up"),
          "Z": ("D", "index finger draws a Z in the air: across, down-left, across")}

#: How to make each letter, in words. The skeletons cannot carry this on their own: they are
#: 2-D projections of a hand held at whatever angle the signer held it, so the six fists
#: (A, E, M, N, S, T) come out as nearly the same blob and the sideways letters (G, H, P, Q)
#: come out foreshortened. The thing that separates those letters is where the THUMB sits and
#: which way the hand points, and a sentence says that better than a silhouette does.
HOW = {
    "A": "Fist, thumb alongside the index finger, not tucked in.",
    "B": "Flat hand, fingers straight and together, thumb folded across the palm.",
    "C": "Fingers and thumb curved into a C, as if holding a cup.",
    "D": "Index straight up, the other fingertips meeting the thumb.",
    "E": "Fingertips curled down onto the thumb, which lies across the palm.",
    "F": "Index and thumb touching in a circle, the other three fingers straight up.",
    "G": "Index and thumb extended parallel and pointing sideways.",
    "H": "Index and middle extended together, pointing sideways.",
    "I": "Little finger straight up, the rest in a fist.",
    "J": "Start at I, then trace a J with the little finger.",
    "K": "Index and middle up in a V, thumb between them.",
    "L": "Index straight up, thumb straight out: an L.",
    "M": "Thumb tucked under three fingers: index, middle and ring.",
    "N": "Thumb tucked under two fingers: index and middle.",
    "O": "All the fingers curved round to meet the thumb in an O.",
    "P": "The K shape, turned to point downward.",
    "Q": "The G shape, turned to point downward.",
    "R": "Index and middle crossed over each other.",
    "S": "Fist with the thumb across the front of the fingers.",
    "T": "Thumb pushed up between the index and middle fingers.",
    "U": "Index and middle straight up, held together.",
    "V": "Index and middle straight up, spread apart.",
    "W": "Index, middle and ring straight up, spread.",
    "X": "Index bent into a hook, the rest in a fist.",
    "Y": "Thumb and little finger out, the rest folded in.",
    "Z": "Point the index finger, then draw a Z in the air.",
}


def medoid(P):
    """The real frame whose palm-normalized shape is closest to all the others. -> (21,2)"""
    S = F.palm_scale(P)[:, None, None]
    Q = (P - F.palm_centre(P)[:, None, :]) / S
    if len(Q) == 1:
        return Q[0]
    # Mean pairwise landmark distance, computed in blocks so a large letter block stays cheap.
    d = np.zeros(len(Q))
    for i in range(len(Q)):
        d[i] = np.linalg.norm(Q - Q[i], axis=-1).mean()
    return Q[int(d.argmin())]


def author_frames():
    """Every letter frame of the author's four sessions, per class."""
    per = T.load()
    merged = {}
    for name in ("static_s2.npz", "static_s3.npz", "static_s4.npz"):
        for c, v in T.load_extra(T.resolve_extra(name)).items():
            merged[c] = np.concatenate([merged[c], v]) if c in merged else v
    return [np.concatenate([P, merged[c]]) if c in merged else P for c, P in enumerate(per)]


def main():
    per = author_frames()
    out, missing = {}, []
    for c, letter in enumerate(T.LETTERS):
        if not len(per[c]):
            missing.append(letter)
            continue
        q = medoid(per[c])
        out[letter] = {"lm": [[round(float(x), 4), round(float(y), 4)] for x, y in q],
                       "how": HOW[letter]}
    for letter, (pose, note) in MOTION.items():
        if pose in out:
            out[letter] = {"lm": out[pose]["lm"], "motion": note, "pose": pose, "how": HOW[letter]}
    if missing:
        print(f"no frames for {missing}; those letters are omitted from the chart")
    without = sorted(k for k in out if not out[k].get("how"))
    if without:
        raise SystemExit(f"no written description for {without}: the skeletons alone do not "
                         f"separate the fists or the sideways letters, so every cell needs one")
    payload = {"letters": out,
               "source": "medoid of the author's own training frames, palm-normalized",
               "note": "one signer's hands; see README.md"}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    n = os.path.getsize(OUT)
    print(f"wrote {OUT}: {len(out)} letters, {n:,} B")
    print(f"  static {sorted(k for k in out if 'motion' not in out[k])}")
    print(f"  motion {sorted(k for k in out if 'motion' in out[k])}")


if __name__ == "__main__":
    main()

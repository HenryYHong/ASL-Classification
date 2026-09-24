"""Why U and R are the two worst letters, and why the obvious fix does not ship.

U and R are 553 of the roughly 1,312 errors a leave-one-signer-out run makes -- 42% of
everything the forest gets wrong on somebody else's hand. crossval_signers.py puts them at
U 0.671 (329 of 1,000 read as R) and R 0.703 (224 read as U), with no other letter under 0.76.

THE CAUSE. U is the index and middle fingers extended side by side; R is the same two fingers
CROSSED. That is a difference of sign, not of magnitude, and static/v4 has no signed
finger-to-finger value in it: 55 of its 112 numbers are pairwise DISTANCES, which are identical
whichever way round the two tips are, and the 42 raw coordinates are in the image frame, so
they express the swap only at one hand orientation. The vector is structurally blind to it.

Measured here: one signed number -- the index/middle offset at the TIP, projected on the hand's
own knuckle axis (index MCP -> middle MCP) so it rotates with the hand -- separates U from R
with balanced accuracy 1.0000 on ASL-HG, at a single fixed threshold, for all ten signers
independently. The same threshold carries to sets it was not fitted on: 0.9907 on ayuraj,
0.9223 on ASLNow, 0.9099 on the author's own sessions. The forest, given 112 values and nine
other signers, manages 0.687.

WHY IT DOES NOT SHIP. Two reasons, both measured, and the second is the one that matters.

  1. The training jitter erases it. static_aug jitters every landmark independently at sigma
     0.12 palm units, and the U/R margin is about 0.086 palm units, so the cue survives at
     1.0000 clean and 0.707 jittered. Averaging the offset over PIP, DIP and TIP recovers
     almost nothing (0.719): the noise is larger than the signal at every height. Note that
     global translation and scale are already invariances of this feature, so independent
     noise is the ONLY part of the jitter that reaches the distance block -- and it is exactly
     the part that destroys crossing.

  2. It fails the idle gate. Added raw, the block lifts leave-one-signer-out U to 0.841 and R
     to 0.809 and pooled 0.9453 -> 0.9517, and pushes the idle maximum for G from 0.7087 to
     0.7623 and for A from 0.5130 to 0.5579 -- both over their floors. Fading the value out
     when the two fingers are not extended (index/middle straightness below 0.80, which is
     every U and every R and no resting hand) fixes the mechanism and does better still:
     U 0.867, R 0.852, pooled 0.9547, and +0.023 to +0.029 on the never-train holdout with a
     signer-clustered CI excluding zero at each of three seeds. It passes the idle gate at
     seeds 0, 1 and 2 -- and emits at seeds 3 and 4, where idle G reaches 0.8038.

That last line is the whole reason this file exists rather than a static/v5 tag. The candidate
is better on every accuracy axis measured and still fails the only gate that matters, and it
fails it in the seeds nobody looked at. It is also what first showed that the SHIPPED recipe
fails at four of eight seeds; idle_gate.MIN_SEEDS came out of this probe.

What would make it shippable, in the order worth trying: a jitter model that perturbs the hand
pose rather than each landmark independently, since the real landmark error is correlated and
the current model destroys fine relative geometry to buy an invariance the feature already has;
or floors set from the idle distribution over many seeds rather than from one draw.

    ./.venv/bin/python temporal/crossing_probe.py              # separation, every set
    ./.venv/bin/python temporal/crossing_probe.py --jitter     # what sigma 0.12 does to it
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402

LETTERS = T.LETTERS
#: Fitted once on ASL-HG and then never refitted; every other set is scored at this value.
THRESHOLD = 0.123
#: Below this straightness the fingers are not extended and a crossing value means nothing.
EXTENDED = 0.80


def crossing(P, gated=False):
    """Signed index/middle tip offset on the hand's own knuckle axis, in palm units.

    Positive when the tips keep the knuckles' left-to-right order (U), negative when they swap
    (R). `gated` fades it to zero as the two fingers curl, which is what keeps a resting hand
    from carrying a confident value the forest can split on.
    """
    k = P[..., 9, :] - P[..., 5, :]
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-9)
    v = ((P[..., 12, :] - P[..., 8, :]) * k).sum(-1) / F.palm_scale(P)
    if not gated:
        return v
    ext = np.minimum(F.finger_straightness(P, "index"), F.finger_straightness(P, "mid"))
    return v * np.clip((ext - EXTENDED) / 0.15, 0.0, 1.0)


def sets():
    """Every per-class block in the repository that has both a U and an R."""
    out = {}
    per = T.load()
    merged = {}
    for name in ("static_s2.npz", "static_s3.npz", "static_s4.npz"):
        for c, v in T.load_extra(T.resolve_extra(name)).items():
            merged[c] = np.concatenate([merged[c], v]) if c in merged else v
    out["mine"] = [np.concatenate([P, merged[c]]) if c in merged else P for c, P in enumerate(per)]
    out["ASL-HG"] = ST.load_aslhg(cap=None)[0]
    out["ASLNow"] = ST.load_aslnow()[0]
    out["ayuraj"] = ST.load_ayuraj()[0]
    return out


def balanced(u, r, thr=THRESHOLD):
    return (float(np.mean(u > thr)) + float(np.mean(r <= thr))) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jitter", action="store_true",
                    help="also report the separation under the shipped training jitter")
    args = ap.parse_args()
    iU, iR = LETTERS.index("U"), LETTERS.index("R")

    print(f"one signed number, threshold {THRESHOLD} fitted on ASL-HG and never refitted:")
    for name, per in sets().items():
        U, R = per[iU], per[iR]
        if not len(U) or not len(R):
            continue
        print(f"  {name:8s} U n={len(U):5d}  R n={len(R):5d}   "
              f"raw {balanced(crossing(U), crossing(R)):.4f}   "
              f"extension-gated {balanced(crossing(U, True), crossing(R, True)):.4f}")
    print(f"  the 112-D forest, leave-one-signer-out, scores 0.687 on the same two letters")

    if args.jitter:
        hg = sets()["ASL-HG"]
        rng = np.random.default_rng(0)
        print(f"\nunder static_aug's jitter (independent per landmark, palm units):")
        for sigma in (0.0, 0.06, T.JITTER_SIGMA, 0.24):
            def j(X):
                return X + rng.normal(0, sigma, X.shape) * F.palm_scale(X)[:, None, None]
            b = balanced(crossing(j(hg[iU])), crossing(j(hg[iR])))
            mark = "   <- what training actually sees" if sigma == T.JITTER_SIGMA else ""
            print(f"  sigma {sigma:.2f}: {b:.4f}{mark}")


if __name__ == "__main__":
    main()

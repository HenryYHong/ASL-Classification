"""Non-letter hand frames from other people's clips: idle negatives that are not the author's.

temporal/idle_holds.npz is 2,180 frames over 43 holds of ONE resting hand -- mine, out of this
page's own browser log. Every floor in thresholds.py is set against it, which makes the whole
emission gate a thing tuned on one person's hand at rest. This project has already been burned
once by exactly that shape of mistake: the defining-geometry training filter was written on my
frames, shipped, and then cost accuracy on every axis as soon as other people's hands arrived.
A gate with the same provenance is the next one waiting to happen.

There is no public "resting hand" dataset. There is, however, video: the OpenHands clips
(Zenodo 10.5281/zenodo.6813108, CC BY 4.0) each show a stranger raising a hand, forming one
letter, and lowering it. The frames at the head and tail of a clip are that stranger's hand,
visible and tracked, NOT making the letter -- which is the condition the gate exists to survive.

WHICH FRAMES COUNT. Taking the first and last frames outright would be wrong: in a short clip
the signer is already holding the letter at frame 0, and scoring that as a false positive would
manufacture a failure. So the letter itself is measured first and then excluded. For each clip:

  1. the HELD shape is the per-landmark median over the middle third of the tracked frames,
     in the canonical palm frame -- the part of the clip that is the letter;
  2. a frame is a NEGATIVE only if it is in the leading or trailing third AND its mean
     per-landmark distance from that held shape is at least MIN_SHAPE_DIST palm units;
  3. frames MediaPipe did not track are dropped -- a lost hand is the segmenter's NO_HAND path
     and has nothing to do with the vote floor.

MIN_SHAPE_DIST is 0.35 palm units, which is roughly twice the jitter sigma the training
augmentation uses and about the distance between two different letters' shapes. Raising it
keeps only frames that are unambiguously not the letter, at the cost of fewer of them; the
count at several values is printed so the choice is visible rather than asserted.

This is NOT a by-signer set and must never be quoted as one: the clips carry no participant
ids (replay_strangers.py's docstring has the detail). What it is, is "hands that are not mine,
tracked, and demonstrably not holding the letter" -- the first idle negatives here that do not
come from a single person.

    ./.venv/bin/python temporal/ingest_idle_strangers.py --root <dir with American/videos>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import train_static as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "idle_strangers.npz")
#: Palm units a frame must differ from its own clip's held shape before it counts as a negative.
MIN_SHAPE_DIST = 0.35
#: Fraction of a clip's tracked span treated as head and as tail.
EDGE = 1.0 / 3.0


def canonical(lm, handed, size):
    """(n,21,2) in the canonical isotropic frame the features are computed from."""
    P = np.asarray(lm)[..., :2].astype(np.float64)
    # One (W,H) pair for the whole clip: every frame of an mp4 has the same size.
    aspect = F.frame_aspect(np.asarray(size, np.int32), len(P), where="openhands")
    # One label for the clip, not per frame: a single mislabeled frame must not mirror the hand
    # mid-clip, which is the same rule the segmenter's handedness latch enforces at runtime.
    return F.canonicalize_handedness(F.apply_aspect(P, aspect), T.modal_handedness(handed))


def clip_negatives(P, min_dist=MIN_SHAPE_DIST, edge=EDGE):
    """-> (negatives (m,21,2), held shape, distances) for one clip's canonical frames."""
    ok = np.isfinite(P[:, 0, 0])
    idx = np.flatnonzero(ok)
    if len(idx) < 9:
        return np.empty((0, 21, 2)), None, np.empty(0)
    S = F.palm_scale(P[idx])[:, None, None]
    m = F.palm_centre(P[idx])[:, None, :]
    Q = (P[idx] - m) / S                      # palm-centered, palm-scaled: shape only
    lo, hi = int(len(idx) * edge), int(len(idx) * (1.0 - edge))
    if hi <= lo:
        return np.empty((0, 21, 2)), None, np.empty(0)
    held = np.median(Q[lo:hi], axis=0)
    dist = np.linalg.norm(Q - held, axis=-1).mean(-1)
    edge_mask = np.zeros(len(idx), bool)
    edge_mask[:lo] = True
    edge_mask[hi:] = True
    keep = edge_mask & (dist >= min_dist)
    return P[idx[keep]], held, dist[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory containing American/videos/<split>")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-dist", type=float, default=MIN_SHAPE_DIST)
    args = ap.parse_args()

    import cv2  # noqa: F401  (imported late: this is the only entry point that needs it)
    import mediapipe as mp
    import replay_strangers as RS

    root = os.path.join(args.root, "American", "videos")
    if not os.path.isdir(root):
        root = args.root
    jobs = RS.find_clips(root)
    print(f"{len(jobs)} static-letter clips under {root}")

    P_out, clip_out, letter_out, dist_out = [], [], [], []
    sweep = {v: 0 for v in (0.25, 0.30, 0.35, 0.40, 0.50)}
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as hands:
        for i, (letter, split, path) in enumerate(jobs):
            got = RS.clip_landmarks(path, hands)
            if got is None:
                continue
            lm, _, handed, size, _ = got
            P = canonical(lm, handed, size)
            neg, _, dist = clip_negatives(P, args.min_dist)
            for v in sweep:
                sweep[v] += len(clip_negatives(P, v)[0])
            name = f"{split}/{letter}/{os.path.basename(path)}"
            for row, d in zip(neg, dist):
                P_out.append(row); clip_out.append(name); letter_out.append(letter); dist_out.append(d)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)} clips, {len(P_out)} negatives so far", flush=True)

    P_arr = np.asarray(P_out, np.float32)
    print(f"\nkept {len(P_arr)} negative frames from {len(set(clip_out))} clips "
          f"at min-dist {args.min_dist}")
    print("  frames at other thresholds: " + ", ".join(f"{v}:{n}" for v, n in sorted(sweep.items())))
    np.savez_compressed(args.out, P=P_arr, clip=np.array(clip_out), letter=np.array(letter_out),
                        dist=np.asarray(dist_out, np.float32),
                        note=np.array(f"OpenHands head/tail frames >= {args.min_dist} palm units "
                                      f"from their own clip's held shape; NOT a by-signer set"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

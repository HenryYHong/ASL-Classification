"""Cut recorded clips into training events by replaying the segmenter over them.

This module exists to prevent one specific, fatal mistake: training the motion classifier on
whole recorded clips.

A recorded clip is 2.0 s and contains a lead-in, the gesture, and a settle. The runtime never
sees anything like that -- Segmenter cuts an event from motion onset to motion offset, which is
shorter, and rejects anything longer than T_MAX = 1.80 s outright. Train on whole clips and
every training example is a shape the deployed system can never produce. The model would score
well in cross-validation and recognize nothing in front of a camera, which is precisely the
failure this repository already documents for Approach A.

So the boundaries come from the same code path either way. label_events replays each clip
through a real Segmenter and harvests the spans it cuts, via the on_event hook.

The diagnostic this prints is the important output, more than the file it writes: for every
J and Z clip it reports whether the segmenter found exactly one event. A clip that yields zero
events is one the runtime would also have missed -- the gate never armed, or the motion never
crossed the trigger. That is worth knowing before training, not after.

    ../.venv/bin/python temporal/label_events.py
    ../.venv/bin/python temporal/label_events.py --clips path/to/motion_clips.npz
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from segmenter import Segmenter
from thresholds import Thresholds, DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLIPS = os.path.join(HERE, "motion_clips.npz")
DEFAULT_OUT = os.path.join(HERE, "events.npz")

#: Recorder label -> event class. NONE clips are negatives; whatever the segmenter cuts out of
#: them is, by construction, motion that is not a letter.
LABEL_TO_CLASS = {"J": "J", "Z": "Z", "NONE": "MOVE"}


def modal_handedness(labels, default="Right"):
    """Reduce the recorder's per-frame labels to one modal real label.

    features.canonicalize_handedness takes a scalar and now raises on a sequence, because
    passing the array used to silently leave mirrored clips unflipped.
    """
    real = [str(x) for x in np.atleast_1d(labels)
            if str(x) not in ("None", "Unknown", "nan", "")]
    return Counter(real).most_common(1)[0][0] if real else default


def frame_sizes(npz, n_clips, override=None):
    """Per-clip (W,H). Returns a list of n_clips pairs, or None if unknown.

    A recording made by collect_motion.py carries one size for the whole file; footage ingested
    from third parties carries one per clip, because it is not one camera. Collapsing the latter
    to the first entry would apply one video's aspect ratio to all of them, and u = x*(W/H) is
    load-bearing -- skipping or mis-applying it triples the Z-gate false-positive rate on this
    repo's own archive.
    """
    if override:
        return [tuple(override)] * n_clips
    for key in ("frame_size", "frame_wh", "wh"):
        if key in npz:
            wh = np.asarray(npz[key]).reshape(-1, 2)
            if len(wh) == n_clips:
                return [(int(a), int(b)) for a, b in wh]
            return [(int(wh[0][0]), int(wh[0][1]))] * n_clips
    return None


def harvest(clip, stamps, handed, th, width, height):
    """Replay one clip and return every motion span the segmenter cuts from it."""
    spans = []
    seg = Segmenter(th, on_event=lambda ev: spans.append(ev))
    lr = modal_handedness(handed)
    clip = np.asarray(clip, dtype=np.float64)
    for i in range(len(clip)):
        lm = clip[i]
        ok = np.all(np.isfinite(lm[:, :2]))
        seg.step(float(stamps[i]), lm if ok else None, lr, width, height)
    # Recorded footage ends; a live stream does not. Score anything still in flight so a clip
    # trimmed tight to the end of the gesture is not silently discarded.
    seg.flush(float(stamps[-1]))
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--thresholds", default=None, help="a Thresholds json from calibrate.py")
    ap.add_argument("--aspect", nargs=2, type=int, metavar=("W", "H"), default=None)
    args = ap.parse_args()

    if not os.path.exists(args.clips):
        print(f"no recording at {args.clips}")
        print("Nothing to cut, and nothing is invented to stand in for it. Record first:")
        print("    ./.venv/bin/python temporal/collect_motion.py --camera 0")
        return 0

    npz = np.load(args.clips, allow_pickle=True)
    clips, stamps, labels = list(npz["clips"]), list(npz["stamps"]), list(npz["labels"])
    handed = list(npz["handed"]) if "handed" in npz else [None] * len(clips)
    sessions = list(npz["sessions"]) if "sessions" in npz else ["S1"] * len(clips)
    signers = list(npz["signers"]) if "signers" in npz else ["signer1"] * len(clips)

    wh = frame_sizes(npz, len(clips), args.aspect)
    if wh is None:
        print("the recording carries no frame size and --aspect was not given.")
        print("Aspect correction is load-bearing: without it the Z-gate's false-positive rate")
        print("triples (118 -> 367 frames on the archive). Refusing to guess.")
        return 1
    th = Thresholds.from_json(args.thresholds) if args.thresholds else DEFAULT
    distinct = sorted(set(wh))
    shown = f"{wh[0][0]}x{wh[0][1]}" if len(distinct) == 1 else f"{len(distinct)} distinct sizes"
    print(f"{len(clips)} clips, frame {shown}, thresholds "
          f"{'from ' + args.thresholds if args.thresholds else 'defaults'}\n")

    X, y, clip_ids, sess_out, signer_out = [], [], [], [], []
    per_label = {}
    for i, (clip, ts, lab) in enumerate(zip(clips, stamps, labels)):
        lab = str(lab).strip().upper()
        cls = LABEL_TO_CLASS.get(lab)
        if cls is None:
            continue
        spans = harvest(np.asarray(clip), np.asarray(ts), handed[i], th, wh[i][0], wh[i][1])
        rec = per_label.setdefault(lab, Counter())
        rec[len(spans)] += 1

        for s in spans:
            if cls in ("J", "Z") and s["arm"] != cls:
                # The gate that armed disagrees with what was being signed. Keep it as a
                # negative rather than discarding: at runtime this same span would be scored
                # under that arm, so the classifier had better have seen it.
                use = "MOVE"
            else:
                use = cls
            try:
                feat = F.event_features(s["times"], s["P"], s["arm"])
            except ValueError:
                continue
            X.append(feat)
            y.append(use)
            clip_ids.append(i)
            sess_out.append(str(sessions[i]))
            signer_out.append(str(signers[i]))

    print("events cut per clip, by recorded label:")
    for lab in sorted(per_label):
        dist = dict(sorted(per_label[lab].items()))
        n = sum(per_label[lab].values())
        clean = per_label[lab].get(1, 0)
        note = ""
        if lab in ("J", "Z"):
            missed = per_label[lab].get(0, 0)
            note = f"   {clean}/{n} gave exactly one event"
            if missed:
                note += f"; {missed} gave NONE -- the runtime would have missed those too"
        print(f"  {lab:5s} {dist}{note}")

    if not X:
        print("\nno events were cut from any clip.")
        print("Every recorded gesture failed to arm a gate or to cross the motion trigger.")
        print("Run calibrate.py --live first: the defaults come from a 1080p 15 fps archive")
        print("and may not fit this camera.")
        return 1

    X = np.stack(X)
    np.savez_compressed(args.out, X=X, y=np.array(y), clip_id=np.array(clip_ids),
                        session=np.array(sess_out), signer=np.array(signer_out),
                        feature=np.array("event/v1"))
    print(f"\nwrote {args.out}: {X.shape[0]} events x {X.shape[1]}-D from "
          f"{len(set(clip_ids))} independent clips {dict(Counter(y))}")
    print("clip_id is stored so the split can partition by clip; never split these rows randomly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

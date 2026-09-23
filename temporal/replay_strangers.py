"""Replay other people's fingerspelling VIDEO through the real Segmenter.

replay_static.py replays the author's own 71 held signs; crossval_strangers.py scores single
frames of other people's hands. Neither answers the question README.md admits it cannot answer:
"nothing here measures a stranger holding a letter through the segmenter." A frame score is not
a runtime score. The runtime has to see the hand appear, settle, hold still long enough for
VOTE_MIN frames, win a vote at the letter's own floor, and survive the launch-pose gates -- and
the frame score says nothing about any of that.

This replays the OpenHands fingerspelling clips exactly the way the page would see them: one
fresh Segmenter per clip (the clips are unrelated videos, so no cooldown, duplicate suppression
or handedness latch may carry between them), one fresh MediaPipe Hands per clip for the same
reason, the SHIPPED temporal/model_static.p and temporal/model_motion.p, the SHIPPED
thresholds.DEFAULT, and a final step D_WAIT past the last frame to release a parked I or D the
way the page's timer would. The motion forest is attached so that a false J or Z track start
counts against the run rather than disappearing.

THIS IS NOT A BY-SIGNER MEASUREMENT, and it must never be quoted as one. The clips carry no
signer ids of any kind: the only structure in the file names is a per-letter index, and while
indices 1, 2 and 3 each cover all 24 static letters (measured here on the 266 clips), from
index 4 up the coverage is partial -- index 4 is missing I, index 14 has three letters, indices
15-23 one each. Whether index 3 of "a" and index 3 of "b" are the same person is not recorded
anywhere in the download, so the clips cannot be grouped, and no leave-one-signer-out split can
be built from them. What this measures is "hands that are not the author's, on video, end to
end". crossval_signers.py is where the by-signer number lives.

Source: Zenodo 10.5281/zenodo.6813108, "OpenHands: Fingerspelling datasets - Poses",
CC BY 4.0, American.zip (63,060,346 B), no account needed. 562 clips over 36 classes
(a-z plus the digit words); 266 of them are the 24 static letters, which is what this reads.
The videos are NOT committed -- temporal/openhands_replay.json, the per-clip result, is.
SOURCES.md carries the recipe for re-downloading and re-extracting them. Two full runs of this
file wrote that JSON byte for byte identically, so the committed rows are reproducible from the
committed code and MediaPipe is not adding noise of its own on this machine.

MEASURED, all 266 clips, three configurations, all three run here. The forest and the vote
floor both changed this release, so separating them is the whole point of the middle column:

                              PREVIOUS RELEASE      old forest,        THIS RELEASE
                              old forest, 0.75      shipped floor      (both new)
    all 266 clips      exact   82 (0.308)           95 (0.357)         96 (0.361)
                       silent 184                  163                170
                       WRONG    0                    8                  0
    tracked >= 0.60    exact   48 (0.345)           58 (0.417)         59 (0.424)
      (139 clips)      WRONG    0                    6                  0
      and >= 1.0 s     exact   29 (0.500)           33 (0.569)         33 (0.569)
      (58 clips)       WRONG    0                    1                  0

    left   --static-model <previous pickle> --floor 0.75
    middle --static-model <previous pickle>
    right  no arguments but --root

The middle column is the one to read twice. Dropping the vote floor from a flat 0.75 to the
shipped per-letter profile buys 13 more correct clips on the OLD forest -- and costs 8 wrong
letters, on strangers' video, where the previous release had none. Six of the eight are in the
tracked cut and are nameable: G read as O, O as E, T as O, V as O, U as V and U as R. Put the
same floor on the retrained forest and 8 becomes 0 while the 13 is kept. The floor drop is safe
only because the forest was retrained underneath it, and
nothing else in this repository shows that; the idle gate cannot, because a relaxed hand is not
a stranger's letter.

Read the right-hand column's three rows together. The headline 0.361 is not an accuracy: 170 of
the 266 clips are silent, and the median clip is 38 frames at 1.23 s -- many end before the
runtime can settle and collect VOTE_MIN frames, so it never gets to vote at all. Cut to the
clips where it can (tracked, at least a second long) and better than half come out exactly
right, on hands the forest has never seen, through the whole segmenter.

The row that matters for a recognizer people have to trust is WRONG. Zero, on all 266 clips, at
every tracking and duration cut. The design is "silent when unsure", and on a stranger's video
it holds absolutely: every clip that speaks says the right letter, and every error is a silence.

What the retrain bought against the previous release as it actually shipped: +14 clips on the
full set, +11 on the tracked set, spread across the alphabet rather than concentrated -- S 1 ->
4, D 4 -> 6, Q 2 -> 3, P, R and X 1 -> 2 each, K and M 0 -> 1 each. C and U are still 0 of 4
and 0 of 6, and they are the two letters at the bottom of crossval_signers.py's per-letter
table as well (seed 0: U 0.671, read as R on 329 of 1,000; C 0.816, read as O on 184 of 1,000).
Two harnesses, two datasets, the same two weak letters.

    ../.venv/bin/python temporal/replay_strangers.py --root <American>/videos \\
        --out temporal/openhands_replay.json
"""
import argparse
import collections
import glob
import json
import os
import pickle
import sys
import time
from dataclasses import replace

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_static as T  # noqa: E402
from segmenter import Segmenter  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = T.LETTERS
DEFAULT_OUT = os.path.join(HERE, "openhands_replay.json")


def clip_landmarks(path, hands):
    """MediaPipe over one video. Returns (landmarks, stamps, handedness labels, (W,H), fps).

    Untracked frames are kept as NaN rows rather than dropped, because the segmenter's own
    NO_HAND / detection-gap logic is part of what is being measured: a clip where MediaPipe
    loses the hand for half a second must be allowed to lose it here too.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    lms, stamps, handed, size, i = [], [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # The container's own timestamp first; the nominal frame rate only as a fallback.
        # Every threshold downstream is in seconds, and a frame index is not a clock.
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if ms and ms > 0 else (i / fps if fps > 0 else i / 30.0)
        size = (frame.shape[1], frame.shape[0])
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lms.append([(p.x, p.y, p.z) for p in res.multi_hand_landmarks[0].landmark])
            handed.append(res.multi_handedness[0].classification[0].label
                          if res.multi_handedness else "Unknown")
        else:
            lms.append([(np.nan, np.nan, np.nan)] * 21)
            handed.append("None")
        stamps.append(t)
        i += 1
    cap.release()
    if not lms:
        return None
    return np.array(lms, np.float64), np.array(stamps, np.float64), handed, size, fps


def find_clips(root):
    """[(letter, split, path)] for the 24 STATIC letters. J and Z are skipped on purpose.

    A J or Z clip replayed here would be scored by the motion branch, whose own cross-signer
    evidence comes from ingest_external.py on video that is not redistributable. Mixing the two
    would put a motion result in a file whose every other row is a static one.
    """
    jobs = []
    for split in ("train", "test"):
        for d in sorted(glob.glob(os.path.join(root, split, "*"))):
            letter = os.path.basename(d).upper()
            if letter not in LETTERS:
                continue
            for f in sorted(glob.glob(os.path.join(d, "*.mp4"))):
                jobs.append((letter, split, f))
    return jobs


def replay_clip(seg, lm, stamps, handed, w, h, th):
    """Step one clip through one Segmenter. Returns [(letter, seconds since first frame)]."""
    out = []
    for j in range(len(lm)):
        tracked = bool(np.isfinite(lm[j, 0, 0]))
        em = seg.step(float(stamps[j]), lm[j] if tracked else None,
                      handed[j] if tracked else None, w, h)
        if em is not None:
            out.append((em.letter, float(stamps[j]) - float(stamps[0])))
    # Release a parked I or D, the way the page's D_WAIT timer would when the hand leaves.
    t_end = float(stamps[-1]) + th.D_WAIT + 0.01
    em = seg.step(t_end, None, None, w, h)
    if em is not None:
        out.append((em.letter, t_end - float(stamps[0])))
    return out


def run(root, static_path=None, motion_path=None, th=DEFAULT, progress=True, limit=None):
    blob = pickle.load(open(static_path or os.path.join(HERE, "model_static.p"), "rb"))
    motion = pickle.load(open(motion_path or os.path.join(HERE, "model_motion.p"), "rb"))
    jobs = find_clips(root)
    if limit:
        jobs = jobs[:limit]
    if progress:
        print(f"{len(jobs)} clips over {len(set(j[0] for j in jobs))} static letters under {root}")
        print(f"static forest: {blob['feature']}, {len(blob['classes'])} classes, "
              f"{sum(e.tree_.node_count for e in blob['model'].estimators_)} nodes")
    rows = []
    t0 = time.time()
    for k, (letter, split, path) in enumerate(jobs):
        # One Hands per clip: the tracker carries state between frames, and these are unrelated
        # videos. Reusing one instance lets a hand found in clip k seed the search in clip k+1.
        with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.3) as hands:
            got = clip_landmarks(path, hands)
        if got is None:
            if progress:
                print(f"  unreadable: {path}")
            continue
        lm, stamps, handed, (w, h), fps = got
        seg = Segmenter(th, static_model=blob["model"], motion_model=motion["model"],
                        static_classes=list(blob["classes"]),
                        motion_classes=list(motion["classes"]),
                        static_feature_tag=blob["feature"])
        ems = replay_clip(seg, lm, stamps, handed, w, h, th)
        emitted = [e[0] for e in ems]
        rows.append({
            "letter": letter,
            "split": split,
            "clip": os.path.relpath(path, root),
            "n": int(len(lm)),
            "tracked": float(np.isfinite(lm[:, 0, 0]).mean()),
            "wh": [int(w), int(h)],
            "fps": round(float(fps), 2),
            "dur": float(stamps[-1] - stamps[0]),
            "emitted": emitted,
            # "exact": the letter came out and NOTHING else. "among": it came out at all.
            # "wrong": how many emissions were some other letter -- the column that has to be 0.
            "exact": emitted == [letter],
            "among": letter in emitted,
            "first_wrong": bool(emitted) and emitted[0] != letter,
            "silent": not emitted,
            "wrong": sum(1 for x in emitted if x != letter),
            "latency": ems[0][1] if ems else None,
        })
        if progress and (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return rows


def summarize(rows, label):
    if not rows:
        print(f"{label:<30} (no clips)")
        return
    lat = [r["latency"] for r in rows if r["latency"] is not None]
    print(f"{label:<30} clips {len(rows):>3}  exact {sum(r['exact'] for r in rows):>3} "
          f"({np.mean([r['exact'] for r in rows]):.3f})  among {sum(r['among'] for r in rows):>3} "
          f"  first wrong {sum(r['first_wrong'] for r in rows):>2}"
          f"  silent {sum(r['silent'] for r in rows):>3}"
          f"  WRONG LETTERS {sum(r['wrong'] for r in rows):>2}"
          f"  latency median {np.median(lat) if lat else float('nan'):.2f} s")


def report(rows, min_tracked=0.60, min_dur=1.0):
    print(f"\n{len(rows)} clips read; median clip "
          f"{np.median([r['n'] for r in rows]):.0f} frames, "
          f"{np.median([r['dur'] for r in rows]):.2f} s")
    res = collections.Counter(tuple(r["wh"]) for r in rows)
    print("resolutions: " + ", ".join(f"{w}x{h}={n}" for (w, h), n in res.most_common()))
    usable = [r for r in rows if r["tracked"] >= min_tracked]
    long_enough = [r for r in usable if r["dur"] >= min_dur]
    summarize(rows, "all clips")
    summarize(usable, f"tracked >= {min_tracked:.2f}")
    summarize(long_enough, f"  and >= {min_dur:.1f} s")
    print(f"\nper letter (tracked >= {min_tracked:.2f}):")
    for L in LETTERS:
        rs = [r for r in usable if r["letter"] == L]
        if not rs:
            continue
        out = collections.Counter(x for r in rs for x in r["emitted"])
        print(f"  {L}: {len(rs):>2} clips  exact {sum(r['exact'] for r in rs):>2}  "
              f"silent {sum(r['silent'] for r in rs):>2}  emitted {dict(out) or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="the American/videos directory of the Zenodo download")
    ap.add_argument("--out", default=None,
                    help=f"write the per-clip rows as JSON (the committed copy is {DEFAULT_OUT})")
    ap.add_argument("--min-tracked", type=float, default=0.60,
                    help="the reporting cut; every clip is replayed and written either way")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="second reporting cut: a clip shorter than this may end before the "
                         "runtime can settle and collect VOTE_MIN frames")
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (a smoke run)")
    ap.add_argument("--static-model", default=None,
                    help="a letter pickle other than the shipped one; this is how the "
                         "previous release's column in the docstring was measured")
    ap.add_argument("--motion-model", default=None, help="likewise for the motion pickle")
    ap.add_argument("--floor", type=float, default=None,
                    help="a FLAT floor what-if: VOTE_PROB and VOTE_PROB_FLOOR both move here "
                         "and VOTE_PROB_LETTER is cleared, the same knob replay_static.py has. "
                         "--floor 0.75 with --static-model is the previous release exactly")
    args = ap.parse_args()
    t0 = time.time()
    # One knob, both fields, per replay_static.py: they are equal in DEFAULT on purpose, so
    # moving the floor alone would leave the confident route where it was and flatten the sweep.
    th = DEFAULT if args.floor is None else replace(
        DEFAULT, VOTE_PROB=args.floor, VOTE_PROB_FLOOR=args.floor, VOTE_PROB_LETTER={})
    print(f"VOTE_PROB {th.VOTE_PROB} / VOTE_PROB_FLOOR {th.VOTE_PROB_FLOOR} / "
          f"VOTE_PROB_LETTER {th.VOTE_PROB_LETTER}")
    rows = run(args.root, static_path=args.static_model, motion_path=args.motion_model,
               th=th, limit=args.limit)
    report(rows, args.min_tracked, args.min_dur)
    print(f"\n{time.time() - t0:.0f}s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

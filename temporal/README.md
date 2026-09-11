# temporal/ — all 26 letters, live

The original pipeline classifies one frame at a time from a feature that discards where the hand
is. J and Z are motion letters — J is the `I` handshape tracing a hook, Z is an extended index
drawing a zigzag — so a single frame of a J *is* an I. This directory adds a motion branch
alongside the static classifier, and rebuilds the static branch to survive a change of day.

## Setup

```
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python temporal/live_demo.py --camera 0
```

Python 3.11 and `mediapipe==0.10.18` specifically: MediaPipe 1.0 removed the `mp.solutions`
namespace, which breaks this code and both original notebooks.

## How it works

**Static letters.** Every frame becomes a 101-D vector: 21 landmarks palm-centered and divided by
palm width, plus all pairwise distances between the fingertips, knuckles and wrist. The distance
block matters because a forest splits one coordinate at a time, so "how far apart are these two
fingertips" — the whole difference between U and V — otherwise costs it a deep chain of splits.
A letter is emitted when a hold is first established, never per frame.

**Motion letters.** A per-frame *gate* (six inequalities on finger geometry, no model) decides
whether the hand is in a launch pose. A track starts only on a **rising edge** — parked in that
pose, then moving — which is why ordinary hand travel almost never creates a scoring opportunity.
The fingertip path becomes a 79-D descriptor: resampled by arc length, centered on itself, divided
by hand size, so a J traced anywhere in frame gives the same numbers.

`I` and `D` are held back 350 ms before being emitted, because they are the launch poses for J
and Z. Without that delay every J reads as "IJ".

## Results

Measured, with the split each number came from:

| | result | split |
| --- | --- | --- |
| Static letters, in-session | 0.968 | held-out tail of each capture burst |
| **Static letters, leave-one-session-out** | **0.759** | `crossval_static.py`: train on all but one session, test on it |
| — the fold that tests all 24 letters | 0.659 | hold out the archive, train on the later sessions |
| Motion letters {J, Z, MOVE} | 0.864 | GroupKFold over 140 independent gestures |
| Motion, at the runtime operating point | 92% correct when it fires, 16% abstain | same |
| False J/Z on held-out negatives | 2 of 55 | same |
| Gate: `J_GATE` on held `I` | 100/100, 0 false of 2,278 | committed archive |
| Segmenter over 157 s of held signs | 0 false triggers, 24/24 letters | committed archive |

**The 0.759 is the honest number, and the 0.659 is how to read it.** Two of the four sessions
are targeted re-recordings covering six and four letters; folds that test four well-separated
shapes score 1.000 and mean nothing, which is exactly the criticism the top-level README makes
of a 100.00% measured with a letter missing from the test set. `crossval_static.py` prints the
per-fold table for that reason, and prints the unweighted mean beside a warning not to quote it. The in-session figure is inflated the same way the original
99.58% was: consecutive frames of one held sign are near-duplicates, so they sit on both sides of
any random split. Chasing the in-session number actively hurt: adding absolute hand extent took it
from 0.956 to 0.983 while *halving* cross-session accuracy, 0.520 to 0.262.

Training data is five sessions — the November archive, a full-alphabet pass months later, and a
targeted pass over the letters that were still confusable. One session is what limited this; the
second one is what fixed it.

## Run order

```
./.venv/bin/python temporal/calibrate.py --live --camera 0        # fit thresholds to your camera
./.venv/bin/python temporal/collect_motion.py --camera 0 --continuous --letters J --clips 30
./.venv/bin/python temporal/collect_motion.py --camera 0 --static-letters --reps 3
./.venv/bin/python temporal/label_events.py                        # cut events; READ its output
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz
./.venv/bin/python temporal/crossval_static.py                     # the number worth quoting
./.venv/bin/python temporal/live_demo.py --camera 0 --log tracks.jsonl
```

`label_events.py`'s diagnostic is the important output: for every recorded gesture it says whether
the segmenter cut exactly one event. A clip yielding none is one the runtime would also miss.

`live_demo.py --log` records every hold and every attempted track with its landmarks and the
reason it was accepted or rejected. Every hard bug here was found that way and none was found by
reasoning about the code.

## What is still weak

- **M, N and S** are the weakest across sessions. They are fists distinguished only by where the
  thumb sits, so a centimetre of thumb drift between sittings is a different letter. G, K and T
  were in this list until the recordings were made consistent.
- **Check new recordings against the letter's defining geometry before training on them.** Every
  G frame in one session had an extended middle finger, which is an H; two thirds of a targeted
  re-recording did too. The correct frames were outvoted and G read as H everywhere. Dropping
  them by that rule moved cross-session accuracy 0.680 -> 0.702, and consistent K/M/S/T
  recordings took it to 0.759.
- **The static model is over-confident in-session and under-confident live**, which is why a
  letter can be correct on 100 of 100 frames and still score 0.41. Emission therefore accepts
  either a confident winner or a decisive one — see `VOTE_MARGIN_CLEAR` in `thresholds.py`.
- **One signer.** Nothing here says anything about a different person's hands.
- `thresholds.NEEDS_GESTURE_DATA` lists the constants that are still reasoned rather than
  measured.
- **Thresholds guessed before the data existed were the single largest source of missed
  letters.** P_EMIT was 0.70 and dropped about one genuine gesture in three; T_MAX was 1.80 and
  sat *below* the p95 of the training set's own Z durations, clipping real gestures out of the
  distribution the classifier was fitted on. Both are now swept against recorded events. Check
  any constant against the data it is supposed to describe before trusting a live failure.

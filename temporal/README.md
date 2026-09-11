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

**Static letters.** Every frame becomes a 101-D vector: 21 landmarks palm-centred and divided by
palm width, plus all pairwise distances between the fingertips, knuckles and wrist. The distance
block matters because a forest splits one coordinate at a time, so "how far apart are these two
fingertips" — the whole difference between U and V — otherwise costs it a deep chain of splits.
A letter is emitted when a hold is first established, never per frame.

**Motion letters.** A per-frame *gate* (six inequalities on finger geometry, no model) decides
whether the hand is in a launch pose. A track starts only on a **rising edge** — parked in that
pose, then moving — which is why ordinary hand travel almost never creates a scoring opportunity.
The fingertip path becomes a 79-D descriptor: resampled by arc length, centred on itself, divided
by hand size, so a J traced anywhere in frame gives the same numbers.

`I` and `D` are held back 350 ms before being emitted, because they are the launch poses for J
and Z. Without that delay every J reads as "IJ".

## Results

Measured, with the split each number came from:

| | result | split |
| --- | --- | --- |
| Static letters, in-session | 0.968 | held-out tail of each capture burst |
| **Static letters, leave-one-session-out** | **0.680** | train one session, test the other |
| Motion letters {J, Z, MOVE} | 0.889 | GroupKFold over 118 independent gestures |
| Motion, at the runtime operating point | 96% correct when it fires, 22% abstain | same |
| False J/Z on held-out negatives | 0 | same |
| Gate: `J_GATE` on held `I` | 100/100, 0 false of 2,278 | committed archive |
| Segmenter over 157 s of held signs | 0 false triggers, 24/24 letters | committed archive |

**The 0.680 is the honest number.** The in-session figure is inflated the same way the original
99.58% was: consecutive frames of one held sign are near-duplicates, so they sit on both sides of
any random split. Chasing the in-session number actively hurt: adding absolute hand extent took it
from 0.956 to 0.983 while *halving* cross-session accuracy, 0.520 to 0.262.

Training data is three sessions — the November archive, a full-alphabet pass months later, and a
targeted pass over the letters that were still confusable. One session is what limited this; the
second one is what fixed it.

## Run order

```
./.venv/bin/python temporal/calibrate.py --live --camera 0        # fit thresholds to your camera
./.venv/bin/python temporal/collect_motion.py --camera 0 --continuous --letters J --clips 30
./.venv/bin/python temporal/collect_motion.py --camera 0 --static-letters --reps 3
./.venv/bin/python temporal/label_events.py                        # cut events; READ its output
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz
./.venv/bin/python temporal/live_demo.py --camera 0 --log tracks.jsonl
```

`label_events.py`'s diagnostic is the important output: for every recorded gesture it says whether
the segmenter cut exactly one event. A clip yielding none is one the runtime would also miss.

`live_demo.py --log` records every hold and every attempted track with its landmarks and the
reason it was accepted or rejected. Every hard bug here was found that way and none was found by
reasoning about the code.

## What is still weak

- **G, M, S, T** are the letters that change most between sittings (G→H, M→E, S→E, T→N across
  sessions). More sessions is the only fix; thresholds cannot separate them.
- **The static model is over-confident in-session and under-confident live**, which is why a
  letter can be correct on 100 of 100 frames and still score 0.41. Emission therefore accepts
  either a confident winner or a decisive one — see `VOTE_MARGIN_CLEAR` in `thresholds.py`.
- **One signer.** Nothing here says anything about a different person's hands.
- `thresholds.NEEDS_GESTURE_DATA` lists the constants that are still reasoned rather than
  measured.

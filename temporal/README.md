# temporal/ — adding J and Z

The 24-letter pipeline classifies one frame at a time from a feature that deliberately discards
where the hand is. J and Z are motion letters: J is the `I` handshape tracing a hook, Z is an
extended index finger drawing a zigzag. A single frame of a J is an `I`, and a single frame of a
Z is a `D`. Neither the representation nor the model can reach them, so this is an addition
rather than a retrain.

## Setup

```
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Python 3.11 and `mediapipe==0.10.18` specifically. MediaPipe 1.0 removed the `mp.solutions`
namespace, which breaks both original notebooks as well as this code.

## How it works

Two stages. The 24-class static classifier is untouched; a motion layer sits beside it.

A **gate** decides, per frame, whether the hand is in a launch pose — six inequalities on
fingertip distances, no model. Measured on the committed archive, aspect-corrected:

| gate | fires on | false positives |
| --- | --- | --- |
| `J_GATE` | 100/100 held `I` frames | 0 of 2,278 |
| `Z_GATE` | 100/100 held `D` frames | 118 of 2,278 (X 61, L 23, G 16, P 16, K 2) |

Deliberately inequalities rather than `predict_proba`. The 24-class forest has never seen a
rotating wrist mid-J, and its probabilities out there are undefined in exactly the way Approach
A's softmax was. Six inequalities fail visibly; an out-of-distribution probability fails
confidently.

A **track** starts only on a rising edge: the hand was parked in a gate-passing pose and *then*
began to move. That conjunction is why replaying all 157 s of held signs produces **zero** false
triggers, and it is what allows the motion threshold to sit low enough (1.30 palm/s) that a slow,
deliberate J still registers.

Cheap **vetoes** run before any model — duration, path length, straightness — so most motion is
rejected without ever receiving a probability.

`I` and `D` are **held back 350 ms** before being emitted, because they are the launch poses. Skip
this and every J reads as `IJ`. Only those two letters pay the latency.

## Run order

```
./.venv/bin/python temporal/extract_static_sequences.py   # done: static_sequences.npz
./.venv/bin/python temporal/train_static.py               # done: model_static.p
./.venv/bin/python temporal/calibrate.py --offline        # threshold provenance table
./.venv/bin/python temporal/calibrate.py --live --camera 0  # fit thresholds to YOUR camera
./.venv/bin/python temporal/collect_motion.py --camera 0 --session S1
./.venv/bin/python temporal/label_events.py               # cut events; read its diagnostic
./.venv/bin/python temporal/train_motion.py               # model_motion.p
./.venv/bin/python temporal/live_demo.py --camera 0
./.venv/bin/python temporal/evaluate.py                   # needs a second session
```

Run `calibrate.py --live` before recording. The defaults are percentiles of a 1080p 15 fps
archive and may not fit another camera — and it catches a wrong `--camera` index immediately,
which otherwise wastes two minutes looking busy while writing nothing.

`label_events.py`'s diagnostic is the important output: for each J and Z clip it reports whether
the segmenter found exactly one event. A clip yielding zero events is one the runtime would also
have missed. Read that before training, not after.

## What is verified, and what is not

Verified against the committed archive, reproducible today:

- gate rates in the table above
- 157 s of held signs replayed through the segmenter: 0 false track starts, 24/24 letters
  emitting exactly one correct static letter
- `shape42` is exactly scale-invariant (0.908 at every rescale); the legacy feature spans
  0.340–0.926 across the same rescales
- `tests/`: 21 feature assertions, 14 leakage assertions

**Not verified: anything about J or Z accuracy.** No motion footage exists yet. Five constants —
`RIGID_VETO`, `T_MIN`, `T_MAX`, `P_EMIT`, `MARGIN` — cannot be derived from held signs and are
currently reasoned defaults, listed in `thresholds.NEEDS_GESTURE_DATA`. They need recalibrating
against real gestures.

## Recording notes

Roughly 17 minutes across two sittings **on different days**. Two sittings is not redundancy: the
train/test split is by session, and it is the only route to a generalization number that means
anything. The README of this repository documents at length why the existing 99.58% does not.

The negatives matter more than the positives. Recording "hold an `I` and move it around without
ever tracing a J", and the same for `D` and Z, trains the classifier *on* the confusion rather
than asking it to extrapolate into it. A negative example is evidence; a feature is only a
hypothesis. Note that a moving hand in a non-launch pose never arms a gate at all, so those
frames are free — the negatives that count are the ones in a launch pose.

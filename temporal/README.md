# temporal/ — all 26 letters, live

The original pipeline classifies one frame at a time from a feature that discards where the hand
is. J and Z are motion letters — J is the `I` handshape tracing a hook, Z is an extended index
drawing a zigzag — so a single frame of a J *is* an I. This directory adds a motion branch
alongside the static classifier, and rebuilds the static branch to survive a change of day. It
also carries an experimental numbers mode (0-9), described at the end, which is the one part of
the repository not trained on my own hand.

## Setup

```
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python temporal/live_demo.py --camera 0
```

Python 3.11 and `mediapipe==0.10.18` specifically: MediaPipe 1.0 removed the `mp.solutions`
namespace, which breaks this code and both original notebooks. `live_demo.py --mode digits`, or
the M key while it runs, switches to the numbers mode.

## How it works

**Static letters.** Every frame becomes a 112-D vector (`features.static_feature_v4`, tagged
`static/v4`): 21 landmarks palm-centered and divided by palm width, all pairwise distances
between the fingertips, knuckles and wrist, a straightness ratio per finger, and an 11-value
thumb block — thumb straightness, each fingertip's distance from the palm center, and the thumb
tip's distance to the index and middle knuckles. The distance block matters because a forest
splits one coordinate at a time, so "how far apart are these two fingertips" — the whole
difference between U and V — otherwise costs it a deep chain of splits. The thumb block is there
because the fists (A, E, M, N, S, T) and D/X differ only in where the thumb sits, and the fists
keep every fingertip curled in the same place, so the thumb has to be measured against the
knuckles. The previous 101-D vector is still `static/v3`; every consumer resolves the transform
from the tag a pickle carries (`features.STATIC_FEATURES`), so a forest can never be fed the
wrong layout silently. A letter is emitted when a hold is first established, never per frame.

**Training the static forest.** The training set is my four sessions plus other people's
hands (`strangers.py`): the 1,874 static-letter records of the ASLNow set, multiple participants
captured with the Web Hand Landmarker the page runs, and the 687 photographs of 218 students'
hands from the digits set whose digit is a letter (0/O, 2/V, 6/W, 9/F). Every training frame
gets four Gaussian-jittered copies, each landmark coordinate moved by N(0, (0.12 x palm
width)^2) and the vector recomputed (`static_aug.py`); that replaced the four-angle rotation
augmentation two releases ago, which had lowered leave-one-session-out accuracy (0.782 without
it, 0.759 with it), and on my sessions alone it was worth +0.09 pooled / +0.14 on the cross-day
fold over no augmentation. The strangers are worth more: with them on the training side of every
fold, the cross-day fold goes 0.778 -> 0.873 and the pooled figure 0.867 -> 0.913 (same recipe,
`crossval_static.py --henry-only` for the first pair), because other people's hands teach
invariances that transfer back to my own unseen day. The previous release's defining-geometry
filter (137 of my frames dropped for X, G, Q, U, V, K, R, P and D) is retired: its thresholds
were set on one hand, they discard half the strangers' X, P and R frames, and with strangers in
the training set the filter costs on every axis (ASLNow five-fold 0.945 -> 0.903, three seeds);
`--ruleset strong` keeps the ablation runnable. The forest is 80 trees, at least five samples
per leaf: 199,528 nodes on 37,195 rows, every one of which ships to the browser.

**Motion letters.** A per-frame *gate* (six inequalities on finger geometry, no model) decides
whether the hand is in a launch pose. A track starts only on a **rising edge** — parked in that
pose, then moving — which is why ordinary hand travel almost never creates a scoring opportunity.
The fingertip path becomes a 79-D descriptor: resampled by arc length, centered on itself, divided
by hand size, so a J traced anywhere in frame gives the same numbers.

**Where the two branches meet.** `I` and `D` are held back 350 ms before being emitted, because
they are the launch poses for J and Z. The timer alone was not enough: a J's track cannot resolve
inside 350 ms, so the parked I used to be released mid-track and every J read "IJ". A parked I or
D whose own pose arms a track is now held until the track resolves — a J/Z emission cancels it,
an abort or an abstention releases it — and a rise that is still being confirmed when the timer
runs out gets the same treatment for at most one `V_MOVE_ARMED_SUSTAIN` (0.13 s), which on the
takes turned 2 of the remaining "IJ" into "J". An I held longer than 350 ms before the stroke
is still released, by design: deferring I and D to the end of the hold was measured and rejected
for its latency (S1 I 0.92 -> 4.01 s), and 2 "IJ" plus 3 "IJI" of the 60 J items remain. After a
J or Z is emitted, the static branch stays quiet while the hand rests in the finishing pose; the
suppression clears on a reshape or on the hand moving again, so a hand that morphs straight into
the next letter is not delayed.

**Handedness is latched, not read per frame.** `features.canonicalize_handedness` mirrors a
"Left" hand, and MediaPipe assigns that label per frame; on the committed takes 166 of the
13,692 detected frames carry the other hand's label for a frame or two, and mirroring one frame
mid-stroke pushed `sigma_rigid` past the veto and lost the gesture (6 of 113 in the per-frame
replay). The segmenter keeps the label it has and switches only once the other label has
persisted for `HAND_SWITCH_S` (0.50 s; the longest flip inside an unbroken detection is 5 frames,
0.20 s), and forgets it on a detection gap, which is the only way a signer changes hands. Training
still cuts events with one modal label per recording; the latch makes the live path agree with
it on every frame that is not a genuine change of hand.

## Results

Measured, with the split each number came from. Every static number is `crossval_static.py`
or `crossval_strangers.py` at seed 0 unless it says otherwise; "S1" is the November 2024
archive, the only session recorded on a different day from the other three, and "the
strangers" are the two public sets in `strangers.py`, always on the training side of a fold
that holds out one of my sessions and never on the test side.

| | result | split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.913** (4,454/4,878); hold-level 95% CI [0.858, 0.958] over 57 session x letter bursts; 3 seeds 0.910 ± 0.004 (0.904-0.914) | train on three of my sessions plus the strangers, test on every frame of the fourth; jitter on the training folds only |
| — cross-day fold (S1 held out, all 24 letters) | 0.873 (2,075/2,378); macro per-letter recall 0.871; 3 seeds 0.870 ± 0.002; below 0.6 recall: M 0.01, R 0.58 | train on S2+S3+S4 and the strangers, test on the archive |
| — the same recipe, my sessions only | 0.867 pooled, 0.778 cross-day | `crossval_static.py --henry-only`; the strangers' contribution is the difference |
| — previous releases, same folds | 0.861 / 0.763 (one signer, geometry filter, RF100); 0.759 / 0.659 (rotation, static/v3, RF400) | `crossval_static.py --legacy` reproduces the older pair exactly |
| **Static letters, other people's hands, held out** | **0.790** ± 0.003 over 3 seeds (0.788 at seed 0) on 1,874 ASLNow records; the one-signer forest 0.781. Weakest: G 0.43, R 0.45, U 0.48, D 0.49, S 0.55, M 0.60 | `crossval_strangers.py`: train on my four sessions and the 218-signer O/V/W/F photos, test on a set the forest never saw, 24 letters, the browser's own landmarker, one frame per record |
| — the vote gate on those records | emits on 0.46 of single frames, right on 0.962 of those | same forest, `thresholds.DEFAULT`; a held sign offers many windows and a record offers one, so this is a floor on what a visitor sees |
| — 218 signers, O/V/W/F | 0.936 ± 0.004 (F 1.00, O 0.93, W 0.94, V 0.83); the one-signer forest 0.766 (V 0.17) | train on my sessions and ASLNow, test on the digit photos |
| — ASLNow, five folds | 0.946 ± 0.002 | my sessions, the photos and four fifths of ASLNow in training, the fifth held out; no participant id, so a signer may sit on both sides: an upper bound |
| Static emission, the 71 held-out holds | exactly the right letter 64/71; first emission wrong 1/71; one wrong letter in all 71; silent 6/71 (S1-E, S1-N, S2-K, S3-D x3); latency 0.46 s median, 1.27 s p90; cross-day 21/24 | `replay_static.py`: fold models with the strangers, real timestamps, one fresh `Segmenter` per hold, shipped thresholds. Previous release: 65/71, first wrong 3, four wrong letters, silent 2, cross-day 19/24 |
| Idle hand, 43 clean holds from a live browser log | 0/43 holds, 0/2,064 votes emit at the 0.75 floor; at the previous 0.55 floor this forest emits on 5/43 (42 votes, all G, max 0.695) | `idle_gate.py` on `idle_holds.npz`: one signer, one ~2-minute stretch, Tasks-API landmarks |
| Motion letters {J, Z, MOVE} | 0.951 accuracy; J+Z recall 0.946 at `P_EMIT` 0.55; MOVE read as J 2/28, as Z 0/28 | `GroupKFold(5)` by prompted item over the S1 takes: 102 events in 80 items. The label set changed in the previous release (one credited event per item, unreachable spans dropped), so this replaces the older 0.864 / 140 gestures rather than improving on it |
| Motion, end to end on the five takes | 85 of 113 items produce their letter (was 78); 0 doubles (was 9); 0 rest-phase J/Z over 5.5 min | in-sample for the motion forest. Replayed with the per-frame MediaPipe handedness label the live path feeds, through the handedness latch; without the latch the same replay credits 79, and with it all 113 item strings are identical to a replay with one modal label per take, which is how training events are cut. `evaluate.py` reproduces it directly: S1 74 of 90 (`--train-session S5 --test-session S1`) plus S5 11 of 23 (the default run), 0 doubles, 0.00 J/Z per minute over 5.25 and 1.51 min of negative footage |
| Motion, held-out session S5 (Z only) | 8 of the 11 credited Z items emit Z (0.727); 8 of 23 prompted items end to end; 0 doubles; 0 false J/Z per minute over 1.51 min of negatives | a forest fitted on S1 alone (`train_motion.py --fit-sessions S1 --out <scratch>`, then `evaluate.py --motion-model <scratch> --test-session S5`); the shipped forest is fitted on S1 and S5, so the default run's S5 row (11 of 23) is in-sample and says so |
| `J_GATE` on held `I` | 100/100; 13 of 2,278 other frames pass, all Y (13 of Y's 100) | committed archive, thumb ceiling 1.30; a Y held still and then moved could arm a track that no negative example resembles, and no such footage exists |
| Segmenter over 157 s of held signs | every one of the 24 letters emitted exactly once, 0 track starts | committed archive, in-sample for the static forest; the S2 holds 23/23 |
| Browser landmarker vs training landmarker | gap +0.001 over 3 seeds on the cross-day fold (paired 95% CI about ±0.03); 0.775 with the page's handedness swap, 0.730 without | the previous release's forest on the archive re-extracted with the MediaPipe Tasks API, measured with the CPU build of the Tasks landmarker from Python in VIDEO mode, not the page's GPU delegate. The ASLNow records are that landmarker on other people's hands, and the forest now trains on them |
| Browser input resolution, hand distance | 1920x1080 to 426x240: within ±0.004; hand shrunk to the browser log's typical palm size: −0.02 | same fold, previous release's forest; the shrink is synthetic |

**0.91 is the figure worth quoting, 0.87 is the one to plan around for my own hand, and 0.79
is the one to plan around for anyone else's.** Two of the four sessions are short re-recordings
covering six and four letters, and a fold that tests four well-separated shapes scores 1.000
without telling us much; `crossval_static.py` prints the unweighted mean beside a note not to
publish it. S2, S3 and S4 were all recorded on the same evening, 2026-09-10, about two and a
half hours apart (S2 and S3 first committed at 18:40 in `ce4c26c`, S4 at 21:14 in `8be6a85`),
so the pooled figure is dominated by same-evening folds, and S1 (2024-11-04) is the only fold
that tests a different day and the only one that tests all 24 letters. The 0.79 is a forest
that never saw the ASLNow set; the shipped forest trains on it, so on a new person it is better
than that by an amount that cannot be measured until another multi-signer set exists, and the
five-fold 0.946 is its ceiling. Two implementations of the same recipe differed by 0.01 from
the jitter RNG alone, so the third decimal of any one run means nothing; the CI and the seed
spread are the honest uncertainty. The in-session figure an old README carried (0.968) is not
reported any more: consecutive frames of one held sign are near-duplicates, so it measured
re-identification, and chasing it actively hurt — adding absolute hand extent raised it while
halving the cross-session number.

The training data is my four sessions — the November 2024 archive (24 letters, 2,378 frames)
and, on one evening nearly two years later, a full-alphabet pass (23 letters, 1,385 frames)
plus two targeted passes over the letters that were still confusable (6 and 4 letters, 728
and 387 frames) — and 2,561 frames of other people's hands. One session is what limited this;
the second one fixed the letters I sign the same way every day, and the strangers fixed the
ones I do not.

## Run order

From the repository root. Recording needs a camera; everything after it does not.

```
./.venv/bin/python temporal/calibrate.py --live --camera 0 --out temporal/thresholds_live.json
./.venv/bin/python temporal/collect_motion.py --camera 0 --continuous --letters J Z --clips 30 --session S6
./.venv/bin/python temporal/collect_motion.py --camera 0 --static-letters --reps 3 --session S6   # -> temporal/static_s6.npz
./.venv/bin/python temporal/ingest_aslnow.py                        # only to rebuild aslnow.npz (network)
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz static_s6.npz
./.venv/bin/python temporal/train_digits.py
./.venv/bin/python temporal/crossval_static.py --seeds 3            # the numbers above
./.venv/bin/python temporal/crossval_strangers.py --seeds 3         # the cross-signer numbers
./.venv/bin/python temporal/idle_gate.py                            # must PASS, or VOTE_PROB_FLOOR moves up 0.05
./.venv/bin/python temporal/replay_static.py                        # what the floor costs on the 71 held signs
./.venv/bin/python temporal/label_events.py                         # cut events; READ its output
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/evaluate.py                             # per session; in-sample for the shipped forest, and each block says so
./.venv/bin/python docs/export_models.py                            # -> docs/models.json, docs/golden.json
./.venv/bin/python temporal/live_demo.py --camera 0 --log tracks.jsonl
```

`train_motion.py` fits the shipped forest on every recorded session, so the `evaluate.py` line
above measures nothing held out; it reports, and labels, in-sample rows. A held-out motion row
needs a forest fitted on one session, written to scratch rather than over the shipped pickle:

```
./.venv/bin/python temporal/train_motion.py --fit-sessions S1 --out /tmp/motion_S1.p
./.venv/bin/python temporal/evaluate.py --motion-model /tmp/motion_S1.p --test-session S5
```

That pair is what the held-out S5 row above came from.

`calibrate.py --live` writes a thresholds JSON and nothing reads it on its own: pass it as
`--thresholds` to `label_events.py`, `train_motion.py`, `evaluate.py` and `live_demo.py`, all of
them or none, because events must be cut with the constants the runtime will use. Without it,
every script runs `thresholds.DEFAULT`, which is what ships. `thresholds_archive.json` is the
provenance record of those defaults (`calibrate.py --offline` regenerates it, and a test holds it
to `DEFAULT` on every field the offline pass does not measure), not a tuned profile to pass.
`collect_motion.py --static-letters`
writes `temporal/static_<session>.npz` (the tag lowercased) and refuses to overwrite an existing
file; the motion recorder appends to `motion_clips.npz`. `crossval_static.py` adds sessions from
the `EXTRA` list
at the top of the file, so a new session has to be added there to enter the measurement.
`evaluate.py` reports on every session other than `--train-session` and labels each block
held-out or in-sample from the pickle's own record of the sessions it was fitted on, which for
the shipped `model_motion.p` is both of them.

`label_events.py`'s diagnostic is the important output: for every prompted item it says whether
the segmenter cut a creditable event. An item yielding none is one the runtime would also miss.

`live_demo.py --log` records every hold and every attempted track with its landmarks and the
reason it was accepted or rejected. Every hard bug here was found that way and none was found by
reasoning about the code.

`make_oof.py` writes the leave-one-session-out posteriors and `simulate_words.py` spells words
out of them; that pair is where the word layer's constants came from (see `docs/README.md`).

## What is still weak

- **M on the cross-day fold, and the letters strangers sign differently from me.** The
  archive's M was recorded with the thumb where the textbook handshape does not put it (it
  fails its own defining-geometry rule on every frame) and reads at 0.01 across days under
  every forest; re-recording it is the only fix. E, N, S and O had the same problem in the
  previous release (0.12, 0.20, 0.54, 0.55) and are read now that other people's versions of
  them are in the training set. On other people's hands the weak letters are different: G
  read as X, R and U as V, D and X as each other, S as A.
- **Check new recordings against the letter's defining geometry before training on them.** Every
  G frame in one session had an extended middle finger, which is an H; two thirds of a targeted
  re-recording did too. The correct frames were outvoted and G read as H everywhere. The rules
  in `static_aug.RULES` still do that check (`train_static.py --ruleset strong`), but they are
  no longer applied to the shipped training set: they were written for one hand, and applied to
  the strangers they discard half their X, P and R frames.
- **28 of the 113 prompted motion items have no reachable span.** Was 35 before the gate ceiling
  moved to 1.30. The remaining misses are segmentation — the gate never armed, no rising edge, a
  veto, or a detection gap — and none of this release's fixes address them.
- **The launch and finishing poses of J and Z are letters too, and they get emitted.** The
  pending-letter rule holds a parked I or D back while its track is in flight, so with the
  shipped forest a wrong letter inside an item is rare (`evaluate.py` on the takes: one Z item
  read as "QZ", one as "ZT", out of 53). What remains is the correct static read of the pose the
  hand rests in afterward: "JI" on 25 of the 60 J items and "ZD" or "DZD" on 5 of the Z items,
  because the signer re-forms the launch shape and holds it. The previous letter forest read
  that parked D as a low-confidence T and stayed silent instead; the shipped one reads it as D.
- **The idle floor has 0.06 of headroom, and it is a G.** `VOTE_PROB_FLOOR` (0.75, with the
  confident route at the same value) sits 0.055 above the most confident vote this forest
  produced on the 43 idle holds, and every idle vote that crossed the previous 0.55 floor was
  a G: a forest that has seen many hands reads a relaxed one as a loose G. Filtering the
  strangers' G frames and a REST class trained on the motion recordings' rest windows were both
  tried and did not remove it (`thresholds.py`). One signer, one two-minute stretch;
  `idle_gate.py` is the thing to re-run after the next live session.
- **Six of the 71 held signs are silent at that floor** (E and N once each on the cross-day
  fold, S2-K, and the three S3 D holds, which were silent at 0.55 as well), against one wrong
  letter in all 71. Silence on a hold the forest is unsure of is the design; the previous
  forest emitted four wrong letters and was silent on two.
- **Other people's hands: a small set, one frame each, no video.** The ASLNow records are
  multiple participants (the source does not say how many), one captured frame per record, so
  nothing here measures a stranger holding a letter through the segmenter, and the shipped
  forest's own cross-signer accuracy is bounded rather than measured (0.79 for a forest that
  never saw the set, 0.95 with a participant possibly on both sides). The digits are the
  mirror image: trained on 218 strangers and never verified on mine.
- `thresholds.NEEDS_GESTURE_DATA` lists the motion constants that are still reasoned rather than
  measured.
- **Thresholds guessed before the data existed were the single largest source of missed
  letters.** `P_EMIT` was 0.70 and dropped about one genuine gesture in three; `T_MAX` was 1.80
  and sat *below* the p95 of the training set's own Z durations. Both are now swept against
  recorded events, and the vote floor in this release was set the same way, against idle holds
  from a live log. Check any constant against the data it is supposed to describe before
  trusting a live failure.

## Numbers mode

`model_digits.p` is a separate ten-class forest — 100 trees, at least five samples per leaf,
23,296 nodes, the same `static/v4` feature and jitter as the letters — trained by
`train_digits.py` on `digits_ankara.npz`: MediaPipe landmarks of 1,805 photographs from the
Sign Language Digits Dataset (218 students, one photo per digit; MediaPipe found a hand in
1,805 of 2,062; see `SOURCES.md`). Leave-signer-out, `GroupKFold(5)` over 222 signer runs: 0.986
(1,780/1,805), worst digit 6 at 0.965. Every feature, jitter and forest-size variant lands at
0.984-0.987, so the dataset does not discriminate between them and the smallest forest ships.

It is trained on 218 signers from a public dataset, not on the author, and it has not been
verified on the author's hand or on live video. No frame of me signing a digit exists. The two
stand-in checks: my O, V, W, F and B letter frames read as 0, 2, 6, 9 and 4 on 0.956 of frames —
weak evidence, because my C, R, X and U frames read as 0, 2, 1 and 2 just as confidently (C as 0
on 1.00 of frames, R and U as 2 on 0.90, X as 1 on 0.99; they emit at the digits gate on
0.83-0.93 of their frames), so the proxy shows the forest has a digit for many handshapes, not
that it reads mine; and on the 2,890 hold records (148 holds) of the letters-mode browser log of
my hand, where every digit emission is a false one, the forest emits a digit (mostly 0 or 1: 57%
and 40% of the emissions) on 0.162 of single frames at the digits-mode vote gate — about one
frame in six — against 0.883 at the previous release's letter gate. That is
why the mode runs with `VOTE_MARGIN_CLEAR` 0.40 and `VOTE_PROB_FLOOR` 0.60
(`thresholds.DIGITS_OVERRIDES`, applied with `dataclasses.replace` so the letters' constants
never change), why the J/Z gates are disarmed (`Segmenter(arm_gates=False)`: the '1' handshape
passes the Z gate on 88% of its frames and would park the machine in a track), and why the
page opens in letters mode on every load. Duplicate suppression bounds the defect to one
spurious digit per hand-raise. Recording my own digits is the first follow-up.

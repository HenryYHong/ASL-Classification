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

**Training the static forest.** The training set is my four sessions, capped, plus three
sources of other people's hands (`strangers.py`, `SOURCES.md`):

| | frames | what it is |
| --- | --- | --- |
| mine, capped | 3,807 | four sessions, at most 160 per letter on the **pooled** per-letter block (4,878 uncapped) |
| ASLNow | 1,874 | multiple participants, one static-letter record each, through the Web Hand Landmarker the page runs |
| Ankara O/V/W/F | 687 | 218 students' digit photographs whose digit is a letter exactly (0/O, 2/V, 6/W, 9/F) |
| ASL-HG | 6,000 | 10 named signers x 24 letters x 25 frames, capped from 23,984 |
| | **12,368** | **-> 61,840 rows after jitter** |

The author cap is new and it buys on both axes at once. My own block is lopsided — 400 frames of
T and 308 of C against 100 of most letters — so the letters I happened to re-record most were
weighted most. Capping at 160 took ASLNow held out from 0.7882 to 0.8063 and the forest from
199,528 nodes to 173,184, at seed 0 on the same recipe with no ASL-HG; `train_static.AUTHOR_CAP`
carries that measurement beside the constant.

The ASL-HG cap is new too, and it is a ceiling rather than a saving. Uncapped, 24,000 frames from
ten hands outvote my 3,807 and my own pooled fold falls from **0.9256** at the cap to 0.9147 at
100 per (signer, letter) and 0.9086 with every frame in (`crossval_static.py --aslhg-cap 100` and
`--aslhg-cap 0`, seed 0; `strangers.ASLHG_CAP` carries it). The cap's own draw is fixed at
`CAP_SEED` 0 rather than following the forest seed, so a seed sweep measures the forest and the
jitter and not which 25 photographs were kept. Every
training frame still gets four Gaussian-jittered copies, each landmark coordinate moved by
N(0, (0.12 x palm width)^2) and the vector recomputed (`static_aug.py`); that replaced the
four-angle rotation augmentation two releases ago, which had lowered leave-one-session-out
accuracy (0.782 without it, 0.759 with it), and on my sessions alone it was worth +0.09 pooled /
+0.14 on the cross-day fold over no augmentation.

Other people's hands are still the largest lever, and each set adds on top of the last. Same
recipe, seed 0, pooled / cross-day:

    my four sessions alone            0.873 / 0.782     crossval_static.py --henry-only
    + ASLNow + Ankara                 0.913 / 0.873     crossval_static.py --no-aslhg
    + ASL-HG at the cap               0.926 / 0.895     crossval_static.py

The previous release's defining-geometry filter (137 of my frames dropped for X, G, Q, U, V, K,
R, P and D) is still retired: its thresholds were set on one hand, they discard half the
strangers' X, P and R frames, and with strangers in the training set the filter costs on every
axis (ASLNow five-fold 0.945 -> 0.903, three seeds); `--ruleset strong` keeps the ablation
runnable. The forest is **60 trees, at least five samples per leaf: 245,064 nodes on 61,840
rows** (3-seed mean 245,040), and every node ships to the browser. That setting was **chosen by
a rule, not inherited** (`train_static.py`'s docstring carries it and the table): a hard gate
first — a candidate that emits on any of the 43 clean idle holds at any of three jitter seeds is
out, whatever it scores — then accuracy, counted only where the gap exceeds the candidate's own
three-seed range, then nodes. 60 trees / leaf 8 was the cheapest candidate and the gate threw it
out (seed 1: 1 hold, 12 votes, all G, 0.7667). Of the three that passed, none separates on
accuracy — the largest gap on any axis is 0.0031 against a three-seed range of 0.0101 on that
same axis — so the node count decided: 245,040 for 60/5 against 245,809 for 80/8 and 326,719 for
80/5. The gate goes first because the previous release's own recipe passes it at seed 0 with a
most confident idle vote of 0.6950 and **emits at seed 2** (1 hold, 1 vote, a G, 0.7536 over its
own 0.75 floor). That headroom was a property of one draw, not of the recipe.

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

Measured, with the split each number came from. Static numbers come from `crossval_static.py`,
`crossval_strangers.py` and `crossval_signers.py`; where a row says "3 seeds" it is the mean and
standard deviation over jitter seeds 0, 1 and 2, and where it quotes a single figure the seed is
named. "S1" is the November 2024 archive, the only session recorded on a different day from the
other three. "The strangers" are the three public sets `strangers.py` puts on the training side
(ASLNow, the Ankara O/V/W/F photos and ASL-HG); a set a row holds out is never also in that row's
training side. A fourth set, `ayuraj`, is in `strangers.py` only to be refused: nothing trains on
it, ever.

**My own hand, holding out a whole recording session.**

| | result | split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.9233** ± 0.0018 over 3 seeds; at seed 0, 0.926 (4,515/4,878), hold-level 95% CI [0.877, 0.965] over 57 session x letter bursts | train on three of my sessions plus the strangers, test on every frame of the fourth; the caps and the jitter on the training folds only |
| — cross-day fold (S1 held out, all 24 letters) | **0.8897** ± 0.0041 over 3 seeds; at seed 0, 0.895 (2,128/2,378), macro per-letter recall 0.893; only letter below 0.6 recall: M 0.12 | train on S2+S3+S4 and the strangers, test on the archive |
| — the same recipe, my sessions only | 0.873 pooled, 0.782 cross-day (seed 0) | `crossval_static.py --henry-only`; the strangers' contribution is the difference |
| — the same recipe without ASL-HG | 0.913 pooled, 0.873 cross-day (seed 0) | `crossval_static.py --no-aslhg`; ASL-HG's contribution is the difference |
| — previous release, same folds | 0.9103 pooled, 0.8702 cross-day, macro S1 0.8685, over 3 seeds; 199,539 nodes | its own recipe (no ASL-HG, no author cap, 80 trees), re-measured on this machine rather than quoted from its own README. `crossval_static.py --legacy` still reproduces the much older 0.759 / 0.659 pair exactly |

**Other people's hands. These are the numbers to plan around.**

| | result | split |
| --- | --- | --- |
| **Leave-one-SIGNER-out, 10 known signers** | **0.9406** ± 0.0033 over 3 seeds; at seed 0, 0.9453 pooled over all 23,984 held-out frames, per signer 0.8796 to 1.0000 (mean 0.9452 ± 0.0397) | `crossval_signers.py`: hold out all 2,400 frames of one ASL-HG signer, train on the other nine at the shipped cap plus my sessions and the other stranger sets. The first by-signer number the letters have ever had |
| — per signer, seed 0 | P1 0.8879, P2 0.9583, P3 0.9396, P4 1.0000, P5 0.9946, P6 0.9333, P7 0.9175, P8 0.8796, P9 0.9829, P10 0.9583 | the spread describes these ten volunteers, not an interval for the next visitor |
| — the letters that fail there, seed 0 | U 0.671 (read as R on 329 of 1,000), R 0.703 (as U on 224), C 0.816 (as O on 184), S 0.814 (as N on 111), M 0.833 (as N on 166); nothing else under 0.90 | nine other signers in training does not fix U/R, so it is a feature defect and not a data shortage |
| **ASLNow held out entirely** | **0.8292** ± 0.0061 over 3 seeds; 0.836 at seed 0 | `crossval_strangers.py`: train on my sessions, the Ankara photos and ASL-HG; test on 1,874 records the forest never saw, through the browser's own landmarker, one frame per record |
| — the vote gate on those records | emits on 0.75 of single frames, right on 0.907 of those (seed 0) | same forest, `thresholds.DEFAULT`; one frame per record against a held sign's many windows, so this is a floor on what a visitor sees |
| **The permanent never-train holdout** | **0.889** on the committed `model_static.p`; the previous release's pickle reads **0.797** on the same 1,111 frames | `crossval_strangers.ayuraj_report()` over the CC0 `ayuraj` set, 5 signers. Nothing may ever train on it (`strangers.NEVER_TRAIN`), so the two figures are the same measurement and the gap is the retrain |
| — 218 signers, O/V/W/F held out | 0.9413 ± 0.0048 over 3 seeds; 0.940 at seed 0 (F 1.00, W 0.96, O 0.91, V 0.88) | train on my sessions, ASLNow and ASL-HG; test on the digit photos |
| — ASLNow, five folds | 0.936 at seed 0 | my sessions, the other stranger sets and four fifths of ASLNow in training, the fifth held out; no participant id, so a signer may sit on both sides: an upper bound |
| *(historical)* previous forest on ASL-HG | 0.8434 (20,229/23,984) | that forest had never seen ASL-HG. **This release trains on it**, so the figure is history, not a held-out number. The held-out cross-signer figure is the by-signer row above |

**Through the runtime, not the classifier.**

| | result | split |
| --- | --- | --- |
| **A stranger holding a letter through the segmenter** | 266 clips: exactly the right letter 94 (0.353), silent 172, **0 wrong letters**. Tracked ≥ 0.60 (139 clips): 58 (0.417), 0 wrong. Tracked and ≥ 1.0 s (58 clips): 33 (0.569), 0 wrong | `replay_strangers.py` over the OpenHands fingerspelling clips: one fresh `Segmenter` and one fresh MediaPipe per clip, shipped pickles, shipped thresholds. The committed result is `openhands_replay.json`. No signer ids, so this is not a by-signer number |
| — the previous release on the same clips | 82 (0.308) / 48 (0.345) / 29 (0.500), 0 wrong. The **old** forest at the **new** floor: 95 (0.357) and **8 wrong letters**, 6 of them in the tracked cut (G→O, O→E, T→O, V→O, U→V, U→R) | the middle column is the point: the floor drop buys 13 clips on the old forest and costs 8 wrong letters on strangers' video. On the retrained forest it costs none. The floor is safe only because the forest was retrained under it, and the idle gate cannot show that — a relaxed hand is not a stranger's letter |
| Static emission, my 71 held-out holds | exactly the right letter 66/71; first emission wrong 1/71; one wrong letter in all 71; silent 4/71 (S1-G, S2-K, and two S3-G); cross-day 22/24; latency 0.50 s median, 0.97 s p90 (seed 0) | `replay_static.py`: fold models, real timestamps, one fresh `Segmenter` per hold, shipped thresholds. The two extra silences are G's own holds, paying for G's 0.88 floor |
| Idle hand, 43 clean holds from a live browser log | 0/43 holds and 0/2,064 votes emit, at all three seeds | `idle_gate.py` on `idle_holds.npz`: one signer, one ~2-minute stretch, Tasks-API landmarks. Gated idle maxima: G 0.7087 / 0.7019 / 0.7101, and nothing else above 0.5130 |
| Motion letters {J, Z, MOVE} | 0.951 accuracy; J+Z recall 0.946 at `P_EMIT` 0.55; MOVE read as J 2/28, as Z 0/28 | `GroupKFold(5)` by prompted item over the S1 takes: 102 events in 80 items. The label set changed in an earlier release, so this replaces the older 0.864 / 140 gestures rather than improving on it |
| Motion, end to end on the five takes | 85 of 113 items produce their letter (was 78); 0 doubles (was 9); 0 rest-phase J/Z over 5.5 min | in-sample for the motion forest. Replayed with the per-frame MediaPipe handedness label the live path feeds, through the handedness latch; without the latch the same replay credits 79. `evaluate.py` reproduces it directly: S1 74 of 90 (`--train-session S5 --test-session S1`) plus S5 11 of 23 |
| Motion, held-out session S5 (Z only) | 8 of the 11 credited Z items emit Z (0.727); 8 of 23 prompted items end to end; 0 doubles; 0 false J/Z per minute over 1.51 min of negatives | a forest fitted on S1 alone (`train_motion.py --fit-sessions S1 --out <scratch>`, then `evaluate.py --motion-model <scratch> --test-session S5`) |
| **Motion, other people's video** | J emitted on 3 of the 3 clips that produced a runtime-reachable span; Z on 0 of 4. 5 of the 8 clips emit no J and no Z at all. 0.00 false J/Z per minute over 160.1 s of the same signers' non-J/Z fingerspelling (5 spans cut, 0 emitted) | four sources, **four clips per letter** — say it as a direction, never as a percentage. The one reachable Z span read MOVE (J 0.404 / MOVE 0.423 / Z 0.173). **This counts the motion branch only**: end to end those 8 clips also emit 8 non-target static letters (readings `HJ`, `W`, `IJL`, `YX`, `IJ`, `Y`), of which the parked `I` is the documented launch-pose read and `H`, `W`, `Y` and `X` are not. Never quote "0 wrong letters" off this row without that sentence |
| `J_GATE` on held `I` | mine 100/100, 13 of 2,278 other frames, all Y. **Strangers: 65/68 = 0.956 on ASLNow's I, and 1 of 1,806 other records, that one a Y** | committed archive and `aslnow.npz`, thumb ceiling 1.30 (`tests/test_gates_cross_signer.py`). The one piece of J/Z geometry that survives leaving my hand |
| `Z_GATE` on held `D` | mine 100/100. **Strangers: 35/72 = 0.486 on ASLNow's D**, and 70 of 1,802 other records (0.0389, against my own 0.0399) | same file. Decomposed: index straight 71/72, thumb 58/72, **curled fingers 37/72** — 22 of the 37 misses fail that clause alone. I curl the idle fingers to a median extension ratio of 0.663 and strangers sit at 1.194, right on the 1.20 line. It is not the straightness clause, which is what I would have blamed |
| The J and Z stills the forest has no class for | **0 of 93 J stills** emit anything. **39 of 155 Z stills do** (0.252): X 26, P 10, D 3 — a stranger's D read as X, as P, or correctly as D | `aslnow.npz`'s 248 J/Z records, scored through the vote gate. The previous flat 0.75 floor gave 12 of 155 on this same pickle (X 10, P 1, D 1), so the whole move from 12 to 39 is the floor drop. The forest is **not** silent on these stills, and only the J half of that sentence was ever true; a relaxed hand still never emits, a stranger's D does |
| Segmenter over 157 s of held signs | every one of the 24 letters emitted exactly once, 0 track starts | committed archive, in-sample for the static forest; the S2 holds 23/23 |
| Browser landmarker vs training landmarker | gap +0.001 over 3 seeds on the cross-day fold (paired 95% CI about ±0.03); 0.775 with the page's handedness swap, 0.730 without | the previous release's forest on the archive re-extracted with the MediaPipe Tasks API, measured with the CPU build from Python in VIDEO mode, not the page's GPU delegate. The ASLNow records are that landmarker on other people's hands, and the forest trains on them |
| Browser input resolution, hand distance | 1920x1080 to 426x240: within ±0.004; hand shrunk to the browser log's typical palm size: −0.02 | same fold, previous release's forest; the shrink is synthetic |

**0.9406 with one of ten signers held out, and 0.8292 with a whole set held out, are the
numbers to plan around for somebody else's hand. 0.9233 and 0.8897 are mine, and they are not
what a visitor gets.** The two
cross-signer figures are far apart because they are different conditions, not because one of
them is wrong: ASL-HG is ten volunteers photographed to one protocol in one place, and ASLNow is
an unknown number of people signing into their own webcams through the browser's landmarker. The
permanent holdout sits between them at 0.889. Quote all three or quote the lowest; quoting only
the highest is the mistake this table exists to prevent.

On my own folds, two of the four sessions are short re-recordings covering six and four letters,
and a fold that tests four well-separated shapes scores 1.000 without telling us much;
`crossval_static.py` prints the unweighted mean beside a note not to publish it. S2, S3 and S4
were all recorded on the same evening, 2026-09-10, about two and a half hours apart (S2 and S3
first committed at 18:40 in `ce4c26c`, S4 at 21:14 in `8be6a85`), so the pooled figure is
dominated by same-evening folds, and S1 (2024-11-04) is the only fold that tests a different day
and the only one that tests all 24 letters. Two implementations of the same recipe differed by
0.01 from the jitter RNG alone, so the third decimal of any one run means nothing; the CI and the
seed spread are the honest uncertainty. The in-session figure an old README carried (0.968) is
not reported any more: consecutive frames of one held sign are near-duplicates, so it measured
re-identification, and chasing it actively hurt — adding absolute hand extent raised it while
halving the cross-session number.

One number that must not be quoted: the shipped forest reads 0.9997 of ASL-HG's 23,984 frames.
It trains on 6,000 of them and the other 17,984 are drawn from the same hundred-frame bursts, so
that is a re-identification score. The by-signer row is what ASL-HG measures.

The training data is my four sessions — the November 2024 archive (24 letters, 2,378 frames) and,
on one evening nearly two years later, a full-alphabet pass (23 letters, 1,385 frames) plus two
targeted passes over the letters that were still confusable (6 and 4 letters, 728 and 387
frames), capped to 3,807 — and 8,561 frames of other people's hands. One session is what limited
this; the second one fixed the letters I sign the same way every day, and the strangers fixed the
ones I do not.

## Run order

From the repository root. Recording needs a camera; everything after it does not.

```
./.venv/bin/python temporal/calibrate.py --live --camera 0 --out temporal/thresholds_live.json
./.venv/bin/python temporal/collect_motion.py --camera 0 --continuous --letters J Z --clips 30 --session S6
./.venv/bin/python temporal/collect_motion.py --camera 0 --static-letters --reps 3 --session S6   # -> temporal/static_s6.npz
./.venv/bin/python temporal/ingest_aslnow.py                        # only to rebuild aslnow.npz (network)
./.venv/bin/python temporal/ingest_aslhg.py                         # only to rebuild aslhg.npz (877 MB of zip)
./.venv/bin/python temporal/ingest_ayuraj.py                        # only to rebuild ayuraj.npz (57 MB of zip)
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz static_s6.npz
./.venv/bin/python temporal/train_digits.py
./.venv/bin/python temporal/crossval_static.py --seeds 3            # my own folds
./.venv/bin/python temporal/crossval_signers.py --seeds 3           # leave one of ten signers out
./.venv/bin/python temporal/crossval_strangers.py --seeds 3         # each stranger set held out, and the never-train set
./.venv/bin/python temporal/idle_gate.py --seeds 16                  # must PASS at EVERY seed
./.venv/bin/python temporal/replay_static.py                        # what the floor costs on my 71 held signs
./.venv/bin/python temporal/replay_strangers.py --root <American>/videos --out temporal/openhands_replay.json
./.venv/bin/python temporal/label_events.py --out /tmp/events_check.npz   # cut events; READ its output; ALWAYS --out
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/evaluate.py                             # per session; in-sample for the shipped forest, and each block says so
./.venv/bin/python docs/export_models.py                            # -> models.json/.gz, golden.json, models.bin/.gz, models.meta.json
./.venv/bin/python docs/parity_probes.py                            # -> docs/parity.json, stamped with the new build id
cd docs && node test_forest.mjs                                     # FAILS on a stale parity.json, and says so
./.venv/bin/python temporal/live_demo.py --camera 0 --log tracks.jsonl
```

Three things in that list will bite if they are skipped or run wrong.

`idle_gate.py --seeds 16` is a **hard gate**, not a report. A candidate that emits on any of the
43 clean idle holds at any of the three jitter seeds is disqualified, and the fix is to raise
**the emitting letter's own** floor by 0.05 in `thresholds.VOTE_PROB_LETTER` and re-run — not to
raise the flat floor, which charges the whole alphabet for one letter's habit. A single-seed pass
is not evidence: the previous release's recipe passes at seed 0 and emits at seed 2.

`parity_probes.py` has to run after every export, because `docs/parity.json` is stamped with the
build id of the binary it was measured on. Skip it and `node docs/test_forest.mjs` fails on the
stale id. That is deliberate.

`label_events.py --out` defaults to `temporal/events.npz`, the committed motion training set, so
running the diagnostic bare on other footage used to replace 70,488 bytes of prompted events
with whatever it had just cut, print the path as though that were routine, and exit 0. It now
**refuses**: `--clips` away from the committed recording with `--out` left at the default stops
before anything is read, names the flag and exits non-zero
(`label_events.default_out_refusal`). `train_motion.py` carried the same one line of argparse
with the shipped `model_motion.p` on the end of it and refuses the same way. The rule is narrow
on purpose — the ordinary retrain (default in, default out) and any explicit `--out` both still
run — and the line in the recipe above keeps its `--out` anyway, because a backstop is not a
reason to stop naming the file you meant. `tests/test_label_events.py` proves both refusals
against the real committed files.

`replay_strangers.py` needs the OpenHands `American.zip` unpacked somewhere outside the
repository (`SOURCES.md` section 10); the committed `openhands_replay.json` is its output, and
two full runs wrote that file byte for byte identically.

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

- **U and R, on everybody's hands.** This is the one weakness that got worse relative to the
  others, because the by-signer split finally showed it. Held out one signer at a time over ten
  signers, U reads 0.671 (as R on 329 of 1,000) and R reads 0.703 (as U on 224); nothing else is
  under 0.80. Nine other signers in the training set do not fix it, so it is the feature and not
  the data: U and V differ by a fingertip gap the pairwise-distance block does measure, but U and
  R differ by which finger crosses over which, and nothing in the 112 values says so. `C` (0.816,
  read as O) is the next one down. The same two letters are at the bottom of the OpenHands replay
  (C 0 of 4 clips, U 0 of 6).
- **M on my own cross-day fold.** The archive's M was recorded with the thumb where the textbook
  handshape does not put it (it fails its own defining-geometry rule on every frame) and it is
  still the only letter under 0.6 recall across days, at 0.12 at seed 0 — up from 0.01 last
  release, which is the strangers' doing and not a fix. Re-recording it is the fix. E, N, S and O
  had the same problem two releases ago (0.12, 0.20, 0.54, 0.55) and are read now.
- **Check new recordings against the letter's defining geometry before training on them.** Every
  G frame in one session had an extended middle finger, which is an H; two thirds of a targeted
  re-recording did too. The correct frames were outvoted and G read as H everywhere. The rules
  in `static_aug.RULES` still do that check (`train_static.py --ruleset strong`), but they are
  no longer applied to the shipped training set: they were written for one hand, and applied to
  the strangers they discard half their X, P and R frames.
- **28 of the 113 prompted motion items have no reachable span.** Was 35 before the gate ceiling
  moved to 1.30. The remaining misses are segmentation — the gate never armed, no rising edge, a
  veto, or a detection gap — and none of this release's fixes address them.
- **Z does not transfer, and the reason is one constant.** `z_gate` fires on 100 of 100 of my own
  D frames and on 35 of 72 of ASLNow's, so on a stranger about half of every Z is unreachable
  before any model runs. Decomposed on those 72: index straight 71/72, thumb 58/72, curled
  fingers 37/72 — and 22 of the 37 misses fail the curled clause and nothing else. I curl the
  three idle fingers to a median extension ratio of 0.663; strangers sit at 1.194, right on the
  1.20 line the gate uses. Moving that ceiling to 1.40 takes D from 35/72 to 47/72 while the
  other 23 letters go 70/1,802 to 74/1,802, which looks nearly free — and it has not been checked
  against my archive, the prompted takes or a segmenter replay, so it has not been moved.
  `j_gate` transfers almost intact by contrast: 65 of 68 of ASLNow's I, one false positive in
  1,806, and that one a Y, which is the same letter it costs on my own hand.
- **The launch and finishing poses of J and Z are letters too, and they get emitted.** The
  pending-letter rule holds a parked I or D back while its track is in flight, so with the
  shipped forest a wrong letter inside an item is rare (`evaluate.py` on the takes: one Z item
  read as "QZ", one as "ZT", out of 53). What remains is the correct static read of the pose the
  hand rests in afterward: "JI" on 25 of the 60 J items and "ZD" or "DZD" on 5 of the Z items,
  because the signer re-forms the launch shape and holds it. The previous letter forest read
  that parked D as a low-confidence T and stayed silent instead; the shipped one reads it as D.
- **The idle floor has 0.04 of headroom, and it is still a G.** A relaxed hand hanging at a
  laptop is a loose G, and a forest that has seen many hands is confident about it. Over the 43
  clean idle holds, the letters a RESTING hand is read as are G, Q and A, and nothing else comes
  near: over sixteen jitter seeds the worst gated idle vote is G 0.8139, Q 0.5701, A 0.5189 and
  then M 0.4350. That is not noise, it is the geometry — a relaxed hand is a loose fist with the
  thumb somewhere (A), angled down (Q), or with index and thumb apart (G). So the floor is per
  letter (`VOTE_PROB_LETTER`: G 0.85, Q 0.62, A 0.60, everything else 0.55), each set above the
  worst draw rather than above one.
- **A 25th class for "not a letter" also fixes it, and still loses.** With 24 classes a resting
  hand must be assigned *some* letter, so the obvious fix is to give the forest somewhere else
  to put it: a REST class trained on `idle_strangers.npz`, 530 frames of OTHER people's hands
  caught not making a letter, cut from the OpenHands clips by `ingest_idle_strangers.py`. It
  works — one jittered copy takes `idle_gate.py --seeds 16` from failing at 4 of 8 to passing
  all 16, and leaves replay at 68/71 rather than 66/71. It also costs **2 wrong letters on the
  266 OpenHands clips against 0**, at seeds 0, 1 and 2 alike, and the margin it buys is margin
  the three floors above already buy. So it is not what ships. `rest_probe.py` has the whole
  measurement, including what does not work: 2 copies costs a D hold, 8 copies FAILS outright
  (the REST mass reshapes the letter boundaries and idle G climbs back to 0.7927), and filtering
  the negatives to drop frames near a real letter breaks the gate, because the near-misses are
  what do the work. The reasoning is sound and the numbers still say no; that is why the file is
  committed instead of deleted.
  One signer, one two-minute stretch; `idle_gate.py --seeds 16` is the thing to re-run after the
  next live session.
- **The floor drop is safe only because the forest was retrained under it.** Dropping 23 of the
  24 letters from 0.75 to 0.55 on the *previous* forest buys 13 more correct OpenHands clips and
  costs **8 wrong letters** on strangers' video (G→O, O→E, T→O, V→O, U→V, U→R in the tracked
  cut). On the retrained forest the same floor costs none. The idle gate cannot detect that
  difference, because a relaxed hand is not a stranger's letter; only `replay_strangers.py` can.
  Anyone who retrains must re-run both.
- **Two of the 71 held signs are silent, and one of them is G paying G's own bill.** S1-G and
  S2-K, against one wrong letter in all 71. Silence on a hold the forest is unsure of is the
  design. The previous release's note here said "six silent at that floor" and was read ever
  after as the floor's price, which it was not — *silent at* a floor is not *silenced by* it. On
  that forest, swept flat (`replay_static.py --floor`, seed 0, held-out fold models): of its six,
  only **S1-E** came back as soon as the floor moved at all (0.72). S1-N needed 0.59, the three
  S3-D holds needed 0.50, and S2-K never gave its letter at any floor — silent from 0.40 up, and
  a wrong H at 0.35 and 0.30. One hold, not six.
- **The cross-signer numbers are real now, and they disagree with each other.** Leave-one-signer-
  out over ten signers reads 0.941 and a whole set held out reads 0.829; the never-train holdout
  sits at 0.889. Those are three different conditions and the spread is the honest uncertainty
  about a visitor. What is still missing on the letters is a by-signer split that is not one
  dataset shot to one protocol, and an OpenHands replay with signer ids (it has none). The digits
  are the mirror image of all of this: trained on 218 strangers and never verified on mine.
- `thresholds.NEEDS_GESTURE_DATA` lists the motion constants that are still reasoned rather than
  measured.
- **Thresholds guessed before the data existed were the single largest source of missed
  letters.** `P_EMIT` was 0.70 and dropped about one genuine gesture in three; `T_MAX` was 1.80
  and sat *below* the p95 of the training set's own Z durations. Both are now swept against
  recorded events, and the vote floor is set the same way, against idle holds from a live log —
  at three seeds now, because a floor whose evidence is one draw is not evidence about the
  recipe. Check any constant against the data it is supposed to describe before trusting a live
  failure.

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
frame in six — against 0.883 at the letter gate as it stood when that was measured. That is
why the mode runs with `VOTE_MARGIN_CLEAR` 0.40, `VOTE_PROB_FLOOR` 0.60 and `VOTE_PROB` 0.70
(`thresholds.DIGITS_OVERRIDES`, applied with `dataclasses.replace` so the letters' constants
never change). That block **clears** this release's per-letter floor, and the reason is
bookkeeping rather than behavior: `VOTE_PROB_LETTER`'s keys are letters and the digit classes
are '0' to '9', so `{"G": 0.75}` could never fire here and the flat pair governed the mode
either way — but `digits_thresholds()` is what `docs/export_models.py` writes into
`models.json` and `models.meta.json`, so a letter's floor was shipping inside the numbers block
for a forest whose classes are 0-9, where the next reader would have to disprove it before
trusting anything else in the block. It is an empty dict now. No digit decision changes. The J/Z gates are disarmed in this mode (`Segmenter(arm_gates=False)`: the '1'
handshape passes the Z gate on 88% of its frames and would park the machine in a track), and the
page opens in letters mode on every load. Duplicate suppression bounds the defect to one
spurious digit per hand-raise. Recording my own digits is the first follow-up.

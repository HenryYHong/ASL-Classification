# docs/ — the recognizer in a browser tab

A port of `temporal/` to plain ES modules. MediaPipe Hands finds the hand in WASM, `features.js`
builds the same 112-D static feature and 79-D event descriptor the Python builds, `forest.js`
walks the three exported forests, `segmenter.js` runs the same state machine that decides when
a letter was actually signed, and `words.js` groups letters into words and names the word when
the frames allow it.

No server, no build step, no bundler. `index.html` loads `app.js` as a module; that is the whole
deployment. It is designed to sit on GitHub Pages.

| file | ports |
| --- | --- |
| `features.js` | `temporal/features.py` |
| `segmenter.js` | `temporal/segmenter.py` |
| `forest.js` | `sklearn.ensemble.RandomForestClassifier.predict_proba` |
| `words.js` | the scoring rule `temporal/simulate_words.py` measures offline |
| `app.js` | `temporal/live_demo.py` (camera, overlay, debug panel, the letters/numbers toggle) |
| `models.json` | `temporal/model_static.p`, `temporal/model_motion.p`, `temporal/model_digits.p`, `temporal/thresholds.py` |

Function names are the Python names in camelCase (`palm_scale` → `palmScale`,
`static_feature_v4` → `staticFeatureV4`) so the two can be diffed by eye. Thresholds keep their
SCREAMING_CASE exactly, because they are data, not code, and they travel inside `models.json`.

## Run it locally

The page needs `http://`, not `file://`: ES modules are blocked by CORS on the filesystem, and
browsers only expose a camera to a secure context, which means https or `localhost`.

```
python3 docs/devserver.py 8000       # from any directory; or: cd docs && python3 -m http.server 8000
open http://localhost:8000/
```

Nothing is installed and nothing is watched. Edit a `.js` file, reload the tab.

`devserver.py` also answers `OPTIONS /log` with 204 and accepts `POST /log`, and the page posts
its diagnostics there — every hold with its vote and every attempted track with its landmarks,
the browser twin of `live_demo.py --log`. The page probes that endpoint once on load and posts
only if the probe answers 204, so under `python -m http.server` (which answers 501) or on the
hosted page nothing is posted and nothing fills the console. The log lands in
`docs/browser_log.jsonl`, which is gitignored; the 43 idle holds that set the vote floor came
out of it.

Three things are fetched from the network on first load and then cached by the browser:
MediaPipe's `tasks-vision@0.10.18` bundle and WASM from jsDelivr (about 9 MB), the hand
landmarker `.task` file from `storage.googleapis.com`, and `models.json` from this directory —
16.8 MB raw, 2.9 MB gzipped in transit, which is how GitHub Pages serves it. `words.txt` is a
further 273 KB (130 KB gzipped), fetched in the background and not required: a page whose
dictionary failed to load still recognizes letters and simply never hints. A blocked CDN and a
slow connection look the same for the first few seconds, so `models.json`, the MediaPipe bundle
and the landmarker start (which is where the WASM and the `.task` model are actually fetched)
each run under a 20 s watchdog that changes the load line to say the download is probably
blocked rather than slow.

Before the camera is ever requested, `app.js` runs the 41 cases in `golden.json` through the
forests it just loaded — 15 frames from the training landmarker, 11 from the MediaPipe Tasks API
the page itself runs, 15 digit photos — and reports the result in the debug panel as
`self-check`. A case whose forest `models.json` does not carry, or whose feature width or class
count disagrees with that forest, is reported as a problem on the page, not as "not run"; only a
`golden.json` that cannot be fetched is "not run". A page that loads but computes the wrong
numbers is the failure this project exists to catch.

## Run the tests

Node 20. No dependencies; they read the JSON files in this directory directly.

```
cd docs
node test_features.mjs      # 41 golden cases + 20 invariants
node test_forest.mjs        # 41 golden cases, tree-walk edge cases, the gzip leg
node test_segmenter.mjs     # the state machine at 15 / 30 / 60 fps, the pending and post-motion rules, the handedness latch, gates off
node test_app.mjs           # the glue, the handedness swap, the self-check's refusals, index.html's own consistency
node test_words.mjs         # the word layer, and that models.json carries its three constants
```

All five pass. `test_forest.mjs`'s gzip leg reads `models.json.gz`, which `export_models.py`
writes beside `models.json` and which is gitignored; on a fresh clone that one check prints a
skip line until the export has been run.

Measured divergence from Python: every golden case reproduces in Python from `golden.json`'s own
6-decimal landmarks to 5.0e-7 on the feature vectors and 4.8e-7 on the probabilities (that is the
file's output rounding, not port error), and `forest.js` reproduces the stored probabilities to
2.0e-6 at worst over the 41 cases. The 11 Tasks-API cases store the label the Tasks API actually
reported, and go through the same swap the camera path uses; fed their raw label without the
swap they miss by 0.52 in probability with the argmax intact on all 11, which is why the check
is on probabilities and not on the predicted letter.

The whole `Segmenter` was replayed against the Python `Segmenter` on a recorded clip with the
same three forests on both sides: 587 frames, identical state, post-motion and pending-letter
flags on every frame, `v_bar` agreeing to 2.8e-14 and both sigmas to 3.3e-16, 33 holds with the
same winner, verdict and blocked flag, 3 cut spans and 7 emissions identical (confidence within
1.0e-6). `staticFeatureV4` matches `static_feature_v4` to 1.3e-15 on 200 random archive frames
under both handedness conventions.

## Regenerating models.json

`models.json` is derived from the three pickles in `temporal/`. Regenerate it whenever any model
is retrained or `thresholds.py` changes — the thresholds travel in the same file, so the page
cannot pick up a new model with stale constants, and a page that loads an export without
`WORD_PRIOR` never hints. `segmenter.js` refuses to build on an export missing any field in its
`REQUIRED_THRESHOLDS` list, which is how a constant added on the Python side (this release's
`HAND_SWITCH_S`, the handedness latch; 45 fields now) reaches the page only through a re-export.

```
./.venv/bin/python docs/export_models.py     # from the repository root -> docs/models.json, docs/models.json.gz, docs/golden.json
cd docs && node test_forest.mjs              # confirms the new export still matches sklearn
```

Measured on the shipped export: `models.json` is 16.80 MB raw and 2.94 MB gzipped. The letter
forest is 199,528 nodes in 80 trees (about 15.3 MB raw, 76 bytes per node; the exporter refuses
past 200,000 nodes, and the forest is trained on 37,195 rows, so 80 trees rather than 100 is
what fits), the motion forest 10,390 nodes in 300 trees (0.33 MB), and the digit forest 23,296
nodes in 100 trees (1.10 MB raw, which adds 0.27 MB gzipped over a letters-only export).
`golden.json` is 162 KB, 41 cases. Running the export twice gives byte-identical files; the
`.gz` is written with a zeroed timestamp so it is a pure function of the `.json`.

`export_models.py` flattens each tree to the four arrays a walk needs (`f`, `t`, `l`, `r`) plus
leaf class distributions, because a sklearn pickle is a Python object graph and cannot run in a
browser. The walk is sklearn's rule exactly: node `i` is internal when `f[i] >= 0`, and
`x[f[i]] <= t[i]` goes **left**.

Each forest carries a `feature` tag: `static/v4` for both static forests, `event/v1` for the
motion forest. `features.js` builds every tag in its registry and the segmenter picks the
transform by the tag, so a registered tag can never produce train/serve skew; an unregistered tag
is the one failure that produces no symptom at all — the vector is the right length, the forest is
confident, every letter is wrong — and `app.js` refuses to start on one.

`models.json.gz` is written alongside as a convenience for hosts that can serve it with
`Content-Encoding: gzip`; `forest.js` sniffs the gzip magic number and decompresses with the
platform's `DecompressionStream`, so either file works at `./models.json`. GitHub Pages gzips
`models.json` in transit on its own, so the plain file is what you want there.

## Words: the break, and the dictionary hint

The page groups letters into words and, when it can, names the word. Both halves are presentation
layers sitting on top of recognition; neither changes a letter the model emitted.

**The break.** A word ends when no hand has been detected for `SPACE_GAP`, currently 1.20 s.
That number is measured rather than chosen. The committed recordings separate into two
populations: while a hand is up and being tracked, consecutive frames are 0.041 s apart at the
median and 0.076 s at p99, and the longest tracking dropout across 13 minutes of continuous
recording is 0.996 s; deliberate hand-down rests between prompts start at 1.008 s and cluster
between 1.5 and 5 s. 1.20 s clears every dropout anyone recorded and still falls under the
shortest rest anyone took. The break is measured from the last frame with a hand in it, not from
the last emission, because the pause after a word is time spent with the hand down.

**The hint.** Every emitted letter carries the whole 24-class vote that produced it, so a
candidate spelling can be scored letter by letter against what the classifier actually saw, with
a prior that favors common words: `score(word) = sum_i log p_i(word_i) - WORD_PRIOR * ln rank(word)`,
where the rank is the word's line number in `words.txt`. At a word break the page scores every
listed word of the same collapsed length — the segmenter emits a held letter once, so a doubled
letter arrives once and HELLO is scored as HELO — and offers the best one only if it clears two
bars: the frames must make it at least `WORD_MIN_RATIO` (0.10) as likely as the letters actually
read, which bounds how far the prior can move a hint, and it must be `WORD_DOMINANCE` (10x)
likelier than the runner-up. The reading itself is a candidate: a reading that is a listed word
keeps the verdict unless a rival beats it by that same bar, so THO with a weak O is hinted to
THE. THO with a confident O is not confirmed either, on the shipped list: THE outranks it on the
prior alone (2.5 x ln 5235, about 21 nats, more than the frames can recover), and THE then fails
the `WORD_MIN_RATIO` bar because the frames rule it out, so the page shows nothing rather than
confirm a reading a far commoner neighbor outranks. A reading that carries a doubled letter and
is itself a listed word (TOO signed with a hand drop between the O's) is never overridden by its
collapsed neighbor (TO). A hint needs at least `HINT_MIN_LEN` (2) positions;
a single letter can only be `exact` (A or I) or nothing, because a stray N or T would otherwise
be "corrected" to A on the strength of one posterior. A letter the static branch can never emit
(J and Z are not among its 24 classes) is scored at `FLOOR` (1e-4) rather than at zero, so JAZZ
can still be hinted when the motion branch supplied both letters. Both constants live in
`words.js`; the three `WORD_*` constants travel in `models.json`, and an export without all three
makes the layer say `no-thresholds` and never hint.

**The list is a frequency list, and its order is the prior.** `words.txt` is built by
`build_words.py` from Peter Norvig's table of Google Books Ngram 1-gram counts
(<https://norvig.com/mayzner.html>, CC BY 3.0): the top 40,000 entries, a-z and 2 to 10 letters,
plus A and I; the source's two-letter tail (abbreviations, mostly) replaced by a curated set of
35 real two-letter words; entries that are one letter repeated removed; and macOS's
`/usr/share/dict/propernames` inserted at the rank-1000 count, which is what puts HENRY in it
(rank 943). 34,702 entries, 273 KB raw, 130 KB gzipped, in descending frequency; the previous
list was 150,594 Webster headwords with no frequencies (438 KB over the wire), under which almost
every letter string had a same-length neighbor and the layer had to abstain. The file must never
be sorted, merged or hand-edited: line number is rank, and `build_words.py` rebuilds it
byte-identically (its docstring carries the md5). Google Books skews literary — HELLO is rank
9,658.

**The reading is never rewritten.** The letters shown are always the letters emitted; a match
appears underneath as a separate line, and only for two of the eight verdicts (`exact`, `hint`).
`unlikely`, `ambiguous` and `too-short` are the layer working correctly and are shown as silence;
`no-list`, `too-long` and `no-thresholds` mean it could not look. A silent auto-correct would make
the recognizer look better than it measures, which is the failure this repository keeps removing.

**The three constants are measured offline, not on recorded words.** `WORD_MIN_RATIO`,
`WORD_DOMINANCE` and `WORD_PRIOR` (0.2 / 3 / 3.0) come from `temporal/simulate_words.py`,
which spells 2,000 frequency-weighted common words, 600 proper names and 1,000 rare words out of
random held-out holds of the leave-one-session-out letter posteriors, through the same vote gate
the page runs, over five seeds, and scores what the layer would have shown. They come from a
sweep of min ratio {0.02, 0.05, 0.1, 0.2} x dominance {3, 5, 10, 20} x prior {0, 1.1, 1.4, 1.7,
2.0, 2.5, 3.0}, the grid `simulate_words.py --sweep` runs by default, extended to min ratio 0.5
and prior 4.0 once the optimum sat on its edge, re-run for the forest that ships at the 0.75
floor. The tables want a stronger prior than the previous forest did (3.0 in the fresh-hold retry
model, 4.0 in the same-hold one) and a tighter ratio; dominance barely matters once the prior is
that strong. (0.2, 3, 3.0) is within 0.01 of the fresh-hold best and 0.025 of the same-hold best,
and against the previous (0.1, 10, 2.5) on the same posteriors it trades one point of recovered
words for hint precision 0.75 -> 0.83 (same-hold) and 0.86 -> 0.90 (fresh-hold) and 18% fewer
wrong "is a word" confirmations. At the shipped gate and constants, common words are recovered
0.43 of the time and a wrong word is shown 0.08 of the time when a misread letter is retried
from consecutive windows of the same hold (three tries), and 0.66 / 0.05 when every retry is a
fresh hold; names 0.27 / 0.06 and 0.54 / 0.03; rare words 0.14 / 0.05 and 0.36 / 0.03, where the
rare targets are drawn independently of the list and about a sixth of them are not in it at all
(in-list 0.833), so those two recovered figures are capped there and measure list coverage as
well as the layer. The previous forest and constants at their 0.55 floor gave 0.44 / 0.07 and
0.63 / 0.06 for common words: the higher floor drops more letters in the same-hold model, which
no hint can repair, while the more confident forest recovers more in the fresh-hold one. The
layer two releases ago, on its own forest and gate, measured 0.445 / 0.158 and 0.612 / 0.091.
Under the consecutive model most of the shown-wrong comes from a reading that lost a letter,
which the layer cannot see.
Those are simulator numbers — the simulator models the segmenter, and the two retry models
bracket what a signer does — not a recording of somebody
spelling a word with the intended spelling written down. That recording is still the missing
measurement, and until it exists the layer claims nothing beyond the simulation.

## Numbers mode

The `Numbers` button beside the transcript switches the page to the ASL digits 0-9. It is a
separate forest (`models.json`'s `digits` block) on the same feature, and the segmenter is
rebuilt on every toggle: J/Z arming off (`armGates: false`, because the '1' handshape passes the
Z gate on 88% of its frames and would park the machine in a track), no motion model, the two
vote constants raised to `VOTE_MARGIN_CLEAR` 0.40 and `VOTE_PROB_FLOOR` 0.60 from the block's
own `thresholds`, and the word layer off — a space is still written at the break, but nothing is
looked up. The page opens in letters mode on every load; the choice is deliberately not
remembered, so a visitor never returns to a page silently left in numbers mode. The debug panel's
gate line reads `gates off (numbers)` in that mode, and the toggle refuses with a problem
message if the export carries no digit forest.

The forest is trained on 218 signers from a public dataset, not on the author, and it has not
been verified on the author's hand or on live video. It scores 0.986 with each signer held out on
that dataset's photos, and the only evidence that it reads this signer is a proxy: the author's O,
V, W, F and B frames read as 0, 2, 6, 9 and 4 on 0.956 of frames. That is weak evidence, because
the author's C, R, X and U frames read as 0, 2, 1 and 2 just as confidently (C as 0 on 1.00 of
frames, R and U as 2 on 0.90, X as 1 on 0.99, emitting at the digits gate on 0.83-0.93 of them).
A relaxed hand reads as a digit (mostly 0 or 1: 57% and 40% of the emissions) on about one frame
in six at that gate — 0.162, the single-frame rate over the 2,890 hold records (148 holds) of the
letters-mode browser log, where any digit emission is a false one — and duplicate suppression
bounds that to one spurious digit per hand-raise. The page says so under the video before anyone
reads a digit off it.

## Accuracy, and the split each number came from

These are `temporal/README.md`'s numbers. Nothing about putting the model in a browser changes
them; the browser reproduces the Python's arithmetic, not its accuracy.

| | result | split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.913** (4,454/4,878); hold-level 95% CI [0.858, 0.958] over 57 session x letter bursts; 3 seeds 0.910 ± 0.004 | `temporal/crossval_static.py`: train on three of the author's sessions plus the strangers, test on the fourth, every held-out frame counted once |
| — cross-day fold (S1 held out, all 24 letters) | 0.873; macro per-letter recall 0.871 | hold out the 2024 archive, train on the three 2026 sessions and the strangers |
| — the same recipe on the author's sessions only | 0.867 / 0.778 | `crossval_static.py --henry-only` |
| — previous releases, same folds | 0.861 / 0.763; before that 0.759 / 0.659 | `crossval_static.py --legacy` reproduces the older pair |
| **Static letters, other people's hands** | **0.790** ± 0.003 on 1,874 ASLNow records (24 letters, this page's landmarker); the one-signer forest 0.781 | `temporal/crossval_strangers.py`: train on the author's sessions and the 218-signer O/V/W/F photos, test on a set the forest never saw |
| — the vote gate on those records | emits on 0.46 of single frames, right on 0.962 of those | same forest, shipped thresholds; one frame per record, so a floor on what a visitor sees |
| — 218 signers, O/V/W/F | 0.936 ± 0.004; the one-signer forest 0.766 (V 0.17 -> 0.83) | train on the author's sessions and ASLNow, test on the digit photos |
| — ASLNow, five folds | 0.946 ± 0.002 | a participant may sit on both sides (no ids), so an upper bound |
| Static emission, the 71 held-out holds | exactly the right letter 64/71; one wrong letter in 71; silent 6/71; latency 0.46 s median, 1.27 s p90; cross-day 21/24 | `temporal/replay_static.py`: fold models, real timestamps, one fresh segmenter per hold; previous release 65/71, four wrong letters, silent 2, cross-day 19/24 |
| Idle hand, 43 clean holds from this page's own log | 0/43 holds, 0/2,064 votes emit at the 0.75 floor; 5/43 at the previous 0.55 floor, every one a G | `temporal/idle_gate.py`: one signer, one ~2-minute stretch |
| Motion letters {J, Z, MOVE} | 0.951; J+Z recall 0.946 at `P_EMIT` 0.55; 2 false J in 28 MOVE events | `GroupKFold(5)` by prompted item, 102 events in 80 items; the label set changed this release, so this replaces the earlier 0.864 |
| Motion, end to end on the five takes | 85 of 113 items produce their letter, 0 doubles, 0 rest-phase J/Z over 5.5 min | in-sample for the motion forest; replayed with the per-frame handedness label this page feeds, through the segmenter's handedness latch (`HAND_SWITCH_S`, 0.50 s) — without the latch the same replay credits 79, with it all 113 item strings match a replay with one modal label per take, which is how training events are cut; `temporal/evaluate.py` gives the same 85 (S1 74/90 + S5 11/23) |
| `J_GATE` on held `I` | 100/100; 13 of 2,278 other frames pass, all Y | committed archive; a Y held still and then moved could arm a track that no negative example resembles, and no such footage exists |
| Segmenter over 157 s of held signs | 24/24 letters exactly once, 0 track starts | committed archive, in-sample |
| This page's landmarker vs the training landmarker | gap +0.001 over 3 seeds on the cross-day fold (paired 95% CI about ±0.03); 0.775 with the handedness swap, 0.730 without | the previous release's forest on the archive re-extracted with the Tasks API — the CPU build run from Python in VIDEO mode, not the GPU delegate this page runs. The ASLNow records are that landmarker on other people's hands, and the forest now trains on them |
| Input resolution, hand distance | 1920x1080 to 426x240: within ±0.004; hand shrunk to the log's typical palm size: −0.02 | same fold; synthetic shrink |
| Digits, leave-signer-out | 0.986 (1,780/1,805); worst digit 6 at 0.965 | `GroupKFold(5)` by signer over 222 signer runs of a public photo set; nothing on the author |

**0.91 is the figure worth quoting, 0.87 is the one to plan around for the author's hand, and
0.79 is the one to plan around for anyone else's.** Two of the four sessions are targeted
re-recordings covering six and four letters, so their folds score high on a handful of
well-separated shapes; the pooled figure counts every held-out frame once. The three later
sessions were recorded on the same evening, 2026-09-10, about two and a half hours apart, so
the 2024 archive is the only cross-day fold and the only one that tests the whole alphabet. On
it M (0.01) and R (0.58) fall below 0.6 recall; the archive's M was recorded off the textbook
handshape, which is a data limitation, not a threshold to tune, and E, N, S and O, which fell
below 0.6 in the previous release, are read now that other people's versions of them are in the
training set.

The letters are no longer **one signer**: the forest trains on four static sessions of the
author (4,878 frames) plus 2,561 frames of other people's hands from two public landmark sets
(`temporal/strangers.py`; 37,195 rows after jitter), and it is measured on strangers with each
set held out in turn. What is still one signer is the motion branch (five prompted takes, 117
events in 92 items), the idle-hand log, and the replay of held signs. The 0.79 is a forest that
never saw the ASLNow set; the shipped forest trains on it, so a visitor gets something better
than that by an amount that cannot be measured until another multi-signer set exists, and 0.95
is its ceiling. The digits are the reverse: 218 strangers and never the author.

The page repeats these numbers directly under the video, before anyone reads a letter off it,
for that reason.

## What is exported at what precision, and why

Split thresholds are exported at full precision. They were once rounded to 5 decimal places to
keep `models.json` small, and on the previous 400-tree forest that moved a split point past a
feature value on 7 of 1,215 probe vectors, each time by exactly one tree's vote (1/400 =
2.5e-3); 2 of those changed the top-1 label, on inputs where sklearn's own top two classes were an
exact tie. The practical effect was nil, but "reproduces sklearn exactly" is a claim the page
makes to visitors in its own readout, and a self-check that can be wrong on one input in 170 is
not a self-check. `repr()` round-trips a float64 exactly and costs about 40% more bytes before
gzip.

Leaf class distributions are exported at 4 decimal places, and that is now a real quantization:
with `min_samples_leaf` 5, 65% of the letter forest's leaves hold a mixed class count (the
previous forest's leaves were one-hot, which is why the old README could measure 0.00e+0).
Measured over 1,215 probe vectors — the real golden features, convex blends of them, and noisy
copies at four scales — against sklearn: the tree walk lands on sklearn's own leaf in every tree
on all 1,215 probes (0.00e+0 against sklearn's leaf indices read at the exported precision), and
against full-precision `predict_proba` the largest difference is 5.1e-6 with 0 argmax changes.
That 5.1e-6 is entirely the 4-dp leaf rounding, bounded at 5e-5 per leaf and therefore 5e-5 for
the forest average, half the golden tolerance of 1e-4 and three orders of magnitude under any
vote constant. Full-precision leaves would add 0.88 MB raw for nothing the page can act on. The
digit forest measures the same way: 5.0e-6, 0 argmax changes, exact leaf agreement.

**One operational consequence:** `golden.json` is generated from a specific `models.json`, so
the two ship together. Deploy a new forest with an old golden file and the page opens with a
self-check failure, which looks alarming and means only that the reference file is stale. The
golden file also pins the averaging rule: a forest that voted on per-tree argmaxes instead of
averaging per-tree probabilities disagrees with it on 17 of the 41 cases.

## What the debug panel is for

The likeliest failure in this system is a gate or a trigger that never fires: J simply becomes
unreachable while the page looks healthy, because it still recognizes the other 24 letters.
Nothing in the emitted string distinguishes "no J was signed" from "J cannot be signed". The
meters draw each threshold as a tick, so a `v_bar` sitting just under `V_MOVE_ARMED` and never
crossing it is visible rather than merely suspected. That is also how the original 5-sample
smoothing bug was found — it worked at 15 fps and never fired at 30. The `sigma` meter plots the
plain shape deviation with the `SHAPE_STABLE` and `RIGID_VETO` ticks, and the separate `rigid`
readout shows the rotation-aligned deviation the veto actually reads, turning red above it; the
two are different signals — one asks whether the handshape settled, the other whether it stayed
rigid through a stroke once the wrist's rotation is removed.

Thresholds are the archive defaults: percentiles measured on one camera at about 15 fps. They
are in seconds and palm-widths rather than frames and pixels so they transfer between cameras,
but they were not measured on yours. `temporal/calibrate.py --live` refits them to a JSON file;
the page reads only what `export_models.py` writes from `thresholds.py`, so a refit reaches the
page by editing the defaults there and re-exporting.

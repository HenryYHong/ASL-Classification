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
| `models.json` | `temporal/model_static.p`, `temporal/model_motion.p`, `temporal/model_digits.p`, `temporal/thresholds.py`; committed as the readable reference, not fetched by the page |
| `models.bin.gz` + `models.meta.json` | the same three forests and the same thresholds, packed; **this is what the page fetches**, see **The binary format** below |

Function names are the Python names in camelCase (`palm_scale` → `palmScale`,
`static_feature_v4` → `staticFeatureV4`) so the two can be diffed by eye. Thresholds keep their
SCREAMING_CASE exactly, because they are data, not code, and they travel inside the export —
`models.json`'s `thresholds` block, and `models.meta.json`'s, which carry the same 46 fields.

## Run it locally

The page needs `http://`, not `file://`: ES modules are blocked by CORS on the filesystem, and
browsers only expose a camera to a secure context, which means https or `localhost`.

```
python3 docs/devserver.py 8000       # from any directory; or: cd docs && python3 -m http.server
8000 open http://localhost:8000/
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
MediaPipe's `tasks-vision@0.10.18` bundle and WASM from jsDelivr (about 9 MB), the hand landmarker
`.task` file from `storage.googleapis.com`, and the models from this directory — `models.bin.gz`
plus `models.meta.json`, **1,183,438 + 965 = 1,184,403 B on the wire**. That is 62.4% less than
the 3,145,891 B the previous release put there, on a letter forest that grew 22.8% over the same
span. `app.js` calls `loadModels('./models.bin.gz', './models.meta.json')`; see **The binary
format** below.

`docs/models.json` is still committed and still served, as the readable reference — the same
three forests in one file anyone can open, 20,361,559 B raw and 3,703,057 B under Pages' own gzip
(level 5) — but the page does not fetch it. `loadModels` decides by the bytes rather than the file
name and returns the identical object from either container, so pointing that one call back at
`./models.json` is all it takes to load the JSON instead. `docs/models.bin.gz` is committed
pre-compressed because GitHub Pages compresses text media types on the fly and leaves
`application/octet-stream` alone: served raw, `models.bin` would put 2,555,774 B on the wire.

`words.txt` is a further 273 KB (130 KB gzipped), fetched in the background and not required:
a page whose dictionary failed to load still recognizes letters and simply never hints. A blocked CDN and a slow connection look the same
for the first few seconds, so the model files, the MediaPipe bundle and the landmarker start (which
is where the WASM and the `.task` model are actually fetched) each run under a 20 s watchdog that
changes the load line to say the download is probably blocked rather than slow.

Before the camera is ever requested, `app.js` runs the 41 cases in `golden.json` through the
forests it just loaded — 15 frames from the training landmarker, 11 from the MediaPipe Tasks API
the page itself runs, 15 digit photos — and reports the result in the debug panel as
`self-check`. A case whose forest the export does not carry, or whose feature width or class
count disagrees with that forest, is reported as a problem on the page, not as "not run"; only a
`golden.json` that cannot be fetched is "not run". A page that loads but computes the wrong
numbers is the failure this project exists to catch.

## Run the tests

Node 20. No dependencies; they read the JSON files in this directory directly.

```
cd docs node test_features.mjs      # 41 golden cases + 20 invariants node test_forest.mjs
# 41 golden cases, tree-walk edge cases, the gzip leg, the binary and its parity file node
test_segmenter.mjs     # the state machine at 15 / 30 / 60 fps, the pending and post-motion rules,
the handedness latch, gates off node test_app.mjs           # the glue, the handedness swap, the
self-check's refusals, index.html's own consistency node test_words.mjs         # the word layer,
and that models.json carries its three constants
```

All six pass, and so do the 93 checks in `temporal/tests/`. Exactly one check skips on a fresh
clone: `test_forest.mjs`'s gzip leg reads `models.json.gz`, which `export_models.py` writes beside
`models.json` and which is gitignored, so it prints a skip line until the export has been run.
Everything else runs from committed files — the binary leg prefers `models.bin` and falls back to
the committed `models.bin.gz`, which is how it still passes on a clone that has never exported,
and it reads `docs/parity.json` unconditionally. It **fails on a stale `parity.json`**: the file
carries the build id of the binary it was measured on, so an export without a following
`parity_probes.py` run is caught rather than believed. I checked the fresh-clone behavior by
moving both gitignored files aside and rerunning: one skip, `41/41 golden cases match`, exit 0.

Measured divergence from Python: every golden case reproduces in Python from `golden.json`'s own
6-decimal landmarks to 5.0e-7 on the feature vectors and 4.99e-7 on the probabilities (that is
the file's output rounding, not port error), and `forest.js` reproduces the stored probabilities
to 3.3e-6 at worst over the 41 cases through `models.json` and 4.98e-7 through `models.bin`. The
binary is what the page loads, so that second figure is the one it reports: booted in headless
Chrome against `devserver.py`, its `self-check` line reads `41/41 cases (26 static + 15 digits),
forest 5.0e-7, whole path 5.0e-7`. The
11 Tasks-API cases store the label the Tasks API actually reported, and go through the same swap
the camera path uses; fed their raw label without the swap they miss by 0.52 in probability with
the argmax intact on all 11, which is why the check is on probabilities and not on the predicted
letter.

`golden.json` is a **port-parity fixture, not an accuracy measure.** Python's own prediction
equals the stored label on 40 of its 41 cases; that 40/41 is not an accuracy figure and is not a
regression, and it must not be quoted as either. The one case that differs is an archive frame,
`static_sequences_tasks.npz[0,71]`, labeled A and predicted S (S 0.2643 / M 0.2585 / A 0.2193).
The forest trained on that burst — A's pooled block is 159 frames, under the 160-frame author
cap, so every A frame trains — and the `solutions`-API extraction of the same JPEG reads A at
0.8105. The two landmarkers differ by 0.0481 in normalized units on that one hand. What the
fixture is for is the browser agreeing with Python, and on that it is 41 of 41.

The whole `Segmenter` was replayed against the Python `Segmenter` on a recorded clip with the
same three forests on both sides: 587 frames, identical state, post-motion and pending-letter
flags on every frame, `v_bar` agreeing to 2.8e-14 and both sigmas to 3.3e-16, 33 holds with the
same winner, verdict and blocked flag, 3 cut spans and 7 emissions identical (confidence within
1.0e-6). `staticFeatureV4` matches `static_feature_v4` to 1.3e-15 on 200 random archive frames
under both handedness conventions.

## Regenerating the models

`models.json`, `models.bin` and their metadata are all derived from the three pickles in
`temporal/`. Regenerate them whenever any model is retrained or `thresholds.py` changes — the
thresholds travel in the same files, so the page cannot pick up a new model with stale constants,
and a page that loads an export without `WORD_PRIOR` never hints. `segmenter.js` refuses to build
on an export missing any field in its `REQUIRED_THRESHOLDS` list (38 of the 46 fields the
dataclass carries; `words.js` checks its own three). That list is how a constant added on the
Python side reaches the page only through a re-export — this release's addition is
`VOTE_PROB_LETTER`, the per-letter vote floor, and it is the half of that change that matters
most: without the name in `REQUIRED_THRESHOLDS`, a `models.json` written before the field existed
would build a segmenter that silently ran a flat 0.55 floor and emitted on a relaxed hand, which
is precisely what the idle gate exists to prevent. With it, the same file throws.

```
./.venv/bin/python docs/export_models.py     # -> models.json, models.json.gz, models.bin,
models.bin.gz,
                                             #    models.meta.json, golden.json
./.venv/bin/python docs/parity_probes.py     # -> docs/parity.json, stamped with the new build id
cd docs && node test_forest.mjs              # confirms the export still matches sklearn; FAILS on
a stale parity.json
```

Run those three in that order. Skipping the second makes the third fail, and it says so. Running
the export twice gives byte-identical files; the `.gz` files are written with a zeroed timestamp
and no stored name, so each is a pure function of the file beside it.

Measured on the shipped export:

| | raw | on the wire | nodes |
| --- | --- | --- | --- |
| `models.json` | 20,361,559 B | 3,703,057 B (what the deployed site returns) | 278,750 |
| `models.bin` + `models.meta.json` | 2,555,774 + 2,726 B | 1,183,438 + 965 B = **1,184,403 B** | the same 278,750 |
| — the letter forest alone | 18,931,302 B of JSON (77.25 B/node) | 2,264,570 B of binary (9.24 B/node), 1,041,301 B gzipped (4.25 B/node) | 245,064 |
| `golden.json` | 162,184 B, 41 cases | | |

The three forests are 245,064 nodes in 60 trees (letters; 122,502 internal and 122,562 leaves),
10,390 in 300 trees (motion) and 23,296 in 100 trees (digits) — 278,750 nodes, 139,145 of them
internal. `NODE_BUDGET` in `export_models.py` is 260,000 and was 200,000; the export stopped
against the old value at 245,064, which is the point of having it. The remaining 6% of headroom
is room for one retrain's drift, not for another dataset. Against the previous release, measured
on that release's own committed file rather than quoted from it: 16,801,546 raw and 3,145,891 on
the wire, so +21.2% raw and **+17.7% on the wire**, and 1,000,455 to 1,184,403 (+18.4%) by the
binary route.

`export_models.py` flattens each tree to the four arrays a walk needs (`f`, `t`, `l`, `r`) plus
leaf class distributions, because a sklearn pickle is a Python object graph and cannot run in a
browser. The walk is sklearn's rule exactly: node `i` is internal when `f[i] >= 0`, and
`x[f[i]] <= t[i]` goes **left**.

Each forest carries a `feature` tag: `static/v4` for both static forests, `event/v1` for the
motion forest. `features.js` builds every tag in its registry and the segmenter picks the
transform by the tag, so a registered tag can never produce train/serve skew; an unregistered tag
is the one failure that produces no symptom at all — the vector is the right length, the forest is
confident, every letter is wrong — and `app.js` refuses to start on one.

`models.json.gz` is written alongside `models.json` as a convenience for hosts that can serve it
with `Content-Encoding: gzip`; `forest.js` sniffs the gzip magic number and decompresses with the
platform's `DecompressionStream`, so either file works at `./models.json`. It is gitignored,
because GitHub Pages gzips `models.json` in transit on its own and the plain file is what you
want there.

## The binary format

`models.bin` is the same three forests packed into typed arrays instead of JSON numbers. It
exists because the letter forest is 245,064 nodes and JSON spends 77.25 bytes on each of them:
`"f":83,"t":0.41237819194793701` and so on, a decimal string per split threshold, re-parsed on
every page load. The binary spends 9.24.

**`models.bin.gz` is committed and `models.bin` is not.** That is not a preference; it is what
GitHub Pages does. Pages compresses text media types on the fly — which is why `models.json` is
committed raw and its `.gz` is gitignored — but it leaves `application/octet-stream` alone.
Serving the raw binary would put 2,555,774 B on the wire where the pre-compressed file puts
1,183,438 B, so the compressed one is the artifact and the raw one is a build product.
`models.meta.json` (2,726 B: format, version, build id, byte count, thresholds, and per forest
the class list, feature tag, dimension, tree count, node count and section offsets) and
`parity.json` (1,062 B) are small and are committed beside it.

The layout, mirrored in `docs/export_models.py` and `docs/forest.js` — they move together, and
`BIN_VERSION` is bumped whenever a section's dtype, order or meaning changes:

```
offset  0   magic     b"ASLFRST\x00"          8 bytes offset  8   version   u32 LE
1 offset 12   endian    u32 LE 0x04030201       read back through a platform-endian Uint32Array
offset 16   header    u32 LE 40               bytes before the first section offset 20   reserved
u32 offset 24   build id  16 bytes                first 16 bytes of SHA-256 over everything after
the header
```

then, per forest, in write order, each section 8-byte aligned and located by an absolute
`[byteOffset, elementCount]` pair in `models.meta.json`:

```
n     u32[trees]      node count per tree
bits  u8   packbits, MSB first, ceil(n/8) per tree: is node i internal
f     i8[internal]    split feature      (the feature dimension is 112, so i8 is enough)
t     f64[internal]   split threshold    full precision, for the reason below
r     u16[internal]   right child, as an OFFSET from the node
vk    u8[leaves]      how many classes this leaf holds
vi    u8[nnz]         class index
vc    u16[nnz]        exact integer sample count
```

Two of those are worth naming. The left child is **not stored**, because sklearn always lays a
tree out so that `children_left[i] == i + 1`; the export asserts that per tree rather than
assuming it. And `vc` holds **exact integer counts**, not probabilities: a leaf's probability is
recovered as `count / sum(counts of that leaf)` into the same `Float32Array` the JSON path fills.
`models.bin` and `models.meta.json` are bound by the build id, and `forest.js` refuses a
mismatched pair; the current pair is `aa052b8b95cb23a9de227ef07b3c3349`.

Measured on the committed export, my own runs, Node v20:

- **Decode.** 114-160 ms for `models.json` (97-122 ms of that is `JSON.parse`) against 18-38 ms
  for `models.bin` (6.4-9.0 ms of that is gunzip), six interleaved runs of each so a warm or busy
  machine hits both the same way.
- **Memory.** Peak `heapUsed + arrayBuffers` over one format's load in a process that has loaded
  nothing else: 100.4 MB against 28.3 MB. Peak RSS 208.1 MB against 73.4 MB. Both settle at the
  same 19.0 MB of typed arrays, because the decoded model is identical; the difference is the
  garbage JSON makes on the way there.
- **Routing.** 0 differences in `f`, `t`, `l`, `r` or leaf offset over all 278,750 nodes.
- **Leaves.** The two leaf tables differ by at most 4.78e-5 over 3,074,503 cells, and that is
  entirely `models.json`'s 4-decimal rounding of leaf probabilities against the binary's exact
  integer counts. It is bounded at 5e-5 per leaf by construction.

What it is worth against sklearn itself, from `parity.json` and `test_forest.mjs`: over 1,215
probe vectors per forest, the tree walk lands on sklearn's own leaf in every tree on every probe
— 0 of 72,900 leaf visits disagree on the letters and 0 of 121,500 on the digits — and against
full-precision `predict_proba` the largest probability difference is 6.25e-6 through `models.json`
and **4.05e-9 through `models.bin`** on the letters, 5.07e-6 and 4.75e-9 on the digits, with 0
argmax changes anywhere. On the 41 golden cases the same comparison is 3.33e-6 through the JSON
and 4.98e-7 through the binary, so the page's own self-check figure improved by most of an order
of magnitude when `app.js` moved over. The probe set is 26 golden bases plus 325 midpoints
between them plus 864 noisy copies at sigma 1e-4, 1e-3, 1e-2 and 5e-2 from `default_rng(0)`.

`t` stays f64, and the reason is worth writing down because the measurement argues the other way.
Narrowing it to float32 would save 490,008 B on the letter forest and 46,392 B on the digits, and
on *this* forest it moves a leaf on 0 of 72,900 letter tree visits and 0 of 121,500 digit visits,
touching none of the 1,215 probes. It costs nothing measurable here. It stays f64 anyway, and
`export_models.py` records why beside the constant: on the **previous** forest the same narrowing
moved a leaf on 1 of that forest's 97,200 tree visits and touched 1 of the 1,215 probes. A probe
set that catches a defect once and misses it once has measured the defect, not its absence. It is
the same argument that keeps split thresholds off five decimals, where the rounding moved a split
point past a feature value on 7 of 1,215 probes and changed the top-1 label on 2 of them. "No
difference on this draw" is not what the page claims in its own readout — it claims it reproduces
sklearn — and 490 KB of a 1.18 MB wire load is not what that claim is worth trading for.

**The page fetches the binary.** `app.js` calls
`loadModels('./models.bin.gz', './models.meta.json')`, and `loadModels` returns the identical
`{static, motion, thresholds, digits, digitsThresholds}` either way, so nothing else in `app.js`
had to change: the load line says "about 1.18 MB, already compressed", the watchdog around it is
labeled with the two files it fetches, and `test_app.mjs` pins all three together. Booted in
headless Chrome against `devserver.py` — which serves everything raw — the Performance API reports
1,183,738 B transferred for `models.bin.gz` and 3,026 B for `models.meta.json`, headers included,
and no request for `models.json` at all.

The deployed figures are measured on the deployed site, not estimated from a local `gzip`. Asking
henryyhong.com for the two files the way a browser does (`Accept-Encoding: gzip, deflate, br`):
`models.bin.gz` comes back with no `Content-Encoding` at 1,183,438 B, because Pages leaves
`application/gzip` alone, which is exactly why the `.gz` is committed rather than generated at
deploy time; `models.meta.json` comes back `Content-Encoding: gzip` at 965 B. **1,184,403 B**,
and all three of `models.bin.gz`, `models.meta.json` and `parity.json` hash identically to the
committed files. A local `gzip -5 -c` predicts 982 B for the meta file, so the estimate was 17 B
pessimistic; the live number is the one quoted here.

**`models.json` stays committed anyway, as the readable reference.** It is the one file in this
directory a person can open and read the forests out of, and `loadModels` still loads it —
`test_forest.mjs` serves it over HTTP and loads it through the same call, unconditionally,
because the container no page exercises is the one that rots. Pointing that single line in
`app.js` back at `./models.json` is the whole of switching back.

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
candidate spelling can be scored letter by letter against what the classifier actually saw, with a
prior that favors common words: `score(word) = sum_i log p_i(word_i) - WORD_PRIOR * ln
rank(word)`, where the rank is the word's line number in `words.txt`. At a word break the page
scores every listed word of the same collapsed length — the segmenter emits a held letter once, so
a doubled letter arrives once and HELLO is scored as HELO — and offers the best one only if it
clears two bars: the frames must make it at least `WORD_MIN_RATIO` (0.10) as likely as the letters
actually read, which bounds how far the prior can move a hint, and it must be `WORD_DOMINANCE`
(10x) likelier than the runner-up. The reading itself is a candidate: a reading that is a listed
word keeps the verdict unless a rival beats it by that same bar, so THO with a weak O is hinted to
THE. THO with a confident O is not confirmed either, on the shipped list: THE outranks it on the
prior alone (2.5 x ln 5235, about 21 nats, more than the frames can recover), and THE then fails
the `WORD_MIN_RATIO` bar because the frames rule it out, so the page shows nothing rather than
confirm a reading a far commoner neighbor outranks. A reading that carries a doubled letter and is
itself a listed word (TOO signed with a hand drop between the O's) is never overridden by its
collapsed neighbor (TO). A hint needs at least `HINT_MIN_LEN` (2) positions; a single letter can
only be `exact` (A or I) or nothing, because a stray N or T would otherwise be "corrected" to A on
the strength of one posterior. A letter the static branch can never emit (J and Z are not among
its 24 classes) is scored at `FLOOR` (1e-4) rather than at zero, so JAZZ can still be hinted when
the motion branch supplied both letters. Both constants live in `words.js`; the three `WORD_*`
constants travel in `models.json`, and an export without all three makes the layer say
`no-thresholds` and never hint.

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

**The three constants are measured offline, not on recorded words — and the sweep that set them is
now one release old.** `WORD_MIN_RATIO`, `WORD_DOMINANCE` and `WORD_PRIOR` (0.2 / 3 / 3.0) were
swept against the **previous** forest at its flat 0.75 floor. Both of that sweep's inputs have
changed: the posteriors come from the letter forest, and the gate is the vote floor. Every figure
in this section therefore describes that sweep and not this release, and re-running
`simulate_words.py --sweep` on the shipped forest and floor is a follow-up. The layer never
rewrites a letter, so the cost of leaving them inherited is a hint shown too rarely or too often,
never a changed reading. The sweep itself: `temporal/simulate_words.py`, which spells 2,000
frequency-weighted common words, 600 proper names and 1,000 rare words out of random held-out
holds of the leave-one-session-out letter posteriors, through the same vote gate the page runs,
over five seeds, and scores what the layer would have shown. They come from a sweep of min ratio
{0.02, 0.05, 0.1, 0.2} x dominance {3, 5, 10, 20} x prior {0, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0}, the
grid `simulate_words.py --sweep` runs by default, extended to min ratio 0.5 and prior 4.0 once the
optimum sat on its edge, re-run for the forest that shipped last release at its 0.75 floor. That
forest wanted a stronger prior than the one before it (3.0 in the fresh-hold retry model, 4.0 in
the same-hold one) and a tighter ratio; dominance barely matters once the prior is that strong.
(0.2, 3, 3.0) is within 0.01 of the fresh-hold best and 0.025 of the same-hold best, and against
the previous (0.1, 10, 2.5) on the same posteriors it trades one point of recovered words for hint
precision 0.75 -> 0.83 (same-hold) and 0.86 -> 0.90 (fresh-hold) and 18% fewer wrong "is a word"
confirmations. At that gate and these constants, common words are recovered 0.43 of the time and a
wrong word is shown 0.08 of the time when a misread letter is retried from consecutive windows of
the same hold (three tries), and 0.66 / 0.05 when every retry is a fresh hold; names 0.27 / 0.06
and 0.54 / 0.03; rare words 0.14 / 0.05 and 0.36 / 0.03, where the rare targets are drawn
independently of the list and about a sixth of them are not in it at all (in-list 0.833), so those
two recovered figures are capped there and measure list coverage as well as the layer. The forest
and constants before those, at their 0.55 floor, gave 0.44 / 0.07 and 0.63 / 0.06 for common
words: the higher floor drops more letters in the same-hold model, which no hint can repair, while
the more confident forest recovers more in the fresh-hold one. The layer before that, on its own
forest and gate, measured 0.445 / 0.158 and 0.612 / 0.091. Under the consecutive model most of the
shown-wrong comes from a reading that lost a letter, which the layer cannot see. Those are
simulator numbers — the simulator models the segmenter, and the two retry models bracket what a
signer does — not a recording of somebody spelling a word with the intended spelling written down.
That recording is still the missing measurement, and until it exists the layer claims nothing
beyond the simulation.

## Numbers mode

The `Numbers` button beside the transcript switches the page to the ASL digits 0-9. It is a
separate forest (`models.json`'s `digits` block) on the same feature, and the segmenter is rebuilt
on every toggle: J/Z arming off (`armGates: false`, because the '1' handshape passes the Z gate on
88% of its frames and would park the machine in a track), no motion model, the vote constants
raised to `VOTE_MARGIN_CLEAR` 0.40, `VOTE_PROB_FLOOR` 0.60 and `VOTE_PROB` 0.70 from the block's
own `thresholds`, and the word layer off — a space is still written at the break, but nothing is
looked up. The letters' per-letter floor is inert here: `VOTE_PROB_LETTER`'s keys are letters and
the digit classes are '0' to '9', so this mode runs on the flat pair as it did before. The page
opens in letters mode on every load; the choice is deliberately not remembered, so a visitor never
returns to a page silently left in numbers mode. The debug panel's gate line reads `gates off
(numbers)` in that mode, and the toggle refuses with a problem message if the export carries no
digit forest.

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

**Other people's hands. This is what a visitor is, and these are the numbers to plan around.**

| | result | split |
| --- | --- | --- |
| **Static letters, leave-one-SIGNER-out** | **0.9406** ± 0.0033 over 3 seeds; at seed 0, 0.9453 over all 23,984 held-out frames, per signer 0.8796 to 1.0000 | `temporal/crossval_signers.py`: hold out all 2,400 frames of one of ten named signers, train on the other nine plus the author's sessions and the other stranger sets |
| **Static letters, a whole set held out** | **0.8292** ± 0.0061 over 3 seeds; 0.836 at seed 0, on 1,874 ASLNow records through this page's own landmarker | `temporal/crossval_strangers.py`: the forest never saw the set. One frame per record |
| — the vote gate on those records | emits on 0.75 of single frames, right on 0.907 of those (seed 0) | same forest, shipped thresholds; a record offers one window and a held sign offers many, so this is a floor on what a visitor sees |
| **The permanent never-train holdout** | **0.889** on the shipped forest; the previous release's forest reads **0.797** on the same 1,111 frames | the CC0 `ayuraj` set, five signers. Nothing may ever train on it, so the gap between the two figures is the retrain and nothing else |
| **A stranger holding a letter through this page's own segmenter** | 266 video clips: right 96 (0.361), silent 170, **0 wrong letters**. Tracked ≥ 0.60 (139 clips): 59, 0 wrong. And ≥ 1.0 s (58 clips): 33, 0 wrong | `temporal/replay_strangers.py` over the OpenHands clips, shipped models and thresholds, one fresh segmenter per clip. No signer ids, so not a by-signer number |
| — 218 signers, O/V/W/F | 0.9413 ± 0.0048; 0.940 at seed 0 (F 1.00, W 0.96, O 0.91, V 0.88) | train on the author's sessions, ASLNow and ASL-HG; test on the digit photos |
| — ASLNow, five folds | 0.936 at seed 0 | a participant may sit on both sides (no ids), so an upper bound |
| *(historical)* previous forest on ASL-HG | 0.8434 (20,229/23,984) | that forest had never seen ASL-HG; this release trains on it, so the figure is history and the by-signer row replaces it |

**The author's own hand, holding out a whole session.** Not what a visitor gets.

| | result | split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.9233** ± 0.0018 over 3 seeds; at seed 0, 0.926 (4,515/4,878), hold-level 95% CI [0.877, 0.965] over 57 session x letter bursts | `temporal/crossval_static.py`: train on three of the author's sessions plus the strangers, test on the fourth, every held-out frame counted once |
| — cross-day fold (S1 held out, all 24 letters) | **0.8897** ± 0.0041; 0.895 at seed 0, macro per-letter recall 0.893 | hold out the 2024 archive, train on the three 2026 sessions and the strangers |
| — the same recipe on the author's sessions only | 0.873 / 0.782 at seed 0 | `crossval_static.py --henry-only` |
| — the same recipe without ASL-HG | 0.913 / 0.873 at seed 0 | `crossval_static.py --no-aslhg` |
| — previous release, same folds | 0.910 / 0.870 over 3 seeds, 199,539 nodes | its own recipe, re-measured rather than quoted |

**The runtime.**

| | result | split |
| --- | --- | --- |
| Static emission, the 71 held-out holds | exactly the right letter 68/71; one wrong letter in 71; silent 2/71 (S1-G, S2-K); cross-day 22/24; latency 0.46 s median, 0.95 s p90 (seed 0) — the counts identical at all three seeds | `temporal/replay_static.py`: fold models, real timestamps, one fresh segmenter per hold. The flat 0.75 floor it replaces gives 61/71 and 9 silent |
| Idle hand, 43 clean holds from this page's own log | 0/43 holds and 0/2,064 votes at all three seeds | `temporal/idle_gate.py`. Gated idle maxima G 0.7087 / 0.7019 / 0.7101, nothing else above 0.5130, which is why the floor is per letter |
| Motion letters {J, Z, MOVE} | 0.951; J+Z recall 0.946 at `P_EMIT` 0.55; 2 false J in 28 MOVE events | `GroupKFold(5)` by prompted item, 102 events in 80 items; the label set changed in an earlier release, so this replaces the earlier 0.864 |
| Motion, end to end on the five takes | 85 of 113 items produce their letter, 0 doubles, 0 rest-phase J/Z over 5.5 min | in-sample for the motion forest; replayed with the per-frame handedness label this page feeds, through the handedness latch (`HAND_SWITCH_S`, 0.50 s) |
| Motion, other people's video | J on 3 of the 3 clips that produced a runtime-reachable span, Z on 0 of 4; no false J or Z in 160.1 s of the same signers' other fingerspelling | four sources, four clips per letter — a direction, not a percentage. Motion branch only: those eight clips also emit eight non-target static letters |
| `J_GATE` on held `I` | author 100/100 and 13 of 2,278 other frames, all Y; **strangers 65/68**, one false positive in 1,806, also a Y | committed archive and the ASLNow J/Z stills |
| `Z_GATE` on held `D` | author 100/100; **strangers 35/72** | same. The curled-finger ceiling is the binding clause, not the straightness test |
| Segmenter over 157 s of held signs | 24/24 letters exactly once, 0 track starts | committed archive, in-sample |
| This page's landmarker vs the training landmarker | gap +0.001 over 3 seeds on the cross-day fold (paired 95% CI about ±0.03); 0.775 with the handedness swap, 0.730 without | the previous release's forest on the archive re-extracted with the Tasks API — the CPU build run from Python in VIDEO mode, not the GPU delegate this page runs |
| Input resolution, hand distance | 1920x1080 to 426x240: within ±0.004; hand shrunk to the log's typical palm size: −0.02 | same fold; synthetic shrink |
| Digits, leave-signer-out | 0.986 (1,780/1,805); worst digit 6 at 0.965 | `GroupKFold(5)` by signer over 222 signer runs of a public photo set; nothing on the author |

**0.9406 with one of ten signers held out, and 0.8292 with a whole set held out, are what a
visitor should plan around. The author's 0.9233 and 0.8897 are not.** The two cross-signer figures
are far apart because they are different conditions, not because one is wrong: the by-signer set
is ten volunteers photographed to one protocol in one place, and the held-out set is an unknown
number of people signing into their own webcams. The never-train holdout sits between them at
0.889. Quote all three, or quote the lowest.

The letters are no longer **one signer**. The forest trains on four static sessions of the
author, capped to 3,807 frames, plus 8,561 frames of other people's hands from three public
landmark sets (`temporal/strangers.py`; 61,840 rows after jitter), it is measured on strangers
with each set held out in turn, it is measured by signer over ten people, and one set is held out
of training permanently so its number can never quietly become an in-sample one. What is still
one signer is the motion branch (five prompted takes, 117 events in 92 items; eight clips of
other people's video is all there is beside it), the idle-hand log, and the replay of held signs.
The digits are the reverse: 218 strangers and never the author.

The page repeats these numbers directly under the video, before anyone reads a letter off it,
for that reason.

## What is exported at what precision, and why

The page loads the sidecar pair, so start there. **`models.bin.gz` carries the trees and nothing
else**, and nothing in it is rounded: `f` is an `i8` feature index, `t` is a full `f64` threshold,
`r` is a `u16` child offset, and each leaf holds its exact integer sample counts as `u16` rather
than a probability. **`models.meta.json` (2,726 B raw, 965 B under Pages' gzip) carries everything
that is not a tree**: per forest the class list, feature tag, dimension, tree count, node count
and the `[byteOffset, elementCount]` pair for each section; the build id that binds the two files;
and the thresholds. Those thresholds are the 46 fields `thresholds.py` defines, written as
round-tripping float64s — the identical numbers `models.json` carries in its own `thresholds`
block, and `test_forest.mjs` compares the two field by field, because a constant that arrives
rounded is a segmenter that behaves differently from `live_demo.py` and says nothing about it.

So the only rounding anywhere in the export is in `models.json`, and only in its leaves. The rest
of this section is about that, and about why the thresholds were never allowed to join it.

Split thresholds are exported at full precision. They were once rounded to 5 decimal places to
keep `models.json` small, and on the previous 400-tree forest that moved a split point past a
feature value on 7 of 1,215 probe vectors, each time by exactly one tree's vote (1/400 =
2.5e-3); 2 of those changed the top-1 label, on inputs where sklearn's own top two classes were an
exact tie. The practical effect was nil, but "reproduces sklearn exactly" is a claim the page
makes to visitors in its own readout, and a self-check that can be wrong on one input in 170 is
not a self-check. `repr()` round-trips a float64 exactly and costs about 40% more bytes before
gzip.

Leaf class distributions are where the two formats genuinely differ, and it is worth being exact
about how much. **`models.json` stores leaf probabilities rounded to 4 decimal places.
`models.bin` stores the leaf's exact integer sample counts** — `vc` is a `u16` per nonzero class,
and the probability is recovered as `count / sum(counts of that leaf)` — so the binary carries no
quantization at all and the JSON carries a real one. With `min_samples_leaf` 5 the letter forest
has 122,562 leaves holding 254,521 nonzero class counts between them, and 83,223 of those leaves
(67.9%) are mixed rather than one-hot; a forest whose leaves are all one-hot has nothing to round,
which is why an older README here could report 0.00e+0 and this one cannot.

Measured against sklearn over 1,215 probe vectors per forest — the real golden features, convex
blends of them, and noisy copies at four scales:

| | leaf disagreements | max abs probability difference | argmax changes |
| --- | --- | --- | --- |
| letters, `models.json` | 0 of 72,900 tree visits | 6.25e-6 | 0 |
| letters, `models.bin` | 0 of 72,900 | **4.05e-9** | 0 |
| digits, `models.json` | 0 of 121,500 | 5.07e-6 | 0 |
| digits, `models.bin` | 0 of 121,500 | **4.75e-9** | 0 |

The tree walk lands on sklearn's own leaf in every tree on every probe in both formats. The
JSON's 6.25e-6 is entirely the 4-decimal leaf rounding, bounded at 5e-5 per leaf and therefore
5e-5 for the forest average — half the golden tolerance of 1e-4 and three orders of magnitude
under any vote constant. The binary's 4.05e-9 is float64 summation order and nothing else. Read
directly against each other rather than against sklearn, the two leaf tables differ by at most
4.78e-5 over 3,074,503 cells, which is the same rounding seen from the other side.

Exporting the JSON's leaves at full precision instead would add 902,497 B raw to `models.json`
for nothing the page can act on, so it is not done; the binary gets exactness for free by storing
counts instead of probabilities, which is a better trade than either JSON option.

**One operational consequence:** `golden.json` is generated from a specific `models.json`, so
the two ship together. Deploy a new forest with an old golden file and the page opens with a
self-check failure, which looks alarming and means only that the reference file is stale. The
golden file also pins the averaging rule: a forest that voted on per-tree argmaxes instead of
averaging per-tree probabilities disagrees with it on 17 of the 41 cases. `models.bin`,
`models.meta.json` and `parity.json` are bound to each other more strictly, by a build id — the
first 16 bytes of a SHA-256 over everything after the binary's header — so a mismatched set is
refused at load rather than reported as a bad model. Re-deriving all 41 golden cases in Python
from `golden.json`'s own rounded landmarks agrees to 5.0e-7 on the features against the file's
stated 1e-6 and 4.99e-7 on the probabilities against its 1e-4.

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

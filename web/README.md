# web/ — the recognizer in a browser tab

A port of `temporal/` to plain ES modules. MediaPipe Hands finds the hand in WASM, `features.js`
builds the same 101-D static feature and 79-D event descriptor the Python builds, `forest.js`
walks the two exported forests, and `segmenter.js` runs the same state machine that decides when
a letter was actually signed.

No server, no build step, no bundler. `index.html` loads `app.js` as a module; that is the whole
deployment. It is designed to sit on GitHub Pages.

| file | ports |
| --- | --- |
| `features.js` | `temporal/features.py` |
| `segmenter.js` | `temporal/segmenter.py` |
| `forest.js` | `sklearn.ensemble.RandomForestClassifier.predict_proba` |
| `app.js` | `temporal/live_demo.py` (camera, overlay, debug panel) |
| `models.json` | `temporal/model_static.p`, `temporal/model_motion.p`, `temporal/thresholds.py` |

Function names are the Python names in camelCase (`palm_scale` → `palmScale`,
`rolling_shape_sigma_aligned` → `rollingShapeSigmaAligned`) so the two can be diffed by eye.
Thresholds keep their SCREAMING_CASE exactly, because they are data, not code, and they travel
inside `models.json`.

## Run it locally

The page needs `http://`, not `file://`: ES modules are blocked by CORS on the filesystem, and
browsers only expose a camera to a secure context, which means https or `localhost`.

```
cd web
python3 -m http.server 8000        # or: npx serve . , or any static server
open http://localhost:8000/
```

Nothing is installed and nothing is watched. Edit a `.js` file, reload the tab.

Two things are fetched from the network on first load and then cached by the browser:
MediaPipe's `tasks-vision@0.10.18` bundle and WASM from jsDelivr (~9 MB) and the hand landmarker
`.task` file from `storage.googleapis.com`. `models.json` (8 MB) is served from this directory.
A blocked CDN and a slow connection look the same for the first few seconds, so `index.html`
carries a 20 s watchdog that says so rather than sitting on "loading".

Before the camera is ever requested, `app.js` runs the 15 cases in `golden.json` through the
forest it just loaded and reports the result in the debug panel as `self-check`. A page that
loads but computes the wrong numbers is the failure this project exists to catch.

## Run the tests

Node 20. No dependencies; they read the JSON files in this directory directly.

```
node test_features.mjs      # 15 golden cases + 12 invariants
node test_forest.mjs        # 15 golden cases, 24 ambiguous cases, tree-walk edge cases
node test_segmenter.mjs     # the state machine at 15 / 30 / 60 fps
node test_app.mjs           # the glue, plus index.html's own consistency
```

All four pass. Measured divergence from Python on the golden cases: `shape42` and
`static_feature` to ~1e-5 (which is the sensitivity of the transform to `golden.json`'s own
6-decimal rounding of its input landmarks, not port error), `palm_scale` to ~5e-7, `j_gate` and
`z_gate` exactly, `static_probs` to 0.00e+0.

`golden.json` does not cover the motion branch, so the 79-D `eventFeatures` and the signals
feeding it were additionally checked against Python directly — 14 synthetic sequences (J hooks,
Z zigzags and a straight transport, at 15/30/60 fps with a jittered frame clock, including a
3-frame degenerate case) agree to **4.3e-13**. The whole `Segmenter` was then replayed against
the Python `Segmenter` driven by the real pickles: 9 streams, 584 frames, identical state,
buffer length, emissions, cut spans and vote diagnostics on every frame, with `v_bar` agreeing
to 3.6e-15 and both sigmas to 2.8e-16.

## Regenerating models.json

`models.json` is derived from the two pickles in `temporal/`. Regenerate it whenever either
model is retrained or `thresholds.py` changes — the thresholds travel in the same file, so the
page cannot pick up a new model with stale constants.

```
cd ..
./.venv/bin/python web/export_models.py        # -> web/models.json, web/models.json.gz, web/golden.json
./.venv/bin/python web/export_ambiguous.py     # -> web/golden_ambiguous.json
cd web && node test_forest.mjs                 # confirms the new export still matches sklearn
```

`export_models.py` flattens each tree to the four arrays a walk needs (`f`, `t`, `l`, `r`) plus
leaf class distributions, because a sklearn pickle is a Python object graph and cannot run in a
browser. The walk is sklearn's rule exactly: node `i` is internal when `f[i] >= 0`, and
`x[f[i]] <= t[i]` goes **left**.

Each branch carries a `feature` tag (`static/v3`, `event/v1`). `app.js` refuses to start if the
tag does not match the transform `features.js` builds, so a `models.json` exported from a
different feature definition fails loudly at load instead of quietly misclassifying.

`models.json.gz` is written alongside as a convenience for hosts that can serve it with
`Content-Encoding: gzip`; `forest.js` sniffs the gzip magic number and decompresses with the
platform's `DecompressionStream`, so either file works at `./models.json`. GitHub Pages
gzips `models.json` in transit on its own, so the plain file is what you want there.

## Accuracy, and the split each number came from

These are `temporal/README.md`'s numbers. Nothing about putting the model in a browser changes
them; the browser reproduces the Python's arithmetic, not its accuracy.

| | result | split |
| --- | --- | --- |
| Static letters, in-session | 0.968 | held-out tail of each capture burst |
| **Static letters, leave-one-session-out** | **0.680** | train one session, test the other |
| Motion letters {J, Z, MOVE} | 0.889 | GroupKFold over 118 independent gestures |
| Motion, at the runtime operating point | 96% correct when it fires, 22% abstain | same |
| False J/Z on held-out negatives | 0 | same |
| `J_GATE` on held `I` | 100/100, 0 false of 2,278 | committed archive |
| Segmenter over 157 s of held signs | 0 false triggers, 24/24 letters | committed archive |

**0.680 is the honest number.** The in-session figure is inflated: consecutive frames of one
held sign are near-duplicates, so they sit on both sides of any random split. The gap is not
noise — chasing the in-session number actively hurt, and adding absolute hand extent took it
from 0.956 to 0.983 while *halving* cross-session accuracy, 0.520 to 0.262.

Everything above is **one signer**, three sessions, 23,215 static frames and 700 motion events.
Nothing here says anything about a different person's hands, and a visitor to the page is
necessarily a different person. G, M, S and T drift most between sittings (toward H, E, E and N);
that is a data limitation, not a threshold to tune.

The page states the 0.680 above the fold, before the camera button, for this reason.

## Known issue: the export rounds split thresholds

`export_models.py` rounds each split threshold to 5 decimal places to keep `models.json` small.
Every one of the static forest's 55,039 splits is perturbed by up to 5e-6, and a sample that
lands inside that interval takes the other branch in that one tree.

Measured over 1,215 probe vectors (real golden features, blends of them, and noise at four
scales) against `sklearn.predict_proba`:

- the **full-precision** export reproduces sklearn at **0.00e+0** on all 1,215 — so the tree
  walk in `forest.js` is exact, and this is a Python-side artifact, not a port bug;
- the **shipped 5-dp** export differs on 7 of 1,215 probes, always by exactly 1/400 = **2.5e-3**,
  one tree of 400 flipping;
- 2 of those 7 changed the top-1 label, and in both cases sklearn's own top two classes were an
  exact tie (0.2650/0.2650 and 0.1725/0.1725) — at 0.17 and 0.27 probability, below
  `VOTE_PROB_FLOOR` of 0.30, so neither vote could have emitted a letter either way.

So the practical effect is nil, but two claims should be stated accurately: "reproduces sklearn
exactly" holds for the tree walk and for the 15 golden cases, not for the shipped export on
arbitrary input; and `golden.json`'s declared probability tolerance of 1e-4 is 25x tighter than
the export's own rounding can guarantee. If exactness matters more than 0.4 MB of gzip, drop the
`round_to` argument in `export_models.py:flatten` — the cost is 7.97 → 8.68 MB raw, 0.88 → 1.29
MB gzipped.

## What the debug panel is for

The likeliest failure in this system is a gate or a trigger that never fires: J simply becomes
unreachable while the page looks healthy, because it still recognizes the other 24 letters.
Nothing in the emitted string distinguishes "no J was signed" from "J cannot be signed". The
meters draw each threshold as a tick, so a `v_bar` sitting just under `V_MOVE_ARMED` and never
crossing it is visible rather than merely suspected. That is also how the original 5-sample
smoothing bug was found — it worked at 15 fps and never fired at 30.

Thresholds are the archive defaults: percentiles measured on one camera at about 15 fps. They
are in seconds and palm-widths rather than frames and pixels so they transfer between cameras,
but they were not measured on yours. `temporal/calibrate.py --live` refits them.

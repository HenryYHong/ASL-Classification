# Real-time American Sign Language Recognition

![Signing A B C H E N R Y J Z into the hosted page, each letter appearing as it is recognized](docs/demo.gif)

*One take on the [hosted page](https://henryyhong.com/ASL-Classification/). I trimmed and cropped the recording but did not speed it up, so the pace here is the pace it runs at. J and Z come last because they are the two letters that cannot be read from a single frame.*


This repository holds three attempts at the same problem, kept in the order I built them, because each one grew out of a limitation in the one before it.

The first trains a CNN on 28x28 pixel images. It scores between 95% and 99% on its benchmark and still fails in front of a webcam. The second ignores the pixels and classifies the 21 hand landmarks that MediaPipe reports instead. It works live, but it covers only 24 letters, and its 99.58% comes from a split that puts nearly identical frames on both sides. The third adds J and Z, reports its accuracy from a split that holds out an entire recording session, and runs in a browser.

### Try it

The third approach is hosted at **[henryyhong.com/ASL-Classification](https://henryyhong.com/ASL-Classification/)**. There is nothing to install, and the camera frames stay on your machine: the page downloads the models — `models.bin.gz` and a small `models.meta.json`, **1,184,427 B on the wire**, about 1.18 MB — and runs them locally. Chrome and Safari both work. Allow the camera, then sign into the box. An alphabet chart sits under the camera so you can look up a letter without leaving the page; the hands in it are drawn from the landmarks the model was trained on rather than from photographs, so they show the shape the classifier is actually looking for, and each carries a written description because a flat drawing cannot separate the six fists. The forest grew 22.8% this release and the download shrank anyway, because the three forests now ship packed instead of as JSON. JSON is the form the page fetched until this release, and the same trees in it are 20.4 MB raw and 3,703,083 B gzipped, which is how GitHub Pages would serve them. `docs/models.json` stays committed as the readable reference; `docs/README.md` has the format and the parity measurements.

<img width="525" alt="ASL" src="https://github.com/user-attachments/assets/7a6fde28-68aa-4d7b-92a0-115bef967a6f" />
<img width="604" alt="ASL3" src="https://github.com/user-attachments/assets/c56a280c-6667-4226-b86d-fa9c79073faa" />

*Both screenshots are mine — frames from Approach B's live demo (`RandomForest/Mediapipe.ipynb`, cell 5, which draws the bare predicted letter above the landmark bounding box), captured on my own machine. Neither is from Approach A, whose live demo is the one documented as failing below.*

## Scope

**Approaches A and B cover 24 letters:** A B C D E F G H I K L M N O P Q R S T U V W X Y. **Approach C covers all 26 letters, plus an experimental numbers mode (0-9).**

J and Z are motion signs. J traces a hook and Z traces a zigzag, so neither can be read from a single still frame, and a single frame of a J is simply an I. A model that sees one frame at a time cannot represent them at all, which is why the first two approaches stop at 24 letters, and why the third one needed a different design rather than a larger forest.

The numbers mode is a separate ten-class forest behind a toggle on the hosted page. It is trained on 218 signers from a public dataset, not on the author, and it has not been verified on the author's hand or on live video; it is labeled experimental for that reason, and the page opens in letters mode on every load.

All three are fingerspelling letter classifiers, not sign language translators: one letter at a time, no grammar, no facial markers, no two-handed signs. Approach C's page does group letters into words and can name the word the letters most resemble, but that is a presentation layer on top of the letter classifier, and it never changes a letter the model emitted.

The first two pipelines reach the same 24 letters through different index maps. The CNN inherits Sign-MNIST's `label` column, whose values run 0-24 with 9 (J) never present and 25 (Z) never assigned at all; `LabelBinarizer` collapses that onto 24 output units. The Random Forest uses a contiguous 0-23 map built at capture time: `0:A … 8:I, 9:K, 10:L … 23:Y`.

## The three pipelines at a glance

| | **A — CNN on Sign-MNIST** | **B — landmarks + Random Forest** | **C — `temporal/`, two branches** |
| --- | --- | --- | --- |
| Code | `CNN/ASL_Detection.ipynb` | `RandomForest/Mediapipe.ipynb` | `temporal/*.py`, ported to `docs/*.js` |
| Letters | 24 | 24 | **26**, plus 0-9 as an experimental mode |
| Input representation | 28x28 grayscale pixels (784 values) | 42 floats: 21 landmarks x (x, y) | 112 floats per frame (42-value palm-normalized shape, 55 pairwise distances, four finger-straightness ratios, an 11-value thumb block), and a 79-D arc-length path descriptor for J and Z |
| Training data | Sign Language MNIST (public), 27,455 rows | 2,377 usable landmark vectors from 2,400 frames I recorded myself | 3,807 frames of mine over four sessions (capped from 4,878) plus 8,561 frames of other people's hands from three public landmark sets, which become 61,840 rows after jitter; 117 motion events cut from 92 prompted items, all mine. Digits: landmarks from 1,805 photos of 218 strangers (public) |
| Model | 3-block CNN, 264,049 params | `RandomForestClassifier()`, 100 trees, 9,452 nodes | Two forests (60 and 300 trees), a six-inequality launch gate, a four-state segmenter; a third 100-tree forest for the digits |
| Training cost | 20 epochs, ~17-20 s each on an Apple M1 Pro | Seconds | About a minute |
| Headline accuracy | 95.29% saved / 99.69% best epoch | 99.58% random split, 100.00% per-class split (V contributes zero test samples — see caveats) | **0.9406 leave-one-SIGNER-out** over ten other people and **0.8292 on a second set of strangers held out entirely**, which are the numbers to plan around; 0.9233 leave-one-session-out on my own hand (95% CI 0.877-0.965 over 57 bursts) and 0.8897 across days; previous release 0.9103 / 0.8702 on my folds and no by-signer number at all. J/Z: 0.951 by prompted item |
| Works in the live demo | No — collapses onto a few classes | Yes, for the session it trained on | Yes — in a browser, on someone else's machine, and measured on someone else's hands |

The two also differ in provenance, not just representation. Approach A trains on Sign Language MNIST, a public benchmark someone else assembled. Approach B trains on a dataset I built end to end: I wrote the capture cell, recorded all 2,400 frames myself on my own webcam, and labeled them by construction — one folder per letter, 100 frames each. Nothing in `RandomForest/data/` was downloaded.

Both approaches run MediaPipe Hands at inference time, but for different jobs. Approach A uses it only to *locate* the hand, then classifies the pixels inside the box. Approach B classifies the geometry MediaPipe already computed and never looks at a pixel. That difference is the whole repository: raw pixels carry lighting, skin tone, sleeve color and background into the classifier; 42 coordinates do not.

Approach A is invariant to none of those; its crop does remove where the hand sits in the frame, but that crop is then squashed to 28x28 without preserving aspect ratio. Approach B's *features* discard all of them, and hand position with them — though MediaPipe still has to find the hand in the image first, and it found none in 23 of the 2,400 captured frames. Those features are also **not** invariant to distance from the camera, because they are translation-normalized and not scale-normalized. That is its own worst weakness, and it is covered below.

## Repository layout

```
ASL/
├── ASL.code-workspace           VS Code workspace file
├── README.md
├── requirements.txt             the pinned environment for temporal/ and docs/export_models.py
├── CNN/
│   ├── ASL_Detection.ipynb      8 cells: install, train, plots, live demo
│   ├── sign_mnist_train.csv     27,455 rows + header, 785 cols (label + 784 px), 83.3 MB
│   ├── sign_mnist_test.csv      7,172 rows + header, same schema, 21.8 MB
│   ├── smnist.h5                3.2 MB — saved by the training cell, loaded by the live demo
│   ├── smnist.keras             3.2 MB — same architecture, DIFFERENT weights (see below)
│   └── asl_model.keras          1.65 MB — leftover from an earlier experiment (see below)
├── RandomForest/
│   ├── Mediapipe.ipynb          7 cells: install, capture, extract, train x2, live demo
│   ├── data/0 … data/23         24 folders x 100 JPGs = 2,400 frames, each 1920x1080
│   ├── data.pickle              912 KB — {'data': [...42 floats...], 'labels': [...]}
│   └── model.p                  2.4 MB — pickled {'model': RandomForestClassifier}
├── temporal/                    Approach C. See temporal/README.md and temporal/SOURCES.md.
│   ├── features.py              the one shared transform: both branches, both languages
│   ├── thresholds.py            every runtime constant, each with the measurement behind it
│   ├── segmenter.py             four-state machine deciding when a letter was actually signed
│   ├── static_aug.py            landmark jitter for training frames (and the retired geometry filter)
│   ├── strangers.py             other people's hands, loaded into the same canonical frame
│   ├── collect_motion.py        recorder for both static letters and J/Z gestures
│   ├── train_static.py          ships model_static.p; characterizes, never selects
│   ├── crossval_static.py       leave-one-session-out — my own folds
│   ├── crossval_signers.py      leave-one-SIGNER-out over ten known signers — the by-signer number
│   ├── crossval_strangers.py    each stranger set held out in turn, plus the never-train holdout
│   ├── replay_static.py, idle_gate.py   my 71 held signs and the 43 idle holds through the real segmenter
│   ├── replay_strangers.py      other people's VIDEO through the real segmenter, end to end
│   ├── label_events.py          cuts the recorded takes into motion events with the runtime's own segmenter
│   ├── train_motion.py          ships model_motion.p; GroupKFold by prompted item
│   ├── evaluate.py              the motion numbers per session, each row labeled in-sample or held-out from the pickle's record
│   ├── train_digits.py          ships model_digits.p; leave-signer-out on the public digits set
│   ├── ingest_images.py         photos -> landmarks; how digits_ankara.npz was made
│   ├── ingest_aslnow.py         downloads and orients the ASLNow records; how aslnow.npz was made
│   ├── ingest_aslhg.py          ten named signers -> aslhg.npz; the only source with real signer ids
│   ├── ingest_ayuraj.py         the permanent never-train holdout -> ayuraj.npz
│   ├── simulate_words.py, make_oof.py   the word layer measured offline on held-out posteriors
│   ├── calibrate.py, live_demo.py       thresholds from your camera; the desktop demo
│   ├── static_sequences.npz, static_s2-4.npz   the four letter sessions, as landmarks
│   ├── static_sequences_tasks.npz       the archive re-extracted with the browser's landmarker
│   ├── motion_clips.npz, events.npz     the five prompted takes and the events cut from them
│   ├── digits_ankara.npz        landmarks of 1,805 public digit photos (no images)
│   ├── aslnow.npz               2,122 records of other people's letters, browser landmarker (MIT)
│   ├── aslhg.npz                23,984 letter frames, 10 named signers, 24 letters (CC BY 4.0)
│   ├── ayuraj.npz               1,819 frames, 5 signers (CC0) — NOTHING MAY EVER TRAIN ON IT
│   ├── openhands_replay.json    266 clips of strangers' video replayed through the segmenter
│   ├── idle_holds.npz           the 43 relaxed-hand holds the vote floor is GATED against
│   ├── idle_strangers.npz       530 frames of other people's hands not making a letter — the REST class
│   ├── rest_probe.py            a 25th "not a letter" class: it fixes the gate, and why it still loses
│   ├── crossing_probe.py        why U and R fail, and why the obvious fix does not ship
│   ├── model_static.p, model_motion.p, model_digits.p
│   └── tests/                   eight plain-script test files, 93 checks
└── docs/                        the hosted demo, served by GitHub Pages. See docs/README.md.
    ├── index.html, app.js       camera, overlay, live readout, the letters/numbers toggle
    ├── features.js, forest.js, segmenter.js   line-for-line ports of the Python
    ├── words.js, words.txt, build_words.py    the word layer and its 34,702-entry frequency list
    ├── export_models.py         flattens the three forests to models.json and models.bin, writes golden.json
    ├── models.bin.gz            the same three forests as packed binary — what the page fetches, 1.25 MB on the wire
    ├── models.json              the same forests as readable JSON, 21.9 MB; the reference copy, no longer fetched
    ├── reference.json           one real hand per letter, for the alphabet chart beside the camera
    ├── reference.js, make_reference.py   the chart: medoid landmarks drawn as solid hands
    ├── models.meta.json         class lists, feature tags, thresholds and the binary's section offsets
    ├── parity_probes.py, parity.json   1,215 probes per forest against sklearn, stamped with the build id
    ├── golden.json              Python's answers for 41 real frames; the page checks itself on load
    ├── test_*.mjs               six Node test files
    ├── devserver.py             a local server that also collects the page's diagnostics
    └── make_demo.sh             screen recording -> the GIF at the top of this file
```

`CNN/asl_model.keras` is, like `smnist.keras`, **not** produced by any cell in the notebook. It is a different, earlier architecture — Conv 32/64/128 with `padding='valid'`, Dense 256, Dropout 0.5, and a 25-unit softmax rather than 24 — left over from a first attempt. Nothing in the repository loads it. Use `smnist.h5` — the file the training cell writes and the live demo loads. `smnist.keras` shares the architecture but not the weights: it is a separate, earlier training run, saved 35 minutes before `smnist.h5`, and it is the better model of the two — 98.87% on the test CSV against `smnist.h5`'s 95.29%. Nothing loads it, and none of the accuracy figures below describe it.

Both notebooks use paths relative to their own folder (`sign_mnist_train.csv`, `./data`, `model.p`), so each must be run with that folder as the working directory.

---

## Approach A — CNN on Sign Language MNIST

`CNN/ASL_Detection.ipynb`

```
webcam frame (1080p BGR)
  -> MediaPipe Hands -> 21 landmarks -> bounding box, expanded 20 px per side, clamped to the frame
  -> crop -> BGR2GRAY -> resize to 28x28 -> /255 -> reshape (1, 28, 28, 1)
  -> CNN -> softmax over 24 classes -> top-3 letters printed, top-1 drawn on the frame
```

### Data and preprocessing

`sign_mnist_train.csv` (27,455 rows) and `sign_mnist_test.csv` (7,172 rows) each hold a `label` column plus 784 pixel columns — one 28x28 grayscale image per row. Pixels are divided by 255.0 and reshaped to `(-1, 28, 28, 1)`; labels are one-hot encoded with `LabelBinarizer`. The official 7,172-row test CSV is passed straight to `fit()` as `validation_data`, so every "validation accuracy" below is the benchmark test accuracy.

Augmentation is a Keras `ImageDataGenerator` with `rotation_range=10`, `zoom_range=0.1`, `width_shift_range=0.1`, `height_shift_range=0.1` — small perturbations only, nothing that would change which letter a hand shape is. Training runs on `datagen.flow(...)`, so every epoch sees perturbed copies rather than the raw rows.

### Architecture

Sequential. The per-layer counts below are derived from the layer definitions; they sum to the 264,049 parameters read back from the saved weight file, of which 263,749 are trainable and 300 are BatchNorm moving mean/variance.

| # | Layer | Output shape | Params |
| --- | --- | --- | --- |
| 1 | `Conv2D` 75, 3x3, stride 1, `same`, ReLU | 28x28x75 | 750 |
| 2 | `BatchNormalization` | 28x28x75 | 300 |
| 3 | `MaxPool2D` 2x2, stride 2, `same` | 14x14x75 | — |
| 4 | `Conv2D` 50, 3x3, stride 1, `same`, ReLU | 14x14x50 | 33,800 |
| 5 | `Dropout` 0.2 | 14x14x50 | — |
| 6 | `BatchNormalization` | 14x14x50 | 200 |
| 7 | `MaxPool2D` 2x2, stride 2, `same` | 7x7x50 | — |
| 8 | `Conv2D` 25, 3x3, stride 1, `same`, ReLU | 7x7x25 | 11,275 |
| 9 | `BatchNormalization` | 7x7x25 | 100 |
| 10 | `MaxPool2D` 2x2, stride 2, `same` | 4x4x25 | — |
| 11 | `Flatten` | 400 | — |
| 12 | `Dense` 512, ReLU | 512 | 205,312 |
| 13 | `Dropout` 0.3 | 512 | — |
| 14 | `Dense` 24, softmax | 24 | 12,312 |

Filter counts descend (75 -> 50 -> 25) as the map shrinks 28 -> 14 -> 7 -> 4, so the convolutional stack is cheap and 205,312 of the 264,049 parameters — 78% — sit in the single `Flatten` -> `Dense(512)` transition.

### Training configuration

`adam` with Keras defaults, `categorical_crossentropy`, batch size 128, 20 epochs, 215 steps per epoch, roughly 17-20 s per epoch on an Apple M1 Pro — about six minutes end to end. There are **no callbacks**: no early stopping, no checkpointing. The run finishes with `model.save('smnist.h5')`.

### Results

Selected epochs, from the notebook's own captured output:

| Epoch | Train acc | Train loss | Val acc | Val loss |
| --- | --- | --- | --- | --- |
| 1 | 0.4798 | 1.7338 | 0.1464 | 3.8237 |
| 2 | 0.9153 | 0.2590 | 0.5066 | 1.6500 |
| 3 | 0.9649 | 0.1125 | 0.9795 | 0.0819 |
| 6 | 0.9893 | 0.0358 | 0.9378 | 0.1898 |
| 11 | 0.9912 | 0.0254 | 0.9204 | 0.2401 |
| **12** | 0.9927 | 0.0220 | **0.9969** | **0.0092** |
| 18 | 0.9955 | 0.0159 | 0.9905 | 0.0365 |
| **20** | **0.9961** | **0.0127** | **0.9529** | **0.1594** |

**The validation curve is noisy.** It opens at 14.64%, jumps to 50.66%, reaches 97.95% by epoch 3, and then swings between 0.9204 and 0.9969 for the rest of the run while training accuracy first passes 0.99 at epoch 7 and never falls below 0.9899 thereafter. After the opening ramp, the largest single move between neighboring epochs is 0.9204 (epoch 11) to 0.9969 (epoch 12) — 7.65 points of held-out accuracy on a training set that has effectively converged. Any single-run figure from this notebook carries an error bar much wider than its last two digits suggest.

**The saved model is not the best model.** With no `ModelCheckpoint` and no `EarlyStopping`, `model.save()` writes whatever weights epoch 20 ended on: 95.29% validation accuracy, val loss 0.1594. The best epoch was 12, at 99.69% and val loss 0.0092. `smnist.h5` — the file the live demo loads — is the epoch-20 model. **The committed model is the 95.29% model, not the 99.69% one.**

### Why it does not work live

From the notebook's own captured run, three consecutive predictions:

```
Predicted Character 1: F, Confidence 1: 99.32%
Predicted Character 2: P, Confidence 2: 0.68%
Predicted Character 3: G, Confidence 3: 0.00%
Predicted Character 1: F, Confidence 1: 99.40%
Predicted Character 2: P, Confidence 2: 0.58%
Predicted Character 3: Y, Confidence 3: 0.02%
Predicted Character 1: F, Confidence 1: 49.12%
Predicted Character 2: P, Confidence 2: 48.09%
Predicted Character 3: G, Confidence 3: 1.24%
```

The full log is 256 predictions spread over just nine letters: `Y` 101 times, `P` 81, `F` 32, `L` 24, and five other letters between one and eight times each. It opens on `Y` at 99.52%, then alternates between `Y` and `P` for most of its length — including unbroken runs of 18 `Y` and 19 `P` — before settling onto a closing run of seven `F`, the last three of which are quoted above. Top-1 barely tracks the sign being made, and most of the time the model is not uncertain about it. It is confident and wrong.

This is a domain gap, not a bug. Sign-MNIST images are tight, centered, uniformly lit 28x28 crops from one controlled pipeline. The live crop is an arbitrary-aspect-ratio rectangle carved out of a cluttered 1080p scene and squashed to 28x28 by `cv2.resize` with no aspect-ratio preservation, carrying whatever the room's lighting and background happened to be. A tall hand gets stretched; a wide one gets squeezed. Nothing in the training distribution looks like that.

A softmax has no way to say so. Trained only on the Sign-MNIST distribution, it cannot report "this input is unlike anything I have seen" — it puts 99% on whichever class the out-of-distribution activations happen to favor. That is why the failure looks like confidence rather than confusion. **95% on a benchmark split says nothing about a webcam**, and the fix is representational, not architectural.

---

## Approach B — MediaPipe landmarks + Random Forest

`RandomForest/Mediapipe.ipynb`

```
webcam frame (1080p BGR)
  -> MediaPipe Hands -> 21 landmarks (x, y) in normalized image coordinates
  -> for each landmark: (x - min_x, y - min_y) -> 42 floats
  -> RandomForestClassifier -> class index -> letter drawn above the hand's bounding box
```

The premise: stop asking one model to learn perception and geometry at once. Let MediaPipe do the perception — hand detection plus 21 keypoints, a job it already does well on far more varied imagery than I could collect — and give the classifier only the geometry. The nuisance variables that broke the CNN are not trained away here; they are never present in the input.

### Step 1 — Capture (cell 1)

`cv2.VideoCapture(1)`; for each of the 24 classes, print the letter, `time.sleep(5)` as a "get ready", then write 100 consecutive frames to `./data/<class_index>/<n>.jpg`. Frames are full-resolution 1920x1080 JPEGs, written straight from the BGR capture buffer by `cv2.imwrite`. 24 x 100 = 2,400 images, every one of them recorded by me, of my own hand, across two sittings on the same day and in two different rooms (22 classes in one continuous pass, then A and B re-captured a few hours later somewhere else) — this is my own dataset, not a downloaded one. The only pause between frames is the loop's own `cv2.waitKey(25)`, so all 100 land within a few seconds, which matters for evaluation and is dealt with below.

### Step 2 — Landmark extraction (cell 2)

For every image: `imread` -> `BGR2RGB` -> `mp_hands.Hands().process()`. If a hand is found, its 21 landmarks become a 42-D vector:

```python
for i in range(len(hand_landmarks.landmark)):
    x = hand_landmarks.landmark[i].x
    y = hand_landmarks.landmark[i].y
    data_aux.append(x - min(x_))   # x_ = all 21 x values
    data_aux.append(y - min(y_))   # y_ = all 21 y values
```

MediaPipe returns `x` and `y` already normalized to [0, 1] against image width and height, so raw landmarks encode where in the room the hand is. Subtracting `min(x_)` and `min(y_)` moves the origin to the top-left corner of the hand's own bounding box, so the vector describes the hand's internal shape instead of its position in the frame.

**What that buys:** translation invariance. Sign the same letter top-left or bottom-right and you get the same 42 numbers. Lighting, skin tone, sleeve color and background are discarded outright, because none of them survive into a coordinate.

**What it does not buy:** scale invariance. The values are still fractions of the *frame*, not of the hand, so a hand held twice as close produces roughly twice the magnitudes. Dividing by the bounding-box extent would fix it. The notebook does not. This is the single largest weakness of the feature.

Result, saved to `data.pickle` as `{'data': [...], 'labels': [...]}`: **2,377 usable vectors out of 2,400 frames (99.0%)**, all exactly 42-dimensional, values ranging 0.0 to 0.6518, with 100 samples for every class **except** class 0 (A) at 99 and class 20 (V) at 78.

Those 23 missing vectors are frames where MediaPipe returned no hand at all. The perception step that makes this approach work is also its single point of failure: no landmarks, no prediction — at extraction time and at inference time alike.

One detail worth knowing before you recapture: the extraction loop iterates `for hand_landmarks in result.multi_hand_landmarks:` and appends into a single `data_aux`, so a frame containing **two** hands yields an 84-D vector rather than 42. That is why both training cells open by filtering out any entry whose length differs from `data_dict['data'][0]`. Samples with a second hand in shot are silently dropped rather than crashing the run.

### Step 3 — Training (cells 3 and 4)

`RandomForestClassifier()` with every scikit-learn default. Read back from the committed `model.p`:

| Property | Value |
| --- | --- |
| `n_estimators` | 100 |
| `criterion` | gini |
| `max_features` | `'sqrt'` — 6 of the 42 features considered per split |
| `max_depth` | `None` — grown until pure |
| Mean tree depth | 12.67 |
| Total nodes across the forest | 9,452 (roughly 95 per tree) |
| `n_features_in_` / classes | 42 / 24 |

Two cells train and evaluate under two different splits, and **both write `model.p`**. The committed pickle is cell 4's: every tree in it was fitted on 1,918 samples, exactly cell 4's training count (23 x 80 + V's 78), against cell 3's 1,901. The model the live demo loads is therefore the one evaluated without a single held-out V.

| Cell | Split | Reported |
| --- | --- | --- |
| 3 | `train_test_split(test_size=0.2, shuffle=True, stratify=labels)` | **99.58%** on 476 test samples (~2 errors) |
| 4 | Per class: shuffle, then `[:80]` train / `[80:100]` test | **100.00%** |

### Step 4 — Live inference (cell 5)

Webcam -> `BGR2RGB` -> MediaPipe Hands -> the *same* 42-feature construction as cell 2 -> `model.predict` -> the letter drawn above the landmark bounding box with the skeleton overlaid. `q` quits.

Feature parity between training and inference is the point: cell 2 and cell 5 apply the same `(x - min(x_), y - min(y_))` arithmetic (cell 5 resets the vector per hand, so it never produces the 84-D two-hand case), so there is none of the train/serve skew that sinks Approach A. This is the demo that actually tracks the sign being made.

---

## What those accuracies actually mean

99.58% and 100.00% are honest measurements of what they measured. They are not evidence that this generalizes, and three things should be stated before anyone quotes them.

**1. The headline number is stable — that is not the issue.** Re-running cell 3's random stratified split under five seeds gives 99.58%, 99.79%, 99.79%, 100.00%, 100.00%. Variance is low; it is not a lucky split. The *split* is the problem, not the noise.

**2. The 100.00% was measured on 23 of the 24 letters.** Cell 4 slices `[80:100]` per class for its test set. Class 20 (V) has only 78 vectors, so that slice is empty and **V contributes zero test samples** — a letter with no test samples cannot be got wrong. (Class 0, at 99 vectors, contributes 19 instead of 20.) The class dropped from the evaluation is precisely the one MediaPipe found hardest to see, which is the last class you would want to stop measuring.

**3. Near-duplicate frames land on both sides of the split.** All 100 frames of a class come from one continuous burst — same hand, same session, same lighting, same background, captured over a few seconds with only the loop's 25 ms `waitKey` between frames. Consecutive frames are near-identical images. A random split therefore puts almost the same frame in train and in test. What the number measures is *"can the forest re-identify frames from the session it was trained on"*, not *"does it recognize a letter signed by a different person, with a different hand size, in a different room."*

A contiguous per-class holdout — train on the first 80% of each class's frames, test on the last 20% — still scores 99.58%. That does not rescue the figure, because that split is drawn from the same burst too. An honest generalization number needs a second capture session, ideally with a different signer, and I have not run one.

So the defensible claim is narrow: the 42-D landmark representation separates these 24 handshapes well enough that an untuned forest solves them, and the live demo runs on a live webcam feed and tracks the letters signed by the signer it was trained on (no frame rate or latency was measured, and the demo still rebuilds the MediaPipe graph every frame). Nothing here supports a claim about a new person or a different hand. A change of room it does appear to survive: the two screenshots at the top were taken in a library, in different clothing and lighting from either capture session, and the demo still reads `A` and `B` correctly — which is what you would expect from a representation that never sees a pixel, but two letters of anecdote is not a measurement. **A benchmark number describes a distribution, not a capability** — which is the same lesson Approach A taught, arriving from the other direction.

---

## Approach C — motion letters, measured across sessions and across people

`temporal/` addresses both of the problems described above. It covers all 26 letters, it measures itself by holding out an entire recording session rather than 20% of a single burst, and it trains on and measures against other people's hands — this release with **real signer ids**, so it can hold out a whole person rather than a whole dataset. The full write-up is in [`temporal/README.md`](temporal/README.md).

**Other people's hands, which is what a visitor is.** Each row holds out something the forest never saw.

| | result | how it was split |
| --- | --- | --- |
| **Static letters, leave-one-SIGNER-out** | **0.9406** ± 0.0033 over 3 seeds; at seed 0, 0.9453 over all 23,984 held-out frames, per signer 0.880 to 1.000 | `temporal/crossval_signers.py`: hold out all 2,400 frames of one of ASL-HG's ten signers, train on the other nine plus my sessions and the other stranger sets. The first by-signer number the letters have had |
| **Static letters, a whole set held out** | **0.8292** ± 0.0061 over 3 seeds; 0.836 at seed 0, on 1,874 ASLNow records of 24 letters | `temporal/crossval_strangers.py`: train on my sessions, the Ankara photos and ASL-HG; test on a set the forest never saw, through the landmarker the page itself runs, one frame per record |
| **Static letters, the permanent never-train holdout** | **0.889** on the shipped forest; the previous release's forest reads **0.797** on the same 1,111 frames, and the paired difference is **+0.0918, CI [+0.069, +0.134]**, all five signers improving | the CC0 `ayuraj` set, five signers. Nothing in this repository may ever train on it, so the two figures are the same measurement. **All of that difference is the new dataset, none of it the recipe** — see below. `crossval_strangers.py --compare-prev <old pickle>` prints the pair; the interval resamples the five SIGNERS, not the 1,111 frames, which would claim ±0.019 it has not earned |
| **A stranger holding a letter through the whole runtime** | 266 video clips: exactly right on 94, silent on 172, **0 wrong letters**. On the 139 clips tracked at least 60% of frames: 58 right, 0 wrong. On the 58 of those lasting at least a second: 33 right, 0 wrong | `temporal/replay_strangers.py` over the OpenHands fingerspelling clips, one fresh segmenter and one fresh MediaPipe per clip, shipped models and thresholds. The clips carry no signer ids, so this is not a by-signer number |
| — 218 signers, the four digits that are letters | 0.9413 ± 0.0048 over 3 seeds; 0.940 at seed 0 (F 1.00, W 0.96, O 0.91, V 0.88) | train on my sessions, ASLNow and ASL-HG; test on the Ankara photos. The one-signer forest scored 0.716 here, V 0.19 |
| — ASLNow, a new capture of possibly the same people | 0.936 at seed 0 | five folds of ASLNow with everything else in training; the set carries no participant id, so this is an upper bound |
| *(historical)* previous forest on ASL-HG | 0.8434 (20,229 of 23,984) | that forest had never seen ASL-HG. **This release trains on it**, so this is history rather than a held-out figure, and the by-signer row above replaces it |

**My own hand, holding out a whole session.** These are not what a visitor gets.

| | result | how it was split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.9233** ± 0.0018 over 3 seeds; at seed 0, 0.926 (4,515 of 4,878 held-out frames; 95% CI [0.88, 0.97] over the 57 session-by-letter bursts) | `temporal/crossval_static.py`: train on three of my sessions plus the strangers, test on my fourth, every held-out frame counted once |
| — the cross-day fold, all 24 letters | **0.8897** ± 0.0041 over 3 seeds; 0.895 at seed 0 (macro per-letter recall 0.893) | hold out the 2,378-frame November 2024 archive, train on the three September 2026 sessions and the strangers |
| — the same recipe on my sessions alone | 0.873 / 0.782 at seed 0 | `crossval_static.py --henry-only`: what the strangers add is the difference |
| — the same recipe without ASL-HG | 0.913 / 0.873 at seed 0 | `crossval_static.py --no-aslhg`: what ASL-HG adds is the difference |
| — previous release, same folds | 0.910 / 0.870 over 3 seeds, 199,539 nodes | its own recipe, re-measured here rather than quoted. `crossval_static.py --legacy` still reproduces the much older 0.759 / 0.659 pair |

**The runtime, not the classifier.**

| | result | how it was split |
| --- | --- | --- |
| Motion letters {J, Z, MOVE} | 0.951; J+Z recall 0.946 at the operating point; 2 false J in 28 MOVE events | `GroupKFold(5)` by prompted item: 102 events (74 J/Z gestures and 28 movements) in 80 items. The label set changed in an earlier release, so this replaces the older 0.864 rather than improving on it |
| Motion letters, other people's video | J came out on 3 of the 3 clips that produced a span the runtime could score; Z on 0 of 4. No false J or Z in 160.1 s of the same signers' other fingerspelling | four sources, **four clips per letter** — a direction, not a percentage. And it counts **the motion branch only**: replayed end to end those eight clips also emit **eight static letters that were not the target** (the readings are `HJ`, `W`, `IJL`, `YX`, `IJ`, `Y`). A parked `I` before a J is the launch pose read correctly; `H`, `W`, `Y` and `X` are not. "No wrong letters" is false of the transcript a viewer would see |
| Launch gate on held `I` | mine 100/100, 13 of 2,278 other frames, all Y. Strangers: 65 of 68, and one false positive in 1,806, also a Y | committed archive and the ASLNow J/Z stills |
| Launch gate on held `D` | mine 100/100. **Strangers: 35 of 72** — about half of every stranger's Z is unreachable before any model runs | same. The binding clause is the curled-finger ceiling, not the straightness test; `temporal/tests/test_gates_cross_signer.py` decomposes it |
| Segmenter over 157 s of held signs | every one of the 24 letters emitted exactly once, 0 track starts | committed archive, in-sample for the static forest |
| Segmenter on my 71 held-out holds | exactly the right letter 66 of 71, one wrong letter in 71 holds, four silent; 22 of 24 on the cross-day fold | `temporal/replay_static.py`: fold models, real timestamps, one fresh segmenter per hold |
| A relaxed hand, 43 idle holds from a live browser log | 0 emissions and 0 of 2,064 votes at **every one of 16 jitter seeds**, tightest letter G with +0.0361 of room. The 24-class recipe this replaces emitted at 4 of 8 | `temporal/idle_gate.py --seeds 16`: one signer, one stretch of about two minutes. The REST class and the per-letter floors are below, under the vote floor |

**0.9406 and 0.8292 are the numbers to plan around for somebody else's hand.** They are far apart because they are different conditions, not because one is wrong: ASL-HG is ten volunteers photographed to one protocol in one place, and ASLNow is an unknown number of people signing into their own webcams. The never-train holdout sits between them at 0.889. Quote all three, or quote the lowest.

My own 0.9233 and 0.8897 are the older story, and the spread between folds is still more informative than the average. Two of the four sessions are short re-recordings covering only six and four letters, and the fold that tests four well-separated letters scores 1.000; averaging the folds evenly would let that one carry the result, which is the same problem I describe in Approach B's 100.00% above, so the pooled figure counts every held-out frame once. The three later sessions were all recorded on the same evening, 2026-09-10, about two and a half hours apart (S2 and S3 first committed at 18:40 in `ce4c26c`, S4 at 21:14 in `8be6a85`), so the November 2024 archive is the only fold that tests a different day and the only fold that tests all 24 letters. On it one letter falls below 0.6 recall: M at 0.12. The archive's M was recorded with the thumb where the textbook handshape does not put it, so that one is a data ceiling and re-recording it is the only fix. E, N, S and O, which two releases ago could not be read across days (0.12, 0.20, 0.54, 0.55), are read now, and that is the strangers' doing.

One number that must never be quoted: the shipped forest reads 0.9997 of ASL-HG's 23,984 frames. It trains on 6,000 of them and the rest are drawn from the same hundred-frame bursts, so that is re-identification. The by-signer row is what ASL-HG actually measures.

### Other people's hands

Until the previous release every letter frame was mine, and every visitor to the hosted page is someone else. Four public sets now change that, all loaded by `temporal/strangers.py` into the same canonical frame as my sessions, and all documented in [`temporal/SOURCES.md`](temporal/SOURCES.md):

- **ASL-HG** (`temporal/aslhg.npz`, Mendeley DOI 10.17632/j4y5w2c8w9.1, CC BY 4.0), new this release and the important one: 36,000 smartphone photographs of ten named volunteers in Mirpur, Dhaka, 100 per class per person. This project reads the 24 static-letter folders — 24,000 images, 23,984 of which MediaPipe found a hand in, 99.93%, the cleanest of any external set here. **Every file name carries its signer** (`P<k>_<CLASS>_<n>.jpg`), which is the first time anything in this repository could hold out a person. It ships on both sides: 25 frames per (signer, letter) = 6,000 in training, and `temporal/crossval_signers.py` holding out one signer at a time for the 0.941.
- **ASLNow** (`temporal/aslnow.npz`, MIT): 2,122 records from a fingerspelling web app, "multiple participants told to sign ASL letters into a camera", each one the 21 landmarks from MediaPipe's Web Hand Landmarker, which is the landmarker my page runs. 1,874 are static letters; the 248 stills labeled J and Z are not (a single frame of a J is an I) and are never trained on. The set carries no frame size, no handedness label and no participant id; `temporal/ingest_aslnow.py` infers the first two and documents both decisions, and the third cannot be inferred — hand proportions do not cluster into people, because the pose dominates every bone-length ratio.
- **The digit photos** (`temporal/digits_ankara.npz`, the same file numbers mode trains on; Apache-2.0): four ASL digits are letters, exactly. 0 is O, 2 is V, 6 is W and 9 is F, so 687 photographs of 218 students' hands are letter frames from 218 more people. (4 is a spread-fingered B and is left out; 1 is close to D but tucks the thumb.)
- **ayuraj** (`temporal/ayuraj.npz`, Kaggle, CC0), new this release and **never trained on, by construction**: 1,111 letter frames from five signers. `strangers.NEVER_TRAIN` holds the path, `strangers.training_source` raises on it, and a test builds the real training set and asserts not one of its frames is in it. Every other set here crossed onto the training side once it proved useful, and each crossing turned a "never seen" figure into an in-sample one that had to be relabeled historical. This is the one set whose number cannot drift that way, which is why it is the cleanest before-and-after in the table above: 0.797 on the previous forest, 0.889 on this one, same 1,111 frames, 113 of them flipping right against 11 flipping wrong. It is also the only claim in this release that is a genuine paired A/B rather than two numbers measured a release apart under different conditions, so when something here has to carry the weight of "the retrain helped", it should be this one.

**And it says the retrain did not help. The data did.** Holding the holdout fixed and changing one thing at a time, at seed 0:

| | ayuraj | change | signer-clustered 95% CI |
| --- | --- | --- | --- |
| A. the previous release's forest | 0.7975 | | |
| B. this release's recipe, ASL-HG **not** in training | 0.7930 | **-0.0045** | [-0.013, +0.015] |
| C. this release's recipe, ASL-HG in training (what ships) | 0.8893 | **+0.0963** | [+0.065, +0.133] |

Going from 80 trees to 60, adding the 160-per-letter cap on my own frames, and every other recipe change between A and B is worth **nothing** on a stranger's hand: the interval straddles zero and the point estimate is slightly negative. Adding 6,000 frames from ten people is worth the entire +0.0918. That is the honest attribution, and it is the opposite of where the effort went — the forest-selection rule in `train_static.py` is a page of reasoning that bought no accuracy. It earned its place a different way, by catching a robustness failure the old recipe hid (see the idle gate), but not this way — and the gate it earned that credit from turned out to be too short as well. The lever for the next release is another licensed multi-signer set, not another sweep over trees and leaves. Reproduce with `crossval_strangers.py --compare-prev`, and `train_static.py --no-aslhg` for row B.

A fifth is worth naming because it is **not** here. The Pugeault and Bowden Kinect fingerspelling set corroborates the others locally — the shipped forest reads 0.808 of its 65,431 frames, which it has never seen — but its source page states no license at all, so neither its images nor landmarks derived from them are committed. `temporal/SOURCES.md` section 11 carries the regeneration recipe instead, along with the two other reasons it stays out: trained on, it breaks the idle gate outright (a relaxed hand reads as a G at 0.975), and its "five users" may be five sessions of four people.

What the sets are worth, and the order in which they were worth it, on my own folds at seed 0 (pooled / cross-day): my four sessions alone 0.873 / 0.782; add ASLNow and the digit photos, 0.913 / 0.873; add ASL-HG at its cap, 0.926 / 0.895. Other people's hands teach invariances that transfer back to my own unseen day. The same effect runs the other way: a forest trained on my sessions alone reads the 218-signer V at 0.19, and the shipped recipe reads it at 0.88. The letters the forest gets wrong on strangers are not the ones it gets wrong on me: on the by-signer split U reads as R and R reads as U, and those two are the only letters under 0.80 with nine other signers in training, which makes them a defect in the 112-value feature rather than a shortage of data.

### The feature

Every frame becomes a vector of 112 values: the 21 landmarks centered on the palm and divided by palm width (42 values), the distance between every pair of fingertips, knuckles and the wrist (55), a straightness ratio for each of the four fingers, and an 11-value thumb block. The pairwise distances matter because a decision tree splits on one value at a time, so a question such as "how far apart are these two fingertips", which is the whole difference between U and V, would otherwise take a long chain of splits to express. Dividing by palm width also fixes the largest weakness of Approach B, because the features no longer change when the hand moves closer to or further from the camera.

The thumb block: how straight the thumb is, how far each fingertip sits from the palm center, and the distance from the thumb tip to the index and middle knuckles. It exists because the letters the first 101 values confuse are the fists (A, E, M, N, S, T) and D against X, which differ only in where the thumb sits, and the fists keep every fingertip curled in almost the same place, so the thumb has to be measured against the knuckles rather than the tips. Measured leave-one-session-out on my sessions alone it added about +0.02 on the cross-day fold, within one seed spread, and it is kept for the geometry it encodes. The feature is tagged `static/v4` in the pickles and in `models.json`, and every consumer picks the transform by that tag; the previous 101-value feature is still `static/v3`.

### Augmentation

One thing happens to a training frame and never to a test frame: it gets four jittered copies. Each landmark coordinate is perturbed by Gaussian noise with a standard deviation of 0.12 palm widths, so the noise scales with the hand's distance from the camera, and the vector is recomputed from the moved landmarks. This replaced the four-angle rotation augmentation two releases ago. Rotation had lowered leave-one-session-out accuracy, 0.782 without it against 0.759 with it, and had only ever been justified by within-session confidence. On my sessions alone the jitter was the largest single lever, +0.09 pooled and +0.14 on the cross-day fold over no augmentation; with the strangers in the training set it is what lets a forest of 60 trees generalize from 12,368 frames of some 230 hands.

Two caps are new, and they matter as much as the jitter. My own block is lopsided — 400 frames of T and 308 of C against 100 of most letters — so the letters I happened to re-record most were weighted most; capping at 160 per letter on the **pooled** block takes my 4,878 frames to 3,807, moves ASLNow held out from 0.7882 to 0.8063, and cuts 13% of the forest's nodes at the same time. ASL-HG needs a ceiling for the opposite reason: uncapped, its 24,000 frames from ten hands outvote mine and my own pooled fold drops from 0.9256 at the cap to 0.9086 with every frame in. Both measurements are recorded beside their constants, in `train_static.AUTHOR_CAP` and `strangers.ASLHG_CAP`.

The previous release also dropped 137 of my frames whose geometry violated their own letter's rule (a near-straight X, a G with an extended middle finger). Those rules were written for one hand. Applied to the strangers they discard half of their X, P and R frames, and with strangers in the training set the filter costs on every axis, three seeds each: the ASLNow five-fold number falls from 0.945 to 0.903 with it. It is retired; `static_aug.py` keeps the rules and `train_static.py --ruleset strong` runs the ablation.

The forest itself is 60 trees with at least five samples per leaf, 245,064 nodes, trained on 61,840 rows from 12,368 frames (3,807 capped frames of mine and 8,561 of strangers', each with four jittered copies). Every node ships to the browser. Packed, all three forests together — 278,750 nodes — are 1,184,427 B on the wire as `models.bin.gz` plus `models.meta.json`; the same trees as JSON are 20,361,576 B raw and 3,703,083 B gzipped, which is the form the page fetched until this release and what `models.json` still costs anyone who downloads it to read. On the wire the JSON form grew 17.7% this release, because the forest grew 22.8%.

That setting was chosen by a rule rather than inherited, and the rule puts a hard gate first: train the candidate at three jitter seeds, and disqualify it if it emits on any of the 43 clean idle holds at any of them, whatever it scores. The cheapest candidate, 60 trees at leaf 8, was thrown out that way (at seed 1 it emits on one idle hold, 12 votes, every one a G). Of the three that passed, none separated on accuracy by more than its own three-seed range, so the node count decided: 245,040 for 60 trees at leaf 5 against 326,719 for 80 at leaf 5. The gate goes first because the previous release's own recipe passes it at seed 0 and **emits at seed 2** — its headroom was a property of one draw of the jitter RNG, not of the recipe, which is the next section's subject.

### Deciding when a static letter has been signed

A letter is emitted once, when a stable hold is first established, from a vote over its first four frames: at least three of the four frames must agree, the winner must lead the runner-up by 0.10, and its mean probability must reach **the winning letter's own floor**. The floor is the knob that separates a held letter from a relaxed hand, and it is set against a live browser log rather than against the training data. One session of about two minutes of my own hand produced 148 holds; 43 of them are clean idle holds the page was right to stay silent on (holds within 2.5 s before or 1 s after a J/Z track are excluded, because those are the I and D launch poses, read correctly and canceled by the motion branch). Their landmarks are committed as `temporal/idle_holds.npz`, and `temporal/idle_gate.py` replays them through the vote with any forest.

**The floor is now per letter, because exactly one letter was setting it for all 24.** A hand hanging relaxed at a laptop is a loose G, and a forest that has seen many hands is confident about it. Over the 43 idle holds at three jitter seeds, G's most confident idle vote that already clears the agreement and margin tests is 0.7087, 0.7019 and 0.7101 — and no other letter exceeds 0.5130. Sixteen of the 24 letters never win an idle vote at all. A single number therefore has to be set by G and then charged to the alphabet, and on this forest that bill is large. So `VOTE_PROB_LETTER` holds G at 0.75 and everything else sits at 0.55: 0.0399 of headroom for G and 0.0370 for the rest, the same margin on both sides, which is what makes it two numbers and not twenty-four. An entry may only *raise* a letter's floor; the constructor refuses one below the flat value, because a per-letter relaxation would be invisible in the flat constants and is exactly how the idle hand gets back in.

What that buys, replaying my 71 held signs end to end through the segmenter with real timestamps and forests that never saw the session: exactly the right letter on **66 of 71** against 61 for the flat 0.75 it replaces, **4 holds silent** against 9, cross-day 22 of 24 against 20, one wrong letter either way, and 0 of the 43 idle holds either way. Three of the four that stay silent are G's own — S1-G and two S3-G — paying for G's 0.88 floor, which is the honest shape of the trade; the fourth is S2-K, which is silent under every forest and every floor this project ships. A flat 0.55 would give 69 of 71 and emit on 4, 4 and 5 of the 43 idle holds (53 to 55 votes, every one a G).

**Two things I had to correct about the floor before moving it.** The first is that the previous release's note, "six of the 71 holds are silent at that floor", was read ever after as the floor's price, and it was not: *silent at* a floor is not *silenced by* it. Swept flat on that same forest, of its six silent holds only **S1-E** came back as soon as the floor moved at all (at 0.72). S1-N needed 0.59, the three S3-D holds needed 0.50, and S2-K never gave its letter at any floor — silent from 0.40 upward, and a wrong H at 0.35 and 0.30. The floor cost one hold, not six. The second is that a floor's headroom is a property of one draw of the jitter RNG. The previous recipe passes the idle gate at seed 0 with a most confident idle vote of 0.6950, and **emits at seed 2** (one hold, one vote, a G, at 0.7536 over its own 0.75 floor). The gate is run at three seeds now, and it is a hard gate on the forest as well as on the floor.

**And three was the same mistake one level up.** I picked seeds 0, 1 and 2, they were clean, and I called that a gate. Run at eight, the 24-class recipe emits on a clean idle hold at seeds **3, 4, 5 and 7** — four of the eight. Which three seeds you happen to pick decided whether the gate printed PASS. The forest that shipped was seed 0 and genuinely read 0 of 43, so the page was never emitting letters at a resting hand; what was wrong was the claim that the recipe was robust. `idle_gate.py` now refuses to print PASS under `MIN_SEEDS = 8` and says so.

**The fix was three numbers, after a detour that was not.** A resting hand is a loose fist with the thumb somewhere, the same hand angled down, or index and thumb apart — which is to say an A, a Q and a G. Those are exactly the three letters that breach a flat floor on the idle holds, and nothing else comes close. Set each one's floor above its own worst draw over sixteen seeds rather than above one, and the gate holds:

| letter | worst idle over 16 seeds | its floor | headroom |
| --- | --- | --- | --- |
| G | 0.8433 | **0.88** | +0.0367 |
| Q | 0.5504 | **0.62** | +0.0696 |
| A | 0.5310 | **0.60** | +0.0690 |
| O | 0.5105 | 0.55 | +0.0395 |
| M | 0.4524 | 0.55 | +0.0976 |

G is 0.88 rather than 0.85 because that margin is nearly free: 0.85 leaves 0.0067 of room, 0.88 leaves 0.0367, and the difference costs 0.7% of real G on held-out signers (0.9640 to 0.9570) and not one extra hold on replay, whose G sweep is flat from 0.85 all the way to 0.90.

**The detour is worth recording, because the reasoning was good.** With 24 classes a resting hand *has* to be assigned some letter — there is no other answer available — so give the forest a 25th, REST, trained on other people's hands caught not making a letter. It works: 530 such frames, cut from the OpenHands clips by `temporal/ingest_idle_strangers.py`, take the gate from failing at four of eight seeds to passing all sixteen, with the 24 letters no longer summing to 1 and the missing mass doing the suppressing. It also costs **two wrong letters on 266 clips of strangers' video against none**, consistently across three seeds, and buys margin the three constants above already buy. `temporal/rest_probe.py` carries the whole measurement — the settings that work, the ones that do not, and the table that kills it — so the next person to have the idea starts from evidence rather than from the same good reasoning. I had those three worst-idle numbers in front of me before I built it.

What the floors cost, honestly: two of the 71 held signs go silent (both S3-G, G paying its own bill) and two of the 266 stranger clips stop being read. `replay_strangers.py` still reports **0 wrong letters at every tracking and duration cut**, which is the property that matters — silence on a hold the forest is unsure of is the design; a wrong letter is the defect.

**And the drop from 0.75 to 0.55 is safe only because the forest was retrained underneath it.** Put the shipped floor on the *previous* forest and replay 266 clips of strangers' fingerspelling video: it buys 13 more correct clips and costs **eight wrong letters** (G read as O, O as E, T as O, V as O, U as V, U as R, in the tracked cut alone). Put it on the retrained forest and it costs none — zero wrong letters on all 266 clips, at every tracking and duration cut. The idle gate cannot show that, because a relaxed hand is not a stranger's letter; only `temporal/replay_strangers.py` can, and both have to be re-run by anyone who retrains. Silence on a hold the forest is unsure of is the design; a wrong letter is the defect.

### Deciding when a motion letter has begun

Six inequalities on finger geometry decide whether the hand is currently in the launch pose for J or for Z. A gesture is tracked only if the hand was held still in that pose and then started moving, so the ordinary movement between letters almost never begins a track. I kept these as explicit inequalities rather than training a second model, because I can watch each condition on the live overlay and see which one failed. A model asked about an input unlike anything in its training set tends instead to answer confidently and incorrectly, which is the failure mode Approach A demonstrates.

The J gate's thumb ceiling moved from 1.20 to 1.30 palm widths in the previous release, because at 1.20 only 41 of the 60 J items in the prompted takes produced a creditable event; at 1.30, 48 do, and 12 still have none (1.25 buys only 3 items, 1.35 two more for one more Y frame). At 1.30 it passes 100 of 100 held-I frames in the archive and 13 of the 2,278 frames of every other letter, all of them Y; replaying the whole 157 s archive through the segmenter starts 0 tracks from them, because arming also needs a parked frame and a sustained rising edge. A Y held still and then moved could arm a track that no negative example resembles; no such footage exists yet.

### Ratios rather than lengths

A finger pointing toward the camera appears shorter than it is, so a distance measured in the image changes for reasons that have nothing to do with the handshape. Straightness is the distance from fingertip to knuckle divided by the summed length of the bones. Because it is a ratio, it stays stable under that foreshortening.

### Labeling the motion events

The motion forest is trained on events the segmenter itself cuts from five prompted takes (113 items: 60 J and 53 Z, each parked, signed and rested on a fixed rhythm), so every training example has boundaries the runtime can produce. One event per prompted item is credited as that item's letter; every other span the runtime could have scored, a false start inside an item or a move during the rest, is labeled MOVE; spans the runtime could never score, aborted tracks and spans the vetoes reject, are dropped rather than taught as MOVE. That gives 117 events (J 48, Z 37, MOVE 32) in 92 items, and the cross-validation folds by item, so a fragment of a test gesture can never sit in training. An older release labeled fragments as gestures and reported 0.864 over 140 events; those numbers are not comparable with the ones above. Two rules tie the branches together, because the launch pose of a motion letter is itself a static letter. A parked I or D whose own pose arms a track, or is still confirming its rising edge when the timer runs out, is held until the track resolves: a J or Z emission cancels it and an abort releases it. An I held longer than the 0.35 s timer before the stroke begins is still released by design, because deferring every I and D to the end of its hold was measured and rejected for latency; on the takes that leaves 2 "IJ" and 3 "IJI" among the 60 J items, down from 38 releases before the rule. And after a J or Z is emitted the static branch stays quiet while the hand rests in the finishing pose, clearing as soon as the hand reshapes or moves again, so a hand that morphs straight into the next letter is not delayed. One more rule turned out to be needed for the live path to match training: MediaPipe labels handedness per frame, and 166 of the 13,692 detected frames on the takes carry the other hand's label for a frame or two, which mirrored the hand mid-stroke and aborted the track. The segmenter now keeps the label it has and switches only after the other label has persisted for 0.50 s (`HAND_SWITCH_S`; the longest such flip inside an unbroken detection is 0.20 s), clearing on a detection gap, which is how a signer changes hands anyway. Replaying the five takes with the retrained forest, which is in-sample for it, feeding the per-frame label the page and `live_demo.py` feed: 85 of 113 items produce their letter, up from 78, doubles fall from 9 to 0, and there are no J or Z emissions during the rest phases over 5.5 minutes. Without the latch the same replay credits 79; with it, every one of the 113 item strings is identical to a replay with one modal label per take, which is how the training events are cut, and `temporal/evaluate.py` reproduces the count directly (S1 74 of 90 plus S5 11 of 23). The 28 misses are segmentation, the gate never armed, no rising edge, a veto or a detection gap, and none of the fixes address them.

### What the development actually cost

Every letter I found broken while building this turned out to be either an inconsistency in my own recordings or a threshold I had chosen before I had the data to choose it. In one session every `G` frame had an extended middle finger, which makes the sign an `H`; those frames outvoted the correct ones, and `G` was read as `H` everywhere. `T_MAX` was set below the 95th percentile of my own recorded Z durations, which cut genuine gestures out of the range the classifier had been fitted on. `P_EMIT` was 0.70, which quietly discarded about one real gesture in three. The vote floor has now moved twice for the same reason from the other side: each forest that reads letters better also reads a relaxed hand more confidently, and the floor that was harmless for the last one lets the next one emit on a third of idle holds. I found each of these by logging what the running system saw, and none of them by reading the code.

### The browser version

`docs/` is a direct port of the Python: the same feature code, the same thresholds, and the three forests flattened into JSON. Before the camera starts, the page runs 41 real frames through the models it just loaded (15 from the training landmarker, 11 from the landmarker the page itself runs, 15 digits), compares the results with the answers Python gives for those same frames, and reports the outcome in the readout. A silent difference between the two implementations would look exactly like a poor model, so it is worth checking on every load. The worst disagreement on those 41 cases is 3.3e-6 in probability.

The thresholds travel in the same file as the forests, so the page cannot pick up a new model with stale constants, and `segmenter.js` refuses to build on an export missing any field it reads — which is how this release's per-letter floor reaches the page as a thrown error rather than as a silent relaxation back to a flat 0.55.

The forests ship in a packed binary form this release, `docs/models.bin.gz` plus a small `models.meta.json`, and that is what the page fetches. Same numbers, 3.1x smaller on the wire than the JSON (1,184,427 B against 3,703,083 B) and four to nine times faster to decode (15-38 ms against 129-167 ms in Node), with zero routing differences over all 278,750 nodes and a worst probability difference of 4.98e-7 against sklearn on the 41 golden frames, against 3.33e-6 through the JSON. `docs/models.json` is still exported and still committed, because a text file is what makes the trees readable and what every other tool here can open, and `loadModels` returns the identical object from either. `docs/README.md` has the format and the measurements.

The browser condition differs from training in one way that had to be measured rather than assumed. The page runs MediaPipe's Tasks API, while my own training landmarks came from the legacy `solutions` API, and the two disagree about what "Left" means on the same unflipped frame; the page swaps the label before the mirror step. On the archive re-extracted with the Tasks API, the previous release's cross-day model scored 0.775 with the swap and 0.730 without it, so the swap is verified rather than assumed, and the gap between the two landmarkers on that fold was +0.001 over three seeds, with a paired 95% interval of about ±0.03. One caveat on both numbers: the re-extraction ran the CPU build of the Tasks landmarker from Python in VIDEO mode, not the GPU delegate the page runs. The ASLNow records close most of that gap from the other side, since they are the browser's landmarker on other people's hands, and the forest now trains on 1,874 of them. Input resolution from 1920x1080 down to 426x240 moved the same fold by no more than 0.004; shrinking the hand to the palm size the browser log typically shows cost 0.02, and below that the loss is MediaPipe failing to find the hand at all rather than the forest misreading it. The shrink is synthetic (the archive frames scaled onto a gray canvas), not real distant captures.

### Numbers mode

The page has an experimental numbers mode for the digits 0-9. It is a separate 100-tree forest on the same 112-value feature, trained on the Sign Language Digits Dataset: landmarks from 1,805 photographs of 218 students' hands (MediaPipe found a hand in 1,805 of the 2,062 photos; no image is redistributed here). Leave-signer-out on that dataset it scores 0.986, with digit 6 the worst at 0.965. It is trained on 218 signers from a public dataset, not on the author, and it has not been verified on the author's hand or on live video. The only evidence that it reads my hand at all is a proxy: my O, V, W, F and B letter frames read as 0, 2, 6, 9 and 4 on 0.956 of frames. It is weak evidence, because my C, R, X and U frames also read as 0, 2, 1 and 2 just as confidently (C as 0 on 1.00 of frames, R and U as 2 on 0.90, X as 1 on 0.99, emitting at the digits gate on 0.83-0.93 of them), so the forest has a digit for many handshapes that are not one. In this mode the J/Z gates are off (the '1' handshape passes the Z gate and would otherwise park the machine in a track), the word layer is off, and the vote constants are the ones measured for it (margin 0.40, floor 0.60, confident route 0.70), because a relaxed hand reads as a digit, mostly 0 or 1 (57% and 40% of the emissions), on about one frame in six (0.162, the single-frame rate over 2,890 hold records forming 148 holds from the letters-mode browser log, where any digit emission is a false one); duplicate suppression bounds that to one spurious digit per hand-raise. Recording my own digits is the first follow-up.

### Words

The page groups letters into words and, when it can, names the word. A word ends after 1.20 s with no hand in the frame. At the break, every listed word of the same length is scored letter by letter against the votes the classifier actually produced, with a prior that favors common words, and the best one is shown under the reading only if it clears three bars: it must be at least 0.2 as likely as the letters read, at least 3 times likelier than the runner-up, and the prior's weight is 3.0. The list is 34,702 words in frequency order, the most frequent 40,000 of the Google Books counts distributed by norvig.com plus a set of proper names, which is what puts HENRY in it. The three constants were chosen offline by `temporal/simulate_words.py`, which spells words out of held-out letter posteriors through the same vote gate the page runs; its `--sweep` default is the grid they came from, and the triple within 0.01 of the best cell in the fresh-hold table and 0.025 in the same-hold one is the one that ships. **That sweep was run against the previous forest at its flat 0.75 floor, and it has not been re-run for this one.** Both inputs it depends on changed — the posteriors come from the forest and the gate is the floor — so the figures below describe the sweep that set the constants and not this release: on the previous forest, 0.43 of common words recovered with a wrong one shown 0.08 of the time when a misread letter is retried from consecutive windows of the same hold, and 0.66 / 0.05 when every retry is a fresh hold; against the constants before them (0.1, 10, 2.5) on the same posteriors, hint precision 0.75 to 0.83 and 0.86 to 0.90, with 18% fewer wrong "is a word" confirmations, for one point of recovered words. The layer two releases ago, a 150,594-entry dictionary with no frequencies, measured 0.445 / 0.158 and 0.612 / 0.091 under the same two models on its own forest and gate. Re-running the sweep is a follow-up, and until it is run the constants are inherited rather than measured here. The reading itself is never rewritten, so the cost of leaving them stale is a hint that is shown too rarely or too often, never a changed letter.

---

## Running it

### Environment

Two environments, because the notebooks predate the pinned one. The notebooks were run under Anaconda Python 3.11 on macOS (Apple M1 Pro) at the versions observed in their own `pip` output; Approach C and `docs/export_models.py` run in the `.venv` that `requirements.txt` pins.

| Package | Notebooks (observed) | `requirements.txt` (Approach C) |
| --- | --- | --- |
| Python | 3.11 | 3.11 (3.11.14 here; `mediapipe` 0.10.18 publishes no macOS arm64 wheels for 3.12+) |
| `mediapipe` | 0.10.18 | 0.10.18 |
| `numpy` | 1.24.3 | 1.26.4 (the last 1.x, which `mediapipe` 0.10.x is built against) |
| `opencv-python` | 4.10.0.84 | 4.11.0.86 |
| `scikit-learn` | 1.3.0 | 1.9.0 |
| `matplotlib` | 3.7.2 | not pinned |
| `tensorflow` / `keras` | Keras 3.x | not used |

`requirements.txt` pins Approach C's four dependencies; the notebooks are not covered by it. Each notebook opens with a `%pip install` cell, but note that the CNN's install cell covers only `opencv-python` and `mediapipe` while its training cell imports `tensorflow`, `pandas` and `scikit-learn` — those must already be present. Two version notes: `model.p` was pickled under scikit-learn 1.3.0 and raises `InconsistentVersionWarning` when unpickled under a newer version (it still loads and predicts), and `model.save('smnist.h5')` warns that HDF5 is a legacy format. `smnist.keras` holds a different, earlier training run of the same architecture in the native format (98.87% on the test CSV); the live demo loads `smnist.h5`.

### Approach C — all 26 letters

```
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python temporal/live_demo.py --camera 0                 # letters; press M, or pass --mode digits, for numbers
```

Python 3.11 and `mediapipe==0.10.18` specifically: MediaPipe 1.0 removed the `mp.solutions` namespace, which breaks this code and both notebooks. Nothing needs installing to use the [hosted version](https://henryyhong.com/ASL-Classification/).

Retraining everything, from the repository root, in this order (each step reads what the one before it wrote):

```
./.venv/bin/python temporal/ingest_aslnow.py                        # only if aslnow.npz needs rebuilding (network)
./.venv/bin/python temporal/ingest_aslhg.py                         # likewise aslhg.npz (877 MB of zip)
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz
./.venv/bin/python temporal/train_digits.py
./.venv/bin/python temporal/crossval_static.py --seeds 3            # my own folds
./.venv/bin/python temporal/crossval_signers.py --seeds 3           # leave one of ten signers out
./.venv/bin/python temporal/crossval_strangers.py --seeds 3         # each stranger set held out, and the never-train set
./.venv/bin/python temporal/idle_gate.py --seeds 16                  # a HARD GATE: must pass at every seed
./.venv/bin/python temporal/replay_static.py                        # what the floor costs on my 71 held signs
./.venv/bin/python temporal/replay_strangers.py --root <American>/videos --out temporal/openhands_replay.json
./.venv/bin/python temporal/label_events.py --out /tmp/check.npz    # read its diagnostic; ALWAYS pass --out
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/evaluate.py                             # motion numbers per session; in-sample with the shipped forest, and it says so
./.venv/bin/python docs/export_models.py                            # -> models.json/.gz, golden.json, models.bin/.gz, models.meta.json
./.venv/bin/python docs/parity_probes.py                            # -> docs/parity.json, stamped with the new build id
cd docs && node test_forest.mjs                                     # fails on a stale parity.json, on purpose
```

Three of those bite if they are skipped. `idle_gate.py --seeds 16` is a gate and not a report: a forest that emits on any of the 43 idle holds at any seed is disqualified, and the fix is to raise the emitting letter's own floor in `thresholds.VOTE_PROB_LETTER`, not the flat one. `parity_probes.py` must run after every export, because `docs/parity.json` is stamped with the build id of the binary it measured. And `label_events.py --out` defaults to `temporal/events.npz`, the committed motion training set: run the diagnostic bare on somebody else's clips and it used to replace that file and say nothing. It now refuses when `--clips` points away from the committed recording and `--out` is left at the default, and `train_motion.py` refuses the same way for `--data` and the shipped `model_motion.p`. Keep the `--out` in the line above regardless; the refusal is a backstop, not a place to put your thinking.

The shipped `model_motion.p` is fitted on both recorded sessions, so `evaluate.py` labels every row in-sample. A genuinely held-out motion row takes a forest fitted on one session first, written to scratch rather than over the shipped pickle:

```
./.venv/bin/python temporal/train_motion.py --fit-sessions S1 --out /tmp/motion_S1.p
./.venv/bin/python temporal/evaluate.py --motion-model /tmp/motion_S1.p --test-session S5
```

`temporal/README.md` has the recording order; `collect_motion.py --static-letters` writes a new session to `temporal/static_<session>.npz` and refuses to overwrite one that exists.

### Approach A — CNN

Run `jupyter notebook ASL_Detection.ipynb` from inside `CNN/`. The cell numbers below are positions in the notebook file counting from 0 and including the two markdown cells at the top (the title and the `Author:` line) — so cell 2 is the first code cell (`%pip install opencv-python mediapipe`), cell 3 is the training cell, and cell 6 is the live demo, the last cell with any code in it. They are not the `In [n]` execution counts Jupyter displays.

| Cell | What it does | Notes |
| --- | --- | --- |
| 2 | `%pip install opencv-python mediapipe` | Restart the kernel if anything was newly installed |
| 3 | Loads both CSVs, builds and trains the model, saves `smnist.h5` | ~6 minutes for 20 epochs on an M1 Pro |
| 4 | Plots the first three training images | Calls `plt`, but no cell in the notebook runs `import matplotlib.pyplot as plt` — add it or this raises `NameError` |
| 5 | Plots the accuracy and loss curves | Same missing import, and it needs `history` from cell 3, so same session only |
| 6 | Live webcam demo, loads `smnist.h5` | Standalone — skip cells 3-5 to use the committed model |

### Approach B — Random Forest

Run `jupyter notebook Mediapipe.ipynb` from inside `RandomForest/`.

| Cell | What it does | Notes |
| --- | --- | --- |
| 0 | `%pip install mediapipe opencv-python scikit-learn numpy matplotlib` | Restart the kernel if anything was newly installed |
| 1 | Captures 100 frames x 24 classes into `./data/` | **Skip unless recapturing** — it overwrites `data/`. Blocks 5 s per class; be in position |
| 2 | Extracts 42-D landmark vectors -> `data.pickle` | Relies on the `import os` in cell 1 — run it after skipping cell 1 and it raises `NameError`; add `import os` at the top. Slow: it rebuilds the MediaPipe graph per image. `data.pickle` is already committed |
| 3 | Trains on the stratified random split, writes `model.p`, prints accuracy | |
| 4 | Trains on the per-class 80/20 slice, **overwrites `model.p`**, prints accuracy | Run 3 or 4, not both, unless you want the second one's model |
| 5 | Live webcam demo, loads `model.p` | Standalone from the committed pickle |

### Camera index

All three capture sites are hard-coded to the second camera. `RandomForest/Mediapipe.ipynb` cell 1 even says so in a comment:

```python
cap = cv2.VideoCapture(1)  # Change to 0 if you have only the default camera
```

**With only a built-in webcam, this must be `cv2.VideoCapture(0)`** in `CNN/ASL_Detection.ipynb` cell 6 and `RandomForest/Mediapipe.ipynb` cells 1 and 5. A wrong index does not report itself clearly: `cap.read()` returns `ret = False` and `frame = None`. The CNN demo raises at `h, w, c = frame.shape` on a `None` frame. The Random Forest cells are quieter: the live demo (cell 5) hits `if not ret: break` on the first iteration and finishes with no error and no window, and the capture cell (cell 1) breaks out of each inner loop but still walks all 24 classes, printing `Collecting data for class ...` and sleeping 5 s each — about two minutes of looking busy while writing no frames at all.

---

## Known limitations and next steps

In rough order of how much they matter:

- **The landmark features are not scale-normalized.** 42 values expressed as fractions of the frame mean predictions shift with how far the hand is from the camera. Dividing each coordinate by the hand's bounding-box extent after the `(x - min)` shift would make the representation size-invariant, and is the highest-value single change to this repository.
- **The dataset is one signer, one hand, one day.** The two sittings differ in room but not in signer, hand or day, so they do not function as independent sessions and cannot be split against each other. Train on session 1, test on session 2 — different day, different lighting, ideally a different person — is what turns 99.58% into a number that means something.
- **Class 20 (V) has 78 samples, not 100.** It is under-represented in training and, under cell 4's split, untested. Recapturing V is cheap.
- **`mp_hands.Hands()` is constructed inside the per-frame loop** in the Random Forest notebook — the extraction cell and the live demo both rebuild the entire MediaPipe graph every iteration instead of once. The extraction cell's captured output carries 2,400 `gl_context` initialization lines — exactly one per image — and the live demo's carries 448. (The CNN demo already builds it once above its loop.) Hoisting it out is a straight throughput win.
- **The capture loop writes 100 frames back to back,** pausing only for the loop's `cv2.waitKey(25)`, which is why frames within a class are near-duplicates. A substantially longer delay, plus deliberate repositioning between frames, would make 100 images worth 100 images at the same capture cost.
- **The CNN has no checkpointing.** `ModelCheckpoint(save_best_only=True)` on validation accuracy would have saved the epoch-12 model at 99.69% instead of the epoch-20 model at 95.29%; early stopping on validation loss would also cut the wasted epochs.
- **The CNN's live crop is squashed, not letterboxed.** `cv2.resize` to 28x28 ignores aspect ratio, so a tall crop is distorted before the model sees it. Padding to a square first would remove one term of the domain gap; the lighting, resolution and background terms would remain.
- **The camera index is hard-coded** in three places rather than being one constant.
- **The notebooks have no pinned dependencies.** `requirements.txt` pins Approach C's environment (`mediapipe` 0.10.18, `numpy` 1.26.4, `opencv-python` 4.11.0.86, `scikit-learn` 1.9.0); the two notebooks still install from unpinned `%pip` cells, so they are approximately reproducible rather than reproducible.
- **Large artifacts are committed:** `sign_mnist_train.csv` at 83.3 MB, `sign_mnist_test.csv` at 21.8 MB, and 2,400 full-resolution JPEGs. The captured frames have to live somewhere, but Sign-MNIST is a public dataset and could be fetched on demand instead of vendored.

Approach C addresses the first three of these: the features are scale-normalized, there are now four capture sessions rather than one, and V was recorded properly. The notebooks themselves are left as they were, since their failures are the reason for keeping them.

Three things I would build rather than fix, written before `temporal/` existed. All three are now part of it:

- **An unknown / no-hand rejection path**, so a model can decline to answer instead of asserting a letter at 99% confidence. Approach A's failure mode is precisely the absence of one.
- **A small MLP on the same 42-D features**, to find out whether the representation or the classifier is the binding constraint. Nothing in this repository currently distinguishes the two.
- **A temporal buffer over the last N frames** — majority voting would stop single-frame flicker reaching the display, and classifying a *sequence* of 42-D vectors is the only route to J and Z, which the single-frame setting cannot reach at all.

The first became the emission rule, which abstains unless a vote is either confident or clearly ahead of the runner-up. The second became the 112-value feature, whose blocks of pairwise distances and thumb geometry test directly whether the representation or the classifier was the limiting factor. The third became the vote window and the motion branch.

Approach C's own limitations, in the same order:

- **U and R, on everybody's hands.** The by-signer split finally exposed this one. Held out one signer at a time over ten signers, U reads 0.671 (as R on 329 of 1,000) and R reads 0.703 (as U on 224), and nothing else is under 0.80. Nine other signers in the training set do not fix it, so it is a defect in the 112-value feature rather than a shortage of data: U and V differ by a fingertip gap the pairwise-distance block measures, but U and R differ by which finger crosses over which, and nothing in the feature says so. C (0.816, read as O) is next. The same two letters are at the bottom of the video replay as well: 0 of 4 C clips, 0 of 6 U clips.
- **The cross-signer numbers disagree with each other, and that spread is the honest uncertainty.** 0.9406 holding out one of ten named signers, 0.8292 holding out a whole different set, 0.889 on the permanent never-train holdout. Three conditions, three answers. What is still missing is a by-signer split that is not one dataset shot to one protocol, and a video replay with signer ids — the OpenHands clips have none.
- **Z does not transfer to other people, and the cause is one constant.** The Z launch gate fires on 100 of 100 of my own D frames and 35 of 72 of a stranger's, so about half of every stranger's Z is unreachable before any model runs. Decomposed: index straightness 71 of 72, thumb 58 of 72, curled fingers 37 of 72 — and 22 of the 37 misses fail the curled clause alone. I curl the idle fingers to a median extension ratio of 0.663; strangers sit at 1.194, right on the 1.20 line. Moving that ceiling to 1.40 takes D to 47 of 72 while the other 23 letters go from 70 of 1,802 to 74, which looks nearly free and has not been checked against my archive or a segmenter replay, so it has not been moved. The J gate transfers almost intact by contrast: 65 of 68, one false positive in 1,806, and that one a Y, the same letter it costs on my own hand.
- **The digits are the mirror image: trained on 218 strangers and never verified on me.** No frame of my own hand signing a digit exists; the 0.956 proxy is my letter frames whose handshape is a digit. Recording my own digits is still the first follow-up, and until then the numbers mode is a demonstration that a forest trained on other people's hands runs in the browser, not a verified feature.
- **Three of the four letter sessions were recorded on the same evening**, about two and a half hours apart, so the 2024 archive is the only cross-day fold and my pooled figure is dominated by same-evening folds. 0.8897, not 0.9233, is my own number to plan around — and neither of them is a visitor's.
- **The archive's M was recorded with the thumb where the textbook handshape does not put it**, and it is still the only letter under 0.6 recall on the cross-day fold, at 0.12. That is a data ceiling; re-recording it is the fix and no threshold will do it. (It was 0.01 last release. The strangers moved it, which is not the same as fixing it.)
- **The idle floor has 0.04 of headroom and it is still a G.** A relaxed hand is read as a loose G at up to 0.7101 over three seeds, and no other letter exceeds 0.5130, which is why the floor is per letter now. A different relaxed pose or camera could cross it. Re-run `temporal/idle_gate.py --seeds 16` after the next live session, and `temporal/replay_strangers.py` after any retrain — the floor drop is safe only because the forest was retrained underneath it, and the idle gate cannot see that.
- **The word layer's three constants were swept against the previous forest at the previous floor** and have not been re-swept. Both of their inputs changed. The layer never rewrites a letter, so the cost is a hint shown too rarely or too often, but the figures it reports are the old sweep's.
- **A relaxed hand reads as a digit about one frame in six in numbers mode**, bounded to one spurious digit per hand-raise by duplicate suppression, which is why that mode is off by default and labeled experimental.
- **The models are committed, and the reader pays for it at the clone.** `temporal/model_static.p` is 62,768,543 B and `docs/models.json` is 20,361,576 B — 62.8 MB and 20.4 MB of generated file, in version control, replaced in full every time the forest is retrained. `.git` is 736 MB, a 702.8 MiB pack, and a clone has to move all of it before a single command runs: copying it locally on this machine, with no network in the measurement, takes 6.4 s and lands 1.6 GB on disk, so treat that time as a floor. The models are not the whole of that pack — its largest object is `CNN/sign_mnist_train.csv` at 79.4 MiB compressed, ahead of the current `model_static.p` at 27.2 MiB — but they are the part that grows, because every retrain writes another full copy into history. Nothing in the pickle or the JSON is needed to read the code, and both come back from one command each — `./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz` rebuilds the pickle from the committed landmark sets (60 trees, leaf 5, seed 0, the recipe the pickle records), and `./.venv/bin/python docs/export_models.py` rebuilds `models.json`, `models.bin`/`.gz`, `models.meta.json` and `golden.json` from the three pickles. Publishing them as release assets instead would cost the page its one-fetch deploy, which is why they are still here; it is a trade I made, not one I did not notice.

---

## Credits and provenance

Approach A trains on **Sign Language MNIST**, a public dataset. Hand detection and 21-keypoint extraction in all three approaches use **Google's MediaPipe Hands** (the legacy `solutions` API in Python, the Tasks API in the browser).

The numbers mode is the one part of this repository not trained on my own hand. Its forest is trained on the **Sign Language Digits Dataset** by the students of Turkey Ankara Ayrancı Anadolu High School (project executives Zeynep Dikle and Arda Mavi), <https://github.com/ardamavi/Sign-Language-Digits-Dataset>, Apache-2.0: 2,062 100x100 photographs, one per digit per student for 218 students. Cite as Mavi, A. (2020), *A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits*, arXiv:2011.08927. What this repository takes from it is landmarks only: `temporal/ingest_images.py` runs MediaPipe over each photograph and keeps the 21 landmarks, the handedness label and the frame size for the 1,805 images where it found a hand, committed as `temporal/digits_ankara.npz` (0.43 MB). No image from the dataset is redistributed here.

The static letters train on three public landmark sets beside my own recordings, and are measured against a fourth that nothing may ever train on. Every one of them is documented in [`temporal/SOURCES.md`](temporal/SOURCES.md), and in every case only landmarks are committed — no photograph and no video from any of these sets is redistributed here.

**ASLNow** (<https://huggingface.co/datasets/sid220/asl-now-fingerspelling>, MIT), by the authors of the ASLNow! fingerspelling app: 2,122 records of multiple participants signing letters into a webcam, each one the 21 landmarks from MediaPipe's Web Hand Landmarker. `temporal/ingest_aslnow.py` downloads them, infers the frame aspect and each record's handedness (the source stores neither), and commits the landmarks with those two decisions as `temporal/aslnow.npz` (0.9 MB).

**ASL-HG** (<https://data.mendeley.com/datasets/j4y5w2c8w9/1>, DOI 10.17632/j4y5w2c8w9.1, CC BY 4.0), by Md. Famidul Islam Pranto, Md. Rifatul Islam, Md. Ali Akbor and co-authors of Bangladesh University of Business and Technology: 36,000 smartphone photographs of ten volunteers in Mirpur, Dhaka, 100 per class per person. `temporal/ingest_aslhg.py` reads the 24 static-letter folders and commits the landmarks of the 23,984 images MediaPipe found a hand in as `temporal/aslhg.npz` (5.6 MB). It is the only source here that carries real signer ids, and therefore the only one that can be split by person.

The four digits of the **Sign Language Digits Dataset** whose handshape is a letter (0/O, 2/V, 6/W, 9/F) are the third set, from the same file the numbers mode trains on.

**ayuraj/asl-dataset** (<https://www.kaggle.com/datasets/ayuraj/asl-dataset>, by Ayush Thakur, CC0) is the fourth, and it is here to stay out of the training set permanently. `temporal/ingest_ayuraj.py` commits the landmarks of its five signers as `temporal/ayuraj.npz`; `strangers.NEVER_TRAIN` and a test in `temporal/tests/` enforce the rest.

The end-to-end video replay uses the **OpenHands fingerspelling pose datasets** (<https://doi.org/10.5281/zenodo.6813108>, CC BY 4.0). Only the per-clip result is committed, as `temporal/openhands_replay.json`; the video is not.

The word list behind the page's hint, `docs/words.txt`, is built by `docs/build_words.py` from Peter Norvig's table of **Google Books Ngram** 1-gram counts (<https://norvig.com/mayzner.html>; the Ngram data is CC BY 3.0) plus macOS's `/usr/share/dict/propernames`.

Everything else here is my own work: the 2,400-frame dataset in `RandomForest/data/` — captured, labeled and curated by me — the four letter sessions and five prompted J/Z takes behind `temporal/`, the capture, training and inference code in all three approaches, the browser port, and the screenshots above.

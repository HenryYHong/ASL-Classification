# Real-time American Sign Language Recognition

![Signing A B C H E N R Y J Z into the hosted page, each letter appearing as it is recognized](docs/demo.gif)

*One take on the [hosted page](https://henryyhong.com/ASL-Classification/). I trimmed and cropped the recording but did not speed it up, so the pace here is the pace it runs at. J and Z come last because they are the two letters that cannot be read from a single frame.*


This repository holds three attempts at the same problem, kept in the order I built them, because each one grew out of a limitation in the one before it.

The first trains a CNN on 28x28 pixel images. It scores between 95% and 99% on its benchmark and still fails in front of a webcam. The second ignores the pixels and classifies the 21 hand landmarks that MediaPipe reports instead. It works live, but it covers only 24 letters, and its 99.58% comes from a split that puts nearly identical frames on both sides. The third adds J and Z, reports its accuracy from a split that holds out an entire recording session, and runs in a browser.

### Try it

The third approach is hosted at **[henryyhong.com/ASL-Classification](https://henryyhong.com/ASL-Classification/)**. There is nothing to install, and the camera frames stay on your machine: the page downloads one model file (11.6 MB of JSON, 2.1 MB gzipped on the wire, which is how GitHub Pages serves it) and runs the models locally. Chrome and Safari both work. Allow the camera, then sign into the box.

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
| Training data | Sign Language MNIST (public), 27,455 rows | 2,377 usable landmark vectors from 2,400 frames I recorded myself | 4,878 frames over four sessions, which become 23,705 rows after the geometry filter and jitter; 117 motion events cut from 92 prompted items; all mine. Digits: landmarks from 1,805 photos of 218 strangers (public) |
| Model | 3-block CNN, 264,049 params | `RandomForestClassifier()`, 100 trees, 9,452 nodes | Two forests (100 and 300 trees), a six-inequality launch gate, a four-state segmenter; a third 100-tree forest for the digits |
| Training cost | 20 epochs, ~17-20 s each on an Apple M1 Pro | Seconds | About a minute |
| Headline accuracy | 95.29% saved / 99.69% best epoch | 99.58% random split, 100.00% per-class split (V contributes zero test samples — see caveats) | **0.86 leave-one-session-out** (95% CI 0.79-0.92 over the 57 session-by-letter bursts), **0.76 on the cross-day fold**; previous release 0.759 / 0.659. J/Z: 0.951 by prompted item |
| Works in the live demo | No — collapses onto a few classes | Yes, for the session it trained on | Yes — in a browser, on someone else's machine |

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
│   ├── static_aug.py            the training-side geometry filter and landmark jitter
│   ├── collect_motion.py        recorder for both static letters and J/Z gestures
│   ├── train_static.py          ships model_static.p; characterizes, never selects
│   ├── crossval_static.py       leave-one-session-out — the number the README quotes
│   ├── label_events.py          cuts the recorded takes into motion events with the runtime's own segmenter
│   ├── train_motion.py          ships model_motion.p; GroupKFold by prompted item
│   ├── evaluate.py              the motion numbers per session, each row labeled in-sample or held-out from the pickle's record
│   ├── train_digits.py          ships model_digits.p; leave-signer-out on the public digits set
│   ├── ingest_images.py         photos -> landmarks; how digits_ankara.npz was made
│   ├── simulate_words.py, make_oof.py   the word layer measured offline on held-out posteriors
│   ├── calibrate.py, live_demo.py       thresholds from your camera; the desktop demo
│   ├── static_sequences.npz, static_s2-4.npz   the four letter sessions, as landmarks
│   ├── static_sequences_tasks.npz       the archive re-extracted with the browser's landmarker
│   ├── motion_clips.npz, events.npz     the five prompted takes and the events cut from them
│   ├── digits_ankara.npz        landmarks of 1,805 public digit photos (no images)
│   ├── model_static.p, model_motion.p, model_digits.p
│   └── tests/                   six plain-script test files
└── docs/                        the hosted demo, served by GitHub Pages. See docs/README.md.
    ├── index.html, app.js       camera, overlay, live readout, the letters/numbers toggle
    ├── features.js, forest.js, segmenter.js   line-for-line ports of the Python
    ├── words.js, words.txt, build_words.py    the word layer and its 34,702-entry frequency list
    ├── export_models.py         flattens the three forests to models.json and writes golden.json
    ├── golden.json              Python's answers for 41 real frames; the page checks itself on load
    ├── test_*.mjs               five Node test files
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

## Approach C — motion letters, measured across sessions

`temporal/` addresses both of the problems described above. It covers all 26 letters, and it measures itself by holding out an entire recording session rather than 20% of a single burst. The full write-up is in [`temporal/README.md`](temporal/README.md).

| | result | how it was split |
| --- | --- | --- |
| **Static letters, leave-one-session-out** | **0.861** (4,199 of 4,878 held-out frames; 95% CI [0.79, 0.92] over the 57 session-by-letter bursts; 3 seeds 0.863 ± 0.004) | `temporal/crossval_static.py`: train on three sessions, test on the fourth, every held-out frame counted once |
| — the cross-day fold, all 24 letters | 0.763 (macro per-letter recall 0.762; 3 seeds 0.763 ± 0.001) | hold out the 2,378-frame November 2024 archive, train on the three September 2026 sessions |
| — previous release, same folds | 0.759 / 0.659 | `crossval_static.py --legacy` reproduces both to the third decimal |
| Motion letters {J, Z, MOVE} | 0.951; J+Z recall 0.946 at the operating point; 2 false J in 28 MOVE events | `GroupKFold(5)` by prompted item: 102 events (74 J/Z gestures and 28 movements) in 80 items. The label set changed in this release, so this replaces the earlier 0.864 rather than improving on it |
| Launch gate on held `I` | 100/100; 13 of 2,278 other frames pass, all of them Y | committed archive; a Y held still and then moved could arm a track that no negative example resembles, and no such footage exists |
| Segmenter over 157 s of held signs | every one of the 24 letters emitted exactly once, 0 track starts | committed archive, in-sample for the static forest |
| Segmenter on the 71 held-out holds | exactly the right letter 65/71; 19/24 on the cross-day fold | fold models, real timestamps, one fresh segmenter per hold |
| A relaxed hand, 43 idle holds from a live browser log | 0 emissions at the shipped thresholds | one signer, one stretch of about two minutes |

The spread between folds is more informative than the average. Two of the four sessions are short re-recordings covering only six and four letters, and the fold that tests four well-separated letters scores 1.000. Averaging the folds evenly would let that one carry the result, which is the same problem I describe in Approach B's 100.00% above. The pooled figure counts every held-out frame once. The three later sessions were all recorded on the same evening, 2026-09-10, about two and a half hours apart (S2 and S3 were first committed at 18:40 in `ce4c26c`, S4 at 21:14 in `8be6a85`), so the November 2024 archive is the only fold that tests a different day, and it is the only fold that tests all 24 letters; 0.76 is the number to plan around. On that fold five letters fall below 0.6 recall: M 0.00, E 0.12, N 0.20, S 0.54, O 0.55, and M, N, S and O stayed below 0.6 for every recipe I tried. The archive's M, N and S were recorded with the thumb where the textbook handshape does not put it, so that is a data ceiling and re-recording those letters is the only fix.

### The feature

Every frame becomes a vector of 112 values: the 21 landmarks centered on the palm and divided by palm width (42 values), the distance between every pair of fingertips, knuckles and the wrist (55), a straightness ratio for each of the four fingers, and an 11-value thumb block. The pairwise distances matter because a decision tree splits on one value at a time, so a question such as "how far apart are these two fingertips", which is the whole difference between U and V, would otherwise take a long chain of splits to express. Dividing by palm width also fixes the largest weakness of Approach B, because the features no longer change when the hand moves closer to or further from the camera.

The thumb block is new in this release: how straight the thumb is, how far each fingertip sits from the palm center, and the distance from the thumb tip to the index and middle knuckles. It exists because the letters the first 101 values confuse are the fists (A, E, M, N, S, T) and D against X, which differ only in where the thumb sits, and the fists keep every fingertip curled in almost the same place, so the thumb has to be measured against the knuckles rather than the tips. Measured leave-one-session-out with the same augmentation, it adds about +0.02 on the cross-day fold and +0.01 pooled. That is within one seed spread, so it is kept for the geometry it encodes and the consistent direction of the effect, not as a large win. The feature is tagged `static/v4` in the pickles and in `models.json`, and every consumer picks the transform by that tag; the previous 101-value feature is still `static/v3`.

### Augmentation, and a filter on the training frames

Two things happen to a training frame and never to a test frame. First, every kept frame gets four jittered copies: each landmark coordinate is perturbed by Gaussian noise with a standard deviation of 0.12 palm widths, so the noise scales with the hand's distance from the camera and the vector is recomputed from the moved landmarks. This replaces the four-angle rotation augmentation of the previous release. Rotation had lowered leave-one-session-out accuracy, 0.782 without it against 0.759 with it, and had only ever been justified by within-session confidence. Measured at seed 0 with everything else as shipped, the jitter adds +0.03 pooled and +0.07 on the cross-day fold on top of the geometry filter described next (0.830 / 0.690 to 0.861 / 0.763), and +0.09 / +0.14 without the filter (0.781 / 0.650 to 0.874 / 0.790); it is the single largest gain in this release, but the +0.10 / +0.10 over the previous release is the whole recipe of feature, filter, jitter and forest, not the jitter alone. Jitter without the filter scores higher on this protocol; the filter is kept because P recall stays at 1.00 under the page's own landmarker with it and because D stops being read as X. Second, a training frame that violates its own letter's defining geometry is dropped, for nine letters whose off-rule frames sit inside a neighbor's shape: X, G, Q, U, V, K, R, P and D. That removes 137 of the 4,878 frames. The rules were written by inspecting the archive before any cross-session run, so the filter is not tuned to the folds it is measured on. It deliberately un-learns the archive's straight-index X (23 frames) and its together-fingers V, which is why in-sample recall on those two bursts falls; on the held-out folds D stops being read as X. The fists are not filtered on purpose: the archive's M, N and S violate their own rules on 47-100% of frames, so filtering them would delete the letters rather than clean them, and the pooled number fell when I tried.

The forest itself is 100 trees with at least five samples per leaf, 133,680 nodes, trained on 23,705 rows. A 400-tree forest with pure leaves scores the same at several times the size, and every node ships to the browser inside `models.json`.

### Deciding when a static letter has been signed

A letter is emitted once, when a stable hold is first established, from a vote over its first four frames. The vote passes by either of two routes: confident (mean winner probability at least 0.70) or decisive (margin over the runner-up at least 0.20 and mean probability at least 0.55). The floor is the knob that separates a held letter from a relaxed hand under the jitter-trained forest, and it was set against a live browser log rather than against the training data. One session of about two minutes of my own hand produced 148 holds; 43 of them are clean idle holds the page was right to stay silent on (holds within 2.5 s before or 1 s after a J/Z track are excluded, because those are the I and D launch poses, read correctly and canceled by the motion branch). At 0.55 the shipped forest emits on 0 of those 43 holds and 0 of their 2,064 votes. At 0.50, the value the sweep first chose, this particular forest emitted on one of them, six Q votes at a mean probability of 0.509, which is why the floor moved. It now sits 0.041 above this forest's most confident idle vote, so a different relaxed pose or camera could cross it; this is measured on one signer and needs re-measuring after the next live session.

The cost, on the 71 held-out holds of the four sessions: the first four-frame vote is correct on 56 of 71 and wrong on 2. Run end to end through the segmenter with real timestamps, exactly the right letter comes out on 65 of 71 holds, the first emission is wrong on 3, two holds stay silent (S1-E and S2-K), and the median latency from the hand settling to the letter appearing is 0.46 s, p90 0.96 s; on the cross-day fold 19 of 24. Of the 79 holds the page emitted on in that live session, 59 (0.747) still emit at the new floor. Live confidences run lower than the leave-one-session-out ones, and that 59/79 is the number to watch.

### Deciding when a motion letter has begun

Six inequalities on finger geometry decide whether the hand is currently in the launch pose for J or for Z. A gesture is tracked only if the hand was held still in that pose and then started moving, so the ordinary movement between letters almost never begins a track. I kept these as explicit inequalities rather than training a second model, because I can watch each condition on the live overlay and see which one failed. A model asked about an input unlike anything in its training set tends instead to answer confidently and incorrectly, which is the failure mode Approach A demonstrates.

The J gate's thumb ceiling moved from 1.20 to 1.30 palm widths in this release, because at 1.20 only 41 of the 60 J items in the prompted takes produced a creditable event; at 1.30, 48 do, and 12 still have none (1.25 buys only 3 items, 1.35 two more for one more Y frame). At 1.30 it passes 100 of 100 held-I frames in the archive and 13 of the 2,278 frames of every other letter, all of them Y; replaying the whole 157 s archive through the segmenter starts 0 tracks from them, because arming also needs a parked frame and a sustained rising edge. A Y held still and then moved could arm a track that no negative example resembles; no such footage exists yet.

### Ratios rather than lengths

A finger pointing toward the camera appears shorter than it is, so a distance measured in the image changes for reasons that have nothing to do with the handshape. Straightness is the distance from fingertip to knuckle divided by the summed length of the bones. Because it is a ratio, it stays stable under that foreshortening.

### Labeling the motion events

The motion forest is trained on events the segmenter itself cuts from five prompted takes (113 items: 60 J and 53 Z, each parked, signed and rested on a fixed rhythm), so every training example has boundaries the runtime can produce. One event per prompted item is credited as that item's letter; every other span the runtime could have scored, a false start inside an item or a move during the rest, is labeled MOVE; spans the runtime could never score, aborted tracks and spans the vetoes reject, are dropped rather than taught as MOVE. That gives 117 events (J 48, Z 37, MOVE 32) in 92 items, and the cross-validation folds by item, so a fragment of a test gesture can never sit in training. The previous release labeled fragments as gestures and reported 0.864 over 140 events; those numbers are not comparable with the ones above. Two rules tie the branches together, because the launch pose of a motion letter is itself a static letter. A parked I or D whose own pose arms a track, or is still confirming its rising edge when the timer runs out, is held until the track resolves: a J or Z emission cancels it and an abort releases it. An I held longer than the 0.35 s timer before the stroke begins is still released by design, because deferring every I and D to the end of its hold was measured and rejected for latency; on the takes that leaves 2 "IJ" and 3 "IJI" among the 60 J items, down from 38 releases before the rule. And after a J or Z is emitted the static branch stays quiet while the hand rests in the finishing pose, clearing as soon as the hand reshapes or moves again, so a hand that morphs straight into the next letter is not delayed. One more rule turned out to be needed for the live path to match training: MediaPipe labels handedness per frame, and 166 of the 13,692 detected frames on the takes carry the other hand's label for a frame or two, which mirrored the hand mid-stroke and aborted the track. The segmenter now keeps the label it has and switches only after the other label has persisted for 0.50 s (`HAND_SWITCH_S`; the longest such flip inside an unbroken detection is 0.20 s), clearing on a detection gap, which is how a signer changes hands anyway. Replaying the five takes with the retrained forest, which is in-sample for it, feeding the per-frame label the page and `live_demo.py` feed: 85 of 113 items produce their letter, up from 78, doubles fall from 9 to 0, and there are no J or Z emissions during the rest phases over 5.5 minutes. Without the latch the same replay credits 79; with it, every one of the 113 item strings is identical to a replay with one modal label per take, which is how the training events are cut, and `temporal/evaluate.py` reproduces the count directly (S1 74 of 90 plus S5 11 of 23). The 28 misses are segmentation, the gate never armed, no rising edge, a veto or a detection gap, and none of the fixes address them.

### What the development actually cost

Every letter I found broken while building this turned out to be either an inconsistency in my own recordings or a threshold I had chosen before I had the data to choose it. In one session every `G` frame had an extended middle finger, which makes the sign an `H`; those frames outvoted the correct ones, and `G` was read as `H` everywhere. `T_MAX` was set below the 95th percentile of my own recorded Z durations, which cut genuine gestures out of the range the classifier had been fitted on. `P_EMIT` was 0.70, which quietly discarded about one real gesture in three. The vote floor in this release is the same lesson from the other side: a forest trained with jitter is more confident on a relaxed hand than the old one was, and the floor that was harmless before let it emit on a third of idle holds. I found each of these by logging what the running system saw, and none of them by reading the code.

### The browser version

`docs/` is a direct port of the Python: the same feature code, the same thresholds, and the three forests flattened into JSON. Before the camera starts, the page runs 41 real frames through the models it just loaded (15 from the training landmarker, 11 from the landmarker the page itself runs, 15 digits), compares the results with the answers Python gives for those same frames, and reports the outcome in the readout. A silent difference between the two implementations would look exactly like a poor model, so it is worth checking on every load.

The browser condition differs from training in one way that had to be measured rather than assumed. The page runs MediaPipe's Tasks API, while every training landmark came from the legacy `solutions` API, and the two disagree about what "Left" means on the same unflipped frame; the page swaps the label before the mirror step. On the archive re-extracted with the Tasks API, the cross-day model scores 0.775 with the swap and 0.730 without it, so the swap is verified rather than assumed. With it, the gap between the two landmarkers on the cross-day fold is +0.001 over three seeds, with a paired 95% interval of about ±0.03, so there is no browser penalty larger than that. One caveat on both numbers: the re-extraction ran the CPU build of the Tasks landmarker from Python in VIDEO mode, not the GPU delegate the page runs, so the landmarks the page actually produces were not the ones measured. Input resolution from 1920x1080 down to 426x240 moves the same fold by no more than 0.004; shrinking the hand to the palm size the browser log typically shows costs 0.02, and below that the loss is MediaPipe failing to find the hand at all rather than the forest misreading it. The shrink is synthetic (the archive frames scaled onto a gray canvas), not real distant captures.

### Numbers mode

The page has an experimental numbers mode for the digits 0-9. It is a separate 100-tree forest on the same 112-value feature, trained on the Sign Language Digits Dataset: landmarks from 1,805 photographs of 218 students' hands (MediaPipe found a hand in 1,805 of the 2,062 photos; no image is redistributed here). Leave-signer-out on that dataset it scores 0.986, with digit 6 the worst at 0.965. It is trained on 218 signers from a public dataset, not on the author, and it has not been verified on the author's hand or on live video. The only evidence that it reads my hand at all is a proxy: my O, V, W, F and B letter frames read as 0, 2, 6, 9 and 4 on 0.956 of frames. It is weak evidence, because my C, R, X and U frames also read as 0, 2, 1 and 2 just as confidently (C as 0 on 1.00 of frames, R and U as 2 on 0.90, X as 1 on 0.99, emitting at the digits gate on 0.83-0.93 of them), so the forest has a digit for many handshapes that are not one. In this mode the J/Z gates are off (the '1' handshape passes the Z gate and would otherwise park the machine in a track), the word layer is off, and the two vote constants are raised to 0.40 / 0.60, because a relaxed hand reads as a digit, mostly 0 or 1 (57% and 40% of the emissions), on about one frame in six (0.162, the single-frame rate over 2,890 hold records forming 148 holds from the letters-mode browser log, where any digit emission is a false one); duplicate suppression bounds that to one spurious digit per hand-raise. Recording my own digits is the first follow-up.

### Words

The page groups letters into words and, when it can, names the word. A word ends after 1.20 s with no hand in the frame. At the break, every listed word of the same length is scored letter by letter against the votes the classifier actually produced, with a prior that favors common words, and the best one is shown under the reading only if it clears three bars: it must be at least 0.10 as likely as the letters read, at least 10 times likelier than the runner-up, and the prior's weight is 2.5. The list is 34,702 words in frequency order, the most frequent 40,000 of the Google Books counts distributed by norvig.com plus a set of proper names, which is what puts HENRY in it. The three constants were chosen offline by `temporal/simulate_words.py`, which spells words out of held-out letter posteriors through the same vote gate the page runs; its `--sweep` default is the 112-cell grid they came from, run at the 0.20 / 0.50 gate the vote floor was later raised from, and no single cell won every table, so the shipped triple is the one within 0.015 of each table's best. At the shipped gate (0.20 / 0.55) it recovers 0.44 of common words and shows a wrong one 0.07 of the time when a misread letter is retried from consecutive windows of the same hold, and 0.63 / 0.06 when every retry is a fresh hold (0.48 / 0.07 and 0.68 / 0.05 at the 0.50 floor the sweep ran at); the two retry models bracket what a signer does, and this is a simulator of the segmenter, not a recording of spelled words. The previous layer, a 150,594-entry dictionary with no frequencies, measured 0.445 / 0.158 and 0.612 / 0.091 under the same two models on its own forest and gate. The reading itself is never rewritten.

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
./.venv/bin/python temporal/train_static.py --extra static_s2.npz static_s3.npz static_s4.npz
./.venv/bin/python temporal/train_digits.py
./.venv/bin/python temporal/crossval_static.py --seeds 3            # the numbers this README quotes
./.venv/bin/python temporal/label_events.py                         # read its diagnostic before trusting the next step
./.venv/bin/python temporal/train_motion.py
./.venv/bin/python temporal/evaluate.py                             # motion numbers per session; in-sample with the shipped forest, and it says so
./.venv/bin/python docs/export_models.py                            # -> docs/models.json, docs/golden.json
```

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

- **The letters are still one signer.** Four sessions, one hand. Nothing here tells you how the letter models behave on a different person's hands, and everyone who opens the hosted demo is a different person. The cross-signer letter sets listed in `temporal/SOURCES.md` are the next honest test.
- **The digits are the opposite: trained on 218 strangers and never verified on me.** No frame of my own hand signing a digit exists; the 0.956 proxy is my letter frames whose handshape is a digit. Recording my own digits is the first follow-up, and until then the numbers mode is a demonstration that a forest trained on other people's hands runs in the browser, not a verified feature.
- **Three of the four letter sessions were recorded on the same evening, about two and a half hours apart**, so the 2024 archive is the only cross-day fold and the pooled figure is dominated by same-evening folds. 0.76, not 0.86, is the number to plan around.
- **The archive's M, N, S and O recordings violate the textbook handshape**, which is why those letters stay below 0.6 recall on the cross-day fold for every model tried. That is a data ceiling; re-recording them is the fix, and no threshold will do it.
- **The idle floor has 0.04 of headroom.** The vote floor sits 0.041 above the most confident vote this forest produced on 43 idle holds from one two-minute session of one signer; a different relaxed pose or camera could cross it. Re-measure after the next live session.
- **A relaxed hand reads as a digit about one frame in six in numbers mode**, bounded to one spurious digit per hand-raise by duplicate suppression, which is why that mode is off by default and labeled experimental.

---

## Credits and provenance

Approach A trains on **Sign Language MNIST**, a public dataset. Hand detection and 21-keypoint extraction in all three approaches use **Google's MediaPipe Hands** (the legacy `solutions` API in Python, the Tasks API in the browser).

The numbers mode is the one part of this repository not trained on my own hand. Its forest is trained on the **Sign Language Digits Dataset** by the students of Turkey Ankara Ayrancı Anadolu High School (project executives Zeynep Dikle and Arda Mavi), <https://github.com/ardamavi/Sign-Language-Digits-Dataset>, Apache-2.0: 2,062 100x100 photographs, one per digit per student for 218 students. Cite as Mavi, A. (2020), *A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits*, arXiv:2011.08927. What this repository takes from it is landmarks only: `temporal/ingest_images.py` runs MediaPipe over each photograph and keeps the 21 landmarks, the handedness label and the frame size for the 1,805 images where it found a hand, committed as `temporal/digits_ankara.npz` (0.43 MB). No image from the dataset is redistributed here.

The word list behind the page's hint, `docs/words.txt`, is built by `docs/build_words.py` from Peter Norvig's table of **Google Books Ngram** 1-gram counts (<https://norvig.com/mayzner.html>; the Ngram data is CC BY 3.0) plus macOS's `/usr/share/dict/propernames`.

Everything else here is my own work: the 2,400-frame dataset in `RandomForest/data/` — captured, labeled and curated by me — the four letter sessions and five prompted J/Z takes behind `temporal/`, the capture, training and inference code in all three approaches, the browser port, and the screenshots above.

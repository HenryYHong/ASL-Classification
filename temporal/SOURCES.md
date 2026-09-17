# External data sources

Why bother: the repository's README states that nothing in it supports a claim about a signer
other than the one who recorded it. External footage is the only way to change that. For the
letters it is a **test set**, not a training substitute — the training distribution should match
your camera and the runtime's parked-then-move assumption. The one exception is the numbers
mode, which is trained entirely on external data (section 5) and is labeled experimental for
exactly that reason.

Everything ingested is tagged with a non-`S1` session, so the by-session split in
`train_motion.py` and `evaluate.py` holds it out by construction rather than by remembering to.

## Reading a Kaggle dataset's license and size without an account

Kaggle renders its dataset pages client-side, so `curl` on the page returns a shell with the
title and a one-line description and nothing else. The Croissant metadata endpoint is served as
plain JSON without a login, and it carries the license, the archive size and its md5:

```
curl -sL https://www.kaggle.com/datasets/<owner>/<slug>/croissant/download | python3 -m json.tool
```

Every license and size below was read that way. Downloading still needs the one-time auth.

## One-time Kaggle auth

1. <https://www.kaggle.com/settings/account> → **Create New API Token** → downloads `kaggle.json`
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

The CLI is already installed in this project's venv (`./.venv/bin/kaggle`).

## 1. ASL Alphabet Videos [J, Z] — the direct hit

<https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z>

Video, scoped to the two motion letters specifically. The only source found that targets exactly
the gap this feature exists to fill. **CC0 (public domain), one 3.879 GB zip of `.avi` files**,
per its Croissant record; the signer count is not in the metadata and has to be read off the
files.

```
./.venv/bin/kaggle datasets download -d signnteam/asl-sign-language-alphabet-videos-j-z \
    -p ~/Downloads/asl_jz --unzip
./.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/asl_jz/J \
    --label J --signer kaggle_jz --session EXT
./.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/asl_jz/Z \
    --label Z --signer kaggle_jz --session EXT
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz
```

Not yet ingested. Check two things when it lands, because they decide whether it is worth
anything:

- **distinct signers** — one signer adds little over twelve minutes of your own recording; many
  signers makes it the cross-signer test set. Give each signer a distinct `--signer` id, or the
  cross-signer claim is not a cross-signer claim.
- **directory layout** — the commands above assume `J/` and `Z/` subdirectories; adjust.

Video files carry width and height in the container, so `u = x*(W/H)` applies cleanly. That is a
real advantage over the landmark-only sources below.

## 2. Google ASL Fingerspelling Recognition — best format, one blocker

<https://www.kaggle.com/competitions/asl-fingerspelling>

MediaPipe landmarks directly, as parquet: x/y/z for 543 landmarks including 21 per hand, `NaN`
for missing frames — the same convention `collect_motion.py` writes. Over 100 Deaf signers,
fingerspelling at real-world speeds. On paper the best fit of anything available.

Accept the competition rules on the website first, then:

```
./.venv/bin/kaggle competitions download -c asl-fingerspelling -p ~/Downloads/asl_fs
```

**`ingest_external.py` deliberately does not implement this format.** The blocker: the parquet
may not record the source video's frame width and height, and without them the aspect correction
cannot be applied. On this repository's own archive, skipping that correction triples the Z-gate
false-positive rate (118 → 367 frames) and makes `J_GATE` start firing on Y. Shipping a loader
that silently assumes 16:9 would bury that error under a plausible number.

If a frame size *is* recoverable from the metadata, the loader is perhaps thirty lines: read the
parquet, take the 21 `right_hand`/`left_hand` landmarks, and write them in the schema
`ingest_external.py` already emits. Check first.

Labels are phrase-level strings, not per-letter timing — so this feeds the letter-error-rate
metric in `evaluate.py`, not isolated J/Z training.

## 3. ChicagoFSWild / ChicagoFSWild+ — large, free, low quality

<https://home.ttic.edu/~klivescu/ChicagoFSWild.htm>

7,304 sequences from 160 signers; the `+` variant has 55,232 from 260. JPEG frame sequences,
direct download, no access form — but **14 GB and 82 GB**. Scraped from YouTube and Deaf social
media, so expect motion blur and awkward framing; MediaPipe will drop a meaningful fraction of
frames, and `ingest_external.py` rejects any clip tracking under 60%.

```
./.venv/bin/python temporal/ingest_external.py --frames ~/Downloads/fswild/sequences \
    --label NONE --signer fswild --session EXT --fps 30
```

Labels are whole-sequence letter strings. Its value here is negatives and the LER metric.

## 4. YouTube — highest yield per unit of effort

Thousands of "ASL alphabet" tutorials, each a distinct signer demonstrating letters slowly and
deliberately — and typically *parked, then moving*, which is exactly what the segmenter's
rising-edge trigger wants. Ten videos is ten signers.

```
yt-dlp -f 'bv*[height<=720]' -o '~/Downloads/yt/%(id)s.%(ext)s' <url>
./.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/yt \
    --label J --signer yt_<id> --session EXT
```

Trim to the J and Z portions first, or ingest whole and let `label_events.py` cut candidates. Use
a distinct `--signer` per video. Check licensing before redistributing anything; for a local test
set this is the cheapest route to signer diversity.

## 5. Sign Language Digits Dataset — ingested; it is the numbers mode

<https://github.com/ardamavi/Sign-Language-Digits-Dataset> — by the students of Turkey Ankara
Ayrancı Anadolu High School (project executives Zeynep Dikle and Arda Mavi), Apache-2.0 on
GitHub; the Kaggle mirror (`ardamavi/sign-language-digits-dataset`, 16.8 MB) lists CC BY-SA 4.0,
so the GitHub repository is the copy this project used and cites. 2,062 photos, 218 students,
one photo per digit per student, 100x100 RGB (three stray 3024x3024 originals in `7/`, same
aspect). Cite as Mavi, A. (2020), *A New Dataset and Proposed Convolutional Neural Network
Architecture for Classification of American Sign Language Digits*, arXiv:2011.08927. A shallow
clone is about 15 MB to download (a 15.1 MiB pack; the Kaggle mirror lists 16.8 MB for the same
images) and about 45 MB on disk after checkout.

```
git clone --depth 1 https://github.com/ardamavi/Sign-Language-Digits-Dataset ~/Downloads/ardamavi
./.venv/bin/python temporal/ingest_images.py --root ~/Downloads/ardamavi/Dataset \
    --session EXTD --out temporal/digits_ankara.npz
./.venv/bin/python temporal/train_digits.py          # -> temporal/model_digits.p
```

What was derived, and where it lives: `temporal/digits_ankara.npz` (0.43 MB) holds, for the
1,805 photos where MediaPipe found a hand (87.5%; upscaling to 400x400 first found fewer, 1,589,
so the ingest runs at native size), the 21 raw landmarks, the handedness label and score, the
per-image frame size, the class label, the file name and a signer id. **No photograph is
redistributed**, and the landmarks cannot be turned back into one. `ingest_images.py` runs
MediaPipe exactly as `extract_static_sequences.py` runs it over my own archive (static image
mode, one hand, detection confidence 0.3, BGR→RGB, never flipped), so the landmarks land in the
same convention as every other training frame. Handedness is 'Left' on 1,793 of the 1,805 images
(the photos are of the signer's right hand, unmirrored, the same convention as my own captures),
and `canonicalize_handedness` mirrors them per image.

The signer id is derived from the image numbering by **runs, not by `IMG // 10`**: each student's
ten photos are consecutive IMG numbers in digit order 0..9 with a drifting offset, so a student is
a maximal run of images whose numbers step by at most 2 and whose labels strictly increase. That
gives 224 runs over all images (186 of exactly ten) and 222 among the detected ones. The simpler
`IMG // 10` rule mixes two students in 187 of its 219 groups, and a by-group split under it leaks
every student's other digits into the training fold. The leave-signer-out number happened not to
move (0.986 either way), but only the run grouping is a signer split; `train_digits.py` folds by
whatever signer id the npz carries and asserts no signer sits on both sides of a fold, so the id
has to be right at ingest time (`--signer-rule runs` is the default; `imgnum10`, `file` and
`none` exist for other sets and for the comparison).

## 6. Google Books word counts — the word list's frequency prior

<https://norvig.com/mayzner.html> → `https://norvig.com/google-books-common-words.txt`

Peter Norvig's distillation of the Google Books Ngram English 1-grams (version 20120701): the
97,565 distinct a-z words with at least 100,000 mentions, one `WORD<TAB>COUNT` per line, sorted
by count. The Google Books Ngram data is published under CC BY 3.0; Norvig's file is a derived
table of it. The 1.5 MB source file is not committed and the script does not fetch it: download
it into `docs/` with

```
cd docs && curl -L -O https://norvig.com/google-books-common-words.txt
```

(`-L` matters: norvig.com redirects to www.norvig.com, and without it `curl` saves a 795-byte
"301 Moved Permanently" page instead of the 97,565-line table), then run `docs/build_words.py`,
which rebuilds `docs/words.txt` (34,702 entries, byte-identical on every rebuild — the docstring
carries the md5) from its top 40,000 entries plus macOS's `/usr/share/dict/propernames`.
`temporal/simulate_words.py` reads the same file, only to draw the target words it spells.

## The archive re-extracted with the browser's landmarker

`temporal/static_sequences_tasks.npz` (0.69 MB) is not external data but it is a derived file
worth documenting here: the S1 archive (`RandomForest/data/<class>/<i>.jpg`, 1920x1080, BGR→RGB,
never flipped) re-extracted with the MediaPipe Tasks `HandLandmarker` the hosted page runs —
the Python CPU build in running mode VIDEO, not the GPU delegate the page itself uses, one hand,
detection confidence 0.5, presence confidence 0.5, tracking confidence 0.3, the float16
`hand_landmarker.task` (7.8 MB), one fresh landmarker per letter burst (treated as a new video),
frame timestamps from the JPEG mtimes in milliseconds made strictly increasing. Keys: `lm`
(24,100,21,3) raw normalized landmarks, `found` (24,100), `handed` (24,100) and `hscore`
(24,100). 2,387 found frames (V has 87). Every found frame is
labeled 'Right', where the training landmarker labeled the same frames 'Left': the page swaps the
label before canonicalizing, and so must anything that builds cases from this file.
`docs/export_models.py` uses it to build the 11 Tasks-API golden cases without a landmarker, and
the browser-condition rows in the READMEs (gap +0.001 over three seeds on the cross-day fold;
0.775 with the swap, 0.730 without) were measured on it, so they carry the same caveat: CPU
build, VIDEO mode, not the page's GPU delegate. It exists only for S1 because S1 is the only
session with raw frames.

## 7. ASLNow — other people's letters through the browser's own landmarker; ingested

<https://huggingface.co/datasets/sid220/asl-now-fingerspelling> (MIT), the training data of the
ASLNow! fingerspelling web app: 2,122 JSON records, one per capture, each the 21 hand landmarks
MediaPipe's **Web Hand Landmarker** returned for a participant signing a letter into a webcam.
"Collected from multiple participants" is all the source says about who; there is no participant
id, no frame size and no handedness label. It is the one public source found that matches the
hosted page's condition — the Tasks-API landmarker, on other people's hands, in other people's
rooms — and it is what turned "one signer" into a measured number for the static letters.

```
./.venv/bin/python temporal/ingest_aslnow.py            # downloads ~8 MB of JSON -> temporal/aslnow.npz
```

What `ingest_aslnow.py` does with the two missing fields, so the decisions can be revisited:

- **Frame size.** x is normalized by width and y by height, so u = x * (W/H) needs the capture
  size. The palm triangle settles it: over the upright palm-forward letters the palm's
  width-to-height ratio is 0.755 on this project's own frames, and the records give 0.728 at
  4:3, 0.569 square, 0.932 at 16:9. Web apps ask for 640x480; 4:3 is stored as `aspect`.
- **Handedness.** About half the records (56%) are the mirror image of this project's canonical
  frame — left-handed participants, a mirrored video feed in some sessions, or both; for
  training it does not matter which. A forest trained on this project's four sessions, canonical
  frames labeled 0 and their mirror images 1, decides per record (`mirror`, with its
  probability in `mirror_p`); it separates the two on 9,752 of 9,756 held-out frames of the
  author's and is confident on 95% of the records. The decision is reproducible from the
  committed sessions (`tests/test_strangers.py` checks it).
- **J and Z.** 248 records are stills labeled J or Z; a still of a J is an I, and the static
  forest has no J or Z class, so they are stored and never trained on.
- **Participants.** No ids, and hand proportions do not cluster into people (the pose dominates
  every bone-length ratio), so the set cannot be split by signer. It is not bursts either: the
  nearest same-letter neighbor sits at a median 0.31 palm units in shape space, against 0.07
  inside one of the author's own held bursts, so every record is a separate capture.

How it is used (`strangers.py`, `crossval_strangers.py`): held out entirely as the cross-signer
test set — a forest trained on the author's sessions and the digit photos, which never saw it,
reads 0.790 of its 1,874 letter records — and as training data for the shipped forest, which
lifts the author's own cross-day fold from 0.778 to 0.873 and the 218-signer V from 0.17 to 0.83.
With it in training, its five-fold number (0.946) is an upper bound, since a participant may sit
on both sides of a fold.

The **Sign Language Digits Dataset** (section 5) is the other stranger set for the letters: its
0, 2, 6 and 9 are O, V, W and F exactly, so 687 of its photographs are letter frames from 218
more hands. `strangers.load_ankara_letters` maps them.

## Still-image letter sets: the next honest test for the letters

Two more public sets cover the 24 static letters with more than one signer, and this file used to
dismiss them for lacking J and Z. That was the wrong reason. ASLNow (above) gave the letters
their first cross-signer number, but it is one frame per record with no participant id; a set
with known signers, or with video, would give a by-signer split and a held-sign replay on
strangers, which nothing here has. They are not useful for the motion branch, but they are the
next test worth running for the letters, through the same `ingest_images.py` path the digits
used — with a signer rule written for their file layout, since `runs` encodes the Ankara set's
numbering.

- **ASL Fingerspelling A / B** (Pugeault and Bowden, "Spelling It Out"): A is 131k images of
  24 letters from 5 signers, B is 9 signers. Still images, so each is its own hold. A Kaggle
  mirror, `mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out`, is 2.1 GB and states no
  license in its Croissant record, so read the original's terms first.
- **ASL-HG** (<https://data.mendeley.com/datasets/j4y5w2c8w9/1>, CC BY 4.0): 36,000 smartphone
  photos across 36 classes — A-Z and 0-9 — from 10 volunteers, 100 per class per person, indoor
  and outdoor. Its 0 is the two-handed sign, which this project cannot read, and its J and Z are
  single stills, which are mislabeled I and D for this project's purposes; the other 34 classes
  are usable, and the ten signers are known, so a by-signer split is possible.

Sign Language MNIST is still not useful: 28x28 crops carry no landmarks to extract. Kaggle's
29-class ASL Alphabet sets include J and Z as single still images, which is worse than omitting
them. `MSL-AlphaVid` (Malayalam) and `AzSLD` (Azerbaijani) have dynamic letters but different
handshapes; the geometry does not transfer.

## Always run this before trusting a download

```
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz
```

If J and Z clips cut zero events, the footage begins mid-gesture and never arms a gate — the
runtime would miss it too. Ingesting ten clips and reading that diagnostic costs a minute and
tells you whether the remaining gigabytes are worth downloading.

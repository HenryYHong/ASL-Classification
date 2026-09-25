# External data sources

Why bother: the README used to say that nothing in this repository supported a claim about a
signer other than the one who recorded it. External footage is what changed that, and this file
used to say it was for testing only. That is no longer true. For the **letters** these sets are
not only a test set: the shipped forest trains on 8,561 frames of them (section 5's
O/V/W/F, section 7, section 8) and is measured on them with each set held out in turn, plus one
set (section 9) nothing may ever train on and one runtime replay on video (section 10). For the
**motion** branch they are still a test set and a thin one; every J and Z event the forest is
fitted on is the author's.

Each section below says which side of the line its set is on, because that is the thing that
goes stale. A set that crosses onto the training side turns its own "never seen" figure into an
in-sample one, and every number taken from it before the crossing has to be relabeled
historical. Section 9 exists to be the one set that can never do that.

Everything ingested is tagged with a non-`S1` session, so the by-session split in
`train_motion.py` and `evaluate.py` holds it out by construction rather than by remembering to.

## Reading a Kaggle dataset's license and size without an account

Kaggle renders its dataset pages client-side, so `curl` on the page returns a shell with the
title and a one-line description and nothing else. The Croissant metadata endpoint is served as
plain JSON without a login, and it carries the license, the archive size and its md5:

```
curl -sL https://www.kaggle.com/datasets/<owner>/<slug>/croissant/download | python3 -m json.tool
```

Every license and size below was read that way. **Downloading does not need the auth either**,
and the older claim here that it did was wrong. A plain
`GET https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` answers 302 with a signed
`storage.googleapis.com` URL, and following the redirect returns the zip with no credentials:
verified on `ayuraj/asl-dataset`, which came back as 59,642,568 B of valid zip
(`ingest_ayuraj.py` does exactly this), and re-checked on 2026-09-23 with a ranged request, which
answered 206 `application/zip`. `HEAD` on that same URL answers 404, so an existence check has to
be a GET and a HEAD will tell you a dataset that is there is not. None of this is documented
anywhere on Kaggle, which means it can close without notice; the auth section below is kept for
the day it does.

## One-time Kaggle auth (a fallback now, not a requirement)

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
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz \
    --out temporal/external_events.npz
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
a distinct `--signer` per video, and pass `--out`: `label_events.py --out` defaults to
`temporal/events.npz`, the committed motion training set. Running the diagnostic bare on
somebody else's clips used to replace it silently; it now refuses and names the flag, which is a
backstop and not a substitute for typing the path you meant.

**This route has been run.** Three CC BY channels (signer ids `yt_danielparks`,
`yt_inclusivesigntalk`, `yt_lexiemoore`) and one ASL Signbank entry (`signbank_asl`, CC BY-NC-SA),
one J and one Z from each, gave the motion branch its first and so far only cross-signer numbers:
of the eight clips, four cut a span the runtime could score, three of those were J and all three
emitted J, and the one scorable Z span read MOVE (J 0.404 / MOVE 0.423 / Z 0.173). The same
signers' fingerspelling containing no J and no Z — 160.1 s of it — cut five spans and emitted
nothing, so 0.00 false J/Z per minute. Four clips per letter: quote it as a direction, never as a
percentage. Neither the video nor the landmarks are committed; the NC clip could not be, and the
rest are not worth the bytes at this sample size.

**Those numbers are the motion branch alone, and the end-to-end reading is not as clean.**
"J 3 of 3, Z 0 of 4, no wrong letter" counts what the motion classifier did with the spans the
segmenter cut. Replayed end to end through the real `Segmenter` on the shipped forest, the same
eight clips also emit **eight static letters that were not the target** — the readings are `HJ`,
`W`, `IJL`, `YX`, `IJ` and `Y`. Some of that is the documented correct read of a parked launch
pose (the `I` before a J is the pose the hand is actually in, and the bullet on launch poses in
`README.md` covers it), but `H`, `W`, `Y` and `X` are not: they are the static branch reading a
stranger's hand mid-travel. Never publish the motion figure bare. On a stranger's clip the thing
a viewer sees is the transcript, and the transcript has extra letters in it.

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

How it is used (`strangers.py`, `crossval_strangers.py`): **held out entirely** as one of the two
cross-signer test sets — a forest trained on the author's sessions, the digit photos and ASL-HG,
which never saw ASLNow, reads **0.8292 ± 0.0061 over three seeds** of its 1,874 letter records
(0.836 at seed 0), and through the shipped vote gate it emits on 0.75 of them and is right on
0.907 of those — and **as training data** for the shipped forest. Together with the Ankara photos
it lifts the author's own cross-day fold from 0.782 to 0.873, and the 218-signer V goes from 0.19
with the author's sessions alone in training to 0.88 once this set and ASL-HG are added. With it
in training, its own five-fold number (0.936 at seed 0) is an upper bound, since a participant
may sit on both sides of a fold. The previous release's figures on this set were
0.790 held out and 0.946 five-fold, on a forest and a training set that no longer exist; they are
not comparable to the pair above.

The **Sign Language Digits Dataset** (section 5) is the other stranger set for the letters: its
0, 2, 6 and 9 are O, V, W and F exactly, so 687 of its photographs are letter frames from 218
more hands. `strangers.load_ankara_letters` maps them, and held out in turn they read 0.9413 ±
0.0048 over three seeds.

## 8. ASL-HG — ten named signers; ingested, and the first by-signer split

<https://data.mendeley.com/datasets/j4y5w2c8w9/1>, DOI `10.17632/j4y5w2c8w9.1`, **CC BY 4.0**
(the record's `data_licence` reads "You can share, copy and modify this dataset so long as you
give appropriate credit, provide a link to the CC BY license, and indicate if changes were made,
but you may not do so in a way that suggests the rights holder has endorsed you or your use of
the dataset. Note that further permission may be required for any content within the dataset that
is identified as belonging to a third party." — the record identifies no such content).

36,000 smartphone photographs across 36 classes (A-Z and 0-9) from 10 volunteers in Mirpur,
Dhaka, taken May-June 2025, 100 per class per person, indoors and out. This project reads the 24
static-letter folders only: 24,000 images, **23,984 with a hand (99.93%)**, the cleanest external
set here (Ankara is 87.5%, ayuraj 72.3%). The 16 misses are all one signer's H. Its 0 is the
two-handed ASL zero, so the digit-to-letter mapping the Ankara set uses does not apply, and its J
and Z are single stills, which a static classifier reads as I and D.

```
./.venv/bin/python temporal/ingest_aslhg.py        # 877 MB of zip -> temporal/aslhg.npz (5.6 MB)
```

What `ingest_aslhg.py` infers, so the decisions can be revisited (its docstring carries the
measurements):

- **classes.** Only the 24 static letters are read. J and Z are single stills here, and a still
  of a J is an I. The digit folders belong to `train_digits.py`, and this set's 0 is the
  two-handed ASL zero, so the digit-to-letter mapping `strangers.DIGIT_LETTERS` uses for the
  Ankara photos does not apply to it.
- **signer.** The file name is `<CLASS>/P<k>_<CLASS>_<n>.jpg`, so the signer is the prefix
  before the first underscore: P1 to P10, 2,400 frames each, P8 2,384. That rule is the whole
  reason for a second ingester rather than another `--signer-rule` on `ingest_images.py`.
- **frame size.** Every image is stored 300x300, so `u = x * (W/H)` is the identity. That is a
  statement about what MediaPipe normalized by, not a guess that the phone shot square, and the
  palm triangle confirms it: over the upright palm-forward letters the palm's width-to-height
  ratio is 0.755 on the author's frames, and ASL-HG reads 0.719 at 1:1 against 0.913 at 4:3,
  1.165 at 16:9, 0.561 at 3:4 and 0.441 at 9:16 (n = 7,000, the method `ingest_aslnow.py` uses).
- **handedness.** MediaPipe labels 23,974 images 'Left' and 10 'Right', the same unmirrored
  convention as the Ankara photos, so `strangers.load_aslhg` canonicalizes per image by that
  label rather than inferring chirality the way the ASLNow loader has to.

It is the first source here with **real signer ids**, and that is what makes a by-signer split
possible at all. It ships on both sides:

- **training**, capped at 25 frames per (signer, letter) = 6,000 (`strangers.ASLHG_CAP`). Uncapped
  it is worse, not better: 24,000 frames from ten hands outvote the author's own 3,807, and his
  pooled leave-one-session-out falls from 0.9256 to 0.909-0.915.
- **testing**, as `temporal/crossval_signers.py`: hold out all 2,400 of one signer's frames, train
  on the other nine at the shipped cap. Pooled 0.9453 over all 23,984 held-out frames at seed 0,
  per signer 0.8796 to 1.0000. U (0.671, read as R) and R (0.703, read as U) are the only letters
  under 0.80, and they stay there with nine other signers in training — a feature defect, not a
  data shortage.

Only landmarks are committed, never a photograph, the same as the Ankara digits.

## 9. Kaggle `ayuraj/asl-dataset` — the permanent never-train holdout

<https://www.kaggle.com/datasets/ayuraj/asl-dataset>, **CC0** (the Croissant record states it
verbatim as `{"@type": "sc:CreativeWork", "name": "CC0: Public Domain", "url":
"https://creativecommons.org/publicdomain/zero/1.0/"}`). 2,515 pre-cropped 400x400 hand
photographs over 36 folders, five signers named by the filename prefix (`hand1`..`hand5`). The zip
carries every image twice, byte for byte identical, once at `asl_dataset/<class>/` and once one
level deeper; the ingester descends exactly one level.

```
./.venv/bin/python temporal/ingest_ayuraj.py       # 57 MB of zip -> temporal/ayuraj.npz (0.4 MB)
```

**Nothing may ever train on it.** `strangers.NEVER_TRAIN` holds the path, `strangers.training_source`
raises on it, and `tests/test_strangers.py` builds the real training set and asserts that not one
of its 1,111 letter frames is in it. The reason is a rule about measurement rather than about this
dataset: every other stranger set here has crossed onto the training side once it proved useful,
and each crossing turned a "never seen" figure into an in-sample one that had to be relabeled
historical. A set that is never trained on is the only one whose number cannot drift that way.

What it reads, on the two pickles I scored it against (`crossval_strangers.ayuraj_report()`,
the committed `temporal/model_static.p` and the previous release's, same 1,111 frames both
times): **0.889 now, 0.797 before**. Per signer on the committed forest: hand1 0.876 (n = 364),
hand2 0.918 (441), hand3 0.867 (83), hand4 0.922 (77), hand5 0.829 (146). Through the shipped
vote gate it emits on 0.88 of records and is right on 0.961 of those, against 0.69 and 0.904
before. That is the whole point of a never-train set: the two numbers are the same measurement,
so the difference between them is the retrain and nothing else.

Two caveats travel with any number taken from it: MediaPipe finds a hand in only 72.33% of the
images (1,819 of 2,515 over all 36 folders), and the misses concentrate on the fists (T 8 of 65,
S 14 of 70, M 15 of 70), so its per-letter cells for those letters have single-digit n and the
pooled figure is scored on a detection-biased subset. Quote it pooled, never per letter.

## 10. OpenHands fingerspelling clips — a stranger holding a letter through the segmenter

<https://doi.org/10.5281/zenodo.6813108>, "OpenHands: Fingerspelling datasets - Poses",
**CC BY 4.0**, no account needed. The American set is one 63,060,346 B `American.zip` holding
562 clips over 36 classes (a-z plus the digit words), with `videos/train` and `videos/test`
subdirectories and a `glosses.csv`. 266 of the clips are the 24 static letters, and those 266
are what this project reads.

**Two artifacts are derived from these clips and both are committed.** `openhands_replay.json`
is the per-clip result of replaying them through the segmenter. `idle_strangers.npz` is 530
landmark frames cut from the head and tail of 123 of them by `ingest_idle_strangers.py` — the
moments where the hand is tracked but demonstrably not holding the letter, kept only when a
frame is at least 0.35 palm units from its own clip's held shape. They train the REST class
(`letter_model.py`), which is what gives the forest somewhere to put a resting hand instead of
having to call it a letter. The videos themselves are not redistributed; the landmark file is a
derived work of a CC BY 4.0 source, which this attribution covers.

The frames are NOT a by-signer set and must never be quoted as one, for the same reason the
replay is not: these clips carry no participant ids. And they are deliberately not the author's
own idle holds — `idle_holds.npz` is the GATE, and a gate trained on its own test set measures
memorization rather than generalization. Training the REST class on strangers and testing it on
one signer's resting hand is what keeps `idle_gate.py` an out-of-sample measurement.

```
# unzip American.zip somewhere outside the repository, then:
./.venv/bin/python temporal/replay_strangers.py --root <American>/videos \
    --out temporal/openhands_replay.json
```

This is the source that answers a question earlier releases of the README admitted they could
not: **nothing here measured a stranger holding a letter through the segmenter.** A frame score
is not a runtime score. `replay_strangers.py` runs each clip through one fresh
`Segmenter` and one fresh MediaPipe Hands, with the shipped pickles and the shipped thresholds,
so the run sees a hand appear, settle, hold still for `VOTE_MIN` frames and win a vote at the
letter's own floor, exactly as the page would. The result is committed as
`temporal/openhands_replay.json` (266 rows, 79,869 B); **no video and no landmark is
committed**, only the per-clip verdict. Two full runs wrote that file byte for byte identically.

It carries **no signer ids**. The only structure in the file names is a per-letter index, and
whether index 3 of one letter and index 3 of another are the same person is recorded nowhere in
the download, so the clips cannot be grouped and no leave-one-signer-out split can be built from
them. The by-signer number lives in section 8, not here. `temporal/ingest_external.py` also read
15 of the J and Z clips from the same download while it was being exercised; they are not
committed either.

## 11. Pugeault and Bowden, ASL Fingerspelling A — LOCAL ONLY, not committed, never trained on

Pugeault, N. and Bowden, R., *Spelling It Out: Real-Time ASL Fingerspelling Recognition*, ICCV
Workshops 2011; dataset page <https://www.cvssp.org/FingerSpellingKinect2011/>. Kinect color
and depth crops of the 24 static letters, five top-level directories A-E.

**Its landmarks are not committed and must not be, because the source states no license.** The
dataset page carries no license, copyright, terms or permission statement of any kind — the
scouting run that fetched it found zero occurrences of any of those words on the page, and on
2026-09-23 the URL answers 200 with an empty body, so there is nothing to read now either. No
license means no redistribution right, and a derived landmark file is a redistribution. It gets
a local corroboration run and no row in any results table.

Two more reasons it stays out.

**It must not be trained on.** With its frames in the training set the idle gate breaks outright:
3 of the 43 clean idle holds emit, 21 of 2,064 votes, every one of them a G, at a mean winner
probability of 0.975 with the set capped at 100 per (signer, letter) and 1.000 uncapped — against
0 of 43 for the recipe that ships. A relaxed hand read as a confident G is the exact failure the
vote floor exists to prevent, and no accuracy buys that back.

**Its "by signer" may be "by session".** The dataset page describes five users, the published
paper is commonly cited as four persons, and the download settles nothing on its own: five
directories is five directories. I could not re-read the page to check — on 2026-09-23 it
answered 200 with an empty body — so treat the signer count as unverified and check it before
anyone builds a by-signer split on this set. Section 8 is where the by-signer number lives, and
it has file-level signer ids.

The regeneration recipe, so the corroboration can be repeated without the file:

1. Download the set from the page above and unpack it outside the repository.
2. Run MediaPipe over the color crops exactly as `temporal/ingest_images.py` runs it —
   `mp.solutions.hands`, `static_image_mode=True`, `max_num_hands=1`,
   `min_detection_confidence=0.3`, BGR read by OpenCV and converted to RGB, never flipped — with
   one addition the tight Kinect crops force: pad the image by 0.5 of its size before detection
   (the palm detector wants context and these crops have none), then map the landmarks **back**
   into the unpadded crop's own normalized frame before storing them, because
   `features.to_isotropic` multiplies x by W/H and padding to a square would silently change the
   aspect the loader then corrects for.
3. Store `frame_size` **per image**: the crops are variable-size hand bounding boxes, so there is
   no single (W, H) for the set. Correcting per image reads 0.8077; collapsing to the first
   image's size, which is what the shipped `(N,2)`-unaware loaders would do, reads 0.7997, and no
   correction at all reads 0.7861.
4. Decide chirality the way `ingest_aslnow.py` does, with a forest trained on the author's own
   frames and their mirror images, and keep MediaPipe's own label beside it. On this set 0.0104
   of frames are mirrored onto the canonical frame and the detector is confident on 0.9720.

What it corroborates, scored on the committed `temporal/model_static.p`, which never saw it:
**0.8077 over 65,431 detected letter frames** (52,851 right), per directory A 0.8052, B 0.8296,
C 0.8432, D 0.8117, E 0.7457; through the shipped vote gate it emits on 0.7956 of frames and is
right on 0.8874 of those. It is the largest cross-signer read in the project and it agrees with
the two committed ones (ASL-HG by signer 0.9453 at seed 0, ayuraj 0.889) on direction, not on
level. Tight Kinect crops with no context around the hand are a harder condition than either of
those sets, which is what a corroboration is for.

## What the letters still have no source for

This file used to end with a list of still-image sets to try next, because nothing here could
split by signer or replay a stranger's held sign. Both of those are now done: section 8 is the
by-signer split (`temporal/crossval_signers.py`), section 10 is the runtime replay on other
people's video, and section 9 is a holdout that can never quietly become training data. What is
left undone is narrower, and it is all on the motion branch.

- **J and Z on other people's video, at any scale.** Eight clips from four sources is what
  exists (section 4's route: three YouTube channels under CC BY, signer ids `yt_danielparks`,
  `yt_inclusivesigntalk` and `yt_lexiemoore`, plus one ASL Signbank entry under CC BY-NC-SA,
  `signbank_asl`; one J and one Z each). Neither the video nor its landmarks is committed, and
  the NC clip could not be even if it were worth committing. Section 1's CC0 J/Z video set is
  still the one worth downloading.
- **A by-signer motion split.** Every motion event in `events.npz` is the author's.

A Kaggle mirror of the Pugeault set,
`mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out`, is 2.1 GB and states no license in
its Croissant record either, so it inherits section 11's problem rather than solving it.

Sign Language MNIST is still not useful: 28x28 crops carry no landmarks to extract. Kaggle's
29-class ASL Alphabet sets include J and Z as single still images, which is worse than omitting
them. `MSL-AlphaVid` (Malayalam) and `AzSLD` (Azerbaijani) have dynamic letters but different
handshapes; the geometry does not transfer.

## Always run this before trusting a download

```
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz \
    --out temporal/external_events.npz
```

If J and Z clips cut zero events, the footage begins mid-gesture and never arms a gate — the
runtime would miss it too. Ingesting ten clips and reading that diagnostic costs a minute and
tells you whether the remaining gigabytes are worth downloading.

`--out` is on that command because `label_events.py --out` defaults to `temporal/events.npz`,
the committed motion training set. Running the diagnostic bare on somebody else's footage used
to replace it and print the path as though that were routine; it now refuses and names the flag
(`label_events.default_out_refusal`, checked by `tests/test_label_events.py`). Type the `--out`
anyway — the refusal is a backstop, not a reason to stop thinking about where the file goes.

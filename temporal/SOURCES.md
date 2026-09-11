# External data sources

Why bother: the repository's README states that nothing in it supports a claim about a signer
other than the one who recorded it. External footage is the only way to change that. It is a
**test set**, not a training substitute — the training distribution should match your camera and
the runtime's parked-then-move assumption.

Everything ingested is tagged with a non-`S1` session, so the by-session split in
`train_motion.py` and `evaluate.py` holds it out by construction rather than by remembering to.

## One-time Kaggle auth

1. <https://www.kaggle.com/settings/account> → **Create New API Token** → downloads `kaggle.json`
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

The CLI is already installed in this project's venv (`./.venv/bin/kaggle`).

## 1. ASL Alphabet Videos [J, Z] — the direct hit

<https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z>

Video, scoped to the two motion letters specifically. The only source found that targets exactly
the gap this feature exists to fill.

```
./.venv/bin/kaggle datasets download -d signnteam/asl-sign-language-alphabet-videos-j-z \
    -p ~/Downloads/asl_jz --unzip
./.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/asl_jz/J \
    --label J --signer kaggle_jz --session EXT
./.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/asl_jz/Z \
    --label Z --signer kaggle_jz --session EXT
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz
```

**Unverified.** Kaggle serves its dataset pages client-side, so only the title and a one-line
description were readable without an account: *"Video data for training American Sign Language
alphabet character recognition"*. Size, signer count and license were not confirmed. Check three
things when it lands, because they decide whether it is worth anything:

- **distinct signers** — one signer adds little over twelve minutes of your own recording; many
  signers makes it the cross-signer test set. Give each signer a distinct `--signer` id, or the
  cross-signer claim is not a cross-signer claim.
- **license** — this repository is public.
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

## Not useful

`ASL Fingerspelling A` (131k samples, 5 signers) and `B` (9 signers) are **24 classes** — they
exclude J and Z for the same reason this project did. Sign Language MNIST likewise. Kaggle's
29-class ASL Alphabet sets *do* include J and Z, as single still images, which is worse than
omitting them: those are mislabeled I and D. `MSL-AlphaVid` (Malayalam) and `AzSLD`
(Azerbaijani) have dynamic letters but different handshapes; the geometry does not transfer.

## Always run this before trusting a download

```
./.venv/bin/python temporal/label_events.py --clips temporal/external_clips.npz
```

If J and Z clips cut zero events, the footage begins mid-gesture and never arms a gate — the
runtime would miss it too. Ingesting ten clips and reading that diagnostic costs a minute and
tells you whether the remaining gigabytes are worth downloading.

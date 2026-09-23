"""Build temporal/ayuraj.npz: five signers the letter forest is PERMANENTLY forbidden to train on.

This set exists to stay out. Every other stranger set in this project has moved onto the
training side once it proved useful -- ASLNow did, the Ankara digits did, ASL-HG does in this
release -- and each time it moved, the number it used to provide stopped meaning "a dataset
the forest never saw" and became an in-sample figure that has to be relabeled historical. A
set that is never trained on is the only one whose number cannot drift that way: it is
measured the same way this year and next, against whatever forest ships, and a regression in
it is a regression and not a change of protocol. So temporal/strangers.py lists this file in
NEVER_TRAIN, asserts on it rather than trusting the convention, and
temporal/tests/test_strangers.py builds the real training set and checks that not one of
these frames is in it. If a future recipe wants more signers, it takes them from somewhere
else.

Source: "American Sign Language Dataset", Ayush Thakur (ayuraj), Kaggle.

    https://www.kaggle.com/datasets/ayuraj/asl-dataset
    download: https://www.kaggle.com/api/v1/datasets/download/ayuraj/asl-dataset

License, verbatim from the dataset's Croissant record
(https://www.kaggle.com/datasets/ayuraj/asl-dataset/croissant/download):

    {"@type": "sc:CreativeWork", "name": "CC0: Public Domain",
     "url": "https://creativecommons.org/publicdomain/zero/1.0/"}

CC0 is a full waiver, so nothing restricts redistribution; only landmarks are committed
regardless, never a photograph.

Kaggle serves this without an account. A plain GET on the download endpoint above returns
59,642,568 B of valid zip with no credentials -- note that HEAD on the same URL answers 404,
which is why an existence check has to be a GET.

What the set is: 2,515 pre-cropped hand photographs, 400x400, over 36 folders (0-9 and a-z),
70 images per folder except t, which has 65. Five signers, named by the filename prefix:
hand1 900, hand2 895, hand3 180, hand4 180, hand5 360.

What this loader infers, and why:

  duplicates   the zip carries every image TWICE, once under asl_dataset/<class>/ and once
               under asl_dataset/asl_dataset/<class>/, byte for byte identical (verified by
               sha256 on a member of each). walk() descends exactly one level below the root
               it is given, so the nested copy contributes nothing and the count is 2,515 and
               not 5,030. Ingesting both would double every frame's weight in any measurement
               that averages over records.
  classes      the folder name, uppercased: 'a' -> 'A'. strangers.load_ayuraj keeps the 24
               static letters, which is 1,111 of the 1,819 detected frames; the digit folders
               and J/Z are read and stored but are not letters here.
  signer       hand1..hand5, the filename prefix before the first underscore, the same shape
               of rule ingest_aslhg.py applies to ASL-HG's P<k>_ names.
  frame_size   every image is stored 400x400, so u = x * (W/H) is the identity. These are
               crops around a hand rather than camera frames, which is a caveat about what
               the set measures, not about the correction: MediaPipe normalized x and y by
               400 either way.
  handedness   MediaPipe labels 1,782 frames "Left" and 37 "Right"; strangers.load_ayuraj
               canonicalizes per frame by that label, as it does for Ankara and ASL-HG.

The detection rate is 1,819 of 2,515 (72.33%) and it is badly class-skewed, because a crop
tight around a fist gives the palm detector little to find: T 8/65 (12.3%), S 14/70 (20.0%),
M 15/70 (21.4%), Q 19/70 (27.1%), N 20/70 (28.6%), A 24/70 (34.3%), against 60-70 of 70 for
the open shapes. So the per-letter cells for the fists have single-digit n and must not be
quoted per letter, and the pooled figure is scored on a detection-biased subset. Both caveats
belong beside any number taken from this file.

Run (network: 59,642,568 B of zip from Kaggle; 59 s of MediaPipe at 3 workers):

    ./.venv/bin/python temporal/ingest_ayuraj.py                 # -> temporal/ayuraj.npz
    ./.venv/bin/python temporal/ingest_ayuraj.py --from DIR      # from an existing extract
    ./.venv/bin/python temporal/ingest_ayuraj.py --limit 200 --workers 1 --out /tmp/sample.npz

This script is what made the committed temporal/ayuraj.npz (1,819 records, 440,152 B). The
committed file was verified rather than rebuilt: the run above over a 200-image sample
detected 157 of them and reproduced all 9,891 landmark coordinates of those records exactly,
along with their handedness labels, scores and signer ids.
"""
import argparse
import collections
import os
import sys
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest_aslhg as HG  # noqa: E402  -- the identical MediaPipe worker; see extract()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ayuraj.npz")
SLUG = "ayuraj/asl-dataset"
DOWNLOAD = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
CROISSANT = f"https://www.kaggle.com/datasets/{SLUG}/croissant/download"
ARCHIVE_BYTES = 59642568
LICENSE = "CC0 1.0 Public Domain"
SESSION = "EXTAY"
IMAGE_EXT = HG.IMAGE_EXT


def check_license():
    """Read the Croissant record and refuse if it no longer says CC0."""
    with urllib.request.urlopen(CROISSANT, timeout=120) as r:
        import json
        rec = json.load(r)
    lic = rec.get("license") or {}
    print(f"{rec.get('name')}: license {lic.get('name')!r} ({lic.get('url')})")
    if "CC0" not in str(lic.get("name", "")):
        raise SystemExit(f"the Croissant record now reads {lic!r}, not CC0; re-verify before ingesting")


def download(cache):
    """Pull the zip and extract the outer copy of the tree (never the nested duplicate)."""
    check_license()
    os.makedirs(cache, exist_ok=True)
    zpath = os.path.join(cache, "asl-dataset.zip")
    if not (os.path.isfile(zpath) and os.path.getsize(zpath) == ARCHIVE_BYTES):
        print(f"downloading {ARCHIVE_BYTES / 1e6:.0f} MB -> {zpath} (no credentials needed; "
              "HEAD on this URL answers 404, a GET does not)", flush=True)
        with urllib.request.urlopen(DOWNLOAD, timeout=600) as r, open(zpath, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        # exactly two levels below the archive root: asl_dataset/<class>/<image>. The nested
        # asl_dataset/asl_dataset/... copy is the same bytes and is left in the zip.
        members = [n for n in z.namelist() if n.count("/") == 2 and n.lower().endswith(IMAGE_EXT)]
        nested = sum(1 for n in z.namelist() if n.count("/") == 3 and n.lower().endswith(IMAGE_EXT))
        print(f"extracting {len(members)} images; {nested} duplicates one level deeper are skipped",
              flush=True)
        z.extractall(cache, members)
    return os.path.join(cache, "asl_dataset")


def walk(root):
    """[(LABEL, relpath)] over the folders one level below `root`, both sorted. The class label
    is the folder name uppercased, because this set writes it lowercase and every other loader
    here spells a letter 'A'."""
    items = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(IMAGE_EXT):
                items.append((label.upper(), os.path.join(label, name)))
    return items


def signer_of(relpath):
    """hand<k> from <class>/hand<k>_<class>_<...>.jpeg."""
    sid = os.path.basename(relpath).split("_")[0]
    if not (sid.startswith("hand") and sid[4:].isdigit()):
        raise SystemExit(f"{relpath!r}: the signer id is the hand<k> filename prefix and this file "
                         "has none; the set names every image hand<k>_<class>_<...>")
    return sid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", default=None,
                    help="directory of <class>/hand<k>_... folders (skips the download)")
    ap.add_argument("--cache", default=os.path.join(HERE, ".ayuraj_cache"), help="download directory")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many images in walk order -- the sample the verification uses")
    args = ap.parse_args(argv)

    root = args.src if args.src is not None else download(args.cache)
    items = walk(root)
    if not items:
        raise SystemExit(f"no <class>/<image> folders under {root}")
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} images in {len(set(l for l, _ in items))} class folders under {root}", flush=True)

    # The same worker, not a copy of it: one MediaPipe configuration for both external image
    # sets, so "run exactly as extract_static_sequences.py configures it" is one statement to
    # check rather than two to keep in step.
    out, dt = HG.extract(items, root, args.workers)

    lm, stamps, handed, letters, signer, files, wh, score = [], [], [], [], [], [], [], []
    n_by_label = collections.Counter(l for l, _ in items)
    found_by_label, found_by_signer, n_by_signer = collections.Counter(), collections.Counter(), collections.Counter()
    sizes = collections.Counter()
    unreadable = 0
    for kind, i, label, rel, pts, hs, size in out:
        sid = signer_of(rel)
        n_by_signer[sid] += 1
        if kind == "unreadable":
            unreadable += 1
            continue
        sizes[size] += 1
        if kind == "miss":
            continue
        lm.append(pts)
        stamps.append(float(i))
        handed.append(hs[0])
        score.append(hs[1])
        letters.append(label)
        signer.append(sid)
        files.append(rel)
        wh.append(size)
        found_by_label[label] += 1
        found_by_signer[sid] += 1

    wh = np.array(wh, dtype=np.int64)
    ratios = wh[:, 0] / wh[:, 1]
    if not np.allclose(ratios, ratios[0]):
        raise SystemExit(f"images have different aspect ratios: {sorted(set(np.round(ratios, 4)))}")
    modal_wh = max(sizes.items(), key=lambda kv: kv[1])[0]

    print(f"{len(items)} images, {unreadable} unreadable, {len(lm)} with a hand "
          f"({100 * len(lm) / len(items):.2f}%), {dt:.1f}s ({1000 * dt / len(items):.1f} ms/image)")
    print(f"image sizes (W,H): {dict(sizes)}; frame_size {list(modal_wh)} (all ratios {ratios[0]:.4f})")
    print(f"handedness: {dict(collections.Counter(handed))}")
    print(f"signers: {len(set(signer))}")
    for s in sorted(found_by_signer):
        print(f"  signer {s:>6}: {found_by_signer[s]:5d}/{n_by_signer[s]:<5d} "
              f"({100 * found_by_signer[s] / n_by_signer[s]:.1f}%)")
    for label in sorted(n_by_label):
        print(f"  class {label:>4}: {found_by_label[label]:5d}/{n_by_label[label]:<5d} "
              f"({100 * found_by_label[label] / n_by_label[label]:.1f}%)")

    np.savez_compressed(args.out,
                        lm=np.array(lm, dtype=np.float32),
                        stamps=np.array(stamps, dtype=np.float32),
                        handed=np.array(handed),
                        letters=np.array(letters),
                        session=np.array(SESSION),
                        frame_size=np.array(modal_wh, dtype=np.int64),
                        signer=np.array(signer),
                        files=np.array(files),
                        frame_wh=wh,
                        score=np.array(score, dtype=np.float32),
                        source=np.array(f"kaggle:{SLUG}"),
                        license=np.array(LICENSE))
    print(f"wrote {args.out}: {len(lm)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")
    print("NEVER TRAIN ON THIS FILE -- strangers.NEVER_TRAIN is what enforces it")


if __name__ == "__main__":
    sys.exit(main())

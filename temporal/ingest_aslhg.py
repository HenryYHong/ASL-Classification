"""Build temporal/aslhg.npz: ten more signers' letters, from the ASL-HG photographs.

Source: ASL-HG: American Sign Language Hand Gesture Image Dataset, Md. Famidul Islam Pranto,
Md. Rifatul Islam, Md. Ali Akbor and co-authors, Bangladesh University of Business and
Technology.

    https://data.mendeley.com/datasets/j4y5w2c8w9/1
    DOI 10.17632/j4y5w2c8w9.1
    record: https://data.mendeley.com/public-api/datasets/j4y5w2c8w9

License, CC BY 4.0, as the Mendeley record states it verbatim in data_licence:

    "You can share, copy and modify this dataset so long as you give appropriate credit,
     provide a link to the CC BY license, and indicate if changes were made, but you may not
     do so in a way that suggests the rights holder has endorsed you or your use of the
     dataset. Note that further permission may be required for any content within the dataset
     that is identified as belonging to a third party."
    short_name "CC BY 4.0", url http://creativecommons.org/licenses/by/4.0

The record identifies no third-party content. No photograph is redistributed: this writes 21
(x,y,z) floats per image and nothing else, the same thing ingest_images.py writes for the
Ankara digits and the same argument SOURCES.md makes there.

What the set is: 36,000 JPEGs over 36 classes (A-Z and 0-9), 10 volunteers photographed in
Mirpur, Dhaka, Bangladesh in May-June 2025, 100 images per volunteer per class, smartphone
cameras indoors and out. It is the first source this project has found that carries real
signer ids, which is what makes temporal/crossval_signers.py possible at all.

What this loader infers, and why:

  classes      only the 24 static letters are read; the run below takes ASL_Raw_Images.zip's
               asl_dataset/<CLASS>/ folders for A-Y minus J and Z. J and Z are motion letters
               and this project's static classifier has no class for them (a still of a J is
               an I). The digit folders belong to train_digits.py, not here -- and this set's
               "0" is the two-handed ASL zero, which is not the letter O's handshape, so the
               digit-to-letter mapping strangers.DIGIT_LETTERS uses for Ankara does not apply.
               24 classes x 1,000 = 24,000 images read, 23,984 with a hand (99.93%), the
               cleanest detection rate of any external set here (Ankara 87.5%). The 16 misses
               are all one signer's H (P8 2,384/2,400 = 99.3%, class H 984/1,000 = 98.4%).
  signer       the filename is <CLASS>/P<k>_<CLASS>_<n>.jpg, so the signer is the prefix
               before the first underscore: P1..P10, 2,400 frames each (P8 2,384). This is
               the one rule ingest_images.py's four (runs / imgnum10 / file / none) do not
               express, and it is the whole reason for a second ingester rather than a flag.
  frame_size   every image is stored 300x300, so the isotropic correction u = x * (W/H) is
               the identity. That is not an assumption that the phone shot square: it is the
               statement that MediaPipe normalized x by 300 and y by 300, which is what the
               correction has to undo. The palm triangle confirms the crop is square rather
               than a squashed 4:3 -- over the upright palm-forward letters (A B E M N S T)
               the palm's width/height ratio is 0.755 on this project's own frames, and
               ASL-HG reads 0.719 at 1:1 as stored against 0.913 at 4:3, 1.165 at 16:9, 0.561
               at 3:4 and 0.441 at 9:16 (n=7,000; the method is ingest_aslnow.py's). 1:1 is
               the only candidate near this project's own hands.
  handedness   MediaPipe labels 23,974 images "Left" and 10 "Right" -- the same unmirrored
               convention as the Ankara photos -- so strangers.load_aslhg canonicalizes per
               image by that label, exactly as load_ankara_letters does.

MediaPipe runs EXACTLY as temporal/extract_static_sequences.py configures it for the archive
and temporal/ingest_images.py for the Ankara photos: mp.solutions.hands.Hands with
static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, the BGR array
converted to RGB and never flipped. With static_image_mode=True every image goes through the
palm detector and a fresh landmark fit with no state carried between images, so --workers
changes only the wall clock and not one landmark: verified by re-running this script over a
200-image sample in a single process (--workers 1) and comparing against the committed file,
which was made at --workers 3 -- all 200 records were detected and all 12,600 landmark
coordinates matched bit for bit, along with every handedness label, score and signer id.

Run (network: 876,640,391 B of zip from Mendeley, about 20 GB-minutes of disk while it
extracts; 415 s of MediaPipe at 3 workers, 17.3 ms/image):

    ./.venv/bin/python temporal/ingest_aslhg.py                  # -> temporal/aslhg.npz
    ./.venv/bin/python temporal/ingest_aslhg.py --from DIR       # from an existing extract
    ./.venv/bin/python temporal/ingest_aslhg.py --limit 200 --workers 1 --out /tmp/sample.npz

This script is what made the committed temporal/aslhg.npz (23,984 records, 5,559,465 B). The
npz was not re-made from scratch for the commit -- re-downloading 876 MB to reproduce a file
byte for byte is not a measurement -- but it was re-verified the way above, on 200 images
pulled out of the remote zip by HTTP Range requests against its central directory (3.9 MB
instead of 877 MB). Any change to the MediaPipe configuration or the signer rule has to re-run
this in full to stay honest.
"""
import argparse
import collections
import json
import os
import sys
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aslhg.npz")
DOI = "10.17632/j4y5w2c8w9.1"
DATASET = "j4y5w2c8w9"
API = f"https://data.mendeley.com/public-api/datasets/{DATASET}"
ARCHIVE = "ASL_Raw_Images.zip"          # 876,640,391 B; the member tree is asl_dataset/<CLASS>/
ARCHIVE_SHA256 = "594cfa0158044085ed61351315c187a6f3a3f9087795b4f4cdba68ecd04f12b1"
LICENSE = "CC BY 4.0"
SESSION = "EXTHG"
#: The 24 static letters. J and Z are motion letters; the digit folders are train_digits.py's.
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def fetch_record():
    with urllib.request.urlopen(API, timeout=120) as r:
        return json.load(r)


def download(cache):
    """Pull ASL_Raw_Images.zip and extract the 24 static-letter folders into `cache`.

    The download URL is read from the public-api record rather than pasted, so the DOI is the
    only thing this script hard-codes about where the bytes live.
    """
    rec = fetch_record()
    lic = rec.get("data_licence", {})
    print(f"{rec['name']} v{rec['version']}, DOI {rec['doi']['id']}, "
          f"license {lic.get('short_name')} ({lic.get('url')})")
    if rec["doi"]["id"] != DOI:
        raise SystemExit(f"the record now reads {rec['doi']['id']}, not {DOI}; re-verify before ingesting")
    if lic.get("short_name") != LICENSE:
        raise SystemExit(f"the license now reads {lic.get('short_name')!r}, not {LICENSE!r}")
    entry = next((f for f in rec["files"] if f["filename"] == ARCHIVE), None)
    if entry is None:
        raise SystemExit(f"{ARCHIVE} is not in the record any more: {[f['filename'] for f in rec['files']]}")
    os.makedirs(cache, exist_ok=True)
    zpath = os.path.join(cache, ARCHIVE)
    if not (os.path.isfile(zpath) and os.path.getsize(zpath) == entry["size"]):
        url = entry["content_details"]["download_url"]
        print(f"downloading {entry['size'] / 1e6:.0f} MB -> {zpath}", flush=True)
        with urllib.request.urlopen(url, timeout=600) as r, open(zpath, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    root = os.path.join(cache, "asl_dataset")
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist()
                   if n.count("/") == 2 and n.split("/")[1] in LETTERS and n.lower().endswith(IMAGE_EXT)]
        print(f"extracting {len(members)} images of the {len(LETTERS)} static letters "
              f"(the archive holds {len(z.namelist())} over 36 classes)", flush=True)
        z.extractall(cache, members)
    return root


def walk(root):
    """[(label, relpath)] in a deterministic order: label folders sorted, files sorted.

    ingest_images.walk's rule, repeated here rather than imported, because this file's whole
    job is to be the one thing that made aslhg.npz.
    """
    items = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d) or label not in LETTERS:
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(IMAGE_EXT):
                items.append((label, os.path.join(label, name)))
    return items


def signer_of(relpath):
    """P<k> from <CLASS>/P<k>_<CLASS>_<n>.jpg. Raises rather than guessing: a silently wrong
    signer id would turn crossval_signers.py's leave-one-signer-out into a random split, and
    the number would still look reasonable."""
    stem = os.path.basename(relpath)
    sid = stem.split("_")[0]
    if not (len(sid) > 1 and sid[0] == "P" and sid[1:].isdigit()):
        raise SystemExit(f"{relpath!r}: the signer id is the P<k>_ filename prefix and this file "
                         "has none; ASL-HG names every image P<k>_<CLASS>_<n>.jpg")
    return sid


_HANDS = None
_ROOT = None


def _init(root):
    """One Hands instance per worker process. static_image_mode=True carries no state between
    images, so the worker count is a wall-clock choice and not a landmark one (docstring)."""
    global _HANDS, _ROOT
    import mediapipe as mp
    _ROOT = root
    _HANDS = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                      min_detection_confidence=0.3)


def _one(arg):
    import cv2
    i, label, rel = arg
    img = cv2.imread(os.path.join(_ROOT, rel))
    if img is None:
        return ("unreadable", i, label, rel, None, None, None)
    H, W = img.shape[:2]
    res = _HANDS.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return ("miss", i, label, rel, None, None, (W, H))
    pts = res.multi_hand_landmarks[0].landmark
    lm = [(p.x, p.y, p.z) for p in pts]
    if res.multi_handedness:
        cls = res.multi_handedness[0].classification[0]
        hs = (cls.label, float(cls.score))
    else:
        hs = ("Unknown", 0.0)
    return ("hit", i, label, rel, lm, hs, (W, H))


def extract(items, root, workers):
    """Run MediaPipe over `items` and return the per-record tuples in walk order."""
    tasks = [(i, l, r) for i, (l, r) in enumerate(items)]
    t0 = time.time()
    out = []
    if workers <= 1:
        _init(root)
        for k, t in enumerate(tasks):
            out.append(_one(t))
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        with Pool(workers, initializer=_init, initargs=(root,)) as pool:
            for k, rec in enumerate(pool.imap(_one, tasks, chunksize=64)):
                out.append(rec)
                if (k + 1) % 2000 == 0:
                    el = time.time() - t0
                    print(f"  {k + 1}/{len(tasks)}  {el:.0f}s  {1000 * el / (k + 1):.1f} ms/img", flush=True)
    return sorted(out, key=lambda r: r[1]), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", default=None,
                    help="directory of <CLASS>/P<k>_<CLASS>_<n>.jpg folders (skips the download)")
    ap.add_argument("--cache", default=os.path.join(HERE, ".aslhg_cache"), help="download directory")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=3,
                    help="MediaPipe processes; changes the wall clock only (see the docstring)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many images in walk order -- the sample the verification uses")
    args = ap.parse_args(argv)

    root = args.src if args.src is not None else download(args.cache)
    items = walk(root)
    if not items:
        raise SystemExit(f"no <CLASS>/<image> folders for the 24 static letters under {root}")
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} images in {len(set(l for l, _ in items))} class folders under {root}", flush=True)

    out, dt = extract(items, root, args.workers)

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
        stamps.append(float(i))          # index in walk order; photographs have no clock
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
        # ingest_images.py's guard, for the same reason: load_extra reads ONE frame_size pair
        # and u = x*(W/H) would then be wrong for every image of another shape.
        raise SystemExit(f"images have different aspect ratios: {sorted(set(np.round(ratios, 4)))}")
    modal_wh = max(sizes.items(), key=lambda kv: kv[1])[0]

    def sort_key(s):
        return int("".join(c for c in s if c.isdigit()) or 0)

    print(f"{len(items)} images, {unreadable} unreadable, {len(lm)} with a hand "
          f"({100 * len(lm) / len(items):.2f}%), {dt:.1f}s ({1000 * dt / len(items):.1f} ms/image)")
    print(f"image sizes (W,H): {dict(sizes)}; frame_size {list(modal_wh)} (all ratios {ratios[0]:.4f})")
    print(f"handedness: {dict(collections.Counter(handed))}")
    print(f"signers: {len(set(signer))}")
    for s in sorted(found_by_signer, key=sort_key):
        print(f"  signer {s:>4}: {found_by_signer[s]:5d}/{n_by_signer[s]:<5d} "
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
                        source=np.array(DOI),
                        license=np.array(LICENSE))
    print(f"wrote {args.out}: {len(lm)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())

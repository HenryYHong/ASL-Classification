"""Build temporal/aslnow.npz: other people's letters, seen through the browser's landmarker.

Source: the ASLNow! fingerspelling dataset (sid220/asl-now-fingerspelling on Hugging Face,
MIT license): 2,122 records of "multiple participants told to sign ASL letters into a camera",
each one MediaPipe's 21 hand landmarks as captured by the Web Hand Landmarker -- the same
Tasks-API landmarker docs/app.js runs, on other people's hands, in other people's rooms. It
is the only public source found that matches the hosted page's condition, and it is what
makes a cross-signer number possible for the static letters at all.

What the source does not carry, and how each gap is closed:

  frame size   x is normalized by width and y by height, and the capture size was not
               stored, so the isotropic correction u = x * (W/H) needs a guess. The palm
               triangle settles it: over the upright palm-forward letters (A B E M N S T) the
               palm's width/height ratio is 0.755 on this project's own frames; the ASLNow
               records give 0.728 at 4:3, 0.569 at 1:1 and 0.932 at 16:9. Web apps ask for
               640x480, and 4:3 is what the data says. ASPECT below.
  handedness   no label, and about half the records are the mirror image of this project's
               canonical frame (left-handed participants, or a mirrored video feed in some
               sessions -- the source cannot say which, and for training it does not
               matter). A chirality detector settles it per record: a forest trained on this
               project's own four sessions, canonical frames labeled 0 and their mirror
               images labeled 1, scores 9,752 of 9,756 leave-one-session-out and is confident
               (p < 0.2 or > 0.8) on 95% of ASLNow records. `mirror` is its decision.
  J and Z      the source has still frames labeled J (93) and Z (155). A single frame of a J
               is an I, and this project's static classifier has no J or Z class, so they are
               stored but never trained on (strangers.load_aslnow skips them).
  signers      no participant id, and hand-proportion clustering finds none (the pose
               dominates every bone-length ratio), so the set cannot be split by signer. The
               records are not bursts either -- the nearest same-letter neighbor sits at a
               median 0.31 palm units in shape space, against 0.07 inside one of this
               project's own held bursts -- so each record is a separate capture. It is used
               two ways, both honest: held out entirely as the cross-signer test set (train on
               this project's sessions, test here), and as training data for the shipped
               forest, whose cross-signer evidence is then the two held-out numbers in
               crossval_strangers.py.

Run (network: about 8 MB of JSON from huggingface.co):

    ./.venv/bin/python temporal/ingest_aslnow.py                 # -> temporal/aslnow.npz
    ./.venv/bin/python temporal/ingest_aslnow.py --from DIR      # from an existing download

The npz keeps the RAW landmarks; the aspect and mirror decisions travel beside them so the
loader applies them and nothing is baked in that cannot be revisited.
"""
import argparse
import glob
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aslnow.npz")
REPO_ID = "sid220/asl-now-fingerspelling"
REVISION = "9b3c96ae0adb7744a2c9fc72692842e6b3e25e33"   # the HF commit the file was built from
API = f"https://huggingface.co/api/datasets/{REPO_ID}"
RAW = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/"
LICENSE = "MIT"
#: Width / height the records are assumed to have been captured at (see the docstring).
ASPECT = (4, 3)
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def fetch_listing():
    with urllib.request.urlopen(API, timeout=60) as r:
        d = json.load(r)
    return sorted(s["rfilename"] for s in d["siblings"] if s["rfilename"].endswith(".json") and "/" in s["rfilename"])


def download(files, into):
    for i, f in enumerate(files):
        path = os.path.join(into, f)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(RAW + f, timeout=60) as r, open(path, "wb") as fh:
            fh.write(r.read())
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)


def read_records(root):
    letters, names, lm = [], [], []
    for f in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        if len(d) != 21 or any(set(p) != {"x", "y", "z"} for p in d):
            raise SystemExit(f"{f}: not 21 x/y/z landmarks")
        letters.append(os.path.basename(os.path.dirname(f)))
        names.append(os.path.basename(f))
        lm.append([[p["x"], p["y"], p["z"]] for p in d])
    return np.array(letters), np.array(names), np.asarray(lm, dtype=np.float32)


def chirality_detector(seed=0):
    """A forest that tells this project's canonical frame from its mirror image.

    Trained on all four static sessions (train_static.load / load_extra, i.e. the frames as
    the shipped forest sees them), canonical = 0 and x-negated = 1, on the shipped feature.
    Leave-one-session-out on those sessions: 9,752 / 9,756. Its errors sit on the shapes that
    are nearly mirror-symmetric (a flat B), which are also the shapes mirroring changes least.
    """
    from sklearn.ensemble import RandomForestClassifier
    import train_static as T
    per = T.load()
    for tag, name in (("S2", "static_s2.npz"), ("S3", "static_s3.npz"), ("S4", "static_s4.npz")):
        got = T.load_extra(os.path.join(HERE, name))
        per = [np.concatenate([per[c], got[c]]) if c in got else per[c] for c in range(len(LETTERS))]
    X, y = [], []
    for P in per:
        if not len(P):
            continue
        Q = P.copy()
        Q[..., 0] = -Q[..., 0]
        X += [T.FEATFN(P), T.FEATFN(Q)]
        y += [np.zeros(len(P)), np.ones(len(P))]
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=4).fit(
        np.concatenate(X), np.concatenate(y))


def orient(lm, det):
    """Per-record probability that the record is the mirror image of the canonical frame."""
    import train_static as T
    P = F.to_isotropic(lm[..., :2].astype(np.float64), ASPECT[0], ASPECT[1])
    return det.predict_proba(T.FEATFN(P))[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", default=None, help="directory of <LETTER>/<uuid>.json (skips the download)")
    ap.add_argument("--cache", default=os.path.join(HERE, ".aslnow_cache"), help="download directory")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    root = args.src
    if root is None:
        files = fetch_listing()
        print(f"{len(files)} records listed at {REPO_ID}@{REVISION[:8]}; downloading into {args.cache}")
        download(files, args.cache)
        root = args.cache
    letters, names, lm = read_records(root)
    print(f"read {len(lm)} records: " + ", ".join(f"{L}={int((letters == L).sum())}" for L in sorted(set(letters))))

    det = chirality_detector()
    pm = orient(lm, det)
    mirror = pm > 0.5
    conf = ((pm < 0.2) | (pm > 0.8)).mean()
    print(f"chirality: {mirror.mean():.3f} of records are mirrored relative to the canonical frame; "
          f"the detector is confident on {conf:.3f}")
    static = np.isin(letters, LETTERS)
    np.savez(args.out, lm=lm, letters=letters, names=names, mirror=mirror, mirror_p=pm.astype(np.float32),
             aspect=np.array(ASPECT), source=REPO_ID, revision=REVISION, license=LICENSE,
             note="raw MediaPipe Web landmarks; loader applies aspect and mirror; J/Z stills are not static letters")
    print(f"wrote {args.out}: {len(lm)} records ({int(static.sum())} static letters, "
          f"{int((~static).sum())} J/Z stills), {os.path.getsize(args.out) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()

"""Turn a class-folder tree of still images into the npz schema train_static.load_extra reads.

    ./.venv/bin/python temporal/ingest_images.py --root <path to the Dataset folder> \\
        --session EXTD --out temporal/digits_ankara.npz

This is how temporal/digits_ankara.npz was made from the Sign Language Digits Dataset (Ankara
Ayranci Anadolu High School, Apache-2.0; see SOURCES.md). Only landmarks are committed, never
the photographs. The tree is <root>/<label>/<image>; the folder name is the class label as
written ('0'..'9' for the Ankara digits, but any string works). train_static.load_extra
skips every label outside the 24 static letters and says so, so a digit npz can be merged
into the letter forest only by mistake and not silently; train_digits.py is its consumer. MediaPipe runs EXACTLY as
temporal/extract_static_sequences.py runs it -- mp.solutions.hands, static_image_mode=True,
max_num_hands=1, min_detection_confidence=0.3, one Hands instance for the whole run, the BGR
array converted to RGB and never flipped -- so the landmarks and the handedness label land in
the same convention as every other training frame.

Output keys (collect_motion.py --static-letters schema, plus provenance):
    lm          float32 (N,21,3)   raw MediaPipe landmarks, only images where a hand was found
    stamps      float32 (N,)       image index; photos have no clock, but downstream readers that
                                   split holds on stamp gaps (>0.5 s) then see one hold per image
    handed      <U8     (N,)       MediaPipe's Left/Right label per image
    letters     <U..    (N,)       class label = folder name
    session     <U..    ()         the --session tag
    frame_size  int64   (2,)       ONE (W,H) pair: load_extra reads a single pair. Written only
                                   if every image has the same W/H ratio (an aspect mismatch
                                   would silently skew u = x*(W/H)); otherwise this refuses.
    signer      <U..    (N,)       signer id, from --signer-rule
    files       <U..    (N,)       path relative to --root, for provenance
    frame_wh    int64   (N,2)      the true per-image (W,H)
    score       float32 (N,)       MediaPipe's handedness score

Signer rule for the Ankara set ('runs', the default): each student was photographed once per
digit in order 0..9 with consecutive IMG numbers, so a student is a maximal run of images whose
IMG numbers step by at most 2 and whose labels strictly increase. That gives 224 runs (186 of
exactly 10 images, none with a repeated digit) against the README's 218 students. The simpler
IMG // 10 rule the first audit used is WRONG for this set: the numbering offset drifts, so 187 of
its 219 groups mix two students, and a by-group split then leaks every student's other digits
into the training fold. Any cross-signer number computed under IMG // 10 is optimistic.
"""
import argparse
import collections
import os
import re
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def image_number(relpath):
    stem = os.path.splitext(os.path.basename(relpath))[0]
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"no number in {relpath!r}; the signer rules need one")
    return int(nums[-1])


def signers_by_runs(items, max_gap=2):
    """{relpath: signer id}: sort every image by its number; a new signer starts wherever the
    label fails to increase or the number jumps by more than `max_gap`."""
    order = sorted(items, key=lambda it: image_number(it[1]))
    out, sid, prev = {}, 0, None
    for label, rel in order:
        n = image_number(rel)
        key = int(label) if label.isdigit() else label     # '10' must sort after '9'
        if prev is not None and (key <= prev[0] or n - prev[1] > max_gap):
            sid += 1
        out[rel] = str(sid)
        prev = (key, n)
    return out


def signer_of(rule, relpath):
    """Map an image path to a signer id under one of the per-image rules."""
    stem = os.path.splitext(os.path.basename(relpath))[0]
    if rule == "imgnum10":
        nums = re.findall(r"\d+", stem)
        if not nums:
            raise ValueError(f"no number in {relpath!r} for the imgnum10 signer rule")
        return str(int(nums[-1]) // 10)
    if rule == "file":
        return stem
    if rule == "none":
        return "unknown"
    raise ValueError(f"unknown signer rule {rule!r}")


def walk(root):
    """[(label, relpath)] in a deterministic order: label folders sorted, files sorted."""
    items = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(IMAGE_EXT):
                items.append((label, os.path.join(label, name)))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory of <label>/<image> folders")
    ap.add_argument("--out", required=True, help="npz to write")
    ap.add_argument("--session", default="EXTD", help="session tag stored in the npz")
    ap.add_argument("--signer-rule", default="runs", choices=("runs", "imgnum10", "file", "none"))
    ap.add_argument("--log", default=None, help="append the per-class detection table here")
    args = ap.parse_args(argv)

    items = walk(args.root)
    if not items:
        raise SystemExit(f"no images under {args.root}")
    print(f"{len(items)} images in {len(set(l for l, _ in items))} class folders under {args.root}")
    run_ids = signers_by_runs(items) if args.signer_rule == "runs" else None

    lm, stamps, handed, letters, signer, files, wh, score = [], [], [], [], [], [], [], []
    n_by_label = collections.Counter(l for l, _ in items)
    found_by_label = collections.Counter()
    sizes = collections.Counter()
    unreadable = 0
    t0 = time.time()
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=0.3) as hands:
        for i, (label, rel) in enumerate(items):
            img = cv2.imread(os.path.join(args.root, rel))
            if img is None:
                unreadable += 1
                continue
            H, W = img.shape[:2]
            sizes[(W, H)] += 1
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            pts = res.multi_hand_landmarks[0].landmark
            lm.append([(p.x, p.y, p.z) for p in pts])
            stamps.append(float(i))
            if res.multi_handedness:
                cls = res.multi_handedness[0].classification[0]
                handed.append(cls.label)
                score.append(cls.score)
            else:
                handed.append("Unknown")
                score.append(0.0)
            letters.append(label)
            signer.append(run_ids[rel] if run_ids is not None else signer_of(args.signer_rule, rel))
            files.append(rel)
            wh.append((W, H))
            found_by_label[label] += 1
    dt = time.time() - t0

    wh = np.array(wh, dtype=np.int64)
    ratios = wh[:, 0] / wh[:, 1]
    if not np.allclose(ratios, ratios[0]):
        raise SystemExit("images have different aspect ratios; load_extra reads one frame_size "
                         "pair, and the isotropy correction u = x*(W/H) would be wrong for some "
                         f"of them. Ratios seen: {sorted(set(np.round(ratios, 4)))}")
    modal_wh = max(sizes.items(), key=lambda kv: kv[1])[0]

    lines = [f"ingest_images: {args.root}",
             f"  {len(items)} images, {unreadable} unreadable, {len(lm)} with a hand "
             f"({100 * len(lm) / len(items):.1f}%), {dt:.1f}s ({1000 * dt / len(items):.1f} ms/image)",
             f"  image sizes (W,H): {dict(sizes)}; frame_size written as {list(modal_wh)} "
             f"(all ratios {ratios[0]:.3f})",
             f"  handedness: {dict(collections.Counter(handed))}",
             f"  signers: {len(set(signer))} under rule {args.signer_rule}"
             + (f" (run sizes: {sorted(collections.Counter(collections.Counter(signer).values()).items())})"
                if run_ids is not None else "")]
    for label in sorted(n_by_label):
        lines.append(f"  class {label:>4}: {found_by_label[label]:4d}/{n_by_label[label]:<4d} detected")
    text = "\n".join(lines)
    print(text)
    if args.log:
        with open(args.log, "a") as fh:
            fh.write(text + "\n")

    np.savez_compressed(args.out,
                        lm=np.array(lm, dtype=np.float32),
                        stamps=np.array(stamps, dtype=np.float32),
                        handed=np.array(handed),
                        letters=np.array(letters),
                        session=np.array(args.session),
                        frame_size=np.array(modal_wh, dtype=np.int64),
                        signer=np.array(signer),
                        files=np.array(files),
                        frame_wh=wh,
                        score=np.array(score, dtype=np.float32))
    print(f"wrote {args.out}: {len(lm)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

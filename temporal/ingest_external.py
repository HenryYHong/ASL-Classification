"""Convert third-party ASL footage into the schema collect_motion.py writes.

External data cannot be a training substitute -- your own camera, your own hand, your own
runtime assumptions -- but it is the only route to the one number this project has never had:
does the recognizer work for someone who is not the person who trained it. The README says
plainly that nothing in the repo supports that claim. This module is how it gets supported.

The reason external footage is usable at all is that the features are geometric. A pixel model
would need matching lighting, resolution and skin tone; 21 landmarks divided by palm width care
about none of it. Run MediaPipe over someone else's video and the vectors land in the same space
as your own.

Everything ingested is tagged with its own session, so the by-session split in train_motion.py
and evaluate.py treats it as held-out by construction rather than by remembering to.

    # a directory of video files, all showing the same letter
    ../.venv/bin/python temporal/ingest_external.py --videos ~/Downloads/jz/J --label J \\
        --signer kaggle_jz --session EXT --out temporal/external_clips.npz

    # ChicagoFSWild-style directories of numbered frames, one directory per sequence
    ../.venv/bin/python temporal/ingest_external.py --frames ~/Downloads/fswild/seq --label NONE \\
        --signer fswild --session EXT --fps 30

Run label_events.py over the result BEFORE trusting it. Footage that begins mid-gesture never
arms a gate, because a track starts only on a rising edge from a parked launch pose, and
label_events will cut zero events from it. That diagnostic is cheap and it is the fastest way
to find out whether a download was worth the disk.

Not implemented: the Google ASL Fingerspelling parquet format. It ships MediaPipe landmarks
directly, which makes it the most natural fit of all the sources, but it may not record the
source frame width and height -- and without those, u = x*(W/H) cannot be applied. Skipping the
aspect correction triples the Z-gate false-positive rate on this repo's own archive (118 -> 367
frames). Rather than ship a loader that silently guesses 16:9, this refuses to handle the format
until someone checks whether the frame size is recoverable. See SOURCES.md.
"""
import argparse
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "external_clips.npz")

def as_object_array(seq):
    """Pack a list of arrays into a 1-D object array, one entry per clip.

    np.array(list_of_arrays, dtype=object) does NOT do this when the arrays happen to share a
    shape -- it broadcasts them into an N-D object array of Python scalars, and every reader
    downstream then fails on dtype=object. Clips recorded at a steady frame rate all have the
    same length, so the bug appears exactly when the recording went well.
    """
    out = np.empty(len(seq), dtype=object)
    for i, item in enumerate(seq):
        out[i] = item
    return out


VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def numeric_key(path):
    """Sort 0,1,2,...,10 numerically. Frame directories are almost always numbered, and
    lexicographic order would put 10.jpg between 1.jpg and 2.jpg -- which silently scrambles
    time, the one thing a motion feature cannot survive."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 0, stem)


def landmarks_from_frames(frame_iter, hands, drawer=None):
    """Run MediaPipe over an iterable of (timestamp, BGR frame). Returns per-frame arrays."""
    lms, stamps, handed = [], [], []
    size = None
    for t, frame in frame_iter:
        if frame is None:
            continue
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lms.append([(p.x, p.y, p.z) for p in hand.landmark])
            lr = "Unknown"
            if res.multi_handedness:
                lr = res.multi_handedness[0].classification[0].label
            handed.append(lr)
        else:
            lms.append([(np.nan, np.nan, np.nan)] * 21)
            handed.append("None")
        stamps.append(t)
    return (np.array(lms, dtype=np.float32), np.array(stamps, dtype=np.float32),
            np.array(handed), size)


def iter_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Prefer the container's own timestamp; fall back to the nominal frame rate. Frame
        # index alone is not a clock, and every threshold downstream is in seconds.
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if ms and ms > 0 else (i / fps if fps > 0 else i / 30.0)
        yield t, frame
        i += 1
    cap.release()


def iter_frame_dir(path, fps):
    files = sorted((p for p in glob.glob(os.path.join(path, "*"))
                    if p.lower().endswith(IMAGE_EXT)), key=numeric_key)
    for i, f in enumerate(files):
        yield i / fps, cv2.imread(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", help="directory of video files, or a single video file")
    src.add_argument("--frames", help="directory whose SUBDIRECTORIES are numbered frame sequences")
    ap.add_argument("--label", required=True, choices=["J", "Z", "NONE"],
                    help="the label for every clip in this source")
    ap.add_argument("--signer", required=True,
                    help="who is signing; distinct signers must get distinct ids or the "
                         "cross-signer claim is not a cross-signer claim")
    ap.add_argument("--session", default="EXT",
                    help="session tag; anything not S1 is held out by the by-session split")
    ap.add_argument("--fps", type=float, default=30.0, help="assumed fps for frame directories")
    ap.add_argument("--min-tracked", type=float, default=0.6,
                    help="reject a clip if fewer than this fraction of frames tracked a hand")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.videos:
        if os.path.isfile(args.videos):
            sources = [(args.videos, iter_video(args.videos))]
        else:
            files = sorted(p for p in glob.glob(os.path.join(args.videos, "*"))
                           if p.lower().endswith(VIDEO_EXT))
            sources = [(p, iter_video(p)) for p in files]
    else:
        dirs = sorted(p for p in glob.glob(os.path.join(args.frames, "*")) if os.path.isdir(p))
        if not dirs and glob.glob(os.path.join(args.frames, "*")):
            dirs = [args.frames]      # a single sequence directory was passed
        sources = [(p, iter_frame_dir(p, args.fps)) for p in dirs]

    if not sources:
        print(f"no usable input found under {args.videos or args.frames}")
        return 1
    print(f"{len(sources)} source(s), label={args.label}, signer={args.signer}, "
          f"session={args.session}\n")

    clips, stamps, handed, labels, sessions, signers, sizes = [], [], [], [], [], [], []
    if os.path.exists(args.out):
        prev = np.load(args.out, allow_pickle=True)
        clips = list(prev["clips"]); stamps = list(prev["stamps"])
        handed = list(prev["handed"]); labels = list(prev["labels"])
        sessions = list(prev["sessions"]); signers = list(prev["signers"])
        sizes = list(np.asarray(prev["frame_size"]).reshape(-1, 2))
        print(f"appending to {len(clips)} existing clips\n")

    kept = skipped = 0
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.3) as hands:
        for path, it in sources:
            lm, ts, lr, size = landmarks_from_frames(it, hands)
            name = os.path.basename(path)
            if len(lm) < 3 or size is None:
                print(f"  skip {name}: fewer than 3 readable frames")
                skipped += 1
                continue
            tracked = float(np.isfinite(lm[:, 0, 0]).mean())
            if tracked < args.min_tracked:
                print(f"  skip {name}: only {tracked:.0%} of frames tracked a hand")
                skipped += 1
                continue
            clips.append(lm); stamps.append(ts); handed.append(lr)
            labels.append(args.label); sessions.append(args.session); signers.append(args.signer)
            sizes.append(size)
            kept += 1
            print(f"  {name}: {len(lm)} frames, {tracked:.0%} tracked, "
                  f"{size[0]}x{size[1]}, {ts[-1]:.2f}s")

    if not kept:
        print("\nnothing ingested. If most clips failed the tracking threshold the footage is "
              "probably too low-resolution or too motion-blurred for MediaPipe.")
        return 1

    # Per-clip frame sizes, because external sources are not one camera. The aspect correction
    # u = x*(W/H) is applied per clip downstream; a single global size would silently mis-correct
    # every clip that did not match it.
    np.savez_compressed(args.out,
                        clips=as_object_array(clips),
                        stamps=as_object_array(stamps),
                        handed=as_object_array(handed),
                        labels=np.array(labels),
                        sessions=np.array(sessions),
                        signers=np.array(signers),
                        frame_size=np.array(sizes, dtype=np.int32))
    print(f"\nwrote {args.out}: {kept} clips kept, {skipped} skipped, {len(clips)} total")
    print(f"signers: {sorted(set(signers))}   sessions: {sorted(set(sessions))}")
    print("\nNext, and do this before trusting any of it:")
    print(f"  ../.venv/bin/python temporal/label_events.py --clips {args.out}")
    print("If J/Z clips cut zero events, the footage starts mid-gesture and never arms a gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

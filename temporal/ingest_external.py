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

Run label_events.py over the result BEFORE trusting it, AND PASS --out. Footage that begins
mid-gesture never arms a gate, because a track starts only on a rising edge from a parked
launch pose, and label_events will cut zero events from it. That diagnostic is cheap and it is
the fastest way to find out whether a download was worth the disk. It is also destructive if
run bare: label_events.py --out defaults to temporal/events.npz, the committed motion training
set, so running the diagnostic on eight ingested clips replaces 70,488 bytes of prompted events
with 3,610 bytes cut from someone else's YouTube video, and nothing says so. I did exactly that
while testing this file. Send it somewhere else.

IT HAS NOW BEEN RUN, on video, twice, and this paragraph is the record of it. First on four
sources of third-party J and Z footage -- three CC BY YouTube channels and one Signbank entry,
one J and one Z each -- which produced the eight clips in ext2.npz and, through them, the first
cross-signer numbers the motion branch has ever had: 4 of the 8 clips cut a runtime-reachable
span, 3 of 3 reachable J spans emitted J, the one reachable Z span read MOVE (Z 0.173), and
160.1 s of the same signers' fingerspelling containing no J and no Z cut 5 spans and emitted
nothing at all -- 0.00 false J/Z per minute. Second on 15 OpenHands clips (train/j, train/z and
test/j, CC BY 4.0), of which 8 were kept and 7 failed --min-tracked at 0-56%. That second run
exercised the fresh path, the APPEND path onto its own output, the append path onto a
collect_motion.py file carrying one frame size for five clips (frame_sizes broadcast it per
clip, which is what it exists for), and the --frames path on two numbered JPEG sequences, which
had never been run at all. Everything worked as written except the shared MediaPipe tracker,
which is now one per clip and is the only code change this release makes here (fresh_hands()).
The Signbank and YouTube clips are CC BY-NC-SA and CC BY respectively and are NOT committed,
neither the video nor the landmarks; SOURCES.md carries the URLs and the recipe. What is
committed is a result, not footage: temporal/openhands_replay.json, from replay_strangers.py.

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
sys.path.insert(0, HERE)
from label_events import frame_sizes

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


def landmarks_from_frames(frame_iter, hands):
    """Run MediaPipe over an iterable of (timestamp, BGR frame). Returns per-frame arrays.

    `hands` must be an instance nothing else has fed. See fresh_hands().
    """
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


def fresh_hands():
    """One MediaPipe tracker, for ONE clip, with the runtime's own detection settings.

    static_image_mode=False means the tracker carries state between frames: after a hit it
    searches the next frame near the last box instead of re-detecting. That is right inside a
    clip and wrong between clips, because these are unrelated videos and clip k's last frame is
    not a prior for clip k+1's first. One instance was shared across the whole run until this
    release, and the cost is measurable: on 18 OpenHands clips read both ways, 3 lost tracking
    on frames they otherwise held (J/2 0.938 vs 1.000, J/4 0.966 vs 1.000, A/3 0.917 vs 1.000),
    and the mean tracked fraction fell from 0.794 to 0.784. None of those three crossed the
    0.60 --min-tracked line here, but they could have: a shared tracker makes both the
    landmarks and the keep/skip decision depend on what order the files were read in, which is
    not a property an ingest may have.
    """
    return mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.3)


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
    prompts = []
    if os.path.exists(args.out):
        prev = np.load(args.out, allow_pickle=True)
        clips = list(prev["clips"]); stamps = list(prev["stamps"])
        handed = list(prev["handed"]); labels = list(prev["labels"])
        sessions = list(prev["sessions"]); signers = list(prev["signers"])
        prompts = list(prev["prompts"]) if "prompts" in prev else [""] * len(clips)
        # One (W,H) per existing clip. A collect_motion.py file may carry a single pair for
        # the whole file; reshaping that to one row and appending one row per new clip used
        # to leave frame_size with 1+K rows for N+K clips, and every reader then fell back to
        # the first row for all of them -- the ingested footage was featurized at the wrong
        # aspect with nothing printed. label_events.frame_sizes broadcasts it per clip.
        sizes = frame_sizes(prev, len(clips))
        if sizes is None:
            raise SystemExit(f"{args.out} carries no frame_size; it was not written by "
                             "collect_motion.py or this script, and appending to it would "
                             "leave its clips with no aspect")
        print(f"appending to {len(clips)} existing clips (frame sizes {sorted(set(sizes))})\n")

    kept = skipped = 0
    for path, it in sources:
        # One tracker per clip, never one per run: see fresh_hands().
        with fresh_hands() as hands:
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
        sizes.append((int(size[0]), int(size[1]))); prompts.append("")
        kept += 1
        print(f"  {name}: {len(lm)} frames, {tracked:.0%} tracked, "
              f"{size[0]}x{size[1]}, {ts[-1]:.2f}s")

    if not kept:
        print("\nnothing ingested. If most clips failed the tracking threshold the footage is "
              "probably too low-resolution or too motion-blurred for MediaPipe.")
        return 1

    # Per-clip frame sizes, because external sources are not one camera. The aspect correction
    # u = x*(W/H) is applied per clip downstream; a single global size would silently mis-correct
    # every clip that did not match it. `prompts` is written (empty for ingested clips) so a
    # file that started as a prompted recording keeps the key its takes need.
    assert len(sizes) == len(clips) == len(prompts), (len(sizes), len(clips), len(prompts))
    np.savez_compressed(args.out,
                        clips=as_object_array(clips),
                        stamps=as_object_array(stamps),
                        handed=as_object_array(handed),
                        prompts=as_object_array(prompts),
                        labels=np.array(labels),
                        sessions=np.array(sessions),
                        signers=np.array(signers),
                        frame_size=np.array(sizes, dtype=np.int32))
    print(f"\nwrote {args.out}: {kept} clips kept, {skipped} skipped, {len(clips)} total")
    print(f"signers: {sorted(set(signers))}   sessions: {sorted(set(sessions))}")
    # --out is not optional in this suggestion. Without it label_events.py overwrites
    # temporal/events.npz, the committed motion training set, with the events it cut here.
    events_out = os.path.splitext(args.out)[0] + "_events.npz"
    print("\nNext, and do this before trusting any of it:")
    print(f"  ../.venv/bin/python temporal/label_events.py --clips {args.out} --out {events_out}")
    print("If J/Z clips cut zero events, the footage starts mid-gesture and never arms a gate.")
    print("Keep the --out. Bare, it replaces temporal/events.npz and says nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

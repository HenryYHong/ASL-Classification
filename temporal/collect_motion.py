"""Record landmark sequences for the motion letters (J, Z) and for negatives.

Why this exists
---------------
The original capture cell wrote 100 full-resolution JPEGs per class, which is why the
repository carries ~650 MB of images. Motion letters need many short *clips* rather than
many single frames, so recording frames as images would multiply that cost. This script
stores only what any downstream model actually consumes: the 21 hand landmarks per frame,
plus a wall-clock timestamp per frame.

Timestamps matter. The original pipeline had none, so there was no way to tell a fast
gesture from a slow one, or to resample two clips onto a common time base. Webcam frame
rate is not constant, so frame index is not a clock.

Classes recorded
----------------
    J, Z    the two motion letters
    NONE    negatives, and specifically NEAR-MISS negatives: an I moved around that is not a
            J, a D moved around that is not a Z. These are the ones worth your time. A track
            only starts when the hand was parked in a launch pose and then moved, so a
            wandering flat hand never arms a gate and yields no training event at all --
            recording "random motion" produces zero usable negatives. The on-screen prompt
            cycles through four near-miss variants automatically; just follow it.

Usage
-----
    ./.venv/bin/python temporal/collect_motion.py --camera 0
    ./.venv/bin/python temporal/collect_motion.py --camera 0 --letters J --clips 20

Controls: SPACE starts the next clip, R redoes the previous one, Q quits and saves.
"""
import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "motion_clips.npz")

# A held sign drifts by a median of 0.018 hand-size units per frame (measured over all
# 2,378 static frames in RandomForest/data). A real gesture moves far more than that, but
# the recorder does not filter on it -- it records whatever you do and lets training decide.
GUIDANCE = {
    "J": "Pinky extended (the 'I' handshape), then trace the J hook: down, curve left, up.",
    "Z": "Index finger extended, then draw the Z: across, diagonal down-left, across again.",
    "NONE": "",   # set per clip by guidance_for(); see below
}

#: Negatives are only useful if they ARM A GATE. A track starts only when the hand was parked
#: in a launch pose and then moved, so a wandering flat hand or a reach for the mug never
#: creates a scoring opportunity and yields no training event at all -- verified: a drifting
#: B-handshape produces zero cut events. Recording "random motion" as negatives therefore
#: teaches the classifier nothing and leaves the MOVE class empty.
#:
#: The negatives that matter are the near-misses: an I moved around that is NOT a J, and a D
#: moved around that is NOT a Z. Those arm the gate, get cut into events, and are exactly the
#: false positives the classifier has to learn to reject. Alternating keeps them balanced.
NEGATIVE_PROMPTS = [
    "Hold an I (pinky out). MOVE it around -- up, down, across, near, far. Never trace a J.",
    "Hold a D / index point. MOVE it around the frame. Never draw a Z.",
    "Hold an I and reposition sharply, as if between two letters. Still never a J.",
    "Hold a D and reposition sharply, as if between two letters. Still never a Z.",
]


def guidance_for(label, clip_idx):
    if label == "NONE":
        return NEGATIVE_PROMPTS[clip_idx % len(NEGATIVE_PROMPTS)]
    return GUIDANCE.get(label, "")


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


def draw_banner(frame, lines, color=(0, 255, 0)):
    y = 40
    for text, scale, thick in lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        y += int(38 * scale) + 14


def record_one_clip(cap, hands, drawer, label, clip_idx, n_clips, duration, countdown):
    """Record one clip. Returns (landmarks, timestamps, handedness) or None if the user quit."""
    mp_hands = mp.solutions.hands

    # Countdown, so you are in position before frame zero rather than after it.
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        remaining = countdown - elapsed
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed during countdown")
        frame = cv2.flip(frame, 1)
        if remaining <= 0:
            break
        draw_banner(frame, [
            (f"{label}   clip {clip_idx + 1}/{n_clips}", 1.1, 3),
            (guidance_for(label, clip_idx), 0.6, 2),
            (f"starting in {remaining:.1f}", 0.9, 2),
        ], (0, 200, 255))
        cv2.imshow("collect_motion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    frames, stamps, hands_lr = [], [], []
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= duration:
            break
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe sees the UNMIRRORED frame. Flipping the array it processes inverts the
        # handedness label it returns, and features.canonicalize_handedness relies on that
        # label to map left hands onto the right-hand convention. A mirrored recording would
        # train the model on the opposite chirality from what the live demo feeds it -- the
        # exact train/serve skew that sank Approach A, reintroduced through a cosmetic flip.
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            frames.append([(p.x, p.y, p.z) for p in hand.landmark])
            lr = "Unknown"
            if res.multi_handedness:
                lr = res.multi_handedness[0].classification[0].label
            hands_lr.append(lr)
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        else:
            # Keep the slot so the clip's time base stays honest; NaN marks the gap.
            frames.append([(np.nan, np.nan, np.nan)] * 21)
            hands_lr.append("None")
        stamps.append(t)

        # Annotate first, then mirror the whole image for display only, so the preview reads
        # naturally without ever touching the array that was processed.
        pct = t / duration
        cv2.rectangle(frame, (20, frame.shape[0] - 40),
                      (20 + int(pct * (frame.shape[1] - 40)), frame.shape[0] - 20), (0, 0, 255), -1)
        disp = cv2.flip(frame, 1)
        draw_banner(disp, [(f"RECORDING {label}  {clip_idx + 1}/{n_clips}", 1.1, 3)], (0, 0, 255))
        cv2.imshow("collect_motion", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    return (np.array(frames, dtype=np.float32), np.array(stamps, dtype=np.float32),
            np.array(hands_lr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index; the old notebooks hard-coded 1, a built-in webcam is usually 0")
    ap.add_argument("--letters", nargs="+", default=["J", "Z", "NONE"])
    ap.add_argument("--clips", type=int, default=40, help="clips per class")
    ap.add_argument("--duration", type=float, default=2.0, help="seconds per clip")
    ap.add_argument("--countdown", type=float, default=2.5)
    ap.add_argument("--session", default="S1",
                    help="capture session tag; the train/test split is BY SESSION, so this is "
                         "what makes an honest generalization number possible later")
    ap.add_argument("--signer", default="signer1")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}; try --camera 0 or --camera 1")
    ok, probe = cap.read()
    if not ok:
        raise SystemExit(f"camera {args.camera} opened but returned no frame; try a different index")
    print(f"camera {args.camera} ok, frame {probe.shape[1]}x{probe.shape[0]}")

    # Existing clips are kept, so recording can be done across several sessions.
    clips, labels, stamps, handed, sessions, signers = [], [], [], [], [], []
    if os.path.exists(args.out):
        prev = np.load(args.out, allow_pickle=True)
        clips = list(prev["clips"])
        labels = list(prev["labels"])
        stamps = list(prev["stamps"])
        handed = list(prev["handed"]) if "handed" in prev else [None] * len(clips)
        sessions = list(prev["sessions"]) if "sessions" in prev else ["S1"] * len(clips)
        signers = list(prev["signers"]) if "signers" in prev else ["signer1"] * len(clips)
        print(f"resuming: {len(clips)} clips already recorded "
              f"({ {l: labels.count(l) for l in sorted(set(labels))} })")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    aborted = False

    # static_image_mode=False enables MediaPipe's frame-to-frame tracking, which is both
    # faster and temporally smoother than re-detecting every frame -- and smoothness is
    # exactly what a trajectory feature depends on.
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for label in args.letters:
            if aborted:
                break
            i = 0
            while i < args.clips:
                # Wait for SPACE so you control the pace between takes.
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise SystemExit("camera read failed")
                    frame = cv2.flip(frame, 1)
                    draw_banner(frame, [
                        (f"NEXT: {label}   clip {i + 1}/{args.clips}", 1.1, 3),
                        (guidance_for(label, i), 0.6, 2),
                        ("SPACE = record    R = redo last    Q = quit and save", 0.6, 2),
                    ])
                    cv2.imshow("collect_motion", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord(" "):
                        break
                    if k == ord("q"):
                        aborted = True
                        break
                    if k == ord("r") and clips:
                        dropped = labels.pop()
                        clips.pop()
                        stamps.pop()
                        handed.pop()
                        sessions.pop()
                        signers.pop()
                        if dropped == label:
                            i = max(0, i - 1)
                        print(f"dropped last {dropped} clip")
                if aborted:
                    break

                got = record_one_clip(cap, hands, drawer, label, i, args.clips,
                                      args.duration, args.countdown)
                if got is None:
                    aborted = True
                    break
                lm, ts, lr = got
                detected = int(np.isfinite(lm[:, 0, 0]).sum())
                if detected < 0.6 * len(lm):
                    # Better to catch a bad take now than to find it in the training set.
                    print(f"  !! only {detected}/{len(lm)} frames tracked a hand - retaking")
                    continue
                clips.append(lm)
                stamps.append(ts)
                labels.append(label)
                handed.append(lr)
                sessions.append(args.session)
                signers.append(args.signer)
                fps = len(ts) / max(ts[-1], 1e-6)
                print(f"  {label} {i + 1}/{args.clips}: {len(lm)} frames, "
                      f"{detected} tracked, {fps:.1f} fps")
                i += 1

    cap.release()
    cv2.destroyAllWindows()

    if not clips:
        print("nothing recorded")
        return
    np.savez_compressed(args.out,
                        clips=as_object_array(clips),
                        stamps=as_object_array(stamps),
                        handed=as_object_array(handed),
                        labels=np.array(labels),
                        sessions=np.array(sessions),
                        signers=np.array(signers),
                        frame_size=np.array([probe.shape[1], probe.shape[0]]))
    counts = {l: labels.count(l) for l in sorted(set(labels))}
    print(f"\nwrote {args.out}: {len(clips)} clips {counts}")


if __name__ == "__main__":
    main()

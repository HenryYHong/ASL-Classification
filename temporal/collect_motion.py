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
import json
import os
import shutil
import signal
import subprocess
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


def build_schedule(counts, park_s, go_s, rest_s, lead_s):
    """Alternate J, Z and near-miss negatives on a fixed rhythm.

    Each item is three phases, and the PARK phase is the point of the whole design: a track
    only arms on a rising edge out of a parked launch pose, so a recording made without a
    deliberate pause before each gesture produces nothing the segmenter can cut. Nineteen
    minutes of fluent third-party video yielded 11 usable events for exactly this reason.

    REST is not dead time. The hand is moving but no letter is being signed, so whatever the
    segmenter cuts there is a true negative, collected for free.
    """
    labels = [l for l, v in counts.items() if v > 0]
    left = {l: counts[l] for l in labels}
    order = []
    if len(labels) <= 1:
        # One label: a straight block, no alternation. Easier to stay in rhythm, and it means
        # the handshape never changes mid-take, which removes a source of tracking dropout.
        for l in labels:
            order = [l] * left[l]
    else:
        i = 0
        while any(v > 0 for v in left.values()):
            lab = labels[i % len(labels)]
            i += 1
            if left[lab] <= 0:
                continue
            left[lab] -= 1
            order.append(lab)

    items, t = [], float(lead_s)
    for k, lab in enumerate(order):
        items.append({"label": lab, "index": k,
                      "park": [t, t + park_s],
                      "go": [t + park_s, t + park_s + go_s],
                      "rest": [t + park_s + go_s, t + park_s + go_s + rest_s]})
        t += park_s + go_s + rest_s
    return items, t


def speaker(enabled):
    """Announce prompts aloud so the signer never has to look away from their own hand.

    Non-blocking: the capture loop must not stall waiting on speech, or the recording drops
    frames and the tracking rate falls, which is what starves the segmenter of events.
    """
    say = shutil.which("say") if enabled else None
    if not say:
        return lambda text: None

    def _speak(text):
        try:
            subprocess.Popen([say, "-r", "260", text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    return _speak


PHASE_TEXT = {
    "J": ("PARK: hold an I, freeze", "GO: trace the J hook"),
    "Z": ("PARK: hold a D / index point, freeze", "GO: draw the Z"),
    "NONE": ("PARK: hold the pose, freeze", "GO: MOVE it, but do NOT sign the letter"),
}


def record_continuous(cap, hands, drawer, args, probe):
    """One unbroken take. Returns (landmarks, stamps, handedness, schedule) or None."""
    mp_hands = mp.solutions.hands
    counts = {}
    for L in args.letters:
        L = L.strip().upper()
        counts[L] = args.negatives if L == "NONE" else args.clips
    items, total = build_schedule(counts, args.park, args.go, args.rest, args.lead)
    print(f"schedule: {len(items)} items, {total:.0f}s "
          f"({sum(i['label']=='J' for i in items)} J, {sum(i['label']=='Z' for i in items)} Z, "
          f"{sum(i['label']=='NONE' for i in items)} near-miss)")

    # Save on SIGTERM/SIGINT rather than losing the take. Without this, stopping a run part
    # way through discards every frame recorded so far, because the write happens at the end.
    stop = {"now": False}

    def _stop(signum, frame):
        stop["now"] = True
    prev_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev_handlers[sig] = signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass

    speak = speaker(args.speak)
    speak(f"Get ready. Recording starts in {int(args.lead)} seconds.")
    last_key = None
    said_countdown = set()

    frames, stamps, handed = [], [], []
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= total or stop["now"]:
            if stop["now"]:
                print(f"\ninterrupted at {t:.0f}s; keeping the {sum(1 for it in items if it['rest'][1] <= t)} "
                      "completed items", flush=True)
                items = [it for it in items if it["rest"][1] <= t]
            break
        ok, frame = cap.read()
        if not ok:
            break
        if t < items[0]["park"][0]:
            left = int(items[0]["park"][0] - t)
            if left in (5, 3, 2, 1) and left not in said_countdown:
                said_countdown.add(left)
                speak(str(left))
        # Never flip the array MediaPipe sees; flipping inverts the handedness label.
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            frames.append([(p.x, p.y, p.z) for p in hand.landmark])
            lr = "Unknown"
            if res.multi_handedness:
                lr = res.multi_handedness[0].classification[0].label
            handed.append(lr)
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        else:
            frames.append([(np.nan, np.nan, np.nan)] * 21)
            handed.append("None")
        stamps.append(t)

        cur = next((it for it in items if it["park"][0] <= t < it["rest"][1]), None)
        phase_now = None
        if cur is not None:
            phase_now = ("PARK" if t < cur["park"][1]
                         else "GO" if t < cur["go"][1] else "REST")
            key = (cur["index"], phase_now)
            if key != last_key:
                last_key = key
                if phase_now == "PARK":
                    spoken = {"J": "J", "Z": "Z", "NONE": "near miss"}[cur["label"]]
                    speak(spoken)
                    print(f"[{t:6.1f}s] item {cur['index']+1}/{len(items)}  {cur['label']}  PARK",
                          flush=True)
                elif phase_now == "GO":
                    speak("go")
                    print(f"[{t:6.1f}s] item {cur['index']+1}/{len(items)}  {cur['label']}  GO",
                          flush=True)
        disp = cv2.flip(frame, 1)
        if cur is None:
            left = items[0]["park"][0] - t if t < items[0]["park"][0] else 0.0
            draw_banner(disp, [("GET READY", 1.4, 4),
                               (f"starting in {left:.0f}s", 0.9, 2),
                               ("hand in frame, well lit, filling ~1/3 of the height", 0.6, 2)],
                        (0, 200, 255))
        else:
            park_txt, go_txt = PHASE_TEXT[cur["label"]]
            if t < cur["park"][1]:
                phase, txt, col = "PARK", park_txt, (0, 200, 255)
                bar = (t - cur["park"][0]) / max(args.park, 1e-6)
            elif t < cur["go"][1]:
                phase, txt, col = "GO", go_txt, (0, 0, 255)
                bar = (t - cur["go"][0]) / max(args.go, 1e-6)
            else:
                # Keeping the hand in frame is the difference between a usable take and a
                # wasted one: dropping it means the next PARK is spent raising it again, and
                # a track can only arm out of frames that are actually tracked. Measured on a
                # real take, PARK tracking was 48% while GO was 95%.
                phase, txt, col = "REST", "KEEP HAND UP in frame - just relax the shape", (120, 120, 120)
                bar = (t - cur["rest"][0]) / max(args.rest, 1e-6)
            done = sum(1 for it in items if it["rest"][1] <= t)
            draw_banner(disp, [
                (f"{cur['label']}   {phase}", 1.5, 4),
                (txt, 0.75, 2),
                (f"item {cur['index']+1}/{len(items)}   {done} done   {total-t:.0f}s left", 0.6, 2),
            ], col)
            w = disp.shape[1] - 40
            cv2.rectangle(disp, (20, disp.shape[0]-46), (20+int(min(bar,1.0)*w), disp.shape[0]-22), col, -1)
        cv2.imshow("collect_motion", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("aborted by user; saving what was recorded")
            items = [it for it in items if it["rest"][1] <= t]
            break

    for sig, h in prev_handlers.items():
        try:
            signal.signal(sig, h)
        except (ValueError, OSError):
            pass
    if len(frames) < 10:
        return None
    return (np.array(frames, dtype=np.float32), np.array(stamps, dtype=np.float32),
            np.array(handed), items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index; the old notebooks hard-coded 1, a built-in webcam is usually 0")
    ap.add_argument("--letters", nargs="+", default=["J", "Z", "NONE"])
    ap.add_argument("--clips", type=int, default=40, help="clips per class")
    ap.add_argument("--duration", type=float, default=2.0, help="seconds per clip")
    ap.add_argument("--countdown", type=float, default=2.5)
    ap.add_argument("--continuous", action="store_true",
                    help="record ONE unbroken take, prompting J / Z / near-miss on a fixed "
                         "rhythm instead of one clip per keypress. Each item is PARK, GO, REST; "
                         "the PARK phase is what lets the segmenter arm at all, and the REST "
                         "gaps yield true negatives for free.")
    ap.add_argument("--negatives", type=int, default=None,
                    help="continuous: near-miss items (an I moved WITHOUT tracing a J, and the "
                         "same for D and Z). Defaults to half the clip count. These are the "
                         "hardest negatives and the exact false positive seen in use; setting 0 "
                         "leaves only rest-gap motion as MOVE examples.")
    ap.add_argument("--speak", action="store_true",
                    help="announce each letter aloud (macOS `say`), so you can watch your hand "
                         "instead of the screen")
    ap.add_argument("--park", type=float, default=1.2, help="continuous: seconds frozen before signing")
    ap.add_argument("--go", type=float, default=1.8, help="continuous: seconds to perform the sign")
    ap.add_argument("--rest", type=float, default=1.2, help="continuous: seconds to relax between items")
    ap.add_argument("--lead", type=float, default=10.0, help="continuous: lead-in before the first item")
    ap.add_argument("--session", default="S1",
                    help="capture session tag; the train/test split is BY SESSION, so this is "
                         "what makes an honest generalization number possible later")
    ap.add_argument("--signer", default="signer1")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.negatives is None:
        args.negatives = max(1, args.clips // 2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}; try --camera 0 or --camera 1")
    # The first read straight after opening frequently fails on macOS while the device
    # negotiates a format, so retry briefly before declaring the index wrong. Failing here
    # used to send you hunting for a camera index that was already correct.
    probe = None
    for _ in range(30):
        ok, probe = cap.read()
        if ok and probe is not None:
            break
        time.sleep(0.1)
    else:
        raise SystemExit(f"camera {args.camera} opened but returned no frame after 3s; "
                         f"try a different --camera index, or grant camera access to this "
                         f"terminal in System Settings > Privacy & Security > Camera")
    print(f"camera {args.camera} ok, frame {probe.shape[1]}x{probe.shape[0]}")

    # Existing clips are kept, so recording can be done across several sessions.
    clips, labels, stamps, handed, sessions, signers, prompts = [], [], [], [], [], [], []
    if os.path.exists(args.out):
        prev = np.load(args.out, allow_pickle=True)
        clips = list(prev["clips"])
        labels = list(prev["labels"])
        stamps = list(prev["stamps"])
        handed = list(prev["handed"]) if "handed" in prev else [None] * len(clips)
        prompts = list(prev["prompts"]) if "prompts" in prev else [""] * len(clips)
        sessions = list(prev["sessions"]) if "sessions" in prev else ["S1"] * len(clips)
        signers = list(prev["signers"]) if "signers" in prev else ["signer1"] * len(clips)
        print(f"resuming: {len(clips)} clips already recorded "
              f"({ {l: labels.count(l) for l in sorted(set(labels))} })")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    aborted = False

    if args.continuous:
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            got = record_continuous(cap, hands, drawer, args, probe)
        cap.release()
        cv2.destroyAllWindows()
        if got is None:
            print("nothing recorded")
            return
        lm, ts, lr, items = got
        tracked = float(np.isfinite(lm[:, 0, 0]).mean())
        clips.append(lm); stamps.append(ts); handed.append(lr)
        labels.append("CONTINUOUS"); prompts.append(json.dumps(items))
        sessions.append(args.session); signers.append(args.signer)
        np.savez_compressed(args.out,
                            clips=as_object_array(clips),
                            stamps=as_object_array(stamps),
                            handed=as_object_array(handed),
                            prompts=as_object_array(prompts),
                            labels=np.array(labels),
                            sessions=np.array(sessions),
                            signers=np.array(signers),
                            frame_size=np.array([probe.shape[1], probe.shape[0]]))
        print(f"\nwrote {args.out}: {len(ts)} frames, {ts[-1]:.0f}s, {tracked:.0%} tracked, "
              f"{len(items)} prompted items")
        if tracked < 0.90:
            print(f"WARNING: only {tracked:.0%} of frames tracked a hand. Below ~90% the "
                  "segmenter loses tracks to detection gaps. More light, hand larger in frame.")
        print("\nNext:")
        print(f"  ../.venv/bin/python temporal/label_events.py --clips {args.out}")
        return

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
                        prompts.pop()
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
                prompts.append("")
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
                        prompts=as_object_array(prompts),
                        labels=np.array(labels),
                        sessions=np.array(sessions),
                        signers=np.array(signers),
                        frame_size=np.array([probe.shape[1], probe.shape[0]]))
    counts = {l: labels.count(l) for l in sorted(set(labels))}
    print(f"\nwrote {args.out}: {len(clips)} clips {counts}")


if __name__ == "__main__":
    main()

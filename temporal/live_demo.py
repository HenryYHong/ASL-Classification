"""Live webcam demo, and the instrument every threshold in thresholds.py is tuned against.

The loop itself is thin: grab a frame, run one hoisted MediaPipe Hands over the *unflipped*
array, hand (t, landmarks, handedness, W, H) to Segmenter.step, and append whatever it emits
to a rolling string. All the substance is the overlay.

Why the overlay is not optional (spec 5.9, failure mode 3): the highest-probability silent
failure in this system is a gate or a trigger that never fires, which makes J simply
unreachable while the demo looks perfectly healthy -- it still recognizes 24 letters. Nothing
in the emitted string distinguishes "no J was signed" from "J is unreachable". So the two
smoothed signals are drawn as meters with their thresholds as tick marks, the gate fractions
are drawn as numbers next to the fraction required to arm, and the tracked path is drawn on
the frame while it is being accumulated. If v_bar never reaches the armed tick, or the J gate
fraction never reaches GATE_ARM_FRAC, that is visible in one glance instead of one evening.

The display copy is mirrored because signing into a non-mirrored image is disorienting. The
array passed to MediaPipe is never flipped: MediaPipe's handedness label is assigned in image
space, so a flipped frame reports a right hand as "Left", and features.canonicalize_handedness
would then mirror exactly the hands it should leave alone. Pixel coordinates are mirrored at
draw time instead, which is the only place the mirror belongs.

Usage
-----
    ./.venv/bin/python temporal/live_demo.py --camera 0
    ./.venv/bin/python temporal/live_demo.py --thresholds temporal/thresholds_mycam.json
    ./.venv/bin/python temporal/live_demo.py --no-motion

Keys: Q or ESC quits, C clears the output string, SPACE prints the current signal
values (state, v_bar, sigma, gate fractions, fps) to stdout so a moment can be quoted exactly.
"""
import argparse
import os
import pickle
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from segmenter import Segmenter, TRACKING, HOLD, SETTLING, NO_HAND
from thresholds import Thresholds, DEFAULT, NEEDS_GESTURE_DATA

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL = os.path.join(HERE, "model_static.p")
MOTION_MODEL = os.path.join(HERE, "model_motion.p")

FONT = cv2.FONT_HERSHEY_SIMPLEX
#: The display copy is resampled to this width before anything is drawn, so the overlay has one
#: fixed layout instead of one per camera resolution. Nothing measured passes through it: the
#: segmenter is fed MediaPipe's normalized landmarks and the capture frame's true W and H.
DISPLAY_W = 960
STATE_COLOR = {NO_HAND: (120, 120, 120), SETTLING: (0, 200, 255),
               HOLD: (0, 255, 0), TRACKING: (0, 128, 255)}
CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


def load_model(path):
    """Both trainers pickle {'model':, 'classes':, ...}; tolerate a bare estimator too."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if isinstance(blob, dict):
        return blob["model"], list(blob.get("classes", [])) or None, blob
    return blob, None, {}


def text(img, s, x, y, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def meter(img, x, y, w, h, value, vmax, ticks, color):
    """Horizontal bar with the thresholds drawn on it as ticks.

    A number alone does not show that v_bar is sitting just under V_MOVE_ARMED; the whole
    point of tuning is seeing how close a signal comes to a threshold it never crosses.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), (35, 35, 35), -1)
    fill = int(w * min(max(value, 0.0) / vmax, 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    for tv, tcol in ticks:
        tx = x + int(w * min(tv / vmax, 1.0))
        cv2.line(img, (tx, y - 3), (tx, y + h + 3), tcol, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)


def draw_hand(img, pts, color=(200, 200, 200)):
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 2, color, -1, cv2.LINE_AA)


def draw_overlay(img, seg, th, fps, letters, out_str, jf, zf, track, note):
    W = img.shape[1]
    cv2.rectangle(img, (0, 0), (W, 132), (18, 18, 18), -1)
    color = STATE_COLOR.get(seg.state, (255, 255, 255))
    text(img, seg.state, 12, 26, color, 0.8, 2)
    text(img, f"{fps:5.1f} fps", W - 110, 26, (200, 200, 200), 0.6, 1)

    text(img, f"v_bar {seg.v_bar:5.2f}", 12, 52, (255, 255, 255))
    meter(img, 130, 40, 220, 14, seg.v_bar, max(th.V_MOVE_UNARMED * 1.5, 3.0),
          [(th.V_STILL, (0, 255, 0)), (th.V_MOVE_ARMED, (0, 200, 255)),
           (th.V_MOVE_UNARMED, (0, 0, 255))], (255, 180, 60))
    text(img, f"still<{th.V_STILL:.2f}  armed>{th.V_MOVE_ARMED:.2f}  free>{th.V_MOVE_UNARMED:.2f}",
         362, 52, (170, 170, 170), 0.42)

    text(img, f"sigma {seg.sigma:5.3f}", 12, 76, (255, 255, 255))
    meter(img, 130, 64, 220, 14, seg.sigma, max(th.RIGID_VETO * 1.5, 0.3),
          [(th.SHAPE_STABLE, (0, 255, 0)), (th.RIGID_VETO, (0, 0, 255))], (255, 180, 60))
    text(img, f"stable<{th.SHAPE_STABLE:.2f}  veto>{th.RIGID_VETO:.2f}",
         362, 76, (170, 170, 170), 0.42)

    armed = seg._arm if seg.state == TRACKING else None
    ready = "J" if jf >= th.GATE_ARM_FRAC and jf >= zf else ("Z" if zf >= th.GATE_ARM_FRAC else None)
    tag = f"ARMED {armed}" if armed else (f"gate ready {ready}" if ready else "gate none")
    tcol = (0, 128, 255) if armed else ((0, 255, 0) if ready else (140, 140, 140))
    text(img, f"gate J {jf:.2f}   Z {zf:.2f}   arm>={th.GATE_ARM_FRAC:.2f}", 12, 100,
         (255, 255, 255))
    text(img, tag, 330, 100, tcol, 0.6, 2)

    if track is not None:
        elapsed, L = track
        text(img, f"track {elapsed:4.2f}s / {th.T_MAX:.2f}   path {L:5.2f} palm "
                  f"(need {th.L_MIN:.1f}-{th.L_MAX:.0f})", 12, 122, (0, 200, 255), 0.5)
    elif note:
        text(img, note, 12, 122, (0, 165, 255), 0.5)

    h = img.shape[0]
    cv2.rectangle(img, (0, h - 52), (W, h), (18, 18, 18), -1)
    text(img, " ".join(letters[-5:]) or "-", 12, h - 16, (0, 255, 0), 1.1, 3)
    text(img, out_str[-46:], 240, h - 16, (200, 200, 200), 0.6, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index; a built-in webcam is usually 0")
    ap.add_argument("--thresholds", default=None,
                    help="JSON written by Thresholds.to_json (see temporal/calibrate.py)")
    ap.add_argument("--no-motion", action="store_true",
                    help="run the static branch alone; the J/Z tracker still shows on the overlay")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480,
                    help="640x480 is requested by default: shorter exposure means less motion "
                         "blur mid-gesture, and landmarks are normalized so no detail is lost")
    args = ap.parse_args()

    if args.thresholds:
        try:
            th = Thresholds.from_json(args.thresholds)
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"could not load thresholds from {args.thresholds}: {exc}")
        print(f"thresholds: {args.thresholds}")
    else:
        th = DEFAULT
        print("thresholds: archive defaults. V_STILL and SHAPE_STABLE are p95s of a 15 fps "
              "1080p archive and V_MOVE_UNARMED is just above that archive's maximum; none of "
              "the three was measured on this camera. Run calibrate.py and pass --thresholds "
              "to replace them with measurements.")
    print("still guesses until J/Z footage exists: " +
          ", ".join(f"{k}={getattr(th, k)}" for k in NEEDS_GESTURE_DATA))

    if not os.path.exists(STATIC_MODEL):
        raise SystemExit(f"{STATIC_MODEL} not found; run temporal/train_static.py first")
    static_model, static_classes, blob = load_model(STATIC_MODEL)
    if blob.get("feature") not in (None, "shape42/v1"):
        raise SystemExit(f"{STATIC_MODEL} was trained on feature {blob['feature']!r}, but the "
                         "segmenter feeds shape42/v1; retrain before running live")
    print(f"static model: {len(static_classes or [])} classes, "
          f"{blob.get('n_train', '?')} training frames")

    # One standing line on the overlay, so the reason J is unreachable is on screen and not
    # only in the startup log the user scrolled past.
    motion_model = motion_classes = None
    note = "" if args.thresholds else "thresholds: archive defaults, not calibrated on this camera"
    if args.no_motion:
        print("--no-motion: the motion branch is disabled; J and Z will not be emitted")
        note = "--no-motion: J and Z are disabled; tracking still runs so the gates show"
    elif os.path.exists(MOTION_MODEL):
        motion_model, motion_classes, mblob = load_model(MOTION_MODEL)
        if mblob.get("feature") not in (None, "event/v1"):
            raise SystemExit(f"{MOTION_MODEL} was trained on feature {mblob['feature']!r}, but "
                             "the segmenter feeds event/v1; retrain before running live")
        print(f"motion model: classes {motion_classes}, {mblob.get('n_train', '?')} training "
              "rows -- augmented copies included, and the pickle does not record how many "
              "independent clips they came from")
    else:
        print(f"WARNING: {MOTION_MODEL} not found. J and Z CANNOT be recognized until it is "
              "trained (record with collect_motion.py, then train_motion.py). Everything else "
              "runs; the overlay still shows arming and tracking so the gates can be tuned.")
        note = "no model_motion.p: J and Z cannot be recognized until it is trained"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}; try --camera 0 or --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        raise SystemExit(f"camera {args.camera} opened but returned no frame; try another index")
    H, W = probe.shape[:2]
    print(f"camera {args.camera} ok, frame {W}x{H}. Q quits, C clears, SPACE prints signals.")

    DW, DH = DISPLAY_W, int(round(DISPLAY_W * H / W))
    seg = Segmenter(th, static_model, motion_model, static_classes, motion_classes)
    letters, out_str = [], ""
    frame_times = deque(maxlen=30)
    # Normalized landmarks, kept here rather than read back out of the segmenter: its buffer
    # holds isotropic, chirality-canonicalized coordinates, which cannot be drawn on a frame.
    px_hist = deque(maxlen=300)
    misses = 0

    def px(xy):
        """Normalized landmark -> display pixel. x is mirrored here and nowhere else."""
        return int((1.0 - xy[0]) * DW), int(xy[1] * DH)

    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.3) as hands:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > 30:
                    print("camera stopped returning frames; exiting")
                    break
                continue
            misses = 0
            t = time.perf_counter()
            frame_times.append(t)

            frame.flags.writeable = False
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.flags.writeable = True

            lm = handed = None
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = np.array([(p.x, p.y, p.z) for p in hand.landmark], dtype=np.float64)
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label
                px_hist.append((t, lm[:, :2].copy()))

            em = seg.step(t, lm, handed, W, H)
            if em is not None:
                letters.append(em.letter)
                out_str += em.letter
                print(f"[{t:9.3f}] {em.letter}  {em.kind:6s} p={em.confidence:.3f} {em.detail}")

            disp = cv2.resize(cv2.flip(frame, 1), (DW, DH))
            if lm is not None:
                draw_hand(disp, [px(p) for p in px_hist[-1][1]])

            # Gate fractions come from the segmenter's own method over its own buffer. A
            # reimplementation here could drift from the state machine and make the overlay
            # lie about the one thing it exists to report.
            jf = seg._gate_fraction(t, "j_gate") if seg.buf else 0.0
            zf = seg._gate_fraction(t, "z_gate") if seg.buf else 0.0

            track = None
            if seg.state == TRACKING and seg._track_from is not None:
                t0 = seg._track_from
                tip_i = F.TIP_FOR_ARM[seg._arm]
                frames = seg._since(t0)
                if len(frames) >= 2:
                    tip = np.stack([f.P[tip_i] for f in frames])
                    S = float(np.median([f.S for f in frames]))
                    track = (t - t0, F.path_length(tip) / max(S, 1e-9))
                trail = [px(p[tip_i]) for ts, p in px_hist if ts >= t0]
                for a, b in zip(trail, trail[1:]):
                    cv2.line(disp, a, b, (0, 128, 255), 2, cv2.LINE_AA)
                if trail:
                    cv2.circle(disp, trail[-1], 6, (0, 128, 255), -1, cv2.LINE_AA)

            fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6) \
                if len(frame_times) > 1 else 0.0
            draw_overlay(disp, seg, th, fps, letters, out_str, jf, zf, track, note)
            cv2.imshow("live_demo", disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                letters, out_str = [], ""
            if k == ord(" "):
                print(f"v_bar={seg.v_bar:.3f} sigma={seg.sigma:.3f} state={seg.state} "
                      f"jf={jf:.2f} zf={zf:.2f} fps={fps:.1f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nemitted {len(letters)} letters: {out_str or '(none)'}")


if __name__ == "__main__":
    main()

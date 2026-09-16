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

Numbers mode (--mode digits, or the M key while running) swaps in the 10-class digit forest
(temporal/model_digits.p, feature static/v4) with the vote constants in
thresholds.DIGITS_OVERRIDES, no motion model and the launch gates disarmed -- a '1' is the Z
launch pose, and a track armed off it would park the machine in TRACKING where nothing
static is ever voted. The Segmenter is rebuilt on every toggle rather than having its fields
swapped, so a parked D cannot be delivered as a digit and a last-emitted O cannot suppress a
0. That forest was trained on 218 signers of a public photo set and never on this project's
author; the overlay says so for as long as the mode is on.

Usage
-----
    ./.venv/bin/python temporal/live_demo.py --camera 0
    ./.venv/bin/python temporal/live_demo.py --thresholds temporal/thresholds_mycam.json
    ./.venv/bin/python temporal/live_demo.py --no-motion
    ./.venv/bin/python temporal/live_demo.py --mode digits

Keys: Q or ESC quits, C clears the output string, M toggles letters / numbers, SPACE prints
the current signal values (state, v_bar, sigma, sigma_rigid, gate fractions, fps) to stdout
so a moment can be quoted exactly.
"""
import argparse
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import replace

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from segmenter import Segmenter, TRACKING, HOLD, SETTLING, NO_HAND
from thresholds import Thresholds, DEFAULT, NEEDS_GESTURE_DATA, DIGITS_OVERRIDES

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL = os.path.join(HERE, "model_static.p")
MOTION_MODEL = os.path.join(HERE, "model_motion.p")
DIGITS_MODEL = os.path.join(HERE, "model_digits.p")
#: The feature tag both shipped forests are trained on (112-D): the letter forest and the
#: digit forest (train_digits.py SHIP_FEATURE; it was measured and shipped on static/v4, not
#: the static/v3 first planned for it). Any tag in features.STATIC_FEATURES is accepted -- the
#: segmenter builds the vector the pickle names -- but a forest on another tag is called out
#: at startup, so the note below fires only when a pickle genuinely differs from what ships.
LETTERS_TAG = "static/v4"
DIGITS_TAG = "static/v4"
DIGITS_NOTE = "NUMBERS 0-9: J/Z off; trained on 218 signers from a public dataset, not on you"

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


def load_static(path, what, expect_tag):
    """A static forest plus the feature tag its pickle names, checked against the registry.

    The old check compared the tag to one hard-coded string; with two static forests on two
    layouts the registry is the check, and the tag is handed to the Segmenter so the vector it
    builds is the one the forest was fitted on (it also verifies the width against the
    estimator). A pickle with no tag is the original static/v3.
    """
    model, classes, blob = load_model(path)
    tag = blob.get("feature") or F.DEFAULT_STATIC_TAG
    if tag not in F.STATIC_FEATURES:
        raise SystemExit(f"{path} was trained on feature {tag!r}, which this build cannot "
                         f"compute (known: {sorted(F.STATIC_FEATURES)}); retrain before "
                         "running live")
    dim = F.STATIC_FEATURES[tag][1]
    print(f"{what} model: {len(classes or [])} classes, {blob.get('n_train', '?')} training "
          f"rows, feature {tag} ({dim}-D)")
    if tag != expect_tag:
        print(f"  note: the shipped {what} forest is {expect_tag}; this pickle is {tag}")
    return model, classes, tag


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


def draw_overlay(img, seg, th, fps, letters, out_str, jf, zf, track, note, mode="letters"):
    W = img.shape[1]
    cv2.rectangle(img, (0, 0), (W, 132), (18, 18, 18), -1)
    color = STATE_COLOR.get(seg.state, (255, 255, 255))
    text(img, seg.state, 12, 26, color, 0.8, 2)
    text(img, f"{fps:5.1f} fps", W - 110, 26, (200, 200, 200), 0.6, 1)
    if mode == "digits":
        text(img, "NUMBERS", W - 260, 26, (0, 165, 255), 0.7, 2)

    text(img, f"v_bar {seg.v_bar:5.2f}", 12, 52, (255, 255, 255))
    meter(img, 130, 40, 220, 14, seg.v_bar, max(th.V_MOVE_UNARMED * 1.5, 3.0),
          [(th.V_STILL, (0, 255, 0)), (th.V_MOVE_ARMED, (0, 200, 255)),
           (th.V_MOVE_UNARMED, (0, 0, 255))], (255, 180, 60))
    text(img, f"still<{th.V_STILL:.2f}  armed>{th.V_MOVE_ARMED:.2f}  free>{th.V_MOVE_UNARMED:.2f}",
         362, 52, (170, 170, 170), 0.42)

    # Two meters for two signals. Stability (SHAPE_STABLE) reads the plain sigma; the rigidity
    # veto reads the ROTATION-ALIGNED sigma_rigid, and drawing the veto tick on the plain
    # meter showed a bar near the red line while no veto was close (plain sigma exceeds
    # RIGID_VETO on 25 frames of the committed takes, sigma_rigid on 9).
    text(img, f"sigma {seg.sigma:5.3f}", 12, 76, (255, 255, 255))
    meter(img, 130, 64, 170, 14, seg.sigma, max(th.SHAPE_STABLE * 3.0, 0.3),
          [(th.SHAPE_STABLE, (0, 255, 0))], (255, 180, 60))
    text(img, f"stable<{th.SHAPE_STABLE:.2f}", 312, 76, (170, 170, 170), 0.42)
    text(img, f"rigid {seg.sigma_rigid:5.3f}", 440, 76, (255, 255, 255))
    meter(img, 560, 64, 170, 14, seg.sigma_rigid, max(th.RIGID_VETO * 1.5, 0.3),
          [(th.RIGID_VETO, (0, 0, 255))], (255, 180, 60))
    text(img, f"veto>{th.RIGID_VETO:.2f}", 742, 76, (170, 170, 170), 0.42)

    armed = seg._arm if seg.state == TRACKING else None
    ready = "J" if jf >= th.GATE_ARM_FRAC and jf >= zf else ("Z" if zf >= th.GATE_ARM_FRAC else None)
    if not seg.arm_gates:
        tag = "gates off (numbers)"
    else:
        tag = f"ARMED {armed}" if armed else (f"gate ready {ready}" if ready else "gate none")
    tcol = (0, 128, 255) if armed else ((0, 255, 0) if ready and seg.arm_gates
                                        else (140, 140, 140))
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
    ap.add_argument("--log", default=None,
                    help="append every attempted track to this JSONL, with the raw landmark "
                         "span and the reason it was accepted or rejected. This is how a live "
                         "failure gets diagnosed instead of guessed at.")
    ap.add_argument("--thresholds", default=None,
                    help="JSON written by Thresholds.to_json (see temporal/calibrate.py)")
    ap.add_argument("--no-motion", action="store_true",
                    help="run the static branch alone; the J/Z tracker still shows on the overlay")
    ap.add_argument("--mode", choices=("letters", "digits"), default="letters",
                    help="start in letters (24 static + J/Z) or numbers (0-9, gates off); the M "
                         "key toggles while running")
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
    static_model, static_classes, static_tag = load_static(STATIC_MODEL, "static", LETTERS_TAG)

    digits = None
    if os.path.exists(DIGITS_MODEL):
        digits = load_static(DIGITS_MODEL, "digits", DIGITS_TAG)
    elif args.mode == "digits":
        raise SystemExit(f"{DIGITS_MODEL} not found; numbers mode needs it (temporal/"
                         "train_digits.py writes it)")
    else:
        print(f"no {os.path.basename(DIGITS_MODEL)}: the M key (numbers mode) is disabled")

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
    # Retry briefly: on macOS the first read after opening routinely fails while the device
    # negotiates a format, and treating that as a wrong camera index sends you hunting for a
    # problem that does not exist. collect_motion.py already did this; live_demo did not, so
    # the same camera that recorded fine refused to open the demo.
    probe = None
    for _ in range(30):
        ok, probe = cap.read()
        if ok and probe is not None:
            break
        time.sleep(0.1)
    else:
        cap.release()
        raise SystemExit(f"camera {args.camera} opened but returned no frame after 3s; try "
                         f"another --camera index, or grant camera access to this terminal in "
                         f"System Settings > Privacy & Security > Camera")
    H, W = probe.shape[:2]
    print(f"camera {args.camera} ok, frame {W}x{H}. Q quits, C clears, SPACE prints signals.")

    DW, DH = DISPLAY_W, int(round(DISPLAY_W * H / W))
    _logf = open(args.log, 'a') if args.log else None
    def _log_span(ev):
        # Fired before any veto runs, so a rejected gesture is captured too --
        # the rejected ones are exactly the ones worth looking at.
        if _logf is None:
            return
        import json as _json
        _logf.write(_json.dumps({'t': float(ev['t_end']), 'arm': ev['arm'],
            'duration': float(ev['duration']),
            'reason': ev.get('reason', 'scored'),
            'times': [float(x) for x in ev['times']],
            'P': np.asarray(ev['P']).tolist()}) + '\n')
        _logf.flush()
    def _log_hold(h):
        if _logf is None:
            return
        import json as _json
        _logf.write(_json.dumps({"kind": "hold", **h}) + "\n")
        _logf.flush()

    def build_segmenter(mode):
        """A fresh Segmenter for the mode. Numbers: the digit forest, DIGITS_OVERRIDES on the
        vote, no motion model, gates disarmed. Rebuilt whole so no parked letter, cooldown or
        last_emitted crosses from one mode into the other."""
        if mode == "digits":
            d_model, d_classes, d_tag = digits
            return Segmenter(replace(th, **DIGITS_OVERRIDES), d_model, None, d_classes, None,
                             on_event=_log_span, on_hold=_log_hold, static_feature_tag=d_tag,
                             arm_gates=False)
        return Segmenter(th, static_model, motion_model, static_classes, motion_classes,
                         on_event=_log_span, on_hold=_log_hold, static_feature_tag=static_tag)

    mode = args.mode
    seg = build_segmenter(mode)
    letters_note = note
    if mode == "digits":
        note = DIGITS_NOTE
        print(f"numbers mode: {DIGITS_NOTE}")
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
            draw_overlay(disp, seg, seg.th, fps, letters, out_str, jf, zf, track, note, mode)
            cv2.imshow("live_demo", disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                letters, out_str = [], ""
            if k == ord("m"):
                if digits is None:
                    print(f"numbers mode needs {DIGITS_MODEL}; staying in letters")
                else:
                    mode = "digits" if mode == "letters" else "letters"
                    seg = build_segmenter(mode)
                    note = DIGITS_NOTE if mode == "digits" else letters_note
                    print(f"mode: {mode}" + (f" -- {DIGITS_NOTE}" if mode == "digits" else ""))
            if k == ord(" "):
                print(f"v_bar={seg.v_bar:.3f} sigma={seg.sigma:.3f} "
                      f"sigma_rigid={seg.sigma_rigid:.3f} state={seg.state} "
                      f"jf={jf:.2f} zf={zf:.2f} fps={fps:.1f} mode={mode}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nemitted {len(letters)} letters: {out_str or '(none)'}")


if __name__ == "__main__":
    main()

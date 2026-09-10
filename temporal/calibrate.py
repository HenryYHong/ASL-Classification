"""Turn the stillness thresholds from a guess into a measurement.

Two modes, and they answer two different questions.

`--offline` recomputes the constants that the committed archive can settle, and is the
provenance of the defaults in thresholds.py. Nothing here invents a number: each one is a
named percentile of a named signal over RandomForest/data/, and the table prints the signal
and the percentile next to the value so the claim can be checked rather than believed.

`--live` runs the 10-second startup calibration from section 5.8 of the spec: 5 s of holding
a letter still, 5 s of moving the hand around. The archive was captured on one camera at
~15 fps; V_STILL and SHAPE_STABLE are properties of *that* camera's jitter as much as of
signing, so they are re-measured on the camera actually in use. It also fails loudly on a
wrong camera index, which the README documents as two minutes of looking busy while writing
nothing.

THE TIME BASE. The archive has no clock: it is 2,400 JPEGs with no timestamps. The capture
loop wrote one file per frame as it went, so the files' mtimes *are* the inter-frame
intervals -- median 66.4 ms, 15.07 fps, 157.4 s across the 24 bursts. That is a proxy, and
it is destroyed by any operation that rewrites mtimes (a fresh clone, a copy without -p), so
`--offline` checks the spacing before trusting it and refuses rather than reporting garbage.
Every threshold below is in seconds and palm-widths, so it survives the frame rate changing.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from thresholds import DEFAULT as TH_DEFAULT
from thresholds import Thresholds, DEFAULT, NEEDS_GESTURE_DATA

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "static_sequences.npz")
DATA_DIR = os.path.join(HERE, "..", "RandomForest", "data")
N_CLASSES, N_FRAMES = 24, 100
ARCHIVE_WH = (1920, 1080)      # the capture resolution; only the ratio enters to_isotropic

L_WINDOW = 1.2                 # seconds, the window L_MIN's safety claim is made over
L_WINDOW_MIN_COVER = 1.0       # a window shorter than this is a clip edge, not a measurement


def _pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def _ceil2(x):
    """Round a stillness ceiling UP to 2 dp. A measured max is an observed maximum, not a
    bound; rounding one down would put the threshold below a value already seen."""
    return math.ceil(x * 100.0 - 1e-9) / 100.0


# ---------------------------------------------------------------- archive time base

def archive_times(c):
    """(times, frame_indices) for class `c`, seconds from its first frame."""
    idx, ts = [], []
    for f in range(N_FRAMES):
        p = os.path.join(DATA_DIR, str(c), f"{f}.jpg")
        if os.path.exists(p):
            idx.append(f)
            ts.append(os.stat(p).st_mtime)
    if not ts:
        raise FileNotFoundError(f"no frames under {os.path.join(DATA_DIR, str(c))}")
    t = np.asarray(ts, dtype=np.float64)
    return t - t[0], np.asarray(idx)


def check_time_base(spans, dts):
    """Refuse to calibrate off mtimes that were clearly not written frame by frame."""
    med = float(np.median(dts))
    if med < 0.01 or med > 0.5:
        raise SystemExit(
            f"archive mtimes give a median inter-frame interval of {med * 1000:.1f} ms, which is "
            "not a plausible capture rate. The mtimes have been rewritten -- a fresh clone or a "
            "copy without -p does this -- so the archive no longer carries a usable clock and "
            "--offline cannot measure anything. Recover the original mtimes, or recalibrate with "
            "--live on the camera you are actually using.")
    if not (100.0 < spans.sum() < 250.0):
        raise SystemExit(f"archive spans {spans.sum():.1f} s in total, expected ~157 s; "
                         "the mtimes are not the capture times.")


# ---------------------------------------------------------------- archive signals

def held_sign_signals():
    """Every per-frame stillness signal the archive can produce, pooled over 24 held signs.

    v_bar and sigma are computed by the same functions the segmenter calls, so a percentile
    of these arrays is directly comparable to the threshold it sets. Handedness is not
    canonicalized: the archive carries no handedness label, and mirroring changes no distance
    that appears below.
    """
    if not os.path.exists(SEQ):
        raise SystemExit(f"{SEQ} not found -- run extract_static_sequences.py first")
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"{os.path.normpath(DATA_DIR)} not found. The landmarks in "
                         f"{os.path.basename(SEQ)} are not enough: the image files are the only "
                         "clock this archive has.")
    d = np.load(SEQ)
    lm, found = d["lm"], d["found"]

    v_all, sigma_all, path_all = [], [], []
    tip_path = {name: [] for name in ("index", "pinky")}
    spans, dts, n_found = [], [], 0

    for c in range(N_CLASSES):
        t_all, idx_all = archive_times(c)
        keep = found[c][idx_all]
        t, idx = t_all[keep], idx_all[keep]
        if len(t) < 3:
            continue
        spans.append(t_all[-1] - t_all[0])
        dts.append(np.diff(t_all))
        n_found += len(t)

        P = F.to_isotropic(lm[c][idx][:, :, :2], *ARCHIVE_WH)
        m, S = F.palm_centre(P), F.palm_scale(P)

        # palm_speed pads a leading 0.0 so its output aligns with the frames; that zero is a
        # placeholder, not a measured speed, and the 5-sample smoothing spreads it. Drop it.
        v_all.append(F.palm_speed(m, S, t, window_s=TH_DEFAULT.V_SMOOTH_WINDOW)[1:])
        sigma_all.append(F.rolling_shape_sigma(F.shape42(P), t, DEFAULT.SHAPE_WINDOW))

        for i in range(len(t)):
            j = np.searchsorted(t, t[i] - L_WINDOW)
            if t[i] - t[j] < L_WINDOW_MIN_COVER:
                continue
            # Scale by the window's own median palm size, which is what the segmenter does at
            # scoring time (S_evt is the median over the event, not over anything longer).
            s = float(np.median(S[j:i + 1]))
            path_all.append(F.path_length(m[j:i + 1]) / s)
            tip_path["index"].append(F.path_length(P[j:i + 1, F.TIPS["index"], :]) / s)
            tip_path["pinky"].append(F.path_length(P[j:i + 1, F.TIPS["pinky"], :]) / s)

    dts = np.concatenate(dts)
    spans = np.asarray(spans)
    check_time_base(spans, dts)
    return {
        "v_bar": np.concatenate(v_all),
        "sigma": np.concatenate(sigma_all),
        "path12": np.asarray(path_all),
        "tip12": {k: np.asarray(v) for k, v in tip_path.items()},
        "n_found": n_found,
        "total_s": float(spans.sum()),
        "fps": 1.0 / float(np.median(dts)),
    }


# ---------------------------------------------------------------- offline mode

_ROW = "{:<17}{:>9}{:>13}{:>10}  {}"


def run_offline(out_path):
    sig = held_sign_signals()
    v, s = sig["v_bar"], sig["sigma"]

    print(f"archive: {N_CLASSES} held signs, {sig['n_found']} frames with a hand, "
          f"{sig['total_s']:.1f} s at {sig['fps']:.2f} fps (from file mtimes)\n")

    rows = [
        ("V_STILL",        DEFAULT.V_STILL,        _pct(v, 95),   "p95",  "held-sign v_bar"),
        ("V_MOVE_ARMED",   DEFAULT.V_MOVE_ARMED,   _pct(v, 99),   "p99",  "held-sign v_bar"),
        ("V_MOVE_UNARMED", DEFAULT.V_MOVE_UNARMED, float(v.max()), "max", "held-sign v_bar"),
        ("SHAPE_STABLE",   DEFAULT.SHAPE_STABLE,   _pct(s, 95),   "p95",
         f"rolling_shape_sigma, trailing {DEFAULT.SHAPE_WINDOW:.1f} s"),
    ]
    print(_ROW.format("constant", "default", "recomputed", "pctile", "signal"))
    print("-" * 88)
    for name, default, value, pct, source in rows:
        print(_ROW.format(name, f"{default:.3f}", f"{value:.3f}", pct, source))

    # The whole-clip reference is a different quantity, not a looser version of the same one:
    # slow drift across a 6.5 s hold accumulates into it. Printed so the factor-of-three gap
    # is visible rather than being a claim in a docstring.
    seq = np.load(SEQ)
    whole = _pct(np.concatenate([_whole_clip_sigma(seq, c) for c in range(N_CLASSES)]), 95)
    print(f"\n  sigma p95 against a whole-clip median instead: {whole:.3f} "
          f"({whole / _pct(s, 95):.1f}x the trailing-window value). SHAPE_STABLE is defined "
          f"against the trailing window; the two are not interchangeable.")
    print(f"  sigma p99 (trailing window): {_pct(s, 99):.3f}")

    p12 = sig["path12"]
    print(f"\nL_MIN safety, palm-centre path over any {L_WINDOW:.1f} s window of a held sign:")
    print(f"  median {np.median(p12):.3f}   p95 {_pct(p12, 95):.3f}   max {p12.max():.3f} palm"
          f"   ({len(p12)} windows)")
    verdict = "excludes every hold" if p12.max() < DEFAULT.L_MIN else "DOES NOT exclude every hold"
    print(f"  L_MIN = {DEFAULT.L_MIN:.2f} palm {verdict}, margin "
          f"{DEFAULT.L_MIN - p12.max():+.3f} palm.")
    for name in ("index", "pinky"):
        tp = sig["tip12"][name]
        flag = "" if tp.max() < DEFAULT.L_MIN else "   <- above L_MIN"
        print(f"  same window, {name}-tip path: max {tp.max():.3f} palm{flag}")
    print("  The tip figures are the honest caveat: a fidgeting fingertip can travel further\n"
          "  than L_MIN during a hold, so L_MIN alone is not what keeps held signs out of the\n"
          "  motion branch. The rising-edge arming is -- replaying the archive produces zero\n"
          "  track starts, so no held frame ever reaches the L_MIN test at all.")

    print(f"\nNOT calibratable from held signs ({len(NEEDS_GESTURE_DATA)} constants). A held sign "
          "contains no J and no Z,\nso the archive carries no evidence about any of these. They "
          "keep their guessed defaults\nand must be re-measured once J/Z footage exists:")
    for name in NEEDS_GESTURE_DATA:
        print(f"  {name:<15} {getattr(DEFAULT, name):>7}   {_why_not_measurable(name)}")

    th = Thresholds(**{**asdict(DEFAULT),
                       "V_STILL": _ceil2(rows[0][2]),
                       "V_MOVE_ARMED": _ceil2(rows[1][2]),
                       "V_MOVE_UNARMED": _ceil2(rows[2][2]),
                       "SHAPE_STABLE": _ceil2(rows[3][2])})
    _emit(th, out_path,
          "recomputed values, rounded up to 2 dp; the NEEDS-GESTURE-DATA constants are copied\n"
          "from thresholds.DEFAULT untouched. thresholds.py ships slightly rounder numbers "
          "still\n(0.85 / 1.30 / 2.10 / 0.14); the difference is margin, not disagreement.")
    return th


def _whole_clip_sigma(d, c):
    t_all, idx_all = archive_times(c)
    keep = d["found"][c][idx_all]
    P = F.to_isotropic(d["lm"][c][idx_all[keep]][:, :, :2], *ARCHIVE_WH)
    return F.shape_sigma(F.shape42(P))


def _why_not_measurable(name):
    return {
        "RIGID_VETO": "how much a hand reshapes DURING a J or Z; no gesture in the archive",
        "T_MIN": "how long a J or Z takes; nothing in the archive is a gesture",
        "T_MAX": "same, upper end",
        "P_EMIT": "an operating point on the motion classifier, which has no training data yet",
        "MARGIN": "same; sweep on S1 GroupKFold once clips exist, never on the test session",
    }[name]


# ---------------------------------------------------------------- live mode

def _open_camera(index):
    import cv2
    cap = cv2.VideoCapture(index)
    ok, frame = (cap.read() if cap.isOpened() else (False, None))
    if not ok or frame is None:
        cap.release()
        raise SystemExit(
            f"camera {index} opened no frames (cap.read() returned ret=False). This is the "
            "failure the README\ndocuments as silent: with only a built-in webcam the index is "
            "almost always 0. Try --camera 0,\nor close whatever else is holding the camera.")
    return cap, frame.shape[1], frame.shape[0]


def _phase(cap, hands, drawer, prompt, seconds, countdown=2.0):
    """Collect one calibration phase. Returns (times, P) over frames where a hand was found."""
    import cv2
    import mediapipe as mp

    times, pts = [], []
    t0 = None
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera stopped returning frames mid-calibration")
        H, W = frame.shape[:2]
        elapsed = time.perf_counter() - start

        # MediaPipe sees the unmirrored frame; only the displayed copy is flipped. Feeding it a
        # mirrored array inverts the handedness label the segmenter's canonicalization relies on.
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None

        if elapsed >= countdown:
            if t0 is None:
                t0 = time.perf_counter()
            if hand is not None:
                lr = (res.multi_handedness[0].classification[0].label
                      if res.multi_handedness else None)
                P = F.to_isotropic([[p.x, p.y] for p in hand.landmark], W, H)
                pts.append(F.canonicalize_handedness(P, lr))
                times.append(time.perf_counter() - t0)
            if time.perf_counter() - t0 >= seconds:
                break

        if hand is not None:
            drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        disp = cv2.flip(frame, 1)
        if elapsed < countdown:
            banner = [(prompt, 0.9), (f"starting in {countdown - elapsed:.1f}", 0.8)]
            colour = (0, 200, 255)
        else:
            banner = [(prompt, 0.9),
                      (f"{seconds - (time.perf_counter() - t0):.1f} s left", 0.8),
                      ("hand: yes" if hand is not None else "hand: NOT FOUND", 0.8)]
            colour = (0, 255, 0) if hand is not None else (0, 0, 255)
        y = 40
        for text, scale in banner:
            cv2.putText(disp, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
            cv2.putText(disp, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)
            y += int(38 * scale) + 12
        cv2.imshow("calibrate", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("calibration aborted; no thresholds written")

    return np.asarray(times), np.asarray(pts)


def _phase_signals(times, P):
    m, S = F.palm_centre(P), F.palm_scale(P)
    v = F.palm_speed(m, S, times, window_s=TH_DEFAULT.V_SMOOTH_WINDOW)[1:]      # same leading-placeholder drop as offline
    sigma = F.rolling_shape_sigma(F.shape42(P), times, DEFAULT.SHAPE_WINDOW)
    return v, sigma


def run_live(out_path, camera, hold_s, move_s):
    import cv2
    import mediapipe as mp

    cap, W, H = _open_camera(camera)
    print(f"camera {camera}: {W}x{H}")
    try:
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as hands:
            drawer = mp.solutions.drawing_utils
            t_hold, P_hold = _phase(cap, hands, drawer,
                                    "HOLD ANY LETTER STILL", hold_s)
            t_move, P_move = _phase(cap, hands, drawer,
                                    "MOVE YOUR HAND AROUND", move_s)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # A calibration run that saw almost no hand produces confident nonsense: p95 of four
    # samples is one sample. Refuse instead, and say which of the two likely causes it is.
    for name, t, expect in (("hold", t_hold, hold_s), ("move", t_move, move_s)):
        if len(t) < 20 or (len(t) / max(expect, 1e-9)) < 5.0:
            raise SystemExit(
                f"only {len(t)} frames with a hand in the {expect:.0f} s {name} phase. Either the "
                "camera index is\npointing somewhere else, or the hand was outside the frame. "
                "Nothing written.")

    v_hold, s_hold = _phase_signals(t_hold, P_hold)
    v_move, _ = _phase_signals(t_move, P_move)

    still = _pct(v_hold, 95)
    unarmed = 2.0 * float(v_move.max())
    stable = _pct(s_hold, 95)
    # Section 5.8 calibrates three constants. V_MOVE_ARMED still has to sit strictly between
    # the other two (Thresholds enforces it), so take the held-sign p99 -- the archive's own
    # definition of it -- and clamp it into the interval this camera just measured.
    armed = min(max(_pct(v_hold, 99), still * 1.05), unarmed * 0.95)

    print(f"\nhold phase: {len(t_hold)} frames, {t_hold[-1]:.1f} s, "
          f"{len(t_hold) / t_hold[-1]:.1f} fps")
    print(f"move phase: {len(t_move)} frames, {t_move[-1]:.1f} s, "
          f"{len(t_move) / t_move[-1]:.1f} fps")
    print()
    print(_ROW.format("constant", "default", "measured", "pctile", "signal"))
    print("-" * 88)
    print(_ROW.format("V_STILL", f"{DEFAULT.V_STILL:.3f}", f"{still:.3f}", "p95",
                      "v_bar while holding"))
    print(_ROW.format("V_MOVE_ARMED", f"{DEFAULT.V_MOVE_ARMED:.3f}", f"{armed:.3f}", "p99",
                      "v_bar while holding, clamped below V_MOVE_UNARMED"))
    print(_ROW.format("V_MOVE_UNARMED", f"{DEFAULT.V_MOVE_UNARMED:.3f}", f"{unarmed:.3f}", "2xmax",
                      "v_bar while moving"))
    print(_ROW.format("SHAPE_STABLE", f"{DEFAULT.SHAPE_STABLE:.3f}", f"{stable:.3f}", "p95",
                      f"rolling_shape_sigma, trailing {DEFAULT.SHAPE_WINDOW:.1f} s"))

    # The two phases must be distinguishable, and DEFAULT.V_STILL is the wrong yardstick for
    # deciding that: it is the archive camera's jitter, not this one's. Compare instead with
    # what the hold phase alone implies -- if the move phase does not clear twice the hold
    # p95, v_bar cannot tell the two recordings apart, and 2x a jitter peak is not a motion
    # threshold. The factor is the same 2x section 5.8 already applies to the move maximum.
    if _pct(v_move, 95) < 2.0 * still:
        raise SystemExit(
            f"\nthe 'move' phase (p95 {_pct(v_move, 95):.3f} palm/s) never got clearly faster "
            f"than the hold phase\n(p95 {still:.3f}). Whatever it recorded, v_bar cannot "
            "distinguish it from holding still, so\nV_MOVE_UNARMED would be twice this camera's "
            "own jitter. Nothing written -- rerun and move\nthe hand right across the frame.")

    still, armed, unarmed = _ceil2(still), _ceil2(armed), _ceil2(unarmed)
    # Rounding to 2 dp can collapse a strictly ordered triple into a tie on a very quiet
    # camera, which Thresholds rejects. Separate by one rounding step instead of re-deriving
    # anything; raising a stillness ceiling is the safe direction.
    armed = round(max(armed, still + 0.01), 2)
    unarmed = round(max(unarmed, armed + 0.01), 2)

    th = Thresholds(**{**asdict(DEFAULT),
                       "V_STILL": still, "V_MOVE_ARMED": armed,
                       "V_MOVE_UNARMED": unarmed, "SHAPE_STABLE": _ceil2(stable)})
    _emit(th, out_path, "measured on this camera, this session; the other constants are "
                        "thresholds.DEFAULT.")
    return th


# ---------------------------------------------------------------- output

def _emit(th, out_path, note):
    th.to_json(out_path)
    print(f"\nwrote {out_path} -- {note}")
    print(json.dumps(asdict(th), indent=2))
    Thresholds.from_json(out_path)      # a file that cannot be loaded back is not a result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true",
                      help="recompute from the committed archive")
    mode.add_argument("--live", action="store_true",
                      help="10-second startup calibration on a webcam")
    ap.add_argument("--out", default=None, help="where to write the Thresholds json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--move-s", type=float, default=5.0)
    args = ap.parse_args()

    if args.offline:
        run_offline(args.out or os.path.join(HERE, "thresholds_archive.json"))
    else:
        run_live(args.out or os.path.join(HERE, "thresholds_live.json"),
                 args.camera, args.hold_s, args.move_s)


if __name__ == "__main__":
    main()

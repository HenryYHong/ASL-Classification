"""Shared feature construction for the 26-letter pipeline. Pure functions, no I/O.

Every consumer imports from here -- landmark extraction, static training, motion training,
evaluation and the live demo. Approach A in this repository failed because the training
pipeline and the inference pipeline preprocessed differently; the only durable fix is that
there is exactly one implementation of the transform and everything calls it.

Three coordinate corrections are applied, in this order, and all three are load-bearing:

1. ISOTROPY.  MediaPipe normalizes x by frame width and y by frame height. On a 16:9 frame
   that squashes the x axis by 1.78x relative to y, so a circle traced in the air becomes an
   ellipse in feature space and every diagonal is skewed. Z is *defined* by two diagonals and
   the angles between them. Correct with u = x * (W/H).

2. CHIRALITY. A left hand is the mirror of a right hand and produces mirrored features. Fix a
   single convention by negating u for left hands. Note this depends on MediaPipe's handedness
   label, which is inverted if you feed it a mirrored frame -- so never cv2.flip() the array
   that goes to MediaPipe, only the copy you display.

3. SCALE. Divide by the palm triangle S = mean(|p0-p5|, |p0-p17|, |p5-p17|). The README names
   the missing scale normalization as the pipeline's single largest weakness; this closes it.
   The palm triangle is used rather than the hand bounding box because the bbox depends on the
   handshape -- dividing by it would cancel the very extent signal that separates A from B.

Landmark indices (MediaPipe Hands):
    0 wrist | 4 thumb tip | 8 index tip | 12 middle tip | 16 ring tip | 20 pinky tip
    5 index MCP | 9 middle MCP | 13 ring MCP | 17 pinky MCP
"""
import numpy as np

WRIST = 0
TIPS = {"thumb": 4, "index": 8, "mid": 12, "ring": 16, "pinky": 20}
MCPS = [5, 9, 13, 17]
PALM = [0, 5, 9, 13, 17]
TIP_FOR_ARM = {"J": 20, "Z": 8}  # J is signed with the pinky, Z with the index finger

SHAPE_DIM = 42
EVENT_DIM = 79
K_RESAMPLE = 16


# ---------------------------------------------------------------- per-frame geometry

def to_isotropic(lm, width, height):
    """(...,21,>=2) MediaPipe-normalized landmarks -> (...,21,2) square-aspect coordinates."""
    lm = np.asarray(lm, dtype=np.float64)
    out = np.stack([lm[..., 0] * (float(width) / float(height)), lm[..., 1]], axis=-1)
    return out


def canonicalize_handedness(P, handedness):
    """Mirror left hands onto the right-hand convention.

    `handedness` must be a SCALAR label ("Left" / "Right" / "Unknown" / "None" / None), not a
    per-frame array. The recorder stores one label per frame, so passing that array straight in
    is an easy mistake -- and it used to fail silently, because str(ndarray) begins with "[",
    which matches neither branch and leaves a genuinely mirrored clip unflipped. A chirality
    error is invisible downstream and poisons training, so this raises instead.
    """
    if handedness is None:
        return P
    if isinstance(handedness, (list, tuple, np.ndarray)):
        raise TypeError(
            "canonicalize_handedness expects one label, got a sequence of "
            f"{len(handedness)}. Reduce per-frame labels to the modal real label first "
            "(ignoring the 'None'/'Unknown' markers written on undetected frames)."
        )
    label = str(handedness).strip().lower()
    if label.startswith("l"):
        P = P.copy()
        P[..., 0] = -P[..., 0]
    elif not (label.startswith("r") or label in ("unknown", "none", "")):
        raise ValueError(f"unrecognized handedness label {handedness!r}")
    return P


def palm_scale(P):
    """Palm triangle size, in the same units as P. Shape (...,21,2) -> (...)."""
    a = np.linalg.norm(P[..., 0, :] - P[..., 5, :], axis=-1)
    b = np.linalg.norm(P[..., 0, :] - P[..., 17, :], axis=-1)
    c = np.linalg.norm(P[..., 5, :] - P[..., 17, :], axis=-1)
    return (a + b + c) / 3.0


def palm_centre(P):
    """Mean of wrist and the four MCPs. Stable under finger motion, unlike min(x)/min(y),
    which hops between landmarks when fingers cross and injects frame-to-frame noise."""
    return P[..., PALM, :].mean(axis=-2)


def shape42(P):
    """(...,21,2) -> (...,42) palm-centred, palm-scaled landmark shape.

    Deliberately NOT rotation-normalized: orientation is what separates P from K and H from U.
    """
    S = palm_scale(P)[..., None, None]
    m = palm_centre(P)[..., None, :]
    q = (P - m) / S
    return q.reshape(*q.shape[:-2], SHAPE_DIM)


def extension_ratios(P):
    """Fingertip distance from the wrist, in palm units. dict of 5, each shape (...)."""
    S = palm_scale(P)
    w = P[..., WRIST, :]
    return {k: np.linalg.norm(P[..., i, :] - w, axis=-1) / S for k, i in TIPS.items()}


def thumb_pinkymcp(P):
    """Thumb tip to pinky MCP distance, palm units.

    This replaces a thumb *extension* threshold in the J gate. Measured on the committed
    archive, no threshold on thumb extension separates a held I from a Y (I's p95 = 1.74 sits
    above Y's median 1.65), so a thumb-extension gate silently rejects roughly a quarter of
    real I frames. The thumb-to-pinky-MCP distance separates them cleanly instead.
    """
    S = palm_scale(P)
    return np.linalg.norm(P[..., 4, :] - P[..., 17, :], axis=-1) / S


def j_gate(P):
    """Is this frame in a J launch pose (the 'I' handshape: pinky out, others curled)?"""
    e = extension_ratios(P)
    others = np.maximum(np.maximum(e["index"], e["mid"]), e["ring"])
    # 1.20, not the 1.15 the archive alone suggested. Measured on a real 30-gesture take, the
    # signer's thumb drifted from 1.10 early to 1.18 late as the hand tired, and the tighter
    # threshold silently dropped the gate from 70% to 27% -- half the recording lost, with no
    # error anywhere. On the archive 1.20 keeps I recall at 100% and admits a single frame of Y
    # out of 100, which cannot itself produce a false J: arming is only the first of the rising
    # edge, the vetoes and the classifier.
    return (e["pinky"] > 1.50) & (others < 1.30) & (thumb_pinkymcp(P) < 1.20)


def z_gate(P):
    """Is this frame in a Z launch pose (index extended, others curled)?"""
    e = extension_ratios(P)
    others = np.maximum(np.maximum(e["mid"], e["ring"]), e["pinky"])
    return (e["index"] > 1.80) & (others < 1.20) & (e["thumb"] < 1.45)


def hand_orientation(P):
    """Unit vector wrist -> middle MCP. Shape (...,2)."""
    v = P[..., 9, :] - P[..., 0, :]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


# ---------------------------------------------------------------- temporal signals

def shape_sigma(shapes, reference=None):
    """Mean per-landmark deviation of each normalized shape from one reference shape.

    shapes: (T,42) from shape42. Returns (T,) in palm units.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    ref = np.median(shapes, axis=0) if reference is None else np.asarray(reference)
    d = (shapes - ref).reshape(len(shapes), 21, 2)
    return np.linalg.norm(d, axis=-1).mean(axis=-1)


def rolling_shape_sigma(shapes, times, window_s=0.4):
    """Shape deviation from the median shape over the TRAILING `window_s` seconds.

    This is the signal that separates "hand travelling, shape fixed" (a real J: rigid finger,
    the arm moves it) from "shape changing" (an inter-letter transition). Speed alone cannot
    make that distinction, and the rigidity veto rests entirely on it.

    The trailing window is not interchangeable with a whole-clip reference. Measured over the
    committed archive, deviation from a whole-clip median gives p95 = 0.381 because slow drift
    across a 6.6 s hold accumulates; against the trailing 0.4 s it gives p95 = 0.120. A
    threshold calibrated on one and applied to the other is wrong by a factor of three, and
    would leave the machine unable to ever recognize a hold as stable.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    out = np.empty(len(shapes))
    for i in range(len(shapes)):
        j = np.searchsorted(times, times[i] - window_s)
        ref = np.median(shapes[j:i + 1], axis=0)
        out[i] = np.linalg.norm((shapes[i] - ref).reshape(21, 2), axis=-1).mean()
    return out


def moving_average_time(values, times, window_s):
    """Causal moving average over a time window, for variable frame rates.

    Frame index is not a clock: webcam frame rate is not constant, so every smoothing and
    threshold in this pipeline is expressed in seconds, never in frames.
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    out = np.empty_like(values)
    j = 0
    for i in range(len(values)):
        while times[i] - times[j] > window_s:
            j += 1
        out[i] = values[j:i + 1].mean()
    return out


def palm_speed(centres, scales, times, window_s=0.33):
    """Palm-centre speed in palm-widths per second, smoothed over `window_s` SECONDS.

    The window is in seconds, not samples, because a sample count is a different amount of
    smoothing at every frame rate. A 5-sample average is 0.33 s on 15 fps footage and 0.17 s at
    30 fps; the under-smoothed signal crosses the motion threshold constantly without ever
    holding it, so the rising-edge trigger never fires and no gesture is ever detected. That is
    not a tuning question -- it silently breaks the detector on any camera faster than the one
    the thresholds were calibrated on.
    """
    centres = np.asarray(centres, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if len(centres) < 2:
        return np.zeros(len(centres))
    dt = np.maximum(np.diff(times), 1e-6)
    step = np.linalg.norm(np.diff(centres, axis=0), axis=-1) / (scales[1:] * dt)
    v = np.concatenate([[0.0], step])
    return moving_average_time(v, times, window_s)


def resample_arclength(path, k=K_RESAMPLE):
    """Resample a 2-D path to k points equally spaced by ARC LENGTH, not by time.

    Arc-length resampling removes absolute speed from the path shape (duration, mean speed
    and peak speed are added back as explicit features so nothing is lost silently).
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return np.repeat(path[:1] if len(path) else np.zeros((1, 2)), k, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(path[:1], k, axis=0)
    targets = np.linspace(0.0, total, k)
    return np.stack([np.interp(targets, cum, path[:, d]) for d in range(2)], axis=-1)


def turning_angles(pts):
    """Signed turning angle at each interior point of a polyline. len(pts)-2 values."""
    v = np.diff(np.asarray(pts, dtype=np.float64), axis=0)
    a, b = v[:-1], v[1:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=-1)
    return np.arctan2(cross, dot)


def path_length(path):
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())


# ---------------------------------------------------------------- event feature

def event_features(times, P_seq, arm, handedness=None):
    """Build the 79-D descriptor of one segmented motion event.

    times : (T,) seconds
    P_seq : (T,21,2) isotropic, handedness-canonicalized landmarks (NOT yet palm-scaled)
    arm   : "J" or "Z" -- which gate armed the track, selecting the writing fingertip

    Layout is documented in the design spec and asserted by tests/test_features.py.
    """
    times = np.asarray(times, dtype=np.float64)
    P = np.asarray(P_seq, dtype=np.float64)
    if len(P) < 3:
        raise ValueError("event needs at least 3 frames")

    S_t = palm_scale(P)
    S_evt = float(np.median(S_t))
    tip_idx = TIP_FOR_ARM[arm]

    # Smooth the writing tip over 0.15 s before measuring anything from it.
    tip = P[:, tip_idx, :]
    tip_s = np.stack([moving_average_time(tip[:, d], times, 0.15) for d in range(2)], axis=-1)

    duration = float(times[-1] - times[0])
    L = path_length(tip_s) / S_evt
    net = (tip_s[-1] - tip_s[0]) / S_evt
    net_mag = float(np.linalg.norm(net))
    straightness = net_mag / L if L > 1e-9 else 0.0

    # [0:32] path shape: resample by arc length, centre on its own centroid, scale by S_evt.
    # Centring across TIME is the same trick the existing 42-D feature applies across
    # LANDMARKS: a J traced top-left and the same J traced bottom-right give identical
    # numbers. Absolute frame position never enters any model.
    if L * S_evt < 0.05 * S_evt:
        rs = np.repeat(tip_s[:1], K_RESAMPLE, axis=0)  # degenerate; L_MIN rejects it upstream
    else:
        rs = resample_arclength(tip_s, K_RESAMPLE)
    g = (rs - rs.mean(axis=0)) / S_evt

    theta = turning_angles(rs)                      # [32:46], 14 values
    e = extension_ratios(P)
    ext_med = [float(np.median(e[k])) for k in ("thumb", "index", "mid", "ring", "pinky")]
    orient = np.median(hand_orientation(P), axis=0)

    # Same trailing-window definition the segmenter's rigidity veto uses, so [59]/[60] are
    # directly comparable to RIGID_VETO rather than being a differently-scaled quantity.
    shapes = shape42(P)
    sig = rolling_shape_sigma(shapes, times)

    third = max(1, len(tip_s) // 3)
    thirds = []
    for i in range(3):
        lo = i * third
        hi = len(tip_s) if i == 2 else min(len(tip_s), (i + 1) * third)
        d = (tip_s[hi - 1] - tip_s[lo]) / S_evt
        thirds.extend([float(d[0]), float(d[1])])

    dt = np.maximum(np.diff(times), 1e-6)
    tip_step = np.linalg.norm(np.diff(tip_s, axis=0), axis=-1) / S_evt
    speeds = tip_step / dt

    s_first = float(np.median(S_t[:third]))
    s_last = float(np.median(S_t[-third:]))
    scale_ratio = (s_last / s_first - 1.0) if s_first > 1e-9 else 0.0

    # [70:78] articulation: the tip measured RELATIVE to the palm centre. In a genuine J or Z
    # the finger is rigid and the arm carries it, so these are near zero. A wave or a finger
    # wiggle moves the tip relative to the palm and lights this block up.
    rel = (P[:, tip_idx, :] - palm_centre(P)) / S_t[:, None]
    rel_L = path_length(rel)
    rel_net = rel[-1] - rel[0]
    rel_net_mag = float(np.linalg.norm(rel_net))
    rel_dir = rel_net / max(rel_net_mag, 1e-9)
    rel_turn = float(np.abs(turning_angles(rel)).sum()) if len(rel) >= 3 else 0.0

    f = np.concatenate([
        g.reshape(-1),                                                    # [0:32]
        theta,                                                            # [32:46]
        ext_med,                                                          # [46:51]
        [float(orient[0]), float(orient[1])],                             # [51:53]
        [L],                                                              # [53]
        [duration],                                                       # [54]
        [straightness],                                                   # [55]
        [float((np.abs(theta) > 0.6).sum())],                             # [56]
        [float(np.abs(theta).max())],                                     # [57]
        [float(theta.sum())],                                             # [58]  signed net turn
        [float(sig.max())],                                               # [59]
        [float(np.median(sig))],                                          # [60]
        [scale_ratio],                                                    # [61]
        thirds,                                                           # [62:68]
        [float(speeds.mean())],                                           # [68]
        [float(speeds.max())],                                            # [69]
        [rel_L, rel_net_mag, rel_L and rel_net_mag / rel_L or 0.0,        # [70:73]
         float(rel_dir[0]), float(rel_dir[1]), rel_turn,                  # [73:76]
         float(rel[:, 0].std()), float(rel[:, 1].std())],                 # [76:78]
        [0.0 if arm == "J" else 1.0],                                     # [78]
    ]).astype(np.float64)

    assert f.shape == (EVENT_DIM,), f"expected {EVENT_DIM}-D, built {f.shape}"
    return f

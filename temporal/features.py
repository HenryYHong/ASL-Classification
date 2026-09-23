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


def frame_aspect(frame_size, n, where="frame_size"):
    """The W/H each of `n` frames must be corrected by. -> (1,) for one size, (n,) for per-frame.

    A source may store ONE (W,H) pair for the whole file (every frame from one camera at one
    resolution: the author's sessions, the Ankara photos, ASL-HG, ayuraj) or ONE PAIR PER FRAME
    (a set of crops, or a mixed-resolution capture). Both loaders used to write
    np.asarray(d["frame_size"]).ravel()[:2], which on an (N,2) array silently takes image 0's
    size and applies it to every frame. That cannot be seen downstream: u = x*(W/H) is a
    plausible number for any W/H, so the run prints an accuracy rather than an error, which is
    the same failure SOURCES.md refuses the Google parquet set over. Measured on the real path
    -- ASL-HG re-written with its true per-image sizes, image 0 at 106x119 -- taking image 0's
    size for all 65,431 records scores 0.7814 against 0.8212 done properly, and -0.053 with no
    correction at all.

    Anything that is neither (2,) nor (n,2) raises, for the reason load_extra already refuses a
    missing frame_size: the aspect is load-bearing and is not guessed.
    """
    wh = np.asarray(frame_size)
    if wh.ndim == 1 and wh.shape[0] == 2:
        return np.array([float(wh[0]) / float(wh[1])])
    if wh.ndim == 2 and wh.shape == (n, 2):
        return wh[:, 0].astype(np.float64) / wh[:, 1].astype(np.float64)
    raise SystemExit(
        f"{where}: frame_size has shape {wh.shape} for {n} frames. It must be one (W,H) pair "
        "for the file, or one pair per frame; any other shape used to be ravel()'d down to "
        "image 0's size and applied to everything, which is invisible in every number this "
        "pipeline prints.")


def apply_aspect(lm, aspect):
    """(...,21,>=2) -> (...,21,2), scaling x by `aspect` from frame_aspect (one value or one
    per frame). The per-frame case broadcasts over the 21 landmarks."""
    lm = np.asarray(lm, dtype=np.float64)
    a = np.asarray(aspect, dtype=np.float64)
    if a.size == 1:
        a = a.reshape(())
    else:
        if len(a) != lm.shape[0]:
            raise SystemExit(f"apply_aspect: {len(a)} aspects for {lm.shape[0]} frames")
        a = a.reshape((-1,) + (1,) * (lm.ndim - 2))
    return np.stack([lm[..., 0] * a, lm[..., 1]], axis=-1)


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
    """(...,21,2) -> (...,42) palm-centered, palm-scaled landmark shape.

    Deliberately NOT rotation-normalized: orientation is what separates P from K and H from U.
    """
    S = palm_scale(P)[..., None, None]
    m = palm_centre(P)[..., None, :]
    q = (P - m) / S
    return q.reshape(*q.shape[:-2], SHAPE_DIM)


def legacy42(P):
    """The original pipeline's feature: translation-normalized, NOT scale-normalized.

    Kept because removing it was a mistake. Dividing by palm size makes the representation
    distance-invariant, but it also discards absolute extent -- and extent is what separated
    G and D from the shapes they collide with. Measured on a held-out archive split, replacing
    this with the palm-normalized feature alone dropped G from 1.00 to 0.25 and D from 1.00 to
    0.85, which showed up in use as letters that simply stopped being recognized.
    """
    x, y = P[..., 0], P[..., 1]
    out = np.empty((*P.shape[:-2], SHAPE_DIM))
    out[..., 0::2] = x - x.min(axis=-1, keepdims=True)
    out[..., 1::2] = y - y.min(axis=-1, keepdims=True)
    return out


def shape84(P):
    """Both representations concatenated: palm-normalized shape, then raw translated extent.

    The forest picks per split, so scale-invariant evidence is available where it helps and
    absolute extent where that is what distinguishes the letter. Measured on the contiguous
    archive split: 0.945 against 0.926 for extent alone and 0.908 for palm-normalized alone,
    while retaining most of the rescale robustness the palm feature was introduced for
    (0.76 at 0.7x, where the original collapses to 0.34).
    """
    return np.concatenate([shape42(P), legacy42(P)], axis=-1)


#: Landmarks whose pairwise distances carry the handshape distinctions trees struggle to
#: express from raw coordinates: the five fingertips, the five MCP knuckles and the wrist.
KEY_POINTS = [4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 0]


def pair_distances(P):
    """All pairwise distances between KEY_POINTS, in palm units, plus per-finger straightness.

    A forest splits on one coordinate at a time, so "are the index and middle fingertips far
    apart" costs it a deep chain of axis-aligned splits on four separate coordinates -- and that
    single quantity is the whole difference between U and V, as thumb-to-fist distance is between
    M and A. Handing over the distances directly turns those into one split each.

    Measured on the held-out archive split (a within-burst split; static_feature says why that
    number misleads), adding this block took overall accuracy from 0.956 to 0.983 and M from
    0.25 to 0.85. Computing the distances in 3-D (MediaPipe supplies a z, scaled by W/H like x)
    was slightly worse on THAT split only (0.964-0.971 against 0.968-0.981 over three seeds)
    and is neutral-to-positive leave-one-session-out: under the previous release's recipe
    (crossval_static protocol on the author's sessions alone, static/v4 with z inside these
    norms, seeds 0-2) pooled 0.861/0.869/0.860 ->
    0.871/0.874/0.864 (+0.004 to +0.010, hold-level CIs all straddling zero) and the cross-day
    fold 0.763/0.761/0.763 -> 0.783/0.795/0.786 (+0.020 to +0.034, mostly E and O); under the
    retired rotation augmentation +0.024 pooled with a paired hold-level CI of [+0.001, +0.047].
    Raw unscaled z is noise (0.753 against 0.759). So depth is untested as a shipped feature,
    not ruled out: static/v3 and static/v4 stay 2-D because the browser port, the golden cases
    and the digits forest are 2-D, and z was never measured on the page's Tasks landmarks.
    """
    import itertools
    S = palm_scale(P)
    out = [np.linalg.norm(P[..., a, :] - P[..., b, :], axis=-1) / S
           for a, b in itertools.combinations(KEY_POINTS, 2)]
    out += [finger_straightness(P, n) for n in ("index", "mid", "ring", "pinky")]
    return np.stack(out, axis=-1)


def static_feature(P):
    """What the 24-class static classifier consumes: palm-normalized shape + distances.

    Deliberately NOT including legacy42. Adding absolute extent raises the in-session benchmark
    (0.968 -> 0.983) and roughly HALVES cross-session accuracy (0.520 -> 0.262 in the
    train-on-archive / test-on-S2 measurement of the commit that removed it; that protocol was
    not committed and no script here reproduces either endpoint, so the figures are the record,
    not a re-derivable number), because extent encodes how this signer happened to be sitting
    and the within-session split rewards memorizing that.
    The distance block is kept because it costs nothing cross-session and fixes the confusable
    pairs the raw coordinates could not express: G 0.25 -> passing, M 0.20 -> 0.65.

    The measurement that matters here is the cross-session one -- train on the archive, test on
    frames recorded months later on a different day. This repository's README exists largely to
    document how misleading the in-session number is, and it misled me twice while tuning this.
    """
    return np.concatenate([shape42(P), pair_distances(P)], axis=-1)


STATIC_DIM = 101


#: CMC, MCP, IP, TIP of the thumb -- the chain thumb_straightness measures along. Kept apart
#: from FINGER_CHAIN because the thumb has one joint fewer and no PIP/DIP, and because
#: pair_distances deliberately iterates the four fingers only (static/v3 must not change).
THUMB_CHAIN = (1, 2, 3, 4)

#: The five points the thumb tip is measured against in static_feature_v4, in this order:
#: index PIP, index DIP, middle PIP, middle DIP, thumb IP.
THUMB_TARGETS = (6, 7, 10, 11, 3)


def thumb_straightness(P):
    """|tip - CMC| / (sum of the three thumb bones), 1.0 for a straight thumb. Shape (...)."""
    a, b, c, d = THUMB_CHAIN
    tip = np.linalg.norm(P[..., d, :] - P[..., a, :], axis=-1)
    seg = (np.linalg.norm(P[..., b, :] - P[..., a, :], axis=-1)
           + np.linalg.norm(P[..., c, :] - P[..., b, :], axis=-1)
           + np.linalg.norm(P[..., d, :] - P[..., c, :], axis=-1))
    return tip / np.maximum(seg, 1e-9)


def tip_palm_distances(P):
    """Distance of each fingertip (thumb, index, middle, ring, pinky) from the palm center, in
    palm units. Shape (...,5)."""
    S = palm_scale(P)
    m = palm_centre(P)[..., None, :]
    return np.linalg.norm(P[..., [4, 8, 12, 16, 20], :] - m, axis=-1) / S[..., None]


def static_feature_v4(P):
    """static/v4: the 101-D static/v3 vector followed by an 11-value thumb block. 112-D.

    The block, in this order and nothing else:
        [101]      thumb_straightness          |P4-P1| / (|P2-P1| + |P3-P2| + |P4-P3|)
        [102:107]  tip_palm_distances          fingertips 4, 8, 12, 16, 20 to the palm center,
                                               divided by palm_scale
        [107:112]  thumb tip (4) to landmarks 6, 7, 10, 11, 3, divided by palm_scale
    All of it on the canonical isotropic x,y; z is never read.

    Why it exists: the letters the 101-D vector confuses are separated by where the THUMB sits.
    The fists (A, E, M, N, S, T) differ only in whether the thumb lies beside the index, across
    the fingers, or between them; D and X differ in whether the thumb touches the middle finger.
    pair_distances carries thumb-to-fingertip distances, but the fists keep every fingertip
    curled at almost the same place, so what separates them is the thumb against the KNUCKLES
    -- the PIP and DIP of the index and middle finger -- and how straight the thumb is. Measured
    leave-one-session-out over the four sessions with the same jitter augmentation (sigma 0.12,
    x4) and the same X-rule training filter, the block adds about +0.02 on the cross-day fold
    (the November archive held out; 3-seed means 0.746 -> 0.757 in one implementation, seed 0
    0.754 -> 0.789 in a second that differs only in jitter RNG) and +0.01 pooled. That is
    within one realistic seed spread, so it is kept for the geometry it encodes and for the
    consistent direction of the effect (every nested inner fold preferred it by 0.01-0.04),
    not as a large win.

    Both shipped forests -- the 24-letter model_static.p and the 10-digit model_digits.p
    (train_digits.py measured it 0.986 leave-signer-out on either tag and 0.162 against 0.267
    idle false-digit on this one) -- and every golden.json case are on this tag; static/v3
    stays registered for pickles written before the tag existed. docs/features.js reproduces
    the block in exactly this order, so a reordering here is a silent break of the browser
    port -- STATIC_FEATURES is what both sides key on.
    """
    S = palm_scale(P)
    extra = [np.linalg.norm(P[..., 4, :] - P[..., j, :], axis=-1) / S for j in THUMB_TARGETS]
    return np.concatenate([static_feature(P), thumb_straightness(P)[..., None],
                           tip_palm_distances(P), np.stack(extra, axis=-1)], axis=-1)


STATIC_DIM_V4 = 112

#: The static feature a model was trained on, by the 'feature' tag its pickle carries. Every
#: consumer that feeds a static forest (the segmenter, the export, the live demo) resolves the
#: function through this table rather than hard-coding one, so a forest can never be fed a
#: vector of the wrong layout without a KeyError naming the tag.
STATIC_FEATURES = {
    "static/v3": (static_feature, STATIC_DIM),
    "static/v4": (static_feature_v4, STATIC_DIM_V4),
}
DEFAULT_STATIC_TAG = "static/v3"


def static_feature_for(tag):
    """(function, dim) for a static model's feature tag; None means the original static/v3
    (pickles written before the tag existed)."""
    tag = DEFAULT_STATIC_TAG if tag is None else str(tag)
    if tag not in STATIC_FEATURES:
        raise KeyError(f"unknown static feature tag {tag!r}; known: {sorted(STATIC_FEATURES)}")
    return STATIC_FEATURES[tag]


def extension_ratios(P):
    """Fingertip distance from the wrist, in palm units. dict of 5, each shape (...)."""
    S = palm_scale(P)
    w = P[..., WRIST, :]
    return {k: np.linalg.norm(P[..., i, :] - w, axis=-1) / S for k, i in TIPS.items()}


#: MCP, PIP, DIP, TIP for each finger -- the joint chain used by finger_straightness.
FINGER_CHAIN = {"index": (5, 6, 7, 8), "mid": (9, 10, 11, 12),
                "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def finger_straightness(P, name):
    """|tip - MCP| / (sum of the three bone lengths). 1.0 is a perfectly straight finger.

    This exists because wrist-to-tip DISTANCE is not orientation-invariant. MediaPipe gives
    2-D projected coordinates, so a finger pointing toward the camera foreshortens and its
    measured extension collapses -- a live Z read ext_index 1.87 against a threshold of 1.80,
    failing 43% of frames, purely because the finger was not held flat to the lens. The signer
    had to point upward for it to work at all.

    A ratio survives that: foreshortening shrinks the numerator and the denominator together,
    so a straight finger reads ~1.0 whichever way it points. Measured on the same footage the
    distance gate scored 56%, this scores 89%, while admitting FEWER archive false positives
    (91 frames against 118) because a curled finger is unambiguous under this measure.
    """
    a, b, c, d = FINGER_CHAIN[name]
    tip = np.linalg.norm(P[..., d, :] - P[..., a, :], axis=-1)
    seg = (np.linalg.norm(P[..., b, :] - P[..., a, :], axis=-1)
           + np.linalg.norm(P[..., c, :] - P[..., b, :], axis=-1)
           + np.linalg.norm(P[..., d, :] - P[..., c, :], axis=-1))
    return tip / np.maximum(seg, 1e-9)


def thumb_pinkymcp(P):
    """Thumb tip to pinky MCP distance, palm units.

    This replaces a thumb *extension* threshold in the J gate. Measured on the committed
    archive, no threshold on thumb extension separates a held I from a Y (I's p95 = 1.74 sits
    above Y's median 1.65), so a thumb-extension gate silently rejects roughly a quarter of
    real I frames. The thumb-to-pinky-MCP distance separates them cleanly instead.
    """
    S = palm_scale(P)
    return np.linalg.norm(P[..., 4, :] - P[..., 17, :], axis=-1) / S


#: Thumb-tip-to-pinky-MCP ceiling of the J gate, palm units. MEASURED on the five prompted
#: takes (113 J/Z items) and the archive: at 1.20 only 41 of the 60 J items produced a
#: creditable event (19 had none); at 1.30, 48 of 60 (12 still have none), and the count of
#: rest-phase spans the runtime could score is identical at both (J 10, Z 8). The price is 13
#: of 2,278 non-I archive frames passing instead of 1 -- all of them Y (13 of Y's 100), which
#: replayed through the whole segmenter start no track. 1.35 buys 2 more items (50/60) for 1
#: more Y frame; 1.25 only 3 (44/60).
J_THUMB_MAX = 1.30


def j_gate(P):
    """Is this frame in a J launch pose (the 'I' handshape: pinky out, others curled)?"""
    e = extension_ratios(P)
    others = np.maximum(np.maximum(e["index"], e["mid"]), e["ring"])
    # The thumb ceiling is J_THUMB_MAX (see above). 1.15 came from the archive alone; 1.20 was
    # set when a 30-gesture take showed the signer's thumb drifting from 1.10 to 1.18 as the
    # hand tired; the prompted takes then showed 1.20 crediting only 41 of 60 J items, 7 of
    # which 1.30 recovers. A frame passing the gate cannot by itself produce a false J: arming
    # is only the first of the rising edge, the vetoes and the classifier.
    return (e["pinky"] > 1.50) & (others < 1.30) & (thumb_pinkymcp(P) < J_THUMB_MAX)


def z_gate(P):
    """Is this frame in a Z launch pose (index extended, others curled)?

    The index test is straightness, not wrist-to-tip distance, so the gate does not silently
    require the finger to be held flat to the camera. The curled-finger and thumb tests stay on
    extension ratios: straightness is unreliable for a folded finger, whose joints MediaPipe
    often cannot see.
    """
    e = extension_ratios(P)
    others = np.maximum(np.maximum(e["mid"], e["ring"]), e["pinky"])
    return (finger_straightness(P, "index") > 0.90) & (others < 1.20) & (e["thumb"] < 1.45)


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


def rolling_shape_sigma_aligned(shapes, times, window_s=0.4):
    """Shape deviation AFTER removing the best-fit rotation. Palm units.

    rolling_shape_sigma answers "did anything about this hand change", which is the right
    question for "is it parked". It is the wrong question for "is this a rigid gesture",
    because shape42 is deliberately not rotation-normalized: a J hook rotates the wrist, so a
    perfectly rigid hand registers a large deviation purely from turning. Calibrating a rigidity
    veto on that measure forced it up to 1.35 palm, high enough that it stopped rejecting
    inter-letter transitions -- which then starved the static branch, because no letter can be
    emitted while a stale track is still running.

    Removing the rotation (Kabsch, 2-D) separates the two: what remains is only the part of the
    change that a rotation cannot explain, i.e. the fingers actually moving relative to each
    other. A rigid hand being carried through an arc scores near zero however far it turns.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    out = np.empty(len(shapes))
    for i in range(len(shapes)):
        j = np.searchsorted(times, times[i] - window_s)
        ref = np.median(shapes[j:i + 1], axis=0).reshape(21, 2)
        cur = shapes[i].reshape(21, 2)
        # Optimal rotation taking cur onto ref; both are already centered and palm-scaled.
        H = cur.T @ ref
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        out[i] = np.linalg.norm((R @ cur.T).T - ref, axis=-1).mean()
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


def trailing_window(times, t_end, window_s):
    """Indices of the frames the segmenter averages v_bar over at time t_end: every frame with
    t >= t_end - window_s (t_end's own frame included), or the last two frames when fewer than
    two qualify (a frame gap longer than the window). `times` must be ascending and end at or
    after t_end; only frames at or before t_end are considered."""
    times = np.asarray(times, dtype=np.float64)
    n = int(np.searchsorted(times, t_end, side="right"))
    j = int(np.searchsorted(times[:n], t_end - window_s, side="left"))
    if n - j < 2:
        j = max(0, n - 2)
    return np.arange(j, n)


def window_speed(times, centers, scales):
    """Mean per-step palm-center speed over one window of >= 2 frames, palm-widths per second.

    Each step is |m[k] - m[k-1]| / (S[k] * dt), scaled by the LATER frame's palm size, and the
    mean is over the steps between consecutive frames of the window -- the step that enters the
    window from before it is not counted. This is the segmenter's v_bar; every stillness and
    motion threshold in thresholds.py is a percentile of it.
    """
    times = np.asarray(times, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    dt = np.maximum(np.diff(times), 1e-6)
    step = np.linalg.norm(np.diff(centers, axis=0), axis=-1) / (scales[1:] * dt)
    return float(step.mean())


def palm_speed(centres, scales, times, window_s=0.33):
    """v_bar for every frame of a sequence, EXACTLY as Segmenter._update_signals computes it:
    trailing_window + window_speed per frame. The first frame has no step and reads 0.0.

    The window is in seconds, not samples, because a sample count is a different amount of
    smoothing at every frame rate. A 5-sample average is 0.33 s on 15 fps footage and 0.17 s at
    30 fps; the under-smoothed signal crosses the motion threshold constantly without ever
    holding it, so the rising-edge trigger never fires and no gesture is ever detected. That is
    not a tuning question -- it silently breaks the detector on any camera faster than the one
    the thresholds were calibrated on.

    This used to be moving_average_time over per-step speeds with a leading 0.0 placeholder,
    which averages k+1 values (the step entering the window included) where the segmenter
    averages k. Over the archive the two differed on 2,352 of 2,354 frames (max 0.69 palm/s,
    p95 0.841 against the segmenter's 0.859), so calibrate.py was measuring a signal the
    thresholds are never applied to. There is now one arithmetic and both call it.
    """
    centres = np.asarray(centres, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    out = np.zeros(len(centres))
    for i in range(1, len(centres)):
        sel = trailing_window(times, times[i], window_s)
        out[i] = window_speed(times[sel], centres[sel], scales[sel])
    return out


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

    # [0:32] path shape: resample by arc length, center on its own centroid, scale by S_evt.
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

    # The PLAIN trailing-window sigma (rolling_shape_sigma), the segmenter's stability signal
    # -- comparable to SHAPE_STABLE. The runtime's rigidity veto reads the rotation-ALIGNED
    # sigma_rigid instead (rolling_shape_sigma_aligned), so [59]/[60] are not the quantity
    # RIGID_VETO is applied to; they describe how much the handshape changed, rotation included.
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

    # [70:78] articulation: the tip measured RELATIVE to the palm center. In a genuine J or Z
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

"""Golden-vector tests pinning the feature contract in features.py.

features.py is the single implementation every consumer shares, which is the whole point of
it -- but it also means an edit there silently changes what the trained models mean. A model
pickled against shape42/v1 and fed a different shape42 does not crash; it just gets worse, and
the next person measures the drop as a data problem. These tests make that edit fail loudly.

Two kinds of assertion live here, and they are not equally strong:

  PROPERTIES are derived from the design and hold for any correct implementation -- 42, 101,
  112 and 79 dimensions, exact invariance of shape42 and of the static/v4 thumb block to
  translation and to uniform rescale, the gate truth table, the arm flag at [78], to_isotropic
  touching only x. These say the code is right.

  PINNED VALUES are numbers this implementation produces today, recorded so drift is visible.
  They do not say the code is right; they say it has not changed. Where a pinned number is
  independently derivable from the fixture geometry the derivation is written beside it, and
  those few are properties in disguise.

The fixture hands are constructed, not captured. Each finger tip is placed at an exact chosen
distance from the wrist in palm units, so extension_ratios comes back as the round number it
was built from and the gate inequalities are exercised at known distances from their
thresholds rather than at whatever a recorded frame happened to give. build_hand places the
joints by linear interpolation, so every finger it builds is exactly straight (straightness
1.0); the hooked-index fixture bends the joints by hand, because z_gate's straightness clause
can only be exercised by a finger that is not straight.

The last section drives the Segmenter itself with a stub forest: a static letter must reach
the emission block and come out (the block once raised NameError on every static emission
while every test stayed green, because nothing here ever voted), and arm_gates=False must
keep a carried launch pose out of TRACKING.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F
from segmenter import Segmenter, TRACKING, HOLD
from thresholds import DEFAULT


# ---------------------------------------------------------------- fixture hands

#: MCP x-offsets, wrist at the origin and the MCP row at y = 1. Right-hand convention, matching
#: canonicalize_handedness, so no mirroring is needed anywhere in this file.
MCP_X = {"index": -0.45, "mid": -0.15, "ring": 0.15, "pinky": 0.45}
FINGERS = ("index", "mid", "ring", "pinky")
CHAIN = {"index": [5, 6, 7, 8], "mid": [9, 10, 11, 12],
         "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}

#: palm_scale of every fixture below: mean of |p0-p5|, |p0-p17|, |p5-p17| for the MCP row above,
#: i.e. (sqrt(1.2025) + sqrt(1.2025) + 0.9) / 3. Identical across the three hands by construction,
#: which is what lets one set of extension ratios be compared against another.
PALM_S = (2.0 * np.hypot(0.45, 1.0) + 0.9) / 3.0


def build_hand(ratios, thumb_tip):
    """A 21-landmark hand whose fingertips sit at the requested wrist distances, in palm units.

    `ratios` maps finger -> the value extension_ratios must return for it. Each tip is placed
    directly above its own MCP at the height that makes |tip - wrist| / palm_scale equal that
    value; the three intermediate joints are linear interpolations, which is geometrically
    coherent but not anatomically real -- nothing in features.py reads them except shape42.
    """
    P = np.zeros((21, 2), dtype=np.float64)
    mcp = {f: np.array([MCP_X[f], 1.0]) for f in FINGERS}
    for f in FINGERS:
        h = np.sqrt((ratios[f] * PALM_S) ** 2 - MCP_X[f] ** 2)
        tip = np.array([MCP_X[f], h])
        for k, i in enumerate(CHAIN[f]):
            P[i] = mcp[f] + (tip - mcp[f]) * (k / 3.0)
    t = np.asarray(thumb_tip, dtype=np.float64)
    for k, i in enumerate([1, 2, 3, 4]):
        P[i] = t * ((k + 1) / 4.0)
    return P


#: All four fingers extended and the thumb out: passes neither gate, which is the case a gate
#: written with the wrong inequality direction would fail.
FLAT = build_hand({f: 2.20 for f in FINGERS}, (-0.90, 1.20))

#: The I handshape, the J launch pose. Pinky at 1.75 clears j_gate's 1.50; the other three at
#: 1.00 clear its 1.30 ceiling; the thumb tucked to the palm gives thumb_pinkymcp 0.353 against
#: the J_THUMB_MAX ceiling of 1.30. Its straight index (1.0) would pass z_gate's straightness
#: clause, so z_gate rejects it on the pinky (1.75 against the 1.20 ceiling on the others).
I_SHAPE = build_hand({"index": 1.00, "mid": 1.00, "ring": 1.00, "pinky": 1.75}, (0.10, 0.90))

#: The D handshape, the Z launch pose. z_gate tests the index by STRAIGHTNESS (> 0.90, so a
#: finger pointing at the lens still passes), and build_hand's linear joints make it exactly
#: 1.0; the other three at 1.00 clear the 1.20 ceiling, thumb extension 1.265 sits under its
#: 1.45. Pinky at 1.00 keeps it below j_gate's 1.50 floor.
D_SHAPE = build_hand({"index": 2.10, "mid": 1.00, "ring": 1.00, "pinky": 1.00}, (-0.10, 1.30))


def hook_index(P, bone=0.45):
    """D_SHAPE with its index bent into a hook: three equal bones up, across and back down, so
    |tip - MCP| is one bone against three (straightness 1/3). Every other landmark, and so
    every other clause of z_gate, is untouched -- only the straightness clause can fail."""
    Q = P.copy()
    mcp = Q[5]
    Q[6] = mcp + np.array([0.0, bone])
    Q[7] = Q[6] + np.array([bone, 0.0])
    Q[8] = Q[7] + np.array([0.0, -bone])
    return Q


#: The X handshape: a hooked index with everything else as in D. Fails z_gate ONLY on
#: straightness; a z_gate that dropped that clause would admit it.
X_SHAPE = hook_index(D_SHAPE)


def with_thumb_pinkymcp(P, ratio):
    """I_SHAPE with the thumb tip moved so thumb_pinkymcp reads exactly `ratio` (the thumb is
    slid along the MCP row away from the pinky MCP), for probing the J_THUMB_MAX ceiling."""
    Q = P.copy()
    Q[4] = Q[17] + np.array([-ratio * PALM_S, 0.0])
    return Q

TOL = 1e-9


def _translated(P, offsets):
    """(21,2) hand rigidly carried along `offsets` (T,2). The handshape never changes, which is
    the defining property of a real J or Z: a rigid finger moved by the arm."""
    return P[None, :, :] + np.asarray(offsets, dtype=np.float64)[:, None, :]


def _straight_event():
    """21 frames over 1.0 s, the hand translated 3.0 palm units along the unit vector (0.6,-0.8).

    Collinear on purpose. A causal moving average of collinear points is still collinear, so
    the 0.15 s tip smoothing inside event_features cannot bend this path, and every path-shape
    output is derivable in closed form rather than merely recorded.
    """
    times = np.linspace(0.0, 1.0, 21)
    u = np.linspace(0.0, 1.0, 21)[:, None]
    return times, _translated(I_SHAPE, u * np.array([0.6, -0.8]) * 3.0 * PALM_S)


def _j_event():
    """A J-like tip path: 55% of the frames descending, then a 180-degree hook of radius 0.5."""
    times = np.linspace(0.0, 1.0, 21)
    n1 = int(21 * 0.55)
    down = np.stack([np.zeros(n1), -np.linspace(0.0, 1.8, n1)], axis=1)
    ang = np.linspace(np.pi, 0.0, 21 - n1 + 1)[1:]
    hook = np.stack([0.5 + 0.5 * np.cos(ang), -1.8 + 0.5 * np.sin(ang)], axis=1)
    return times, _translated(I_SHAPE, np.concatenate([down, hook]) * PALM_S)


def _z_event():
    """A Z-like tip path: across, diagonally back and down, across again. Deliberately symmetric,
    so its two corners turn by equal and opposite angles and the signed net turn is exactly 0 --
    the property that separates Z from J in feature [58] and that a sign error would destroy."""
    times = np.linspace(0.0, 1.0, 21)
    k = 7
    segs = [np.stack([np.linspace(0.0, 2.0, k), np.zeros(k)], axis=1),
            np.stack([np.linspace(2.0, 0.0, k), np.linspace(0.0, -2.0, k)], axis=1),
            np.stack([np.linspace(0.0, 2.0, k), np.full(k, -2.0)], axis=1)]
    return times, _translated(D_SHAPE, np.concatenate(segs) * PALM_S)


def close(a, b, tol=TOL):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


# ---------------------------------------------------------------- tests

def test_dims_declared():
    assert F.SHAPE_DIM == 42, F.SHAPE_DIM
    assert F.EVENT_DIM == 79, F.EVENT_DIM
    assert F.K_RESAMPLE == 16, F.K_RESAMPLE
    assert F.TIP_FOR_ARM == {"J": 20, "Z": 8}, F.TIP_FOR_ARM


def test_shape42_dimensionality():
    assert F.shape42(FLAT).shape == (42,)
    stack = np.stack([FLAT, I_SHAPE, D_SHAPE])
    assert F.shape42(stack).shape == (3, 42)
    # The batch path must agree with the single-frame path element for element, because
    # training feeds batches and the live demo feeds one frame at a time.
    assert close(F.shape42(stack)[1], F.shape42(I_SHAPE)) < 1e-12


def test_shape42_translation_invariant():
    for P in (FLAT, I_SHAPE, D_SHAPE):
        base = F.shape42(P)
        for d in ([0.3, -0.7], [-12.0, 4.5], [1e3, 1e3]):
            assert close(F.shape42(P + np.array(d)), base) < TOL


def test_shape42_scale_invariant():
    """The property the whole scale fix rests on, asserted numerically.

    In the spec's D4 measurement the legacy 42-D feature collapses from 0.931 to 0.352 when
    test landmarks are rescaled to 0.7x about the hand centroid; shape42 divides by the palm
    triangle, so rescaling about any point cancels exactly. That accuracy gap is the spec's
    number, not this test's -- what is measured here is the exactness the gap rests on. If
    this assertion ever loosens, the README's headline fix is gone.
    """
    for P in (FLAT, I_SHAPE, D_SHAPE):
        base = F.shape42(P)
        centroid = P.mean(axis=0)
        for k in (0.25, 0.5, 0.7, 0.85, 1.2, 2.0, 7.0):
            scaled = centroid + (P - centroid) * k
            assert close(F.shape42(scaled), base) < TOL, (k, close(F.shape42(scaled), base))


def test_shape42_is_not_rotation_invariant():
    """Deliberate, and documented in shape42's own docstring: orientation is what separates P
    from K and H from U. A well-meaning edit that adds rotation normalization would erase
    those distinctions silently, so pin the fact that rotation changes the vector."""
    th = 0.6
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert close(F.shape42(D_SHAPE @ R.T), F.shape42(D_SHAPE)) > 0.1


def test_shape42_pinned():
    """Golden vector. Derivable: landmark 0 is the wrist, palm_centre is the mean of the wrist
    and the four MCPs, so for these fixtures the wrist sits at (0, -0.8) / PALM_S from it."""
    v = F.shape42(I_SHAPE)
    assert close(v[0], 0.0) < TOL, v[0]
    assert close(v[1], -0.8 / PALM_S) < TOL, v[1]
    assert close(v[:6], [0.0, -0.775902731968, 0.024246960374,
                         -0.557680088602, 0.048493920748, -0.339457445236]) < TOL
    assert close(float(np.sum(v)), 3.036075698478) < TOL, float(np.sum(v))


def test_palm_scale_and_centre():
    for P in (FLAT, I_SHAPE, D_SHAPE):
        assert close(F.palm_scale(P), PALM_S) < TOL
        assert close(F.palm_centre(P), [0.0, 0.8]) < TOL
    assert close(PALM_S, 1.031057073315) < TOL, PALM_S


def test_extension_ratios_exact():
    """The fixtures were built from these numbers, so this is an inverse test of the geometry
    in extension_ratios, not a recording of its output."""
    e = F.extension_ratios(I_SHAPE)
    assert close(e["pinky"], 1.75) < TOL
    for k in ("index", "mid", "ring"):
        assert close(e[k], 1.00) < TOL
    assert close(F.extension_ratios(D_SHAPE)["index"], 2.10) < TOL
    assert close(F.extension_ratios(FLAT)["index"], 2.20) < TOL
    assert close(F.thumb_pinkymcp(I_SHAPE), 0.353041072007) < TOL


def test_gate_truth_table():
    """j_gate fires on I and nothing else here, z_gate on D and nothing else."""
    assert bool(F.j_gate(I_SHAPE)) is True
    assert bool(F.j_gate(D_SHAPE)) is False
    assert bool(F.j_gate(FLAT)) is False
    assert bool(F.j_gate(X_SHAPE)) is False
    assert bool(F.z_gate(D_SHAPE)) is True
    assert bool(F.z_gate(I_SHAPE)) is False
    assert bool(F.z_gate(FLAT)) is False

    # Vectorized over a batch, because the segmenter calls these per frame but calibrate and
    # the archive sweeps call them on whole (T,21,2) stacks.
    stack = np.stack([FLAT, I_SHAPE, D_SHAPE, X_SHAPE])
    assert list(F.j_gate(stack)) == [False, True, False, False]
    assert list(F.z_gate(stack)) == [False, False, True, False]


def test_z_gate_straightness_clause():
    """A hooked index fails z_gate on straightness alone. Every straight-finger fixture passes
    the clause trivially (straightness 1.0), so without this hand a z_gate that dropped the
    test would still pass the truth table."""
    assert close(F.finger_straightness(D_SHAPE, "index"), 1.0) < TOL
    st = float(F.finger_straightness(X_SHAPE, "index"))
    assert close(st, 1.0 / 3.0) < TOL, st
    # Only the index changed: the other clauses read exactly as they do for D.
    e_d, e_x = F.extension_ratios(D_SHAPE), F.extension_ratios(X_SHAPE)
    for k in ("thumb", "mid", "ring", "pinky"):
        assert close(e_d[k], e_x[k]) < TOL, k
    assert bool(F.z_gate(X_SHAPE)) is False
    assert bool(F.z_gate(D_SHAPE)) is True


def test_j_gate_thumb_ceiling():
    """j_gate's thumb clause is thumb_pinkymcp < J_THUMB_MAX, and the constant is 1.30: measured
    on the prompted takes, 1.20 lost 13 of 60 J items at the gate and 1.30 admits 13 of 2,278
    other archive frames (all Y) that start no track. The value is pinned so a change to it is
    a deliberate, re-measured one."""
    assert F.J_THUMB_MAX == 1.30, F.J_THUMB_MAX
    for ratio, expect in ((1.25, True), (1.35, False)):
        Q = with_thumb_pinkymcp(I_SHAPE, ratio)
        assert close(F.thumb_pinkymcp(Q), ratio) < TOL
        assert bool(F.j_gate(Q)) is expect, (ratio, expect)


def test_gates_are_scale_and_translation_invariant():
    """Both gates are built from ratios to palm_scale, so standing further from the camera must
    not change which gate fires. This is the same fix as shape42's, on the gate path."""
    for P, j, z in ((I_SHAPE, True, False), (D_SHAPE, False, True)):
        for k in (0.4, 1.0, 3.0):
            Q = P.mean(axis=0) + (P - P.mean(axis=0)) * k + np.array([5.0, -2.0])
            assert bool(F.j_gate(Q)) is j, (k, "j")
            assert bool(F.z_gate(Q)) is z, (k, "z")


def test_static_feature_dims_and_registry():
    """static/v3 is 101-D and unchanged; static/v4 is v3 followed by the 11-value thumb block.
    The registry is what every consumer resolves a model's feature tag through, so its keys,
    functions and widths are part of the contract, as is None meaning v3."""
    assert F.STATIC_DIM == 101 and F.STATIC_DIM_V4 == 112
    for P in (FLAT, I_SHAPE, D_SHAPE, X_SHAPE):
        v3, v4 = F.static_feature(P), F.static_feature_v4(P)
        assert v3.shape == (101,) and v4.shape == (112,)
        assert close(v4[:101], v3) == 0.0, "v4 must begin with v3 bit for bit"
        assert np.isfinite(v4).all()
    stack = np.stack([FLAT, I_SHAPE, D_SHAPE])
    assert F.static_feature_v4(stack).shape == (3, 112)
    assert close(F.static_feature_v4(stack)[1], F.static_feature_v4(I_SHAPE)) < 1e-12

    assert set(F.STATIC_FEATURES) == {"static/v3", "static/v4"}, sorted(F.STATIC_FEATURES)
    assert F.STATIC_FEATURES["static/v3"] == (F.static_feature, 101)
    assert F.STATIC_FEATURES["static/v4"] == (F.static_feature_v4, 112)
    assert F.static_feature_for(None) == (F.static_feature, 101)
    assert F.static_feature_for("static/v4")[1] == 112
    try:
        F.static_feature_for("static/v0")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown tag must raise, not fall back to some layout")


def test_static_feature_v4_block_layout():
    """The thumb block, in the order docs/features.js reproduces: [101] thumb straightness,
    [102:107] tips 4, 8, 12, 16, 20 to the palm center, [107:112] thumb tip to landmarks
    6, 7, 10, 11, 3 -- all over palm_scale. Derivable on the fixtures: build_hand lays the
    thumb joints on one line, so straightness is exactly 1.0; the pinky tip of I_SHAPE sits at
    (0.45, h) with h = sqrt((1.75 S)^2 - 0.45^2) and the palm center at (0, 0.8)."""
    v = F.static_feature_v4(I_SHAPE)
    assert close(v[101], 1.0) < TOL, v[101]
    h = np.sqrt((1.75 * PALM_S) ** 2 - 0.45 ** 2)
    assert close(v[106], np.hypot(0.45, h - 0.8) / PALM_S) < TOL, v[106]
    assert close(v[102], np.linalg.norm(I_SHAPE[4] - np.array([0.0, 0.8])) / PALM_S) < TOL
    for k, j in enumerate((6, 7, 10, 11, 3)):
        assert close(v[107 + k], np.linalg.norm(I_SHAPE[4] - I_SHAPE[j]) / PALM_S) < TOL, j
    assert F.THUMB_TARGETS == (6, 7, 10, 11, 3) and F.THUMB_CHAIN == (1, 2, 3, 4)
    # The block is not degenerate: the hooked index moves the thumb-to-index-PIP/DIP entries.
    assert close(F.static_feature_v4(X_SHAPE)[107:109], v[107:109]) > 0.05


def test_static_feature_v4_translation_and_scale_invariant():
    """Every entry of the block is a ratio of distances between landmarks to palm_scale, so
    carrying the hand or standing closer to the camera must leave all 112 values unchanged --
    the same property the whole scale fix rests on, asserted on the new block explicitly."""
    for P in (FLAT, I_SHAPE, D_SHAPE, X_SHAPE):
        base = F.static_feature_v4(P)
        for d in ([0.3, -0.7], [-12.0, 4.5], [1e3, 1e3]):
            assert close(F.static_feature_v4(P + np.array(d))[101:], base[101:]) < TOL
            assert close(F.static_feature_v4(P + np.array(d)), base) < TOL
        centroid = P.mean(axis=0)
        for k in (0.25, 0.5, 0.7, 0.85, 1.2, 2.0, 7.0):
            scaled = centroid + (P - centroid) * k + np.array([1.0, -2.0])
            got = F.static_feature_v4(scaled)
            assert close(got[101:], base[101:]) < TOL, (k, close(got[101:], base[101:]))
            assert close(got, base) < TOL, k


def test_to_isotropic():
    """u = x * (W/H): x is stretched, y is untouched, and the correction is the identity on a
    square frame. Skipping this is what turned the Z gate's false-positive count from 118 into
    366 in the spec's D1 measurement, so it is not cosmetic."""
    raw = np.stack([I_SHAPE[:, 0] * 0.3 + 0.5, I_SHAPE[:, 1] * 0.3 + 0.5], axis=-1)
    out = F.to_isotropic(raw, 1920, 1080)
    assert out.shape == raw.shape
    assert close(out[:, 1], raw[:, 1]) == 0.0, "y must be bit-identical"
    assert close(out[:, 0], raw[:, 0] * (16.0 / 9.0)) < TOL
    assert close(out[:, 0], raw[:, 0]) > 0.1, "x must actually change on a 16:9 frame"
    assert close(F.to_isotropic(raw, 720, 720), raw) < TOL, "square frame must be the identity"

    # A (T,21,3) landmark array with MediaPipe's z present must come back as (T,21,2).
    xyz = np.concatenate([np.stack([raw] * 4), np.zeros((4, 21, 1))], axis=-1)
    assert F.to_isotropic(xyz, 1920, 1080).shape == (4, 21, 2)


def test_canonicalize_handedness():
    assert close(F.canonicalize_handedness(D_SHAPE, "Right"), D_SHAPE) == 0.0
    flipped = F.canonicalize_handedness(D_SHAPE, "Left")
    assert close(flipped[:, 0], -D_SHAPE[:, 0]) == 0.0
    assert close(flipped[:, 1], D_SHAPE[:, 1]) == 0.0
    assert close(F.canonicalize_handedness(D_SHAPE, None), D_SHAPE) == 0.0
    # Must not mutate the caller's array: the segmenter keeps the same buffer across states.
    assert close(D_SHAPE[:, 0], build_hand({"index": 2.10, "mid": 1.00, "ring": 1.00,
                                            "pinky": 1.00}, (-0.10, 1.30))[:, 0]) == 0.0


def test_event_features_dimensionality():
    times, P = _straight_event()
    for arm in ("J", "Z"):
        f = F.event_features(times, P, arm)
        assert f.shape == (79,), f.shape
        assert f.dtype == np.float64
        assert np.isfinite(f).all(), np.where(~np.isfinite(f))[0]


def test_event_features_arm_flag():
    """[78] tells the classifier which finger drew the path, and it is the only feature that
    can distinguish an I-launched hook from a D-launched one when the paths coincide."""
    times, P = _straight_event()
    assert F.event_features(times, P, "J")[78] == 0.0
    assert F.event_features(times, P, "Z")[78] == 1.0


def test_event_features_too_short():
    times, P = _straight_event()
    for n in (0, 1, 2):
        try:
            F.event_features(times[:n], P[:n], "J")
        except ValueError:
            continue
        raise AssertionError(f"{n}-frame event should have raised ValueError")


def test_event_layout_straight_derivable():
    """Closed-form checks on the collinear event, block by block, against the documented layout.

    L = 2.775 is derivable: the tip travels 3.0 palm units over 21 evenly spaced frames, and the
    0.15 s causal average at the last frame is the mean of the final four samples, i.e. u=18.5/20,
    while at the first frame it is u=0. (18.5/20 - 0) * 3.0 = 2.775.
    """
    times, P = _straight_event()
    f = F.event_features(times, P, "J")

    # [0:32] arc-length-resampled path, centered on its own centroid: 16 points spread from
    # -L/2 to +L/2 along (0.6,-0.8), so the first two points are exactly derivable.
    assert close(f[0:2], [-1.3875 * 0.6, -1.3875 * -0.8]) < TOL
    assert close(f[2:4], [(-1.3875 + 2.775 / 15) * 0.6, (-1.3875 + 2.775 / 15) * -0.8]) < TOL
    assert close(f[0:32].reshape(16, 2).mean(axis=0), [0.0, 0.0]) < TOL, "must be centered"

    assert close(f[32:46], np.zeros(14)) < TOL, "a straight path has no turning"
    assert close(f[46:51], [0.878262258463, 1.0, 1.0, 1.0, 1.75]) < TOL     # ext_med
    assert close(f[51:53], [-0.15 / np.hypot(0.15, 1.0), 1.0 / np.hypot(0.15, 1.0)]) < TOL
    assert close(f[53], 2.775) < TOL                                        # L
    assert close(f[54], 1.0) < TOL                                          # duration
    assert close(f[55], 1.0) < TOL                                          # straightness
    assert f[56] == 0.0 and close(f[57], 0.0) < TOL and close(f[58], 0.0) < TOL
    assert close(f[59:61], [0.0, 0.0]) < TOL, "a rigidly carried hand has zero shape sigma"
    assert close(f[61], 0.0) < TOL, "constant palm size means zero scale ratio"
    assert close(f[62:68], [0.45, -0.6, 0.54, -0.72, 0.495, -0.66]) < TOL   # thirds
    assert close(f[68], 2.775) < TOL                                        # mean tip speed
    assert close(f[69], 4.5) < TOL                                          # peak tip speed
    assert close(f[70:72], [0.0, 0.0]) < TOL, "rigid carry: no tip motion relative to the palm"
    assert close(f[76:78], [0.0, 0.0]) < TOL
    # [72] and [75] are deliberately NOT asserted here. They are a ratio and a turning-angle sum
    # over the tip-relative path, which for a perfectly rigid carry has length ~1e-17 -- so both
    # amplify float noise into arbitrary values. Real footage never reaches that degeneracy
    # (landmark jitter is ~0.01 palm), but a synthetic fixture does, and pinning noise would
    # make this test fail on an unrelated numpy change.


def test_event_j_and_z_shape_signals():
    """The features the event classifier is meant to decide on actually separate the two paths.

    Not an accuracy claim -- two synthetic paths prove nothing about real signing. It checks
    that [55] straightness, [56] corner count and [58] signed net turn carry the geometry the
    spec says they carry, which is the precondition for the model learning geometry rather
    than this signer's tempo.
    """
    fj = F.event_features(*_j_event(), "J")
    fz = F.event_features(*_z_event(), "Z")

    assert fj[55] < 0.90 and fz[55] < 0.90, "neither glyph may look like straight transport"
    assert fj[57] > 1.5, "the J hook must register as one large turn"
    assert fj[58] > 0.3, "a hook turns one way; the signed net turn must be one-sided"
    assert fz[56] >= 2, "the Z must register at least its two corners"
    assert close(fz[58], 0.0) < TOL, "this symmetric Z turns equally both ways by construction"
    assert fz[53] > fj[53], "the three-stroke Z is the longer path"
    assert fj[78] == 0.0 and fz[78] == 1.0


def test_event_pinned():
    """Drift detectors. These are recorded outputs, not derivations: they exist so that an edit
    to smoothing, resampling or the turning-angle convention cannot pass unnoticed."""
    fj = F.event_features(*_j_event(), "J")
    fz = F.event_features(*_z_event(), "Z")
    assert close([fj[53], fj[55], fj[57], fj[58], fj[56]],
                 [2.870059879065, 0.638811650317, 2.392778398852,
                  0.618889815875, 1.0]) < TOL, fj[[53, 55, 57, 58, 56]]
    assert close([fz[53], fz[55], fz[57], fz[56]],
                 [5.929276108385, 0.421636630560, 2.036176961288, 3.0]) < TOL, \
        fz[[53, 55, 57, 56]]
    assert close(fj[0:4], [-0.203893737886, 1.070337076248,
                           -0.203893737886, 0.878999750977]) < TOL, fj[0:4]


def test_rolling_sigma_window_is_trailing():
    """rolling_shape_sigma against a trailing window, not a whole-clip median. features.py
    records p95 = 0.381 over the archive one way and 0.120 the other; SHAPE_STABLE (0.14) is
    calibrated for the trailing one, so swapping them breaks every stillness threshold at once.
    The ordering below is this test's own measurement; the two p95s are not."""
    times = np.linspace(0.0, 2.0, 41)
    shapes = F.shape42(np.repeat(D_SHAPE[None, :, :], 41, axis=0))   # rigid: sigma must be zero
    assert close(F.rolling_shape_sigma(shapes, times), np.zeros(41)) < TOL

    # A slowly reshaping hand: the trailing-window sigma must stay far below what a whole-clip
    # reference reports, because slow drift accumulates against a fixed median and not against
    # a 0.4 s one.
    morph = np.stack([build_hand({"index": 1.0 + 1.1 * u, "mid": 1.0, "ring": 1.0,
                                  "pinky": 1.0}, (-0.10, 1.30))
                      for u in np.linspace(0.0, 1.0, 41)])
    shapes = F.shape42(morph)
    trailing = F.rolling_shape_sigma(shapes, times, window_s=0.4)
    whole = F.shape_sigma(shapes)
    assert trailing.max() < whole.max(), (trailing.max(), whole.max())


def test_event_rigidity_uses_the_trailing_window():
    """[59]/[60] must come from rolling_shape_sigma, not from a whole-clip reference.

    They are the plain trailing-window sigma, the segmenter's stability signal (the runtime
    veto reads the rotation-aligned one; event_features says so). The rigid fixtures elsewhere
    in this file cannot catch a swap to shape_sigma -- both definitions give exactly zero on a
    hand that never reshapes. On a steadily reshaping hand they differ by more than a factor of
    two, so a forest trained on one and fed the other reads a different quantity.
    """
    times = np.linspace(0.0, 1.0, 21)
    morph = np.stack([build_hand({"index": 1.0 + 1.1 * u, "mid": 1.0, "ring": 1.0,
                                  "pinky": 1.0}, (-0.10, 1.30))
                      for u in np.linspace(0.0, 1.0, 21)])
    carry = np.linspace(0.0, 1.0, 21)[:, None] * np.array([0.6, -0.8]) * 3.0 * PALM_S
    f = F.event_features(times, morph + carry[:, None, :], "Z")

    shapes = F.shape42(morph + carry[:, None, :])
    trailing = F.rolling_shape_sigma(shapes, times)
    assert close(f[59], trailing.max()) < TOL, (f[59], trailing.max())
    assert close(f[60], np.median(trailing)) < TOL, (f[60], np.median(trailing))
    assert F.shape_sigma(shapes).max() > 2.0 * trailing.max(), "the two are not distinguished"


def test_resample_arclength_removes_speed():
    """Arc-length resampling must give the same 16 points whether the path was traced fast at
    the start or fast at the end -- that is what lets duration and speed be separate features
    instead of contaminating the path shape."""
    t_even = np.linspace(0.0, 1.0, 60)
    t_eased = t_even ** 2
    line = lambda s: np.stack([s * 2.0, -s], axis=-1)
    a = F.resample_arclength(line(t_even), 16)
    b = F.resample_arclength(line(t_eased), 16)
    assert a.shape == (16, 2)
    assert close(a, b) < 1e-12
    assert close(F.path_length(a), np.hypot(2.0, 1.0)) < 1e-12


# ---------------------------------------------------------------- segmenter contract

class _StubForest:
    """A forest that always answers `winner` with probability 1.0. Enough to drive the static
    branch end to end; the letters are whatever static_classes names them."""

    def __init__(self, n_classes, winner, dim):
        self.n_classes, self.winner, self.n_features_in_ = n_classes, winner, dim

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.shape[1] == self.n_features_in_, X.shape
        p = np.zeros((len(X), self.n_classes))
        p[:, self.winner] = 1.0
        return p


def _stream(seg, frames, W=1280, H=720):
    """Drive a segmenter over [(t, P_or_None)] of ISOTROPIC canonical hands; returns the
    emissions and the set of states visited. The fixtures are already isotropic, so they are
    handed over as if the frame were square and W/H = 1 -- the caller passes W == H."""
    ems, states = [], set()
    for t, P in frames:
        lm = None if P is None else np.concatenate([P, np.zeros((21, 1))], axis=1)
        em = seg.step(t, lm, None, W, W)
        states.add(seg.state)
        if em is not None:
            ems.append(em)
    return ems, states


def _hold(P, t0, seconds, fps=30):
    return [(t0 + k / fps, P) for k in range(int(seconds * fps))]


def _carry(P, t0, seconds, palm_per_s, fps=30):
    """P translated at a steady palm_per_s along +x: v_bar reads exactly palm_per_s and the
    rotation-aligned sigma stays 0, so nothing but the rising edge and the gates decide."""
    out = []
    for k in range(int(seconds * fps)):
        out.append((t0 + k / fps, P + np.array([palm_per_s * PALM_S * k / fps, 0.0])))
    return out


def test_segmenter_static_emission_reaches_the_output():
    """A held letter votes, passes the gate and is emitted exactly once, carrying the whole
    vote in detail['probs']. This is the regression for the NameError that shipped in the
    emission block: no Python test ever built a Segmenter with a static model, so an edit
    that broke every static letter left all 35 tests green."""
    classes = list("ABCDEFGHIKLMNOPQRSTUVWXY")
    forest = _StubForest(len(classes), classes.index("B"), F.STATIC_DIM_V4)
    seg = Segmenter(DEFAULT, static_model=forest, static_classes=classes,
                    static_feature_tag="static/v4")
    ems, states = _stream(seg, _hold(FLAT, 0.0, 2.0))
    assert HOLD in states
    assert [e.letter for e in ems] == ["B"], [e.letter for e in ems]
    assert ems[0].kind == "static" and ems[0].confidence == 1.0
    assert len(ems[0].detail["probs"]) == len(classes) and ems[0].detail["agree"] == 1.0
    # A deferred letter (I is a launch pose) is parked for D_WAIT and then delivered once.
    forest = _StubForest(len(classes), classes.index("I"), F.STATIC_DIM_V4)
    seg = Segmenter(DEFAULT, static_model=forest, static_classes=classes,
                    static_feature_tag="static/v4")
    ems, _ = _stream(seg, _hold(I_SHAPE, 0.0, 2.0))
    assert [e.letter for e in ems] == ["I"], [e.letter for e in ems]
    # The feature the forest is fed is the one its tag names: a v4-tagged stub fed by a v3
    # segmenter would have been handed 101 values and asserted above.
    assert seg.static_feature_tag == "static/v4"


def test_segmenter_arm_gates_off_never_tracks():
    """A parked launch pose that then moves arms a track by design; with arm_gates=False (the
    numbers mode, where '1' is the Z launch pose) the same stream must never enter TRACKING
    and must still emit its static letter."""
    classes = list("0123456789")
    forest = _StubForest(len(classes), 1, F.STATIC_DIM)
    frames = _hold(D_SHAPE, 0.0, 1.2) + _carry(D_SHAPE, 1.2, 0.5, 2.5) \
        + _hold(D_SHAPE + np.array([2.5 * PALM_S * 0.5, 0.0]), 1.7, 1.2)
    seg_on = Segmenter(DEFAULT, static_model=forest, static_classes=classes)
    _, states_on = _stream(seg_on, frames)
    assert TRACKING in states_on, "the carried D must arm a Z track with the gates on"
    seg_off = Segmenter(DEFAULT, static_model=forest, static_classes=classes, arm_gates=False)
    ems, states_off = _stream(seg_off, frames)
    assert TRACKING not in states_off, sorted(states_off)
    assert [e.letter for e in ems] == ["1"], [e.letter for e in ems]


def _hook(P, t0, seconds, radius_palm, fps=30):
    """The whole hand carried along a half circle of `radius_palm` palm widths in `seconds`:
    tip path pi*r, net 2r (straightness 0.64), rotation-aligned sigma 0. With r 1.5 over
    0.6 s the speed is ~7.9 palm/s, well over V_MOVE_ARMED, and the path clears L_MIN."""
    n = int(seconds * fps)
    out = []
    for k in range(n):
        a = np.pi * k / (n - 1)
        off = np.array([radius_palm * PALM_S * (1 - np.cos(a)), radius_palm * PALM_S * np.sin(a)])
        out.append((t0 + k / fps, P + off))
    return out


def _i_then_j(stroke_at):
    """Park in I from t=0, start a J hook at `stroke_at`, then hold the finishing pose. Both
    forests are stubs (I at 1.0, J at 1.0), so only the state machine decides the string.
    Returns (letters, I vote time, its D_WAIT deadline, the time the track armed)."""
    classes = list("ABCDEFGHIKLMNOPQRSTUVWXY")
    seg = Segmenter(DEFAULT, static_model=_StubForest(24, classes.index("I"), F.STATIC_DIM_V4),
                    motion_model=_StubForest(3, 0, F.EVENT_DIM), static_classes=classes,
                    motion_classes=["J", "Z", "MOVE"], static_feature_tag="static/v4")
    frames = _hold(I_SHAPE, 0.0, stroke_at) + _hook(I_SHAPE, stroke_at, 0.6, 1.5)
    frames += _hold(frames[-1][1], frames[-1][0] + 1 / 30, 1.5)
    letters, vote_t, deadline, arm_t, prev = [], None, None, None, None
    for t, P in frames:
        em = seg.step(t, np.concatenate([P, np.zeros((21, 1))], axis=1), None, 1280, 1280)
        if seg._pending is not None and vote_t is None:
            vote_t, deadline = seg._pending.t, seg._pending_until
        if seg.state == TRACKING and prev != TRACKING and arm_t is None:
            arm_t = t
        prev = seg.state
        if em is not None:
            letters.append(em.letter)
    assert vote_t is not None and arm_t is not None, (letters, vote_t, arm_t)
    return "".join(letters), vote_t, deadline, arm_t


def test_segmenter_pending_launch_letter_waits_for_the_arm_decision():
    """The PENDING LAUNCH LETTER rule at its three timings (segmenter.py docstring):
    a J whose track ARMS inside D_WAIT cancels the parked I ('J'); a J whose RISE began
    inside D_WAIT but whose V_MOVE_ARMED_SUSTAIN elapses after the deadline is held too, so
    the arm decision and not the timer settles it ('J', the gray zone that read 'IJ'); an I
    held well past D_WAIT before the stroke is released by the timer, by design ('IJ')."""
    th = DEFAULT
    s, vote_t, deadline, arm_t = _i_then_j(0.45)
    assert arm_t < deadline, (arm_t, deadline)
    assert s == "J", s
    s, vote_t, deadline, arm_t = _i_then_j(0.65)
    # the positive control for the gray zone: the arm lands after the deadline but within
    # one sustain (plus the frame that finds the rise) of it
    assert deadline <= arm_t <= deadline + th.V_MOVE_ARMED_SUSTAIN + 1 / 30 + 1e-9, (deadline, arm_t)
    assert s == "J", s
    s, vote_t, deadline, arm_t = _i_then_j(1.0)
    assert arm_t > deadline + th.V_MOVE_ARMED_SUSTAIN + 1 / 30, (deadline, arm_t)
    assert s == "IJ", s


def test_segmenter_checks_the_feature_width():
    """A forest fitted on 101 values must not be fed 112, and the tag must be a known one."""
    forest = _StubForest(24, 0, F.STATIC_DIM)
    try:
        Segmenter(DEFAULT, static_model=forest, static_feature_tag="static/v4")
    except ValueError:
        pass
    else:
        raise AssertionError("a 101-D forest under a static/v4 tag must be refused")
    try:
        Segmenter(DEFAULT, static_model=forest, static_feature_tag="static/v0")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown feature tag must be refused")
    assert Segmenter(DEFAULT, static_model=forest).static_feature_tag == "static/v3"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Exception:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}")
            print("      " + traceback.format_exc().strip().replace("\n", "\n      "))
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

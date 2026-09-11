"""Golden-vector tests pinning the feature contract in features.py.

features.py is the single implementation every consumer shares, which is the whole point of
it -- but it also means an edit there silently changes what the trained models mean. A model
pickled against shape42/v1 and fed a different shape42 does not crash; it just gets worse, and
the next person measures the drop as a data problem. These tests make that edit fail loudly.

Two kinds of assertion live here, and they are not equally strong:

  PROPERTIES are derived from the design and hold for any correct implementation -- 42 and 79
  dimensions, exact invariance of shape42 to translation and to uniform rescale, the gate
  truth table, the arm flag at [78], to_isotropic touching only x. These say the code is right.

  PINNED VALUES are numbers this implementation produces today, recorded so drift is visible.
  They do not say the code is right; they say it has not changed. Where a pinned number is
  independently derivable from the fixture geometry the derivation is written beside it, and
  those few are properties in disguise.

The three fixture hands are constructed, not captured. Each finger tip is placed at an exact
chosen distance from the wrist in palm units, so extension_ratios comes back as the round
number it was built from and the gate inequalities are exercised at known distances from their
thresholds rather than at whatever a recorded frame happened to give.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F


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

#: The I handshape, the J launch pose. Pinky at 1.75 clears J_GATE's 1.50; the other three at
#: 1.00 clear its 1.30 ceiling; the thumb tucked to the palm gives thumb_pinkymcp 0.353 against
#: a 1.15 ceiling. Index at 1.00 keeps it far below Z_GATE's 1.80.
I_SHAPE = build_hand({"index": 1.00, "mid": 1.00, "ring": 1.00, "pinky": 1.75}, (0.10, 0.90))

#: The D handshape, the Z launch pose. Index at 2.10 clears Z_GATE's 1.80, the other three at
#: 1.00 clear its 1.20 ceiling, thumb extension 1.265 sits under its 1.45. Pinky at 1.00 keeps
#: it below J_GATE's 1.50 floor.
D_SHAPE = build_hand({"index": 2.10, "mid": 1.00, "ring": 1.00, "pinky": 1.00}, (-0.10, 1.30))

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
    assert bool(F.z_gate(D_SHAPE)) is True
    assert bool(F.z_gate(I_SHAPE)) is False
    assert bool(F.z_gate(FLAT)) is False

    # Vectorized over a batch, because the segmenter calls these per frame but calibrate and
    # the archive sweeps call them on whole (T,21,2) stacks.
    stack = np.stack([FLAT, I_SHAPE, D_SHAPE])
    assert list(F.j_gate(stack)) == [False, True, False]
    assert list(F.z_gate(stack)) == [False, False, True]


def test_gates_are_scale_and_translation_invariant():
    """Both gates are built from ratios to palm_scale, so standing further from the camera must
    not change which gate fires. This is the same fix as shape42's, on the gate path."""
    for P, j, z in ((I_SHAPE, True, False), (D_SHAPE, False, True)):
        for k in (0.4, 1.0, 3.0):
            Q = P.mean(axis=0) + (P - P.mean(axis=0)) * k + np.array([5.0, -2.0])
            assert bool(F.j_gate(Q)) is j, (k, "j")
            assert bool(F.z_gate(Q)) is z, (k, "z")


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

    event_features says these are directly comparable to RIGID_VETO, which the segmenter
    applies to the trailing-window signal. The rigid fixtures elsewhere in this file cannot
    catch a swap to shape_sigma -- both definitions give exactly zero on a hand that never
    reshapes. On a steadily reshaping hand they differ by more than a factor of two, so a
    veto calibrated on one and fed the other rejects events the segmenter would have kept.
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

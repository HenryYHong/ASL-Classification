"""Replay tests: the COMMITTED models through the real Segmenter over recorded frames.

test_features.py drives the Segmenter with a stub forest that answers one letter at
probability 1.0; that proves the emission path runs, not that the shipped pickles, feature
tags, thresholds and recordings still fit together. These tests load temporal/model_static.p,
model_motion.p and model_digits.p exactly as live_demo.py does and feed them recorded
landmarks with their real timestamps, so a retrained forest with the wrong tag, a threshold
that stops every hold from emitting, or a NameError in the emission block (which once shipped
green because nothing here ever emitted) fails here before it reaches a camera.

Five streams:
  a static hold from static_s2.npz, one Segmenter per hold as the live path has: the hold's
    letter must come out, exactly once (in-sample for the static forest, so anything less is
    a wiring fault, not a model limit);
  20 s of the prompted J take (motion_clips.npz clip 1, 13-33 s, four J items) with both
    models: at least one J and no exception;
  the same window through evaluate.replay, the entry point evaluate.py's numbers 2 and 3 go
    through, with the pickle's feature tag as main() passes it: a J and no exception; and
    WITHOUT the tag the width check must refuse the 112-D forest (evaluate.py once built the
    Segmenter tagless and crashed there on every run);
  the handedness latch on a recorded hold: one frame labeled the other hand mid-hold must
    change nothing (same letter, same chirality), a sustained flip must switch the chirality
    after HAND_SWITCH_S, and a GAP_RESET must clear the latch;
  the numbers-mode stream: a '1' from the Ankara digits held, carried sideways at about 2.5
    palm widths per second, held again, then a '0' -- with arm_gates=False the machine must
    never enter TRACKING (z_gate passes a '1', so with arming on it would sit in a Z track
    and vote nothing), and '1' must still be emitted. The same stream with arm_gates=True
    must reach TRACKING, which documents why the switch exists.

Every test skips with a message rather than failing if its artifact is missing, because the
pickles are build products; a missing pickle is train_static.py's problem, not this file's.
"""
import os
import pickle
import sys
import traceback
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evaluate as EV                                  # noqa: E402
import features as F                                   # noqa: E402
from segmenter import Segmenter, TRACKING, HOLD        # noqa: E402
from thresholds import DEFAULT, DIGITS_OVERRIDES       # noqa: E402

STATIC = os.path.join(ROOT, "model_static.p")
MOTION = os.path.join(ROOT, "model_motion.p")
DIGITS = os.path.join(ROOT, "model_digits.p")
S2 = os.path.join(ROOT, "static_s2.npz")
CLIPS = os.path.join(ROOT, "motion_clips.npz")
ANKARA = os.path.join(ROOT, "digits_ankara.npz")


class Skip(Exception):
    pass


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise Skip(f"{os.path.basename(p)} is not built")


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _segmenter(static, motion=None, th=DEFAULT, **kw):
    """Built exactly as live_demo.py builds it: the feature function comes from the pickle's
    tag, and a tag/width mismatch raises here."""
    return Segmenter(th, static_model=static["model"], motion_model=motion["model"] if motion else None,
                     static_classes=static["classes"],
                     motion_classes=motion["classes"] if motion else None,
                     static_feature_tag=static.get("feature"), **kw)


def _modal(labels):
    labs = [str(x) for x in labels if str(x)[:1] in ("L", "R")]
    return max(sorted(set(labs)), key=labs.count) if labs else None


# ---------------------------------------------------------------- tests

def test_committed_static_model_emits_a_recorded_hold():
    _need(STATIC, MOTION, S2)
    st, mo = _load(STATIC), _load(MOTION)
    assert st["feature"] in F.STATIC_FEATURES, st["feature"]
    assert st["model"].n_features_in_ == F.STATIC_FEATURES[st["feature"]][1]
    e = np.load(S2, allow_pickle=True)
    L, T, W, H = e["letters"], e["stamps"], *[int(v) for v in np.asarray(e["frame_size"]).ravel()]
    bounds = [0] + [i for i in range(1, len(L)) if L[i] != L[i - 1]] + [len(L)]
    checked = []
    for a, b in list(zip(bounds, bounds[1:]))[:3]:          # the first three holds (A, B, C)
        letter = str(L[a])
        seg = _segmenter(st, mo)
        modal = _modal(e["handed"][a:b])
        ems, states = [], set()
        for i in range(a, b):
            em = seg.step(float(T[i]), e["lm"][i].astype(np.float64), modal, W, H)
            states.add(seg.state)
            if em is not None:
                ems.append(em)
        # a deferred I/D is delivered D_WAIT after its vote; the live stream's next frame does this
        em = seg.step(float(T[b - 1]) + DEFAULT.D_WAIT + 0.01, None, None, W, H)
        if em is not None:
            ems.append(em)
        assert HOLD in states, f"{letter}: the hand never parked"
        got = [x.letter for x in ems]
        assert got == [letter], f"hold {letter!r} at {T[a]:.1f}s emitted {got}"
        assert ems[0].kind == "static" and 0.0 < ems[0].confidence <= 1.0
        assert len(ems[0].detail["probs"]) == len(st["classes"])
        checked.append(letter)
    assert checked, "static_s2.npz has no hold"


def test_committed_models_replay_a_motion_take():
    _need(STATIC, MOTION, CLIPS)
    st, mo = _load(STATIC), _load(MOTION)
    d = np.load(CLIPS, allow_pickle=True)
    W, H = [int(v) for v in np.asarray(d["frame_size"]).ravel()[:2]]
    ci = 1
    raw, stamps, handed = d["clips"][ci].astype(np.float64), d["stamps"][ci].astype(np.float64), d["handed"][ci]
    keep = (stamps >= 13.0) & (stamps < 33.0)
    assert keep.sum() > 200, "clip 1 is shorter than the window this test expects"
    seg = _segmenter(st, mo)
    ems, states = [], set()
    for i in np.where(keep)[0]:
        lm = raw[i] if np.isfinite(raw[i, :, :2]).all() else None
        h = str(handed[i]) if lm is not None else None
        em = seg.step(float(stamps[i]), lm, h if h and h[:1] in ("L", "R") else None, W, H)
        states.add(seg.state)
        if em is not None:
            ems.append(em)
    assert TRACKING in states, "no track ever started on the J take"
    js = [x for x in ems if x.kind == "motion" and x.letter == "J"]
    assert js, f"no J emitted; emissions {[(round(x.t, 2), x.letter, x.kind) for x in ems]}"
    assert all(x.detail.get("end") in ("stopped", "slowed", "flush") for x in js), \
        [x.detail for x in js]


def _clip1_window(d, lo=13.0, hi=33.0):
    """evaluate.Clip for motion_clips.npz clip 1 between lo and hi seconds, with the per-frame
    handedness labels the recorder wrote (what evaluate.py feeds)."""
    W, H = [int(v) for v in np.asarray(d["frame_size"]).ravel()[:2]]
    stamps = d["stamps"][1].astype(np.float64)
    keep = (stamps >= lo) & (stamps < hi)
    assert keep.sum() > 200, "clip 1 is shorter than the window this test expects"
    return EV.Clip(clip_id="S1:CONTINUOUS:1", label="CONTINUOUS", session="S1",
                   times=stamps[keep], lm=d["clips"][1].astype(np.float64)[keep], width=W,
                   height=H, handedness=np.asarray(d["handed"][1]).astype(str)[keep])


def test_evaluate_replay_runs_the_committed_static_model():
    _need(STATIC, MOTION, CLIPS)
    st, mo = _load(STATIC), _load(MOTION)
    clip = _clip1_window(np.load(CLIPS, allow_pickle=True))
    ems, tap, probe = EV.replay(clip, DEFAULT, static=st["model"], motion=mo["model"],
                                static_classes=st["classes"], motion_classes=mo["classes"],
                                static_feature_tag=st.get("feature"))
    assert tap.static_feature_tag == st["feature"], tap.static_feature_tag
    assert [e.letter for e in ems if e.kind == "motion" and e.letter == "J"], \
        [(round(e.t, 2), e.letter, e.kind) for e in ems]
    assert probe.seen and tap.spans, (len(probe.seen), len(tap.spans))
    # main() reads the tag from the pickle and hands it to both consumers of replay()
    import inspect
    for fn in (EV.false_fire_rate, EV.letter_error_rate):
        assert "static_feature_tag" in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(EV.main)
    assert src.count("static_feature_tag=static_tag") == 2, "main() must pass the tag twice"
    assert 'static_blob.get("feature")' in src
    # and without the tag the Segmenter refuses the forest instead of feeding it 101 values
    if st["model"].n_features_in_ != F.STATIC_FEATURES[F.DEFAULT_STATIC_TAG][1]:
        try:
            EV.replay(clip, DEFAULT, static=st["model"], motion=mo["model"],
                      static_classes=st["classes"], motion_classes=mo["classes"])
        except ValueError as exc:
            assert "fitted on" in str(exc), str(exc)
        else:
            raise AssertionError("a tagless replay of the 112-D forest must be refused")


def _s2_hold(index=0):
    """One recorded hold of static_s2.npz: (letter, lm (n,21,3), stamps, modal label, W, H)."""
    e = np.load(S2, allow_pickle=True)
    L, T = e["letters"], e["stamps"]
    W, H = [int(v) for v in np.asarray(e["frame_size"]).ravel()]
    bounds = [0] + [i for i in range(1, len(L)) if L[i] != L[i - 1]] + [len(L)]
    a, b = list(zip(bounds, bounds[1:]))[index]
    return str(L[a]), e["lm"][a:b].astype(np.float64), T[a:b].astype(np.float64), \
        _modal(e["handed"][a:b]), W, H


def _run_hold(seg, lm, T, labels, W, H, capture=None):
    """Feed one hold frame by frame. Returns (emissions, seg.hand after every frame, the
    canonicalized P of frame `capture` as the segmenter buffered it)."""
    ems, hands, P_at = [], [], None
    for i in range(len(T)):
        em = seg.step(float(T[i]), lm[i], labels[i], W, H)
        hands.append(seg.hand)
        if i == capture:
            P_at = seg.buf[-1].P.copy()
        if em is not None:
            ems.append(em)
    em = seg.step(float(T[-1]) + DEFAULT.D_WAIT + 0.01, None, None, W, H)
    if em is not None:
        ems.append(em)
    return ems, hands, P_at


def test_handedness_latch_ignores_a_one_frame_flip():
    _need(STATIC, MOTION, S2)
    st, mo = _load(STATIC), _load(MOTION)
    letter, lm, T, modal, W, H = _s2_hold(0)
    other = "Right" if modal == "Left" else "Left"
    k = len(T) // 3
    assert T[k] - T[0] > DEFAULT.HOLD_SETTLE, "the flip must land inside the settled hold"
    labels = [modal] * len(T)
    labels[k] = other
    seg = _segmenter(st, mo)
    ems, hands, P_k = _run_hold(seg, lm, T, labels, W, H, capture=k)
    assert [e.letter for e in ems] == [letter], [e.letter for e in ems]
    assert set(hands) == {modal}, sorted(set(map(str, hands)))
    # the flipped frame was canonicalized with the latched label, i.e. not mirrored
    ref = F.canonicalize_handedness(F.to_isotropic(lm[k][:, :2], W, H), modal)
    assert np.allclose(P_k, ref), "the flipped frame was mirrored"
    assert not np.allclose(P_k, F.canonicalize_handedness(F.to_isotropic(lm[k][:, :2], W, H), other))


def test_handedness_latch_switches_on_a_sustained_flip_and_clears_on_gap_reset():
    _need(STATIC, MOTION, S2)
    st, mo = _load(STATIC), _load(MOTION)
    letter, lm, T, modal, W, H = _s2_hold(0)
    other = "Right" if modal == "Left" else "Left"
    k = len(T) // 2
    assert T[-1] - T[k] > DEFAULT.HAND_SWITCH_S + 0.2, "the flip must outlast HAND_SWITCH_S"
    labels = [modal] * k + [other] * (len(T) - k)
    seg = _segmenter(st, mo)
    _, hands, _ = _run_hold(seg, lm, T, labels, W, H)
    switched = [i for i, h in enumerate(hands) if h == other]
    assert switched, "the latch never switched on a sustained flip"
    dt = float(T[switched[0]] - T[k])
    assert DEFAULT.HAND_SWITCH_S <= dt < DEFAULT.HAND_SWITCH_S + 0.1, dt
    assert all(h == modal for h in hands[:switched[0]]) and all(h == other for h in hands[switched[0]:])
    # after the switch the frames ARE mirrored relative to the modal canonicalization
    fr = seg.buf[-1]
    i = int(np.argmin(np.abs(T - fr.t)))
    mirrored = F.canonicalize_handedness(F.to_isotropic(lm[i][:, :2], W, H), other)
    assert np.allclose(fr.P, mirrored), "frames after the switch must be canonicalized with the new label"
    # GAP_RESET clears the latch; the first frame after the gap adopts its own label at once
    seg.step(float(T[-1]) + DEFAULT.GAP_RESET + 0.05, None, None, W, H)
    assert seg.hand is None
    seg.step(float(T[-1]) + DEFAULT.GAP_RESET + 0.10, lm[0], modal, W, H)
    assert seg.hand == modal
    # a detected frame with no real label keeps the latched hand
    seg.step(float(T[-1]) + DEFAULT.GAP_RESET + 0.15, lm[1], "Unknown", W, H)
    assert seg.hand == modal


def _digit_stream(one, zero, fps=30):
    """Hold '1' 1.2 s, carry it sideways at ~2.5 palm widths/s for 0.5 s, hold 1.2 s, drop
    the hand past GAP_RESET, hold '0' 1.5 s. Raw Ankara landmarks in a 100x100 frame, where
    the palm spans about a quarter of the frame."""
    rng = np.random.default_rng(0)
    frames, t = [], 0.0

    def hold(P, dur):
        nonlocal t
        for _ in range(int(dur * fps)):
            frames.append((t, P + rng.normal(0, 0.0005, P.shape)))
            t += 1 / fps

    def move(P, dur, v):
        nonlocal t
        Q = P.copy()
        for _ in range(int(dur * fps)):
            Q[:, 0] += v / fps
            frames.append((t, Q + rng.normal(0, 0.0005, P.shape)))
            t += 1 / fps

    hold(one, 1.2)
    move(one, 0.5, 0.25 * 2.5)
    hold(one + np.array([[0.25 * 2.5 * 0.5, 0.0, 0.0]]), 1.2)
    for _ in range(int(0.6 * fps)):
        frames.append((t, None))
        t += 1 / fps
    hold(zero, 1.5)
    return frames


def test_digits_mode_never_tracks_a_carried_one():
    _need(DIGITS, ANKARA)
    dg = _load(DIGITS)
    assert dg["classes"] == [str(i) for i in range(10)]
    assert dg["feature"] in F.STATIC_FEATURES and dg["trained_on_author"] is False
    d = np.load(ANKARA, allow_pickle=True)

    def first(label):
        i = next(i for i in range(len(d["lm"]))
                 if str(d["letters"][i]) == label and str(d["handed"][i]) == "Left")
        return d["lm"][i].astype(np.float64)

    frames = _digit_stream(first("1"), first("0"))
    th = replace(DEFAULT, **DIGITS_OVERRIDES)
    results = {}
    for arm in (True, False):
        seg = _segmenter(dg, None, th, arm_gates=arm)
        ems, states = [], set()
        for t, P in frames:
            em = seg.step(t, P, "Left" if P is not None else None, 100, 100)
            states.add(seg.state)
            if em is not None:
                ems.append(em.letter)
        results[arm] = (ems, states)
    assert TRACKING in results[True][1], "with the gates on, carrying a '1' must arm a Z track"
    ems, states = results[False]
    assert TRACKING not in states, sorted(states)
    assert ems and ems[0] == "1", ems
    assert all(x in dg["classes"] for x in ems), ems


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Skip as why:
            print(f"skip  {fn.__name__}: {why}")
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

"""The labeling rule is one rule, applied to every kind of recording and to the evaluator.

label_events.assign_continuous credits ONE runtime-reachable gesture per prompted item and
labels everything else by the `other` policy. Three things here pin what that means at the
edges the committed recording never exercises (its five takes are all CONTINUOUS and carry no
near-miss item):

  a near-miss (NONE) item is negative footage end to end: its reachable spans stay MOVE under
    every policy (label_events), and evaluate.py counts a J fired inside it as a false fire
    over its whole park..rest span rather than as a "substitution" of a letter that was never
    prompted;
  a per-clip recording (collect_motion.py without --continuous, or ingested footage) goes
    through the same rule via label_events.clip_as_item, in both consumers: no aborted or
    veto-failing span is kept at the default policy, at most one letter event per clip, and a
    clip whose gesture is cut by a detection gap longer than GAP_INTERP yields no letter.

The per-clip cases slice real items out of the committed motion_clips.npz so the spans come
from the real Segmenter over real footage; they skip if the recording is not present.

The last group is about the file the script writes rather than the labels in it. `--out`
defaults to the committed temporal/events.npz, so `label_events.py --clips <anything>` used to
replace the motion training set with events cut from that anything, print the path as if that
were routine and exit 0. It happened here, and the file had to be restored from HEAD. The
refusal is checked end to end -- a real subprocess, and the committed file's bytes compared
before and after -- because the defect was never in the labeling and a unit test of the rule
would not have seen it.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evaluate as EV                                  # noqa: E402
import features as F                                   # noqa: E402
import label_events as LE                              # noqa: E402
import train_motion as TM                              # noqa: E402
from collect_motion import build_schedule              # noqa: E402
from segmenter import Emission                         # noqa: E402
from thresholds import DEFAULT                         # noqa: E402

CLIPS = os.path.join(ROOT, "motion_clips.npz")


class Skip(Exception):
    pass


# ---------------------------------------------------------------- synthetic spans

def _span(t0, arm, dur=0.8, n=12):
    """A runtime-reachable span: a unit palm triangle and a writing tip tracing a half circle
    of ~4 palm units (net ~2.6, straightness ~0.63), between T_MIN..T_MAX and L_MIN..L_MAX."""
    times = np.linspace(t0, t0 + dur, n)
    P = np.zeros((n, 21, 2))
    P[:, 0] = [0.0, 0.0]
    P[:, 5] = [1.0, 0.0]
    P[:, 17] = [0.5, 0.87]
    tip = F.TIP_FOR_ARM[arm]
    ang = np.linspace(0, np.pi, n)
    P[:, tip, 0] = 2.0 + 1.3 * np.cos(ang)
    P[:, tip, 1] = 2.0 + 1.3 * np.sin(ang)
    return {"P": P, "times": times, "arm": arm, "reason": "scored"}


def _schedule():
    items, _ = build_schedule({"J": 2, "Z": 2, "NONE": 2}, 1.2, 1.8, 1.2, 10.0)
    assert any(I["label"] == "NONE" for I in items)
    return items


# ---------------------------------------------------------------- tests: near-miss items

def test_none_item_spans_stay_move_under_every_policy():
    items = _schedule()
    spans = []
    for I in items:
        arm = I["label"] if I["label"] in ("J", "Z") else ("J" if I["index"] % 2 else "Z")
        spans.append(_span(I["go"][0] + 0.2, arm))
    for sp in spans:
        assert LE.span_geometry(sp, DEFAULT)[3], "fixture span is not runtime-reachable"
    n_none = sum(1 for I in items if I["label"] == "NONE")
    for policy in LE.OTHER_POLICIES:
        rows = LE.assign_continuous(spans, items, DEFAULT, other=policy)
        kept = [r for r in rows if r["gid"] != LE.ORPHAN_GID and items[r["gid"]]["label"] == "NONE"]
        assert len(kept) == n_none, (policy, [(r["gid"], r["label"]) for r in rows])
        assert all(r["label"] == "MOVE" and not r["credited"] for r in kept), policy
        credited = [r for r in rows if r["credited"]]
        assert len(credited) == len(items) - n_none, policy


def test_drop_still_discards_a_letter_items_fragment():
    """The policy the previous test constrains must still do its job: a second, shorter
    reachable span inside a J item is a fragment and goes under other='drop'."""
    items = _schedule()
    J = next(I for I in items if I["label"] == "J")
    spans = [_span(J["go"][0] + 0.1, "J"), _span(J["go"][0] + 1.0, "J", dur=0.5, n=8)]
    rows = LE.assign_continuous(spans, items, DEFAULT, other="drop")
    assert [r["credited"] for r in rows] == [True], [(r["label"], r["credited"]) for r in rows]
    rows = LE.assign_continuous(spans, items, DEFAULT, other="move")
    assert sorted(r["label"] for r in rows) == ["J", "MOVE"]


def _take_with_none_item():
    """A CONTINUOUS Clip for evaluate.py whose schedule is [J, NONE]."""
    items, total = build_schedule({"J": 1, "NONE": 1}, 1.2, 1.8, 1.2, 1.0)
    times = np.arange(0.0, total + 1.0, 1 / 30)
    lm = np.full((len(times), 21, 3), np.nan)
    return EV.Clip(clip_id="S9:CONTINUOUS:0", label="CONTINUOUS", session="S9", times=times,
                   lm=lm, width=1280, height=720, handedness="Left", items=items), items


def test_evaluate_treats_a_none_item_as_negative_footage():
    clip, items = _take_with_none_item()
    J, N = items[0], items[1]
    assert J["label"] == "J" and N["label"] == "NONE"
    assert clip.phase_at(J["go"][0] + 0.5) == "item"
    assert clip.phase_at(J["rest"][0] + 0.5) == "rest"
    assert clip.phase_at(N["park"][0] + 0.5) == "none"
    assert clip.phase_at(N["go"][0] + 0.5) == "none"
    assert clip.phase_at(N["rest"][0] + 0.5) == "none"
    win = clip.negative_windows()
    # the NONE item's whole park..rest span lies inside one negative window
    assert any(a <= N["park"][0] and N["rest"][1] <= b for a, b in win), (win, N)
    # and the J item's park..go span lies inside none
    assert not any(a < J["go"][0] + 0.5 < b for a, b in win), win
    expect = (J["park"][0] - clip.times[0]) + (J["rest"][1] - J["rest"][0]) \
        + (clip.times[-1] - J["rest"][1])
    assert abs(clip.negative_seconds - expect) < 1e-6, (clip.negative_seconds, expect)


def test_evaluate_counts_a_j_fired_in_a_none_item_as_a_false_fire():
    clip, items = _take_with_none_item()
    N = items[1]
    onset = N["go"][0] + 0.3
    fired = Emission("J", "motion", onset + 0.8, 0.9, {"arm": "J", "duration": 0.8})
    real = EV.replay
    EV.replay = lambda *a, **kw: ([fired], None, None)
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            EV.false_fire_rate([clip], DEFAULT, None, [], None, ["J", "Z", "MOVE"])
    finally:
        EV.replay = real
    text = out.getvalue()
    assert "false J:   1" in text, text
    assert "J 0, Z 0" in text, text          # not reported as a wrong-letter substitution
    assert f"{clip.negative_seconds / 60:.2f} min" in text, text


# ---------------------------------------------------------------- tests: per-clip recordings

def _per_clip_recording(path, with_gap):
    """Slice the first three J items of take 0 out of the committed recording as per-clip J
    clips (park start .. rest start), their rest gaps as NONE clips, and, with_gap, one more
    J clip with 0.4 s of NaN (> GAP_INTERP) cut through the middle of its only gesture."""
    if not os.path.exists(CLIPS):
        raise Skip("motion_clips.npz is not present")
    d = np.load(CLIPS, allow_pickle=True)
    lm, ts, hd = np.asarray(d["clips"][0]), np.asarray(d["stamps"][0]), np.asarray(d["handed"][0])
    items = [I for I in json.loads(str(d["prompts"][0])) if I["label"] == "J"][:3]
    clips, stamps, handed, labels = [], [], [], []
    for I in items:
        m = (ts >= I["park"][0]) & (ts < I["rest"][0])
        clips.append(lm[m]); stamps.append(ts[m] - ts[m][0]); handed.append(hd[m]); labels.append("J")
        m = (ts >= I["rest"][0]) & (ts < I["rest"][1])
        clips.append(lm[m]); stamps.append(ts[m] - ts[m][0]); handed.append(hd[m]); labels.append("NONE")
    if with_gap:
        # A J item whose intact clip yields exactly ONE reachable J-armed span (an item with a
        # reachable false start would credit that instead once the gesture aborts, which is
        # the rule working, not the case under test), with 0.4 s of NaN cut through the
        # middle of that span.
        W, H = [int(v) for v in np.asarray(d["frame_size"]).ravel()[:2]]
        for I in [I for I in json.loads(str(d["prompts"][0])) if I["label"] == "J"]:
            m = (ts >= I["park"][0]) & (ts < I["rest"][0])
            c, s, h = lm[m].copy(), ts[m] - ts[m][0], hd[m]
            spans = LE.harvest(c, s, h, DEFAULT, W, H)
            reach = [sp for sp in spans if LE.span_geometry(sp, DEFAULT)[3] and sp["arm"] == "J"]
            if len(reach) == 1:
                mid = float(np.mean(reach[0]["times"]))
                c[(s >= mid) & (s < mid + 0.4)] = np.nan
                clips.append(c); stamps.append(s); handed.append(h); labels.append("J")
                break
        else:
            raise Skip("no J item of take 0 yields exactly one reachable span")
    obj = lambda xs: np.array(xs + [None], dtype=object)[:-1]   # noqa: E731
    np.savez_compressed(path, clips=obj(clips), stamps=obj(stamps), handed=obj(handed),
                        prompts=obj([""] * len(clips)), labels=np.array(labels),
                        sessions=np.array([str(d["sessions"][0])] * len(clips)),
                        signers=np.array(["signer1"] * len(clips)),
                        frame_size=np.array(d["frame_size"], dtype=np.int32))
    return len(clips)


def test_per_clip_recording_is_labeled_by_the_credited_rule():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perclip.npz")
        n = _per_clip_recording(path, with_gap=True)
        with contextlib.redirect_stdout(io.StringIO()):
            out, meta = TM.load_clips_cut(path, None, "Left", DEFAULT, other="drop_unreachable",
                                          return_meta=True)
        assert out, "no event was cut from the sliced footage"
        assert all(m["reachable"] for m in meta), [m for m in meta if not m["reachable"]]
        assert all(m["reason"] == "scored" for m in meta), sorted({m["reason"] for m in meta})
        per_clip = {}
        for c in out:
            if c.label in ("J", "Z"):
                per_clip[c.clip_id] = per_clip.get(c.clip_id, 0) + 1
        assert per_clip and max(per_clip.values()) == 1, per_clip
        # the gap copy is the last clip: its gesture aborts at the gap, so no J is credited
        gap_id = LE.group_id(n - 1)
        assert not any(c.clip_id == gap_id and c.label == "J" for c in out), \
            [(c.clip_id, c.label) for c in out if c.clip_id == gap_id]
        # the same rule with the aborted spans kept as MOVE (other='move') keeps more rows,
        # none of them a letter
        with contextlib.redirect_stdout(io.StringIO()):
            more = TM.load_clips_cut(path, None, "Left", DEFAULT, other="move")
        assert len(more) >= len(out)
        assert sum(c.label == "J" for c in more) == sum(c.label == "J" for c in out)


def test_label_events_and_train_motion_agree_on_a_per_clip_recording():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perclip.npz")
        _per_clip_recording(path, with_gap=False)
        ev = os.path.join(tmp, "events.npz")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "label_events.py"), "--clips", path,
                            "--out", ev], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        e = np.load(ev, allow_pickle=True)
        with contextlib.redirect_stdout(io.StringIO()):
            out = TM.load_clips_cut(path, None, "Left", DEFAULT, other="drop_unreachable")
        assert sorted(e["y"].tolist()) == sorted(c.label for c in out), (e["y"], [c.label for c in out])
        assert sorted(e["clip_id"].tolist()) == sorted(c.clip_id for c in out)
        assert "credited with one event" in p.stdout, p.stdout


def test_clip_as_item_covers_the_whole_clip():
    stamps = np.linspace(2.0, 5.0, 91)
    (item,) = LE.clip_as_item("J", stamps)
    assert item["label"] == "J" and item["index"] == 0
    assert item["park"][0] <= stamps[0] and item["rest"][1] > stamps[-1]
    assert item["rest"][0] > stamps[-1], "a span starting anywhere in the clip is in-item"
    assert LE.clip_as_item("NONE", stamps)[0]["label"] == "NONE"
    assert LE.clip_as_item("MOVE", stamps)[0]["label"] == "NONE"


# ---------------------------------------------------------------- the committed default output

def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_label_events_refuses_to_write_the_committed_events_over_foreign_clips():
    """The real command, in a real subprocess, with the real committed file on disk.

    A sliced copy of the recording under a different name is exactly the shape of input that
    caused this: legitimate footage, a legitimate diagnostic run, and --out left at its
    default. The run must refuse, name the flag, exit non-zero, and leave events.npz byte for
    byte as it was.
    """
    events = os.path.join(ROOT, "events.npz")
    if not os.path.exists(events):
        raise Skip("temporal/events.npz is not present")
    before = _digest(events)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perclip.npz")
        _per_clip_recording(path, with_gap=False)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "label_events.py"),
                            "--clips", path], capture_output=True, text=True)
    assert p.returncode != 0, f"exited 0:\n{p.stdout}{p.stderr}"
    said = p.stdout + p.stderr
    assert "--out" in said, said
    assert "refusing to write" in said, said
    assert events in said, said
    assert _digest(events) == before, "events.npz was written by a run that claimed to refuse"


def test_train_motion_refuses_to_write_the_shipped_pickle_from_foreign_clips():
    """The same hazard one file over: --data moved, --out left at the committed pickle."""
    model = os.path.join(ROOT, "model_motion.p")
    if not os.path.exists(model):
        raise Skip("temporal/model_motion.p is not present")
    before = _digest(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perclip.npz")
        _per_clip_recording(path, with_gap=False)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "train_motion.py"),
                            "--data", path], capture_output=True, text=True)
    assert p.returncode != 0, f"exited 0:\n{p.stdout}{p.stderr}"
    said = p.stdout + p.stderr
    assert "--out" in said and "refusing to write" in said, said
    assert _digest(model) == before, "model_motion.p was written by a run that claimed to refuse"


def test_the_refusal_is_narrow_enough_to_leave_every_ordinary_run_alone():
    """Positive control on the rule itself: refuse the mismatch and nothing else.

    A guard that refused the ordinary retrain, or an explicit --out, would pass the two tests
    above while making the script useless, and nothing else here would notice.
    """
    other = os.path.join("/nowhere", "other.npz")
    assert LE.default_out_refusal(LE.DEFAULT_CLIPS, LE.DEFAULT_OUT) is None, "the retrain"
    assert LE.default_out_refusal(other, "/tmp/somewhere_else.npz") is None, "an explicit --out"
    assert LE.default_out_refusal(other, LE.DEFAULT_CLIPS) is None, "not the default output"
    # the default input spelled any other way is still the default input
    rel = os.path.relpath(LE.DEFAULT_CLIPS, os.getcwd())
    assert LE.default_out_refusal(rel, LE.DEFAULT_OUT) is None, rel
    said = LE.default_out_refusal(other, LE.DEFAULT_OUT)
    assert said and "--out" in said and other in said, said
    # and it carries a usable path rather than only a complaint
    assert "/nowhere/other_events.npz" in said, said


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

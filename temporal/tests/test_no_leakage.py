"""Structural guard: every train/test split in this project partitions by clip, not by row.

The README documents this project already reporting an accuracy from a random split over
frames of one held sign, where consecutive frames are near-duplicates and the number measured
re-identification rather than recognition. train_motion.py and evaluate.py defend against a
repeat structurally -- the array handed to the splitter has one row per clip, feature rows are
materialised inside each side afterwards, and augmentation runs on the training side only.
Those are properties of code that can be edited away, so they are tested here.

Three kinds of assertion live in this file:

  BEHAVIOURAL, on the real splitters, run over a synthetic registry of fake clips. No footage
  has been recorded yet, so motion_clips.npz does not exist; a test that skipped until it did
  would be a test that never ran before the mistake it guards against could be made. The clips
  are fabricated, but train_motion.cross_validate and its GroupKFold are not.

  A POSITIVE CONTROL. test_checker_catches_a_row_level_split feeds the same checker a
  deliberately leaky row-level split and requires it to complain. Without it, a checker that
  had quietly stopped checking anything would still report every other test passing.

  SOURCE-TEXT, over the shipped modules: the runtime assertions inside cross_validate and
  evaluate.main still exist, and no module reaches for a row-level splitter. These read the
  source, so they prove a guard is present, not that it is correct -- the behavioural tests
  above are what prove that.
"""
import ast
import contextlib
import inspect
import io
import os
import sys
import tempfile
import textwrap
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import features as F
import train_motion as TM
import train_static as TS
import evaluate as EV


# ---------------------------------------------------------------- synthetic clip registry

#: A plausible right hand, index-extended, in isotropic units. Only its palm triangle and the
#: two writing tips (8, 20) are read by anything here; the rest exists so shape42 and the
#: rigidity signals inside event_features see a hand rather than a point cloud.
BASE = np.array([
    [0.00, 0.00],
    [-0.70, 0.25], [-0.85, 0.45], [-0.95, 0.60], [-1.00, 0.75],
    [-0.45, 1.00], [-0.50, 1.40], [-0.52, 1.70], [-0.53, 2.00],
    [-0.15, 1.00], [-0.16, 1.45], [-0.17, 1.80], [-0.18, 2.10],
    [0.15, 1.00], [0.17, 1.40], [0.18, 1.70], [0.19, 1.95],
    [0.45, 1.00], [0.50, 1.35], [0.53, 1.60], [0.55, 1.80],
])
PALM_S = float(F.palm_scale(BASE))
T_FRAMES = 15


def _path(label, k):
    """A tip path per class, in palm units. Shape only has to be non-degenerate and distinct
    per class -- these clips test the splitter, never the classifier's accuracy."""
    t = np.linspace(0.0, 1.0, T_FRAMES)
    if label == "J":
        return np.stack([0.4 * np.sin(np.pi * t) * (1 + 0.1 * k), -2.0 * t], axis=1)
    if label == "Z":
        return np.stack([2.0 * np.abs(((2 * t + 0.25 * k) % 1.0) - 0.5), -1.5 * t], axis=1)
    return np.stack([1.5 * t, -1.5 * t * (1 + 0.1 * k)], axis=1)


def registry(per_class=2):
    """A small set of fake Clips with unique ids, balanced over {J, Z, MOVE}."""
    clips = []
    for label, arm in (("J", "J"), ("Z", "Z"), ("MOVE", "Z")):
        for k in range(per_class):
            times = np.linspace(0.0, 0.9, T_FRAMES)
            P = BASE[None, :, :] + (_path(label, k) * PALM_S)[:, None, :]
            clips.append(TM.Clip(len(clips), times, P, label, arm))
    return clips


def violations(train_rows, test_rows):
    """Every way a clip-level partition can be broken, as a list of readable complaints.

    Returns a list rather than asserting so the positive control below can require it to be
    non-empty on a split that is known to leak.
    """
    bad = []
    tr = {r.clip_id for r in train_rows}
    te = {r.clip_id for r in test_rows}
    if tr & te:
        bad.append(f"clip id(s) on both sides: {sorted(tr & te)}")
    if any(r.augmented for r in test_rows):
        bad.append(f"{sum(r.augmented for r in test_rows)} augmented row(s) on the test side")
    return bad


def observed_folds(clips, n_splits, n_aug):
    """Run the real cross_validate and record the rows each side of each fold was built from.

    rows_from is wrapped rather than re-implemented: re-deriving the split here would test a
    copy of the logic and pass happily while the shipped splitter leaked. The original is
    restored in a finally, and cross_validate's report is swallowed to keep this runner's
    output readable.
    """
    calls = []
    real = TM.rows_from

    def spy(clips_side, n_aug=0, seed=0):
        rows = real(clips_side, n_aug=n_aug, seed=seed)
        calls.append(rows)
        return rows

    TM.rows_from = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            TM.cross_validate(clips, n_splits, n_aug)
    finally:
        TM.rows_from = real

    assert len(calls) % 2 == 0 and calls, f"{len(calls)} rows_from call(s); expected 2 per fold"
    return [(calls[i], calls[i + 1]) for i in range(0, len(calls), 2)]


def _write_npz(path, sessions=True, n=4):
    """A minimal motion_clips.npz in the shape collect_motion.py writes, for evaluate.py."""
    clips = np.empty(n, dtype=object)
    stamps = np.empty(n, dtype=object)
    for i in range(n):
        lm = np.zeros((T_FRAMES, 21, 3))
        lm[:, :, :2] = BASE[None, :, :] + (_path("J", i) * PALM_S)[:, None, :]
        clips[i] = lm
        stamps[i] = np.linspace(0.0, 0.9, T_FRAMES)
    kw = {}
    if sessions:
        kw["sessions"] = np.array(["S1" if i < n // 2 else "S2" for i in range(n)])
    np.savez(path, clips=clips, stamps=stamps,
             labels=np.array(["J", "Z", "J", "Z"][:n]),
             frame_wh=np.array([1280, 720]), **kw)
    return path


# ---------------------------------------------------------------- tests

def test_registry_is_well_formed():
    clips = registry()
    ids = [c.clip_id for c in clips]
    assert len(set(ids)) == len(ids) == 6, ids
    assert {c.label for c in clips} == {"J", "Z", "MOVE"}


def test_rows_carry_their_parent_clip_id():
    clips = registry()
    rows = TM.rows_from(clips, n_aug=3)
    assert len(rows) == len(clips) * 4, len(rows)
    for c in clips:
        mine = [r for r in rows if r.clip_id == c.clip_id]
        assert len(mine) == 4 and sum(r.augmented for r in mine) == 3
        # An augmented copy that lost its parent id would be split away from its parent and
        # land on the other side of the fold: the leak this whole file exists to prevent.
        assert all(r.y == c.label for r in mine)


def test_rows_from_does_not_augment_by_default():
    rows = TM.rows_from(registry(), n_aug=0)
    assert not any(r.augmented for r in rows)
    assert len(rows) == 6


def test_augmentation_produces_a_different_row():
    """If augment() were a no-op the augmented-row ban would be vacuous rather than enforced."""
    clips = registry()
    rows = TM.rows_from(clips[:1], n_aug=1)
    assert not np.allclose(rows[0].x, rows[1].x)


def test_cross_validate_partitions_by_clip():
    clips = registry()
    for tr_rows, te_rows in observed_folds(clips, n_splits=3, n_aug=2):
        assert not violations(tr_rows, te_rows)


def test_cross_validate_covers_every_clip_exactly_once():
    """A partition, not merely a disjoint pair: no clip is tested twice, none is skipped."""
    clips = registry()
    seen = []
    for _, te_rows in observed_folds(clips, n_splits=3, n_aug=2):
        seen.extend({r.clip_id for r in te_rows})
    assert sorted(seen) == sorted(c.clip_id for c in clips), sorted(seen)


def test_cross_validate_tests_whole_clips():
    """Rows on the test side are materialised per whole clip, one each, from known clips.

    With n_aug=0 a clip yields exactly one row, so this pins the count rather than proving a
    multi-row clip stays intact -- cross_validate never builds a multi-row test clip, which is
    itself what test_augmentation_never_reaches_the_test_side checks.
    """
    clips = registry()
    by_id = {c.clip_id: c for c in clips}
    for _, te_rows in observed_folds(clips, n_splits=3, n_aug=0):
        counts = {}
        for r in te_rows:
            counts[r.clip_id] = counts.get(r.clip_id, 0) + 1
        assert all(n == 1 for n in counts.values()), counts
        assert set(counts) <= set(by_id)


def test_augmentation_never_reaches_the_test_side():
    clips = registry()
    for tr_rows, te_rows in observed_folds(clips, n_splits=3, n_aug=4):
        assert any(r.augmented for r in tr_rows), "n_aug=4 produced no augmented training row"
        assert not any(r.augmented for r in te_rows)


def test_checker_catches_a_row_level_split():
    """Positive control. The split this project is forbidden to make must fail the checker.

    Rows are materialised for every clip first, then split at random over rows -- the exact
    shape of the mistake the README documents. Both failure modes must be reported, otherwise
    the passes above only mean the checker is asleep.
    """
    rows = TM.rows_from(registry(), n_aug=2)
    order = np.random.default_rng(0).permutation(len(rows))
    cut = len(rows) // 2
    tr = [rows[i] for i in order[:cut]]
    te = [rows[i] for i in order[cut:]]
    found = violations(tr, te)
    assert any("both sides" in v for v in found), found
    assert any("augmented" in v for v in found), found


def test_static_contiguous_split_shares_no_frame():
    """train_static's per-class split is contiguous, so no frame lands on both sides.

    It is still a within-session split and train_static.py's own docstring says so; this test
    checks only that it does not interleave, which is what a random split over these frames
    would do -- neighbouring frames of one held sign are near-duplicates of each other.
    """
    per_class = [np.arange(20 + c).reshape(-1, 1, 1) + 0.0 for c in range(3)]
    tr, te = TS.contiguous_split(per_class, frac=0.8)
    for (c1, a), (c2, b), P in zip(tr, te, per_class):
        assert c1 == c2
        assert len(a) + len(b) == len(P)
        assert np.array_equal(np.concatenate([a, b]), P), "the split reordered the frames"
        assert not (set(a.ravel().tolist()) & set(b.ravel().tolist()))
        assert a[-1, 0, 0] < b[0, 0, 0], "the test side must be the tail, not a random subset"


def test_evaluate_refuses_a_split_it_cannot_make_honestly():
    """Without per-clip session labels evaluate.py must refuse rather than split at random."""
    with tempfile.TemporaryDirectory() as d:
        path = _write_npz(os.path.join(d, "clips.npz"), sessions=False)
        try:
            EV.load_recordings(path)
        except SystemExit as exc:
            assert "session" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("load_recordings accepted a recording with no session labels")


def test_evaluate_clip_ids_identify_a_clip_uniquely():
    """The session split is a clip partition only if ids are unique and one clip has one session.

    evaluate.main's split is inline and not importable, so this asserts the precondition it
    rests on, over ids that load_recordings actually synthesized; the assertion inside main
    that no id crosses the split is checked by test_split_assertions_still_exist below.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write_npz(os.path.join(d, "clips.npz"))
        clips = EV.load_recordings(path)
    ids = [c.clip_id for c in clips]
    assert len(set(ids)) == len(ids) == 4, ids
    train = [c for c in clips if c.session == "S1"]
    test = [c for c in clips if c.session == "S2"]
    assert train and test
    assert not ({c.clip_id for c in train} & {c.clip_id for c in test})


def test_split_assertions_still_exist():
    """The runtime guards inside the shipped splitters are present.

    A source check: it cannot tell whether an assertion is correct, only that deleting it is
    visible. The behavioural tests above are what establish that the split itself is sound.
    """
    want = {
        TM.cross_validate: [("train_ids", "test_ids"), ("augmented",)],
        EV.main: [("train_ids", "test_ids")],
    }
    for fn, groups in want.items():
        src = textwrap.dedent(inspect.getsource(fn))
        asserts = [ast.get_source_segment(src, n) or ""
                   for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assert)]
        for names in groups:
            assert any(all(nm in a for nm in names) for a in asserts), \
                f"{fn.__qualname__} no longer asserts on {names}"


def test_no_row_level_splitter_anywhere():
    """The standing rule: no random split over frames, windows or feature rows, in any module.

    Only group-aware splitters may appear. train_test_split on a feature matrix is the single
    call that would undo every structural defence in this file.
    """
    allowed = {"GroupKFold", "GroupShuffleSplit", "StratifiedGroupKFold",
               "LeaveOneGroupOut", "LeavePGroupsOut"}
    forbidden = ("KFold", "ShuffleSplit", "train_test_split")
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(ROOT, name)) as fh:
            try:
                tree = ast.parse(fh.read(), filename=name)
            except SyntaxError as exc:
                # A module that does not parse cannot be cleared of a row-level split, so this
                # is a failure and not a skip -- but say which file and why, because a bare
                # SyntaxError out of a leakage test reads as a broken test.
                raise AssertionError(f"{name} does not parse ({exc}); it cannot be scanned")
        for node in ast.walk(tree):
            ident = node.id if isinstance(node, ast.Name) else \
                    node.attr if isinstance(node, ast.Attribute) else \
                    node.name if isinstance(node, ast.alias) else None
            if ident and any(f in ident for f in forbidden):
                assert ident in allowed, f"{name} references {ident}"


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

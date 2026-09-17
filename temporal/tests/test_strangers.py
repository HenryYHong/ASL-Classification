"""The other people's hands: provenance, canonicalization and the two gates that ship with them.

Three things this pins that nothing else would catch:

  1. temporal/aslnow.npz's per-record chirality decision is reproducible from the committed
     detector (ingest_aslnow.chirality_detector on the author's sessions). A silent change to
     the detector, the sessions, or the feature would re-orient records and the training set
     with them, without any letter changing.
  2. Strangers never reach a test fold. crossval_static.run merges them into the TRAINING side
     of every leave-one-session-out fold; the held-out frames are the author's 4,878 and
     nothing else, so the published number is still leave-one-SESSION-out for the author.
  3. The committed forest passes the idle-hand gate at the committed thresholds (idle_gate.py,
     0 of 43 clean idle holds), and the shipped constants agree with each other (VOTE_PROB ==
     VOTE_PROB_FLOOR, which is what makes the floor real -- see thresholds.py).
"""
import os
import pickle
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crossval_static as CV  # noqa: E402
import features as F  # noqa: E402
import idle_gate as G  # noqa: E402
import ingest_aslnow as IA  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

TEMPORAL = os.path.dirname(HERE)


def test_aslnow_schema():
    d = np.load(ST.ASLNOW, allow_pickle=True)
    assert d["lm"].shape == (2122, 21, 3) and d["lm"].dtype == np.float32
    assert d["letters"].shape == (2122,) and d["mirror"].dtype == bool and d["mirror_p"].shape == (2122,)
    assert tuple(int(v) for v in d["aspect"]) == IA.ASPECT == (4, 3)
    assert str(d["license"]) == "MIT" and str(d["source"]) == IA.REPO_ID and str(d["revision"]) == IA.REVISION
    letters = np.asarray(d["letters"])
    assert set(letters) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert int(np.isin(letters, ST.LETTERS).sum()) == 1874 and int((~np.isin(letters, ST.LETTERS)).sum()) == 248


def test_aslnow_orientation_is_reproducible():
    d = np.load(ST.ASLNOW, allow_pickle=True)
    det = IA.chirality_detector()
    pm = IA.orient(d["lm"], det)
    agree = ((pm > 0.5) == np.asarray(d["mirror"])).mean()
    # the forest is seeded, the sessions are committed: the decision must not drift
    assert agree == 1.0, f"mirror decisions differ from the committed detector on {1 - agree:.4f} of records"
    assert np.abs(pm - np.asarray(d["mirror_p"])).max() < 1e-5


def test_loaders_are_canonical_and_static_only():
    per, P, letters, names = ST.load_aslnow()
    assert len(P) == 1874 and P.shape[1:] == (21, 2) and not np.isin(letters, ["J", "Z"]).any()
    assert sum(len(p) for p in per) == 1874 and len(per) == 24
    # mirrored records were mirrored: the detector must call the LOADED frames canonical
    det = IA.chirality_detector()
    canonical = det.predict_proba(T.FEATFN(P))[:, 1] < 0.5
    assert canonical.mean() > 0.97, f"only {canonical.mean():.3f} of loaded records read as canonical"
    ank, Pa, La, signer = ST.load_ankara_letters()
    assert set(La) == set(ST.DIGIT_LETTERS.values()) and len(Pa) == sum(len(p) for p in ank)
    assert len(set(signer)) > 200, "the Ankara set should carry the run-based signer ids"
    merged, sources = ST.load_strangers()
    assert sources == ["ankara:FOVW", "aslnow"] and sum(len(p) for p in merged) == 1874 + len(Pa)


def test_strangers_never_reach_a_test_fold():
    src = CV.sessions()
    henry_frames = sum(len(src[s][0][c]) for s in src for c in range(24))
    strangers, _ = ST.load_strangers()
    # a cheap forest: the test is about the split, not the score
    T_FOREST = dict(T.FOREST)
    T.FOREST.update(n_estimators=5)
    try:
        res = CV.run(seed=0, src=src, n_jobs=2, verbose=False, strangers=strangers)
    finally:
        T.FOREST.clear()
        T.FOREST.update(T_FOREST)
    assert res["total"] == henry_frames == 4878, res["total"]
    assert set(res["folds"]) == {"S1", "S2", "S3", "S4"}
    assert all(f["dropped"] == 0 for f in res["folds"].values()), "no filter ships"


def test_committed_forest_passes_the_idle_gate():
    blob = pickle.load(open(os.path.join(TEMPORAL, "model_static.p"), "rb"))
    featfn, dim = F.static_feature_for(blob["feature"])
    assert blob["model"].n_features_in_ == dim
    assert blob.get("strangers") == ["ankara:FOVW", "aslnow"] and blob.get("filter") == "none"
    holds = G.load_holds()
    assert len(holds) == 43 and sum(len(h) for h in holds) == 2180
    eh, nh, ev, nv, maxp, letters = G.gate(blob["model"], featfn, list(blob["classes"]), holds=holds)
    assert nv == 2064
    assert eh == 0 and ev == 0, f"{eh}/{nh} clean idle holds emit ({letters}); raise VOTE_PROB_FLOOR"
    assert maxp < DEFAULT.VOTE_PROB_FLOOR, (maxp, DEFAULT.VOTE_PROB_FLOOR)


def test_floor_is_real():
    # a floor above the confident route would be a no-op (thresholds.py, VOTE_PROB)
    assert DEFAULT.VOTE_PROB_FLOOR <= DEFAULT.VOTE_PROB, (DEFAULT.VOTE_PROB_FLOOR, DEFAULT.VOTE_PROB)
    assert DEFAULT.VOTE_PROB == DEFAULT.VOTE_PROB_FLOOR == 0.75


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

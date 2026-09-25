"""The other people's hands: provenance, canonicalization and the gates that ship with them.

What this pins that nothing else would catch:

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
  4. ASL-HG's schema, its signer ids and its cap: the cap is part of the recipe, so it has to
     be the same 6,000 frames every time it is drawn, and the leave-one-signer-out fold has to
     contain none of the signer it is scoring.
  5. ayuraj is never trained on. Not "load_strangers does not call load_ayuraj" -- the REAL
     training set is built here and every one of its 12,368 frames is checked against ayuraj's
     1,111. That is the only form of this test that survives someone adding a loader.
  6. A frame_size of shape (N,2) is applied per image or refused, never ravel()'d down to
     image 0's size (features.frame_aspect; the cost of the old behavior was 0.8212 -> 0.7814).
"""
import os
import pickle
import sys
import traceback

import numpy as np
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crossval_static as CV  # noqa: E402
import features as F  # noqa: E402
import idle_gate as G  # noqa: E402
import ingest_aslhg as IHG  # noqa: E402
import ingest_aslnow as IA  # noqa: E402
import ingest_ayuraj as IAY  # noqa: E402
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
    assert sources == ["ankara:FOVW", "aslnow", f"aslhg:cap{ST.ASLHG_CAP}"], sources
    assert sum(len(p) for p in merged) == 1874 + len(Pa) + 10 * 24 * ST.ASLHG_CAP
    # ASLNow and Ankara are NOT capped: only the two sets with many frames per signer are
    assert sum(len(p) for p in ST.load_strangers(aslhg=False)[0]) == 1874 + len(Pa)


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
    assert blob.get("strangers") == ["ankara:FOVW", "aslnow", f"aslhg:cap{ST.ASLHG_CAP}"]
    assert blob.get("filter") == "none"
    assert blob.get("author_cap") == T.AUTHOR_CAP and blob.get("aslhg_cap") == ST.ASLHG_CAP
    assert blob["forest"]["n_estimators"] == T.FOREST["n_estimators"]
    assert blob["forest"]["min_samples_leaf"] == T.FOREST["min_samples_leaf"]
    holds = G.load_holds()
    assert len(holds) == 43 and sum(len(h) for h in holds) == 2180
    classes = list(blob["classes"])
    eh, nh, ev, nv, maxp, letters = G.gate(blob["model"], featfn, classes, holds=holds)
    assert nv == 2064
    assert eh == 0 and ev == 0, f"{eh}/{nh} clean idle holds emit ({letters}); raise the floor"
    # ... and with headroom, per letter: the floor is only real if the most confident idle vote
    # THAT ALREADY CLEARS agree and margin sits under the floor of the letter that wins it.
    maxima = G.idle_maxima(blob["model"], featfn, classes, holds=holds)
    assert maxima, "no idle vote clears agree and margin at all; the gate would be vacuous"
    for letter, best in maxima.items():
        assert best < DEFAULT.vote_prob_for(letter), (letter, best, DEFAULT.vote_prob_for(letter))
    # G is the letter a relaxed hand is read as, and it is why the profile exists at all
    assert max(maxima, key=maxima.get) == "G", maxima


def _frame_keys(per_class):
    """One hashable key per frame of a per-class list, exact to the byte."""
    return {np.ascontiguousarray(P[i, :, :2], dtype=np.float64).tobytes()
            for P in per_class for i in range(len(P))}


def test_aslhg_schema():
    d = np.load(ST.ASLHG, allow_pickle=True)
    assert d["lm"].shape == (23984, 21, 3) and d["lm"].dtype == np.float32
    assert str(d["source"]) == IHG.DOI == "10.17632/j4y5w2c8w9.1"
    assert str(d["license"]) == IHG.LICENSE == "CC BY 4.0"
    assert tuple(int(v) for v in np.asarray(d["frame_size"])) == (300, 300)
    assert np.asarray(d["frame_wh"]).shape == (23984, 2)
    letters = np.asarray([str(x) for x in d["letters"]])
    assert set(letters) == set(ST.LETTERS), "J, Z and the digit folders are not letters here"
    # 100 per (signer, letter) less MediaPipe's 16 misses, all of them one signer's H
    assert len(letters) == 23984 and int((letters == "H").sum()) == 984
    assert all(int((letters == L).sum()) == 1000 for L in ST.LETTERS if L != "H")


def test_aslhg_signer_ids_come_from_the_filenames():
    d = np.load(ST.ASLHG, allow_pickle=True)
    signer = np.asarray([str(x) for x in d["signer"]])
    files = np.asarray([str(x) for x in d["files"]])
    assert set(signer) == {f"P{k}" for k in range(1, 11)}
    # the id is the P<k>_ prefix and nothing else; a wrong one would turn crossval_signers.py
    # into a random split that still printed a plausible number
    assert all(IHG.signer_of(f) == s for f, s in zip(files[::97], signer[::97]))
    n = {s: int((signer == s).sum()) for s in signer}
    assert n == dict({f"P{k}": 2400 for k in range(1, 11)}, P8=2384), n


def test_aslhg_cap_is_the_recipe_and_is_stable():
    per_all, P_all, L_all, S_all = ST.load_aslhg()
    assert len(P_all) == 23984 and P_all.shape[1:] == (21, 2)
    per, P, L, S = ST.load_aslhg(cap=ST.ASLHG_CAP)
    assert ST.ASLHG_CAP == 25
    assert len(P) == 6000 == 10 * 24 * 25, len(P)
    counts = {(s, l) for s, l in zip(S, L)}
    assert len(counts) == 240
    for s in set(S):
        for l in set(L):
            assert int(((S == s) & (L == l)).sum()) == 25
    # the same 6,000 frames every time: the cap is part of the recipe, not a per-run draw
    again = ST.load_aslhg(cap=ST.ASLHG_CAP)[1]
    assert np.array_equal(P, again)
    assert _frame_keys([P]) <= _frame_keys([P_all]), "the capped draw must be a subset"
    # ... and it is a permutation draw, not the first 25 in file order: ASL-HG's filenames run
    # over one continuous shoot, so the first 25 would be 25 consecutive shutter presses
    sel = ST.cap_per_signer_letter(L_all, S_all, 25)
    first = np.concatenate([np.where((S_all == s) & (L_all == l))[0][:25]
                            for s in sorted(set(S_all)) for l in ST.LETTERS])
    assert len(sel) == len(first) == 6000
    assert not np.array_equal(np.sort(sel), np.sort(first))


def test_aslhg_is_declared_for_training_and_for_testing():
    """ASL-HG ships on BOTH sides, and the two sides never touch the same frame."""
    per, sources = ST.load_strangers()
    assert sources == ["ankara:FOVW", "aslnow", f"aslhg:cap{ST.ASLHG_CAP}"], sources
    import crossval_signers as CG
    _, P_all, L_all, S_all = ST.load_aslhg()
    _, P_cap, L_cap, S_cap = ST.load_aslhg(cap=ST.ASLHG_CAP)
    held = "P4"
    train_side = ST.per_class_of(P_cap[S_cap != held], L_cap[S_cap != held])
    base, _ = CG.base_training_set(author_cap=T.AUTHOR_CAP)
    fold = ST.merge(base, train_side)
    test_side = [P_all[S_all == held]]
    assert sum(len(p) for p in test_side) == 2400
    assert not (_frame_keys(fold) & _frame_keys(test_side)), \
        f"{held}'s own frames are in the fold that scores {held}"
    assert sum(len(p) for p in train_side) == 9 * 24 * 25


def test_ayuraj_is_never_in_a_training_set():
    """The permanent holdout: build the REAL training set and look for its frames in it."""
    assert ST.NEVER_TRAIN == (ST.AYURAJ,)
    per, label = CV.shipped_training_set()
    ay_per, ay_P, ay_L, ay_S = ST.load_ayuraj()
    assert len(ay_P) == 1111 and len(set(ay_S)) == 5
    train_keys, ay_keys = _frame_keys(per), _frame_keys(ay_per)
    assert ay_keys, "ayuraj loaded nothing; this test would pass vacuously"
    assert not (train_keys & ay_keys), \
        f"{len(train_keys & ay_keys)} ayuraj frames are in the shipped training set"
    # and the same for every fold's training side, not just the full fit
    src = CV.sessions()
    strangers, _ = ST.load_strangers()
    for held in src:
        fold = ST.merge(*[src[s][0] for s in src if s != held])
        assert not (_frame_keys(ST.merge(fold, strangers)) & ay_keys)
    # the structural gate, not the convention
    try:
        ST.training_source(ST.AYURAJ)
    except AssertionError as e:
        assert "NEVER_TRAIN" in str(e)
    else:
        raise AssertionError("strangers.training_source accepted the never-train file")
    d = np.load(ST.AYURAJ, allow_pickle=True)
    assert str(d["license"]) == IAY.LICENSE and str(d["source"]) == f"kaggle:{IAY.SLUG}"


def test_the_harnesses_and_train_static_build_the_same_block():
    """crossval_static.shipped_training_set must equal what train_static.py itself fits on."""
    per_class = T.load()
    merged = {}
    for name in ("static_s2.npz", "static_s3.npz", "static_s4.npz"):
        for c, v in T.load_extra(os.path.join(TEMPORAL, name)).items():
            merged[c] = np.concatenate([merged[c], v]) if c in merged else v
    per_class = [np.concatenate([P, merged[c]]) if c in merged else P
                 for c, P in enumerate(per_class)]
    import static_aug as A
    train = ST.merge(A.cap_per_letter(per_class, T.AUTHOR_CAP), ST.load_strangers()[0])
    got, _ = CV.shipped_training_set()
    assert [len(P) for P in train] == [len(P) for P in got]
    assert all(np.array_equal(a, b) for a, b in zip(train, got))


def _two_row_npz(path, letters, handed, frame_size, signer=None):
    """A minimal load_extra/load_ankara_letters npz with two records and a given frame_size."""
    lm = np.zeros((2, 21, 3), dtype=np.float32)
    lm[0, :, 0] = 0.5          # x of every landmark, image 0
    lm[1, :, 0] = 0.5          # ... and image 1: identical normalized coordinates
    lm[:, :, 1] = 0.25
    kw = dict(lm=lm, letters=np.array(letters), handed=np.array(handed),
              stamps=np.arange(2, dtype=np.float32), frame_size=np.asarray(frame_size))
    if signer is not None:
        kw["signer"] = np.array(signer)
    np.savez(path, **kw)
    return path


def test_per_image_frame_size_is_not_image_zeros():
    """A frame_size of shape (N,2) must scale each image by ITS OWN aspect, or refuse.

    Both loaders used to do np.asarray(d["frame_size"]).ravel()[:2], which on an (N,2) array
    silently takes image 0's (W,H) and applies it to every record. Nothing downstream can see
    that: u = x*(W/H) is a plausible number for any W/H, so the run prints an accuracy instead
    of an error. Measured cost of the collapsed behavior on the real path (the judge's
    verification, ASL-HG re-written with per-image sizes): 0.8212 -> 0.7814.
    """
    import tempfile
    d = tempfile.mkdtemp()
    # image 0 is 100x200 (aspect 0.5), image 1 is 200x100 (aspect 2.0); same landmarks
    path = _two_row_npz(os.path.join(d, "mixed.npz"), ["A", "A"], ["Right", "Right"],
                        [[100, 200], [200, 100]])
    got = T.load_extra(path)
    P = got[T.LETTERS.index("A")]
    assert abs(P[0, 0, 0] - 0.5 * 0.5) < 1e-12, f"image 0 should be scaled by 100/200: {P[0, 0, 0]}"
    assert abs(P[1, 0, 0] - 0.5 * 2.0) < 1e-12, (
        f"image 1 must be scaled by ITS OWN 200/100, not image 0's: {P[1, 0, 0]}")

    path = _two_row_npz(os.path.join(d, "ank.npz"), ["0", "2"], ["Right", "Right"],
                        [[100, 200], [200, 100]], signer=["s0", "s1"])
    _, Pa, La, _ = ST.load_ankara_letters(path)
    order = {L: i for i, L in enumerate(La)}
    assert abs(Pa[order["O"], 0, 0] - 0.5 * 0.5) < 1e-12, Pa[order["O"], 0, 0]
    assert abs(Pa[order["V"], 0, 0] - 0.5 * 2.0) < 1e-12, (
        f"the second Ankara image must use its own aspect: {Pa[order['V'], 0, 0]}")


def test_frame_size_of_an_unexpected_shape_is_refused():
    """Anything that is neither (2,) nor (N,2) raises loudly rather than being ravel()'d."""
    import tempfile
    d = tempfile.mkdtemp()
    for bad in ([[100, 200, 3], [200, 100, 3]], [100, 200, 300], [[[100, 200]]]):
        path = _two_row_npz(os.path.join(d, "bad.npz"), ["A", "A"], ["Right", "Right"], bad)
        for fn in (lambda: T.load_extra(path), lambda: ST.load_ankara_letters(path)):
            try:
                fn()
            except SystemExit as e:
                assert "frame_size" in str(e), str(e)
            else:
                raise AssertionError(f"frame_size {np.asarray(bad).shape} was accepted")


def test_floor_is_real():
    # a floor above the confident route would be a no-op (thresholds.py, VOTE_PROB)
    assert DEFAULT.VOTE_PROB_FLOOR <= DEFAULT.VOTE_PROB, (DEFAULT.VOTE_PROB_FLOOR, DEFAULT.VOTE_PROB)
    assert DEFAULT.VOTE_PROB == DEFAULT.VOTE_PROB_FLOOR == 0.55
    # G, Q and A are the three letters a RESTING hand resembles -- a loose fist with the thumb
    # somewhere is an A, the same hand angled down is a Q, index and thumb apart is a G -- and
    # they are exactly the three that breach a flat floor on the idle holds. Measured over 16
    # jitter seeds of the shipped recipe: worst idle G 0.8433, Q 0.5504, A 0.5310, and nothing
    # else above 0.5105. The floors sit above those, and idle_gate.py --seeds 16 is the check.
    # G is 0.88 rather than 0.85 because the margin is nearly free: 0.85 leaves 0.0067 of room
    # and 0.88 leaves 0.0367, for 0.7% of real G on held-out signers (0.9640 -> 0.9570) and not
    # one extra hold on replay_static.py, whose G sweep is flat from 0.85 to 0.90.
    assert DEFAULT.VOTE_PROB_LETTER == {"G": 0.88, "Q": 0.62, "A": 0.60}
    assert DEFAULT.vote_prob_for("G") == DEFAULT.vote_floor_for("G") == 0.88
    assert DEFAULT.vote_prob_for("Q") == DEFAULT.vote_floor_for("Q") == 0.62
    assert DEFAULT.vote_prob_for("A") == DEFAULT.vote_floor_for("A") == 0.60
    # a letter nobody named keeps the flat floor
    assert DEFAULT.vote_prob_for("B") == DEFAULT.vote_floor_for("B") == 0.55


def test_both_vote_routes_read_the_per_letter_floor():
    """The profile is worth nothing if only one route consults it, and a segmenter that read
    th.VOTE_PROB directly would pass every number this file checks."""
    src = open(os.path.join(TEMPORAL, "segmenter.py")).read()
    verdict = src[src.index("def _vote_verdict"):]
    verdict = verdict[:verdict.index("\n    def ")]
    assert "th.vote_prob_for(letter)" in verdict and "th.vote_floor_for(letter)" in verdict
    assert "meanp >= th.VOTE_PROB" not in verdict
    # and behaviorally, through the pure vote function the idle gate uses
    classes = list(ST.LETTERS)
    flat55 = replace(DEFAULT, VOTE_PROB=0.55, VOTE_PROB_FLOOR=0.55, VOTE_PROB_LETTER={})
    flat75 = replace(DEFAULT, VOTE_PROB=0.75, VOTE_PROB_FLOOR=0.75, VOTE_PROB_LETTER={})

    def window(letter, p):
        w = np.full((DEFAULT.VOTE_MIN, len(classes)), (1.0 - p) / (len(classes) - 1))
        w[:, classes.index(letter)] = p
        return w

    assert G.vote(window("G", 0.70), flat55, classes)[0], "0.70 clears a flat 0.55"
    assert not G.vote(window("G", 0.70), DEFAULT, classes)[0], "G's own floor is 0.75"
    assert G.vote(window("A", 0.60), DEFAULT, classes)[0], "A is not charged G's bill"
    assert not G.vote(window("A", 0.60), flat75, classes)[0]


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

"""Smoke tests for the digits data and trainer.

The digit forest is trained on somebody else's hands (temporal/digits_ankara.npz, landmarks
from the Sign Language Digits Dataset) and nothing about it is verified on the author, so the
one thing tests can hold is that the artifacts stay what they were measured as: the npz has
the frames and signers the ingest log recorded, its labels are exactly '0'..'9', the trainer's
loader canonicalizes and groups them, a fit produces a forest over all ten classes, and
train_static.load_extra refuses to fold digits into the letter forest (it skips every label
outside the 24 letters and says so, rather than returning zero classes silently).

The frame and signer counts below are the committed file's (ingest_images.py --signer-rule
runs, ingest.log: 1,805 of 2,062 photos detected, 222 signer runs among them). A re-ingest
that changes them must change these numbers on purpose.
"""
import contextlib
import io
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import features as F                  # noqa: E402
import train_digits as TD             # noqa: E402
import train_static as TS             # noqa: E402

NPZ = os.path.join(ROOT, "digits_ankara.npz")
N_FRAMES, N_SIGNERS = 1805, 222


def test_npz_is_the_ingested_set():
    d = np.load(NPZ, allow_pickle=True)
    for key in ("lm", "stamps", "handed", "letters", "session", "frame_size", "signer", "files"):
        assert key in d.files, f"{key} missing from digits_ankara.npz"
    assert d["lm"].shape == (N_FRAMES, 21, 3), d["lm"].shape
    assert sorted(set(str(x) for x in d["letters"])) == [str(i) for i in range(10)]
    assert len(set(str(x) for x in d["signer"])) == N_SIGNERS
    assert np.asarray(d["frame_size"]).ravel().tolist() == [100, 100]
    assert all(str(x) in ("Left", "Right") for x in d["handed"])
    assert all(str(f).lower().endswith((".jpg", ".jpeg", ".png")) for f in d["files"])


def test_loader_canonicalizes_and_groups():
    P, y, g = TD.load_digits(NPZ)
    assert P.shape == (N_FRAMES, 21, 2) and y.shape == (N_FRAMES,) and g.shape == (N_FRAMES,)
    assert sorted(set(y.tolist())) == list(range(10))
    assert len(set(g.tolist())) == N_SIGNERS
    # every digit appears for many signers, so a signer split leaves every class in every fold
    for c in range(10):
        assert len(set(g[y == c].tolist())) > 100, c
    # isotropic + canonical: a square frame leaves x untouched, and the palm is a finite size
    assert np.isfinite(P).all() and (F.palm_scale(P) > 0).all()


def test_fit_covers_every_digit_on_each_registered_feature():
    P, y, g = TD.load_digits(NPZ)
    for tag in F.STATIC_FEATURES:
        model, X = TD.fit_all(P, y, tag, sigma=0.0, copies=0, seed=0, n_jobs=2)
        assert list(model.classes_) == list(range(10))
        assert X.shape == (N_FRAMES, F.STATIC_FEATURES[tag][1]), (tag, X.shape)
        blob = TD.blob_for(model, tag, len(X), len(P), len(set(g)), 0.0, 0, 0)
        for key in ("model", "classes", "feature", "n_train", "n_frames", "signers", "aspect",
                    "augment", "source", "trained_on_author"):
            assert key in blob, key
        assert blob["classes"] == [str(i) for i in range(10)]
        assert blob["trained_on_author"] is False and blob["aspect"] == [1, 1]


def test_committed_digit_model_matches_its_tag():
    path = os.path.join(ROOT, "model_digits.p")
    if not os.path.exists(path):
        print("      (model_digits.p not built; skipping the pickle check)")
        return
    import pickle
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    fn, dim = F.static_feature_for(blob["feature"])
    assert blob["model"].n_features_in_ == dim, (blob["feature"], blob["model"].n_features_in_)
    assert blob["classes"] == [str(i) for i in range(10)]
    assert blob["n_frames"] == N_FRAMES and blob["signers"] == N_SIGNERS
    assert blob["trained_on_author"] is False
    P, y, g = TD.load_digits(NPZ)
    pr = blob["model"].predict_proba(fn(P[:50]))
    assert pr.shape == (50, 10) and np.allclose(pr.sum(1), 1.0)


def test_load_extra_ignores_a_digit_file():
    """train_static must never merge digits into the letter forest: the file yields nothing,
    and the loader says so instead of returning an empty dict silently."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        got = TS.load_extra(NPZ)
    assert got == {}, sorted(got)
    assert "no frame" in out.getvalue() and "train_digits" in out.getvalue(), out.getvalue()


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

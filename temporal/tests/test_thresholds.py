"""thresholds_archive.json is the provenance record of thresholds.DEFAULT, not a tuned profile.

calibrate.py --offline writes it as DEFAULT with the four stillness constants it can measure
on the committed archive (calibrate.OFFLINE_FIELDS) recomputed; every other field is copied
from DEFAULT untouched. Twice the file has silently stopped being that: once it carried 14
retired pre-data guesses, once it carried VOTE_PROB_FLOOR 0.50 after the default moved to
0.55, and label_events.py, train_motion.py, evaluate.py and live_demo.py all accept it via
--thresholds, so the stale values ran without a word. These tests hold the committed file to
the contract so a default can no longer move without the file following it.

Plain functions with a main(), like the other files here; pytest collects them as well.
"""
import os
import sys
import traceback
from dataclasses import asdict, fields

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import calibrate as C                                  # noqa: E402
from thresholds import DEFAULT, Thresholds             # noqa: E402

ARCHIVE = os.path.join(ROOT, "thresholds_archive.json")


def test_offline_fields_are_thresholds_fields():
    known = {f.name for f in fields(Thresholds)}
    assert set(C.OFFLINE_FIELDS) <= known, sorted(set(C.OFFLINE_FIELDS) - known)
    assert len(set(C.OFFLINE_FIELDS)) == len(C.OFFLINE_FIELDS)


def test_archive_json_loads_as_thresholds():
    assert os.path.exists(ARCHIVE), f"{ARCHIVE} is missing; run calibrate.py --offline"
    th = Thresholds.from_json(ARCHIVE)
    assert isinstance(th, Thresholds)


def test_archive_json_differs_from_default_only_in_the_calibrated_fields():
    got, want = asdict(Thresholds.from_json(ARCHIVE)), asdict(DEFAULT)
    drift = sorted(k for k in want if got[k] != want[k] and k not in C.OFFLINE_FIELDS)
    assert not drift, (f"thresholds_archive.json differs from thresholds.DEFAULT in {drift}, "
                       f"which calibrate.py --offline does not measure; a default moved without "
                       f"the file following it. Rerun ./.venv/bin/python temporal/calibrate.py "
                       f"--offline")


def test_archive_json_carries_every_default_field():
    """A field added to Thresholds after the last regeneration loads as its default (from_json
    tolerates a missing key), so the drift test above cannot see it; this one can."""
    import json
    with open(ARCHIVE) as fh:
        keys = set(json.load(fh))
    missing = sorted({f.name for f in fields(Thresholds)} - keys)
    assert not missing, (f"thresholds_archive.json lacks {missing}; rerun calibrate.py "
                         "--offline so the record carries every constant that ships")


def test_checker_catches_a_drifted_file():
    """Positive control: a copy of the archive with one non-calibrated field moved must fail
    the same check, or the check above proves nothing."""
    import json
    import tempfile
    with open(ARCHIVE) as fh:
        data = json.load(fh)
    data["VOTE_PROB_FLOOR"] = round(data["VOTE_PROB_FLOOR"] - 0.05, 2)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(data, fh)
        path = fh.name
    try:
        got, want = asdict(Thresholds.from_json(path)), asdict(DEFAULT)
        drift = sorted(k for k in want if got[k] != want[k] and k not in C.OFFLINE_FIELDS)
        assert drift == ["VOTE_PROB_FLOOR"], drift
    finally:
        os.unlink(path)


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

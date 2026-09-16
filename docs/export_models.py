"""Export the trained forests and a set of golden vectors for the browser port.

Two outputs:

  models.json    the forests as flat typed arrays, gzip-friendly. A sklearn pickle cannot run
                 in a browser -- it is a Python object graph -- so the trees are flattened to
                 the four arrays a tree walk actually needs. The letter forest ('static'), the
                 J/Z forest ('motion') and, when temporal/model_digits.p exists, the numbers
                 forest ('digits'), plus the thresholds the page runs: 'thresholds' is
                 thresholds.DEFAULT as a dict, 'digits.thresholds' is DEFAULT with
                 DIGITS_OVERRIDES applied. The thresholds travel in the same file as the trees
                 so the page can never pick up a new forest with stale constants.

  golden.json    inputs paired with the exact outputs Python produces for them. The browser port
                 is checked against these rather than against a reading of the code, because the
                 expensive failures in this project have all been silent divergences between two
                 implementations that looked equivalent: training mirrored while inference did
                 not, a vote window that completed at one frame rate and never at another.

                 Three kinds of case, told apart by two optional fields:
                   model  'static' (default) | 'digits'   which forest, and so which feature
                                                           function (STATIC_FEATURES by the
                                                           forest's tag) the case is for
                   api    'solutions' (default) | 'tasks'  which MediaPipe API produced the
                                                           landmarks and the handedness label
                 A 'solutions' case stores `handedness`, the label the recorder wrote, and the
                 page feeds it to canonicalizeHandedness as is. A 'tasks' case stores
                 `reported_handedness`, the RAW label the Tasks API reported on the unflipped
                 frame, and the expected values were computed AFTER the swap the page applies to
                 every live label (docs/app.js swapTasksHandedness: Left <-> Right): the legacy
                 `solutions` API that produced every training landmark labels a hand as if the
                 image were mirrored (the selfie convention) while the Tasks API labels the
                 frame as given, so the same hand on the same frame gets the opposite word. A
                 port that feeds the raw label straight in mirrors the hand the wrong way and
                 misses these cases by probability (worst 0.52 at the time of writing, argmax
                 intact on all 11), which is the point of storing the raw label.

Run:  ./.venv/bin/python docs/export_models.py     (from the repository root; any cwd works)

The three pickles must exist first (train_static.py, train_motion.py, train_digits.py), and
golden.json is regenerated together with models.json: its expected probabilities come from
these exact forests, so the two files ship together. Deploying a new forest with an old golden
file makes the page's self-check fail on load, which is what it is for.
"""
import gzip
import json
import os
import pickle
import sys
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPORAL = os.path.join(HERE, "..", "temporal")
sys.path.insert(0, TEMPORAL)

import features as F                                  # noqa: E402
from thresholds import DEFAULT, digits_thresholds     # noqa: E402

#: Node budget for the letter forest. At 76 bytes per node in models.json (full-precision
#: thresholds, 4-dp leaves) the letter forest is 10.2 MB raw at its shipped 133,680 nodes
#: (23,705 training rows, min_samples_leaf 5); past 200k the page's first load on a phone
#: connection is the defect, not the accuracy.
NODE_BUDGET = 200_000

#: Pixel size of the frames behind the letter golden cases. static_sequences.npz and
#: static_sequences_tasks.npz are both extracted from the RandomForest/data JPEGs (1920x1080);
#: neither file records the size, and the size is load-bearing because to_isotropic corrects
#: MediaPipe's normalized x by the aspect ratio.
ARCHIVE_WH = (1920, 1080)

#: Letter index -> class of the static forest, for the archive files (24 bursts in this order).
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

#: The Tasks-API frames that become golden cases, as (letter, frame) into
#: static_sequences_tasks.npz. A 71, L 89, P 88/93/96 and U 53 are six of the 11 frames (of the
#: 2,378 the Tasks API found) whose argmax under the previous letter forest FLIPPED when the
#: swap was left out, all at p 0.33-0.46: the frames a port that drops the swap failed by
#: letter, not only by probability. The shipped forest happens to keep their argmax without the
#: swap (0 of 11), so they now catch the omission by probability alone, which is why the check
#: compares distributions. The first found frame of A, D, I, N and V pairs each solutions case
#: below with the same burst seen through the page's landmarker.
TASKS_PICKS = [("A", 71), ("L", 89), ("P", 88), ("P", 93), ("P", 96), ("U", 53)]
TASKS_FIRST = "ADINV"

#: The Ankara photos behind the digit cases, by file name in digits_ankara.npz: three each of
#: 0, 1, 2, 6 and 9 -- the digits that share a handshape with a letter (0/O, 2/V, 6/W, 9/F)
#: plus 1, the only digit whose pose passes z_gate (0.88 of its 159 frames; no other digit above
#: 0.01, no digit passes j_gate). Named rather than indexed so a re-ingest that changes the
#: detection set cannot silently swap the cases.
DIGIT_FILES = ["0/IMG_1118.JPG", "0/IMG_1128.JPG", "0/IMG_1138.JPG",
               "1/IMG_1119.JPG", "1/IMG_1129.JPG", "1/IMG_1139.JPG",
               "2/IMG_1120.JPG", "2/IMG_1130.JPG", "2/IMG_1140.JPG",
               "6/IMG_1124.JPG", "6/IMG_1134.JPG", "6/IMG_1144.JPG",
               "9/IMG_1127.JPG", "9/IMG_1137.JPG", "9/IMG_1147.JPG"]


def swap_tasks_handedness(raw):
    """The page's convention for a Tasks-API label, mirrored from docs/app.js
    swapTasksHandedness: Left <-> Right, anything else unchanged. The two must agree, and
    golden.json is how that is checked: every 'tasks' case's expected values go through this."""
    return {"Left": "Right", "Right": "Left"}.get(raw, raw)


def flatten(forest):
    """A forest as the four arrays a tree walk needs, plus leaf class distributions.

    Internal node i: feature[i], threshold[i], left[i], right[i]. Leaves carry -1 for feature
    and an index into `leaves`. sklearn's rule is `x[feature] <= threshold -> left`, which the
    port must match exactly; an inverted comparison is the kind of error that still classifies
    most inputs correctly and quietly ruins the rest.
    """
    trees = []
    for est in forest.estimators_:
        t = est.tree_
        feat = t.feature.astype(int).tolist()
        # Thresholds are NOT rounded. Rounding to 5 decimals moved a split point past a
        # feature value on roughly one probe vector in a thousand, so the exported forest
        # stopped reproducing sklearn on arbitrary input while still matching every golden
        # case -- the unanimous cases are exactly the ones a nudged threshold cannot flip.
        # repr() round-trips a float64 exactly and costs about 40% more bytes before gzip.
        thr = [float(v) for v in t.threshold]
        left = t.children_left.astype(int).tolist()
        right = t.children_right.astype(int).tolist()
        leaves = {}
        for i, f in enumerate(feat):
            if f < 0:
                # Leaves ARE rounded, to 4 decimals. With min_samples_leaf 5 most leaves hold a
                # mixed class count (65% of the letter forest's), so this is a real
                # quantization: at most 5e-5 per leaf, and the forest average cannot exceed the
                # worst leaf, so the page's probabilities sit within 5e-5 of sklearn's (5.1e-6
                # measured over 1,215 probes, 0 argmax changes) -- half the golden tolerance of
                # 1e-4 and three orders under any vote constant. Full-precision leaves would add
                # 0.9 MB raw for nothing the page can act on. Where a leaf falls, not what
                # it holds, is what the walk must get exactly right, and that is checked at
                # 0.00e+0 against sklearn's own leaf indices.
                v = t.value[i][0]
                s = v.sum()
                leaves[i] = [round(float(x / s), 4) for x in v] if s > 0 else [0.0] * len(v)
        trees.append({"f": feat, "t": thr, "l": left, "r": right,
                      "v": {str(k): v for k, v in leaves.items()}})
    return trees


def node_count(forest):
    return int(sum(est.tree_.node_count for est in forest.estimators_))


def forest_block(blob, static=True):
    """One forest as the page loads it: {classes, feature, dim, trees}.

    A static forest's tag must be one STATIC_FEATURES can build, and its width must be that
    function's, or the export stops here rather than shipping a forest the page cannot feed.
    Every consumer (the segmenter, the page, the golden cases below) reads the class of a
    probability by POSITION, so column i of predict_proba must be `classes[i]`: the pickle's
    classes_ is either 0..n-1 (the static forests are fit on the index into `classes`) or the
    labels themselves in the same order (the motion forest is fit on the strings). A forest
    trained on data missing a class would skip an index and mislabel everything after the gap.
    """
    model = blob["model"]
    dim = int(model.n_features_in_)
    tag = blob.get("feature")
    if static:
        _, want = F.static_feature_for(tag)
        if dim != want:
            raise SystemExit(f"{tag} builds {want}-D but the forest takes {dim}-D")
    classes = [str(c) for c in blob["classes"]]
    fitted = list(model.classes_)
    if fitted != list(range(len(classes))) and [str(c) for c in fitted] != classes:
        raise SystemExit(f"forest classes_ {fitted} do not line up with classes {classes}; the "
                         "positional class lookup every consumer uses would mislabel")
    return {"classes": classes, "feature": tag, "dim": dim, "trees": flatten(model)}


def make_case(raw_lm, label, width, height, blob, extra):
    """One golden case: raw landmarks -> every intermediate the page computes for them.

    `label` is the handedness label the Python canonicalizes with (already swapped for a Tasks
    frame); `extra` carries the fields that tell the page how to reach the same label.

    Round FIRST, then compute. Storing rounded landmarks beside outputs derived from the
    unrounded originals makes the file's own stated tolerance unreachable: re-running Python
    on golden.json's landmarks missed its stored values by 1.9e-5 against a claimed 1e-6,
    which reads as a port bug in any language.
    """
    raw_lm = np.round(np.asarray(raw_lm, dtype=np.float64), 6)
    P = F.to_isotropic(raw_lm[None, :, :2], width, height)
    P = F.canonicalize_handedness(P, label)
    feature_fn, _ = F.static_feature_for(blob.get("feature"))
    feat = feature_fn(P)[0]
    probs = blob["model"].predict_proba(feat.reshape(1, -1))[0]
    case = dict(extra)
    case.update({
        "landmarks": [[round(float(v), 6) for v in p] for p in raw_lm[:, :3]],
        "width": int(width), "height": int(height),
        "shape42": [round(float(v), 6) for v in F.shape42(P)[0]],
        "static_feature": [round(float(v), 6) for v in feat],
        "palm_scale": round(float(F.palm_scale(P)[0]), 6),
        "j_gate": bool(F.j_gate(P)[0]), "z_gate": bool(F.z_gate(P)[0]),
        "static_probs": [round(float(v), 6) for v in probs],
        "predicted": str(blob["classes"][int(probs.argmax())]),
    })
    return case


def letter_cases(static):
    """(a) solutions-API frames and (b) Tasks-API frames of the November archive."""
    W, H = ARCHIVE_WH
    cases = []
    seq = np.load(os.path.join(TEMPORAL, "static_sequences.npz"))
    for letter in TASKS_FIRST:                        # A, D, I, N, V
        c = LETTERS.index(letter)
        for i in np.where(seq["found"][c])[0][:3]:
            handed = str(seq["handed"][c, i])
            if handed not in ("Left", "Right"):
                raise SystemExit(f"static_sequences.npz {letter}[{i}]: no handedness label")
            cases.append(make_case(seq["lm"][c, i], handed, W, H, static, {
                "model": "static", "api": "solutions", "label": letter,
                "source": f"static_sequences.npz[{c},{i}]", "handedness": handed}))

    tasks = np.load(os.path.join(TEMPORAL, "static_sequences_tasks.npz"))
    picks = list(TASKS_PICKS)
    for letter in TASKS_FIRST:
        picks.append((letter, int(np.where(tasks["found"][LETTERS.index(letter)])[0][0])))
    for letter, i in picks:
        c = LETTERS.index(letter)
        if not tasks["found"][c, i]:
            raise SystemExit(f"static_sequences_tasks.npz {letter}[{i}]: no hand found")
        reported = str(tasks["handed"][c, i])
        if reported not in ("Left", "Right"):
            raise SystemExit(f"static_sequences_tasks.npz {letter}[{i}]: no handedness label")
        cases.append(make_case(tasks["lm"][c, i], swap_tasks_handedness(reported), W, H, static, {
            "model": "static", "api": "tasks", "label": letter,
            "source": f"static_sequences_tasks.npz[{c},{i}]",
            "reported_handedness": reported}))
    return cases


def digit_cases(digits):
    """(c) Ankara photos through the numbers forest, with the label the photo was filed under."""
    d = np.load(os.path.join(TEMPORAL, "digits_ankara.npz"), allow_pickle=True)
    W, H = [int(v) for v in d["frame_size"]]
    files = [str(f) for f in d["files"]]
    cases = []
    for name in DIGIT_FILES:
        if name not in files:
            raise SystemExit(f"digits_ankara.npz has no frame for {name}")
        i = files.index(name)
        handed = str(d["handed"][i])
        if handed not in ("Left", "Right"):
            raise SystemExit(f"digits_ankara.npz {name}: no handedness label")
        cases.append(make_case(d["lm"][i], handed, W, H, digits, {
            "model": "digits", "api": "solutions", "label": str(d["letters"][i]),
            "source": f"Sign-Language-Digits-Dataset/Dataset/{name}", "handedness": handed}))
    return cases


def load_blob(name):
    """A pickle from temporal/, or None when it is absent. The forest is switched to one job:
    the trainers pickle n_jobs 4, and a threaded predict_proba adds the per-tree distributions
    in whichever order the threads finish, so the last bit of a golden probability could differ
    between two runs. One vector at a time gains nothing from threads anyway."""
    path = os.path.join(TEMPORAL, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    blob["model"].n_jobs = 1
    return blob


def main():
    static = load_blob("model_static.p")
    motion = load_blob("model_motion.p")
    if static is None or motion is None:
        raise SystemExit("temporal/model_static.p and model_motion.p must exist (train_static.py, "
                         "train_motion.py)")
    digits = load_blob("model_digits.p")

    payload = {
        "static": forest_block(static),
        "motion": forest_block(motion, static=False),
        # Full precision, like the split thresholds: a constant the page compares a probability
        # against must be the one the Python measured with.
        "thresholds": asdict(DEFAULT),
    }
    if digits is not None:
        payload["digits"] = forest_block(digits)
        payload["digits"]["thresholds"] = asdict(digits_thresholds())

    n_static = node_count(static["model"])
    if n_static > NODE_BUDGET:
        raise SystemExit(f"letter forest has {n_static} nodes, over the {NODE_BUDGET} budget; "
                         "raise min_samples_leaf or cut trees before exporting")

    out = os.path.join(HERE, "models.json")
    raw = json.dumps(payload, separators=(",", ":"))
    with open(out, "w") as fh:
        fh.write(raw)
    # mtime 0 and no name in the gzip header: the .gz is then a pure function of the .json, so
    # two runs from the same pickles are byte-identical and a diff means the content moved.
    with open(out + ".gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(raw.encode("utf-8"))
    print(f"models.json      {len(raw)/1e6:6.2f} MB raw")
    print(f"models.json.gz   {os.path.getsize(out + '.gz')/1e6:6.2f} MB")
    print(f"static forest    {n_static:7d} nodes, {len(payload['static']['trees'])} trees, "
          f"{payload['static']['feature']} {payload['static']['dim']}-D, "
          f"{len(payload['static']['classes'])} classes  (budget {NODE_BUDGET})")
    print(f"motion forest    {node_count(motion['model']):7d} nodes, "
          f"{len(payload['motion']['trees'])} trees, {payload['motion']['feature']} "
          f"{payload['motion']['dim']}-D")
    if digits is not None:
        dth = payload["digits"]["thresholds"]
        print(f"digits forest    {node_count(digits['model']):7d} nodes, "
              f"{len(payload['digits']['trees'])} trees, {payload['digits']['feature']} "
              f"{payload['digits']['dim']}-D; VOTE_MARGIN_CLEAR {dth['VOTE_MARGIN_CLEAR']} / "
              f"VOTE_PROB_FLOOR {dth['VOTE_PROB_FLOOR']}")
    else:
        print("digits forest    none (temporal/model_digits.p absent; the page ships letters only)")

    # ---- golden vectors -------------------------------------------------------------
    cases = letter_cases(static)
    if digits is not None:
        cases += digit_cases(digits)
    golden = {"note": "browser port must reproduce every field to the stated tolerance; "
                      "'model' names the forest (static | digits), 'api' the landmarker "
                      "(solutions | tasks); a tasks case stores the raw Tasks label and its "
                      "expected values assume the page's Left<->Right swap",
              "tolerance": {"features": 1e-6, "probabilities": 1e-4},
              "cases": cases}
    gout = os.path.join(HERE, "golden.json")
    with open(gout, "w") as fh:
        json.dump(golden, fh, indent=1)
    kinds = {}
    for c in cases:
        k = f"{c['model']}/{c['api']}"
        kinds[k] = kinds.get(k, 0) + 1
    agree = sum(c["predicted"] == c["label"] for c in cases)
    print(f"golden.json      {os.path.getsize(gout)/1e3:6.1f} KB, {len(cases)} cases "
          f"({', '.join(f'{v} {k}' for k, v in kinds.items())}); predicted == label on "
          f"{agree}/{len(cases)}")


if __name__ == "__main__":
    main()

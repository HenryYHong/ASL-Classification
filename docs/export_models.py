"""Export the trained forests and a set of golden vectors for the browser port.

Three outputs:

  models.json    the forests as flat typed arrays, gzip-friendly. A sklearn pickle cannot run
                 in a browser -- it is a Python object graph -- so the trees are flattened to
                 the four arrays a tree walk actually needs. The letter forest ('static'), the
                 J/Z forest ('motion') and, when temporal/model_digits.p exists, the numbers
                 forest ('digits'), plus the thresholds the page runs: 'thresholds' is
                 thresholds.DEFAULT as a dict, 'digits.thresholds' is DEFAULT with
                 DIGITS_OVERRIDES applied. The thresholds travel in the same file as the trees
                 so the page can never pick up a new forest with stale constants.

  models.bin     the same three forests as a binary sidecar, with models.bin.gz beside it and
                 the class lists and threshold blocks in a small models.meta.json. forest.js
                 reads either format and decides from the bytes rather than the file name;
                 docs/app.js fetches the binary pair as of this release, through the same
                 loadModels('./models.bin.gz', './models.meta.json'). models.json stays anyway:
                 it is the readable form, it is what every parity check is run against, and it
                 is a format forest.js must keep reading -- test_forest.mjs loads it over HTTP
                 on every run so the unfetched container cannot rot.

                 Measured on this export, the one that carries the 245,064-node letter forest.
                 20,361,559 B of JSON become 2,555,774 B of binary, 8.0x. On the wire,
                 3,703,069 B become 1,184,420 B, 3.1x -- GitHub Pages compresses the JSON on
                 the fly at gzip level 5 (that figure is `gzip -5 models.json` here) and does
                 not compress octet-stream at all, so the binary ships as the committed
                 1,183,438 B models.bin.gz plus 982 B of meta. Decoding the three forests into
                 the typed arrays the page walks: 124-150 ms through models.json against
                 17-28 ms through models.bin, six interleaved runs of each on Node 20
                 (101-114 ms of that is JSON.parse alone; the binary has no parse step, only a
                 6.5-8.7 ms gunzip and a 2.7 KB meta file). Peak memory over the load, one
                 format per process so neither pays for the other's garbage: 94.3-100.4 MB of
                 heapUsed + arrayBuffers against 28.3-28.4 MB, 216.6 MB against 75.8 MB of
                 RSS, both ending at the same 19.0 MB of typed arrays.

                 The forest grew 23% in nodes between the last release and this one, and those
                 two figures are how the growth is paid for: the JSON route, which the page
                 used until this release, went 3,145,891 -> 3,703,069 B on the wire, and the
                 binary route the page now takes 1,000,455 -> 1,184,420 B. The larger forest
                 through models.bin.gz is still a third of the smaller one through models.json,
                 which is why the download fell 62.4% in a release that grew the forest 22.8%.

                 The leaf probabilities also get BETTER on the way, because exact integer class
                 counts are both smaller than the 4-decimal probabilities they replace and
                 lossless -- see pack_binary_forest. Against sklearn over the 1,215 probes of
                 parity_probes.py: 6.25e-6 through models.json, 4.05e-9 through models.bin.

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
                 misses these cases by probability, worst |dp| 0.630, and on two of the 11 by
                 letter as well (this export: A[71] S -> M, U[53] U -> R), which is the point
                 of storing the raw label.

Run:  ./.venv/bin/python docs/export_models.py     (from the repository root; any cwd works)

The three pickles must exist first (train_static.py, train_motion.py, train_digits.py), and
golden.json is regenerated together with models.json: its expected probabilities come from
these exact forests, so the two files ship together. Deploying a new forest with an old golden
file makes the page's self-check fail on load, which is what it is for.
"""
import gzip
import hashlib
import io
import json
import os
import pickle
import struct
import sys
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPORAL = os.path.join(HERE, "..", "temporal")
sys.path.insert(0, TEMPORAL)

import features as F                                  # noqa: E402
from thresholds import DEFAULT, digits_thresholds     # noqa: E402

#: Node budget for the letter forest: past it the page's first load on a phone connection is the
#: defect, not the accuracy. It stood at 200,000 while models.json was the only wire format, and
#: the previous comment asked whoever raised it to raise it against a wire figure and to measure
#: that leg again. This forest is 245,064 nodes (12,368 training frames, 61,840 rows after
#: jitter, 60 trees, min_samples_leaf 5), so the old number stops the export; here is that leg,
#: measured on this export, letter forest only:
#:
#:   models.json block   18,931,302 B raw, 77.25 B/node;  3,111,150 B gzipped, 12.70 B/node
#:   models.bin sections  2,264,550 B raw,  9.24 B/node;  1,041,301 B gzipped,  4.25 B/node
#:
#: and for the whole file, which is what a phone actually waits for: 3,145,891 -> 3,703,069 B at
#: `gzip -5` (models.json, the route the page left behind this release: +17.7%), 1,000,455 -> 1,184,420 B
#: through models.bin.gz plus its meta (+18.4%). The per-node cost barely moved; the node count
#: did, and the binary route is what keeps the bill under a megabyte and a fifth.
#:
#: 260,000 leaves about 6% of headroom over what ships -- room for one more retrain's drift,
#: not for another dataset. The next forest that needs more than this should buy it by moving
#: the page to models.bin.gz (loadModels('./models.bin.gz', './models.meta.json')), which is
#: worth 2.5 MB on the wire, rather than by raising this line again.
NODE_BUDGET = 260_000

#: Pixel size of the frames behind the letter golden cases. static_sequences.npz and
#: static_sequences_tasks.npz are both extracted from the RandomForest/data JPEGs (1920x1080);
#: neither file records the size, and the size is load-bearing because to_isotropic corrects
#: MediaPipe's normalized x by the aspect ratio.
ARCHIVE_WH = (1920, 1080)

#: Letter index -> class of the static forest, for the archive files (24 bursts in this order).
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

#: The Tasks-API frames that become golden cases, as (letter, frame) into
#: static_sequences_tasks.npz. A 71, L 89, P 88/93/96 and U 53 are six of the 11 frames (of the
#: 2,378 the Tasks API found) whose argmax under an older letter forest FLIPPED when the swap
#: was left out, all at p 0.33-0.46: the frames a port that drops the swap failed by letter,
#: not only by probability. Under the forest this export ships, two of the 11 flip again
#: without the swap (A 71 S -> M, U 53 U -> R) and the other nine miss by probability alone,
#: worst |dp| 0.630 -- which is why the check compares whole distributions rather than letters.
#: A 71 is also the one golden case whose argmax is not its own label: this forest reads that
#: frame as S 0.264 / M 0.258 / A 0.219. The SAME JPEG through the solutions landmarker -- the
#: extraction that is in training, since A's pooled block is 159 frames and the author cap is
#: 160 -- reads A 0.811, and the two landmarkers put that hand's points 0.048 apart in
#: normalized units, which on a closed fist is most of the distance between A, M and S. So the
#: case stays: a golden case is a port-parity fixture, the page must reproduce these
#: probabilities whatever they say about the frame, and dropping the cases a forest gets wrong
#: is how a fixture set stops being a test.
#: The first found frame of A, D, I, N and V pairs each solutions case below with the same
#: burst seen through the page's landmarker.
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
                # mixed class count (83,223 of the letter forest's 122,562, 67.9%), so this is a
                # real quantization: at most 5e-5 per leaf, and the forest average cannot exceed
                # the worst leaf, so the page's probabilities sit within 5e-5 of sklearn's
                # (6.3e-6 measured over 1,215 probes, 0 argmax changes) -- half the golden
                # tolerance of 1e-4 and three orders under any vote constant. Full-precision
                # leaves would add 1.73 MB raw for nothing the page can act on. Where a leaf
                # falls, not what it holds, is what the walk must get exactly right, and that
                # is checked at 0.00e+0 against sklearn's own leaf indices.
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


# ---- models.bin: the binary sidecar ---------------------------------------------------------

#: models.bin's magic and format version. forest.js refuses a body whose first 8 bytes are not
#: BIN_MAGIC, and refuses a VERSION it does not implement rather than reading the sections at
#: whatever layout it happens to know. A section read at the wrong dtype or offset is not a
#: crash: it is a forest that walks to the wrong leaves and answers confidently, which is the
#: shape of every expensive bug in this project. Bump VERSION whenever a section's dtype, order
#: or meaning changes -- an old page then says so instead of guessing.
BIN_MAGIC = b"ASLFRST\x00"
BIN_VERSION = 1

#: Fixed part of models.bin: magic(8) + VERSION(u32) + endian probe(u32) + header length(u32) +
#: reserved(u32) + build id(16) = 40 bytes. 40 is a multiple of 8, so the first section can
#: carry a Float64Array view over the same ArrayBuffer with no copy.
BIN_HEADER_BYTES = 40

#: Little-endian 0x04030201. A platform-endian Uint32Array view reads it back as 0x04030201 on
#: a little-endian machine and 0x01020304 on a big-endian one. Every view forest.js makes over
#: this buffer is platform-endian, so a big-endian reader would byte-swap all 139,145
#: thresholds in silence; the probe is how it finds out instead.
BIN_ENDIAN_PROBE = 0x04030201

#: Sections, in the order they are written, with the dtype each is stored at. Each one is the
#: concatenation over the forest's trees, so the page makes ONE typed-array view per section
#: instead of one per tree:
#:
#:   n     uint32[trees]     nodes per tree
#:   bits  uint8[]           leaf bitmap, np.packbits order (MSB first), byte-aligned per tree
#:   f     int8[internal]    split feature, internal nodes only, in node order
#:   t     float64[internal] split threshold, full precision. Narrowing it to float32 would
#:                           save 490,008 B on the letter forest and is the one saving here
#:                           that was measured and then refused. On the PREVIOUS forest it
#:                           moved a leaf on 1 of that forest's 97,200 tree visits and touched
#:                           1 of the 1,215 probes; on this one it moves none (0 of 72,900
#:                           letter visits, 0 of 121,500 digit visits, 0 probes). The refusal
#:                           stands on the earlier hit rather than on this forest's clean
#:                           sweep: a probe set that catches a defect once and misses it once
#:                           has measured the defect, not its absence, and it is the same
#:                           defect `flatten` records rejecting 5-decimal thresholds over. The
#:                           page claims to reproduce sklearn, and 490 KB of a 1.18 MB wire
#:                           load is not what that claim is worth trading for
#:   r     uint16[internal]  right child as an offset from the node; the left child is i + 1
#:   vk    uint8[leaves]     how many classes are nonzero in this leaf
#:   vi    uint8[nnz]        their class indices
#:   vc    uint16[nnz]       their EXACT integer sample counts; the page divides by their sum
BIN_SECTION_DTYPES = {"n": "<u4", "bits": "u1", "f": "i1", "t": "<f8", "r": "<u2",
                      "vk": "u1", "vi": "u1", "vc": "<u2"}


def pack_binary_forest(model, name):
    """One forest as the eight flat sections models.bin stores: {section name: numpy array}.

    Every narrowing here is checked on this forest rather than assumed, because each one fails
    silently when it is wrong: an out-of-range feature index, right-child offset or class count
    wraps into a legal-looking value and the walk lands somewhere else with no error at all.
    The export stops instead, and says which section to widen.

    Three things about the layout are worth knowing.

    Only internal nodes carry f/t/r and only leaves carry a class vector, so neither block pays
    for the other half of the tree. A leaf bitmap, one bit per node, is what tells the two
    apart on the way back in.

    The left child is not stored. sklearn's DepthFirstTreeBuilder emits a node's left subtree
    immediately after the node, so children_left[i] == i + 1 on every internal node -- 139,145
    of them across the three forests. That is an artifact of the builder and not a documented
    guarantee: max_leaf_nodes switches sklearn to BestFirstTreeBuilder, which grows the most
    promising leaf next and interleaves the subtrees, and the identity dies quietly. So it is
    asserted per tree. Refusing to export costs one run; a wrong left child sends about half of
    every walk into the wrong subtree and the forest still answers.

    Leaves store exact integer class counts rather than probabilities. sklearn keeps tree_.value
    normalized, and `value * weighted_n_node_samples` puts the counts back; the page divides by
    their sum and gets the same rational number sklearn's own predict_proba averages, so the
    counts are both smaller than the 4-decimal probabilities models.json ships AND lossless,
    which is not a trade one usually gets. Most of that block was zeros before it was made
    sparse: the letter forest holds 2.08 nonzero classes per leaf out of 24 (254,521 nonzero
    cells in 122,562 leaves), 91.3% off a dense table.
    """
    n_classes = int(model.n_classes_)
    if n_classes > 256:
        raise SystemExit(f"{name}: {n_classes} classes, but a leaf's class indices are stored "
                         "in uint8 (`vi`); widen the section and bump BIN_VERSION")
    cols = {k: [] for k in BIN_SECTION_DTYPES}
    for k, est in enumerate(model.estimators_):
        t = est.tree_
        n = int(t.node_count)
        feat = t.feature.astype(np.int64)
        thr = t.threshold.astype(np.float64)
        left = t.children_left.astype(np.int64)
        right = t.children_right.astype(np.int64)
        isleaf = feat < 0
        internal = np.flatnonzero(~isleaf)
        leaves = np.flatnonzero(isleaf)
        where = f"{name} tree {k}"

        bad = internal[left[internal] != internal + 1]
        if len(bad):
            raise SystemExit(
                f"{where}: children_left[{bad[0]}] is {left[bad[0]]}, not {bad[0] + 1}. "
                f"{len(bad)} of {len(internal)} internal nodes break the left-child-is-next "
                "identity models.bin relies on, so this forest was not grown depth-first "
                "(max_leaf_nodes is what does that). Store children_left as its own section "
                "and bump BIN_VERSION, or drop max_leaf_nodes")
        if not ((left[leaves] == -1).all() and (right[leaves] == -1).all()):
            raise SystemExit(f"{where}: a leaf has a child other than sklearn's TREE_LEAF (-1); "
                             "the decoder writes -1 into l and r at every leaf")
        if not ((feat[leaves] == -2).all() and (thr[leaves] == -2.0).all()):
            raise SystemExit(f"{where}: a leaf carries feature {int(feat[leaves].max())} / "
                             f"threshold {float(thr[leaves].max())} rather than sklearn's "
                             "TREE_UNDEFINED -2 / -2.0; the decoder writes those constants back "
                             "at every leaf, and forest.js tells a leaf from an internal node "
                             "by f < 0")
        if len(internal) and int(feat[internal].max()) > 127:
            raise SystemExit(f"{where}: splits on feature {int(feat[internal].max())}, but `f` "
                             "is int8; widen it and bump BIN_VERSION")
        off = right[internal] - internal
        if len(internal) and int(off.max()) > 0xFFFF:
            raise SystemExit(f"{where}: a right child sits {int(off.max())} nodes ahead, over "
                             "the uint16 `r` section; widen it and bump BIN_VERSION")

        val = t.value[:, 0, :].astype(np.float64)
        worst_row = float(np.abs(val.sum(axis=1) - 1.0).max())
        if worst_row > 1e-9:
            raise SystemExit(
                f"{where}: tree_.value rows miss 1.0 by {worst_row:.3e}, so this sklearn does "
                "not store a normalized class distribution and `value * "
                "weighted_n_node_samples` is not the class counts. Recover them the way this "
                "version stores them, and bump BIN_VERSION")
        counts = val * t.weighted_n_node_samples.astype(np.float64)[:, None]
        resid = float(np.abs(counts - np.rint(counts)).max())
        if resid > 1e-6:
            raise SystemExit(f"{where}: a recovered leaf count misses an integer by {resid:.3e}. "
                             "The counts are only exact for an unweighted fit; ship leaf "
                             "probabilities instead and bump BIN_VERSION")
        ci = np.rint(counts).astype(np.int64)[leaves]
        nzr, nzc = np.nonzero(ci)
        per_leaf = np.bincount(nzr, minlength=len(leaves))
        if not (per_leaf > 0).all():
            raise SystemExit(f"{where}: a leaf holds no samples at all, so the page would "
                             "divide its class counts by zero")
        if int(per_leaf.max()) > 255:
            raise SystemExit(f"{where}: a leaf holds {int(per_leaf.max())} nonzero classes, over "
                             "the uint8 `vk` section; widen it and bump BIN_VERSION")
        if int(ci.max()) > 0xFFFF:
            raise SystemExit(f"{where}: a leaf holds {int(ci.max())} samples of one class, over "
                             "the uint16 `vc` section; widen it and bump BIN_VERSION")

        cols["n"].append(np.array([n], np.uint32))
        cols["bits"].append(np.packbits(isleaf.astype(np.uint8)))
        cols["f"].append(feat[internal].astype(np.int8))
        cols["t"].append(thr[internal])
        cols["r"].append(off.astype(np.uint16))
        cols["vk"].append(per_leaf.astype(np.uint8))
        cols["vi"].append(nzc.astype(np.uint8))
        cols["vc"].append(ci[nzr, nzc].astype(np.uint16))

    return {k: (np.concatenate(v).astype(BIN_SECTION_DTYPES[k]) if v
                else np.zeros(0, BIN_SECTION_DTYPES[k]))
            for k, v in cols.items()}


def write_models_bin(blobs, payload):
    """Write models.bin, models.bin.gz and models.meta.json. Returns (raw, gz, meta) byte counts.

    `payload` is the JSON export, already built and NOT modified here: the class lists, feature
    tags, dims and threshold blocks are copied out of it, so the two formats cannot end up
    describing different forests. What the meta file does not carry is the trees; those are the
    binary.

    models.meta.json ships beside models.bin for the reason the thresholds ride inside
    models.json today -- a constant the page compares a probability against must be the one the
    Python measured with, so the class lists and the threshold block travel with the forest
    rather than living in the page. Two files have one failure mode that one file does not,
    which is deploying a new models.bin over a stale models.meta.json, so the pair is bound by
    a build id: the first 16 bytes of the SHA-256 of the section payload, written into both.
    forest.js refuses a pair whose ids differ.

    models.bin.gz is committed, unlike models.json.gz. GitHub Pages compresses text media types
    on the fly but leaves application/octet-stream alone, so an uncompressed models.bin would
    cross the wire at its full size. Serving the .gz as an opaque body and letting
    DecompressionStream inflate it is the same route forest.js already takes when it is handed
    a .gz, so it costs the page nothing new.
    """
    buf = io.BytesIO()
    buf.write(b"\0" * BIN_HEADER_BYTES)
    tables, nodes = {}, {}
    for key in ("static", "motion", "digits"):
        if key not in payload:
            continue
        secs = pack_binary_forest(blobs[key]["model"], key)
        table = {}
        for nm in BIN_SECTION_DTYPES:
            # 8-byte align every section start. A Float64Array view over the page's ArrayBuffer
            # is only legal at a multiple of 8, and one rule for all eight sections is cheaper
            # to keep right than a per-dtype one; the padding costs at most 7 bytes a section.
            while buf.tell() % 8:
                buf.write(b"\0")
            table[nm] = [buf.tell(), int(secs[nm].size)]
            buf.write(secs[nm].tobytes())
        tables[key] = table
        nodes[key] = int(secs["n"].sum())

    raw = bytearray(buf.getvalue())
    build = hashlib.sha256(bytes(raw[BIN_HEADER_BYTES:])).digest()[:16]
    raw[0:8] = BIN_MAGIC
    raw[8:12] = struct.pack("<I", BIN_VERSION)
    raw[12:16] = struct.pack("<I", BIN_ENDIAN_PROBE)
    raw[16:20] = struct.pack("<I", BIN_HEADER_BYTES)
    raw[20:24] = struct.pack("<I", 0)
    raw[24:40] = build
    raw = bytes(raw)

    meta = {"format": "asl-forest/bin", "version": BIN_VERSION, "build": build.hex(),
            "bytes": len(raw), "thresholds": payload["thresholds"]}
    for key in ("static", "motion", "digits"):
        if key not in payload:
            continue
        blk = {"classes": payload[key]["classes"], "feature": payload[key]["feature"],
               "dim": payload[key]["dim"], "trees": len(payload[key]["trees"]),
               "nodes": nodes[key], "sections": tables[key]}
        if "thresholds" in payload[key]:
            blk["thresholds"] = payload[key]["thresholds"]
        meta[key] = blk

    bout = os.path.join(HERE, "models.bin")
    with open(bout, "wb") as fh:
        fh.write(raw)
    # Same gzip convention as models.json.gz: level 9, mtime 0, no stored name, so the .gz is a
    # pure function of the .bin and two runs from the same pickles are byte-identical.
    with open(bout + ".gz", "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    mout = os.path.join(HERE, "models.meta.json")
    mraw = json.dumps(meta, separators=(",", ":"))
    with open(mout, "w") as fh:
        fh.write(mraw)
    return len(raw), os.path.getsize(bout + ".gz"), len(mraw), build.hex()


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
    print(f"models.json      {len(raw)/1e6:6.2f} MB raw ({len(raw)} B)")
    print(f"models.json.gz   {os.path.getsize(out + '.gz')/1e6:6.2f} MB "
          f"({os.path.getsize(out + '.gz')} B)")

    blobs = {"static": static, "motion": motion}
    if digits is not None:
        blobs["digits"] = digits
    n_bin, n_gz, n_meta, build = write_models_bin(blobs, payload)
    print(f"models.bin       {n_bin/1e6:6.2f} MB raw ({n_bin} B), build {build[:16]}")
    print(f"models.bin.gz    {n_gz/1e6:6.2f} MB ({n_gz} B)  <- the file to serve; GitHub "
          f"Pages does not compress application/octet-stream, so this .gz is committed")
    print(f"models.meta.json {n_meta/1e3:6.1f} KB ({n_meta} B): classes, feature tags, "
          f"thresholds, section offsets")
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

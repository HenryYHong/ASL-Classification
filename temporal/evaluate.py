"""The three reported numbers, and nothing flattering.

This repository's README documents, at length, that its old 100.00% and 99.58% figures were
measured on near-duplicate frames from a single capture burst: what they scored was the
forest's ability to re-identify frames it had already seen, not to recognize a handshape.
This module is the correction, so every decision in it is made in the conservative direction
and the count of *independent groups* (prompted items, clips) behind a figure is printed
beside the figure itself -- the habit that would have caught the old number, which was
computed over 23 of 24 letters.

Three numbers, all on held-out sessions only (every session that is not the training session,
by default; --test-session narrows it):

1. Event-level 3-class {J, Z, MOVE} metrics -- confusion matrix, per-class precision and
   recall. Reported next to them, because they are the part a confusion matrix cannot show:
   how many prompted reps the state machine never cut an event out of at all, and how many
   cut events the cheap vetoes killed before the classifier ever saw them. A recall figure
   computed only over events that reached the model would silently exclude every gesture the
   segmenter missed, which is the more likely failure. Truth for a prompted take comes from
   its prompt schedule through label_events.assign_continuous -- the same rule that labels
   the training events -- so one credited event per item, and everything else MOVE.

2. False J/Z emissions per minute of negative footage, produced by replaying `Segmenter` --
   the same object the live demo drives -- over the recorded frames. The number then
   describes the deployed system rather than a classifier in isolation. `Segmenter` is driven
   entirely by the timestamps passed to `step()`, so a replay reproduces a live run exactly.

3. Letter error rate on the continuous fingerspelling takes, by Levenshtein alignment of the
   emitted string against the written-down intended string. Substitutions, insertions and
   deletions are reported separately because they mean different things: an insertion is
   usually a spurious motion fire or a double-emit, a deletion a gate or a hold that never
   fired, a substitution a genuine classifier error. One aggregate rate hides which.

The split is by SESSION and nothing else. If only one session has been recorded, this refuses
to print a generalization number rather than falling back to a random split over clips -- a
random split here would measure re-identification of one sitting, which is the exact error
this repository already published a correction for. Whether a test session is genuinely held
out from the MOTION forest is read from the pickle (train_motion.py records the sessions it
fitted on); a forest fitted on every session makes every row in-sample, and the report says
so on each one rather than leaving it to the reader. The static forest is fitted on all four
static sessions, so static letters are in-sample everywhere here.

Data contract. `motion_clips.npz` as written by collect_motion.py:

    clips     object (N,)   each (T,21,3) MediaPipe-normalized landmarks, NaN rows where the
                            hand was not detected
    stamps    object (N,)   each (T,) seconds from the start of the clip
    labels    (N,) str      "J" | "Z" | "NONE" (negative) | "SPELL" (continuous fingerspelling)
                            | "CONTINUOUS" (a prompted take: PARK / GO / REST items)
    prompts   object (N,)   for a CONTINUOUS take, the JSON schedule the recorder followed: a
                            list of {label, index, park, go, rest} items with the window
                            bounds in clip seconds. J/Z items are the positives; NONE
                            (near-miss) items, the REST windows and the time outside every
                            item are the negatives.

    handed    object (N,)   per-frame MediaPipe handedness label, or one label per clip
    sessions  (N,) str      "S1", "S2", ...  -- without it there is no honest split
    signers   (N,) str      who signed the clip. Read only so the closing caveat states the
                            signer count it measured instead of asserting "one signer"
    frame_size (2,) or (N,2)  the capture frame size; u = x*(W/H) is load-bearing (an
                            uncorrected Z-gate admits L, X and G at ~100%, a corrected one
                            at 23%, 61% and 16%), so it is never defaulted silently

Two fields the recorder does not write, and what happens without them:

    clip_ids  (N,) str      optional; synthesized from session/label/index when absent
    intended  (N,) str      the intended letter string of a continuous take. Nothing in the
                            capture path can know it, so it comes from --intended, a JSON
                            object mapping clip_id to the string that was actually signed.
                            Without it number 3 is reported as not computable; it is never
                            guessed from the emissions, which would score the system against
                            its own output, and it is never read out of `prompts` either --
                            that key is a schedule, not a spelling, and reading it as one
                            produced a 630-letter "LABELJINDEXPARK..." reference.

Run:  ./.venv/bin/python temporal/evaluate.py                 # test = every session but S1
      ./.venv/bin/python temporal/evaluate.py --test-session S5
      ./.venv/bin/python temporal/evaluate.py --train-session S5 --test-session S1

The shipped model_motion.p is fitted on S1 and S5, so with it every row above is IN-SAMPLE
for the motion forest and the block headers say so. For a genuinely held-out motion row fit a
forest on one session first and point --motion-model at it:

      ./.venv/bin/python temporal/train_motion.py --fit-sessions S1 --out /tmp/motion_S1.p
      ./.venv/bin/python temporal/evaluate.py --motion-model /tmp/motion_S1.p --test-session S5
"""
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from segmenter import Segmenter
from thresholds import Thresholds, DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "motion_clips.npz")
STATIC_MODEL = os.path.join(HERE, "model_static.p")
MOTION_MODEL = os.path.join(HERE, "model_motion.p")

MOTION_CLASSES = ["J", "Z", "MOVE"]
NEGATIVE_LABELS = {"NONE", "NEG", "NEGATIVE", "MOVE"}
SPELL_LABELS = {"SPELL", "SPELLING", "TAKE", "FS"}
#: The recorder's label for a prompted take (collect_motion.py --continuous), the only mode
#: that has been used: all five committed clips carry it. Its positives and negatives come
#: from the prompt schedule, not from the label.
CONTINUOUS_LABELS = {"CONTINUOUS"}

#: Position of path length inside features.event_features' 79-D layout. This module reads one
#: number out of that vector and nothing else; the width check in _Probe is what turns a layout
#: change into a failure instead of silently ranking reps by whatever moved into slot 53.
EVENT_PATH_LEN = 53


# ---------------------------------------------------------------- recordings

@dataclass
class Clip:
    clip_id: str
    label: str
    session: str
    times: np.ndarray          # (T,) seconds
    lm: np.ndarray             # (T,21,3) raw MediaPipe, NaN where undetected
    width: int
    height: int
    handedness: object         # one label, or one per frame as the recorder writes it
    intended: str = ""
    signer: str = "?"
    items: list = field(default_factory=list)   # a CONTINUOUS take's prompt schedule

    def hand_at(self, k):
        """MediaPipe's label for frame k. Chirality is not cosmetic -- features.py mirrors
        left hands onto the right-hand convention, so a clip replayed under the wrong label
        is a clip of the other hand."""
        if isinstance(self.handedness, str):
            return self.handedness
        return str(self.handedness[k]) if k < len(self.handedness) else str(self.handedness[-1])

    @property
    def duration(self):
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def is_take(self):
        return self.label in CONTINUOUS_LABELS

    def item_at(self, t):
        """The prompted item whose PARK..REST span contains t, or None (REST is part of the
        item so a settle after the gesture is attributed to it; see phase_at)."""
        return next((I for I in self.items if I["park"][0] <= t < I["rest"][1]), None)

    def phase_at(self, t):
        """'item' inside a J/Z item's PARK/GO window, 'rest' inside its REST window, 'none'
        everywhere else. The last two are the take's negative footage, and a near-miss
        (NONE) item is negative footage end to end: its GO is "MOVE it, but do NOT sign the
        letter" (collect_motion.py), so a J fired there is a false fire, not a substitution
        of a letter that was never prompted. label_events.assign_continuous labels the same
        span MOVE for training."""
        it = self.item_at(t)
        if it is None or it["label"] not in ("J", "Z"):
            return "none"
        return "rest" if t >= it["rest"][0] else "item"

    def negative_windows(self):
        """[(t0, t1)] of every stretch outside the prompted J/Z items: their REST windows,
        the time between items, and every near-miss (NONE) item whole (park..rest)."""
        if not self.items:
            return [(float(self.times[0]), float(self.times[-1]))]
        out = []
        cursor = float(self.times[0])
        for I in sorted(self.items, key=lambda I: I["park"][0]):
            if I["label"] not in ("J", "Z"):
                continue            # the cursor sweeps over it: the whole item is negative
            if I["park"][0] > cursor:
                out.append((cursor, float(I["park"][0])))
            out.append((float(I["rest"][0]), float(I["rest"][1])))
            cursor = float(I["rest"][1])
        if float(self.times[-1]) > cursor:
            out.append((cursor, float(self.times[-1])))
        return out

    @property
    def negative_seconds(self):
        return sum(b - a for a, b in self.negative_windows())


def _pick(npz, names):
    for k in names:
        if k in npz.files:
            return npz[k]
    return None


def _hand_field(entry, fallback):
    """Keep per-frame handedness per-frame, and a single label single.

    The recorder stores MediaPipe's label for every frame, including "None" where no hand was
    found. Those frames replay as detection gaps, so the label is carried through untouched
    rather than filled in: guessing chirality is the same class of silent default as guessing
    the frame aspect, and it mirrors the whole hand when it is wrong. Debouncing the label is
    the Segmenter's job (HANDEDNESS LATCH in segmenter.py), so this replay sees exactly what
    live_demo.py and the page see; feeding the modal label per take here instead would hide
    the flips the live path has to survive.
    """
    a = np.asarray(entry)
    if a.ndim == 0:
        return str(a)
    return a.astype(str) if a.size else fallback


def _norm_session(s):
    """'s1', 'S1', '1' and 1 all name the same sitting."""
    return "S" + str(s).strip().upper().lstrip("S")


def load_recordings(path, width=None, height=None, handedness="Right"):
    if not os.path.exists(path):
        raise SystemExit(
            f"no recordings at {path}.\n"
            "Nothing has been captured yet, so there is nothing to evaluate. Record with\n"
            "  ../.venv/bin/python temporal/collect_motion.py --camera 0\n"
            "across two sessions on different days, then run this again.")

    d = np.load(path, allow_pickle=True)
    clips = _pick(d, ("clips",))
    stamps = _pick(d, ("stamps", "times"))
    labels = _pick(d, ("labels", "label"))
    if clips is None or stamps is None or labels is None:
        raise SystemExit(f"{path} is missing one of clips/stamps/labels; it was not written "
                         "by collect_motion.py")
    if len(clips) == 0:
        raise SystemExit(f"{path} holds zero clips.")

    sessions = _pick(d, ("sessions", "session"))
    if sessions is None:
        raise SystemExit(
            f"{path} carries no per-clip session label.\n"
            "Refusing to report a generalization number: the only honest split here is by\n"
            "recording session (different day, room, lighting, camera distance), and a random\n"
            "split over clips from one sitting measures re-identification of that sitting, not\n"
            "recognition. That is the error this repository's README already documents.\n"
            "Re-record so each clip carries its session, or add the labels by hand.")

    ids = _pick(d, ("clip_ids", "clip_id", "ids"))
    intended = _pick(d, ("intended", "intent"))
    prompts = _pick(d, ("prompts",))
    hands = _pick(d, ("handed", "handedness", "hands"))
    signers = _pick(d, ("signers", "signer"))
    wh = _pick(d, ("frame_wh", "frame_size", "wh", "size"))

    if wh is None and (width is None or height is None):
        raise SystemExit(
            f"{path} does not record the capture frame size, and --width/--height were not "
            "given.\nThe isotropy correction u = x*(W/H) cannot be applied without it, and "
            "every gate\nthreshold in features.py is calibrated on corrected coordinates -- "
            "guessing 16:9 would\nquietly change what the gates admit. Pass the size the "
            "footage was captured at.")
    wh = np.asarray(wh) if wh is not None else None

    out = []
    for i in range(len(clips)):
        lm = np.asarray(clips[i], dtype=np.float64)
        ts = np.asarray(stamps[i], dtype=np.float64)
        if lm.ndim != 3 or lm.shape[1] != 21:
            raise SystemExit(f"clip {i} has shape {lm.shape}, expected (T,21,3)")
        sess = _norm_session(sessions[i])
        lab = str(labels[i]).strip().upper()
        if wh is None:
            w, h = int(width), int(height)
        elif wh.ndim == 1:
            w, h = int(wh[0]), int(wh[1])
        else:
            w, h = int(wh[i][0]), int(wh[i][1])
        items = []
        if lab in CONTINUOUS_LABELS:
            raw = str(prompts[i]).strip() if prompts is not None else ""
            if not raw:
                raise SystemExit(f"clip {i} is a CONTINUOUS take with no prompt schedule; "
                                 "nothing says where its J/Z items and rest windows are")
            items = json.loads(raw)
        out.append(Clip(
            clip_id=str(ids[i]) if ids is not None else f"{sess}:{lab}:{i}",
            label=lab, session=sess, times=ts, lm=lm, width=w, height=h,
            handedness=_hand_field(hands[i], handedness) if hands is not None else handedness,
            intended="".join(ch for ch in str(intended[i]).upper() if ch.isalpha())
                     if intended is not None else "",
            signer=str(signers[i]) if signers is not None else "?",
            items=items))
    return out


def attach_intended(clips, path):
    """Read the written-down intended strings for the continuous takes.

    A key that matches no clip is an error rather than a warning. A mistyped id would silently
    leave its take out of the LER, and the takes most likely to be mistyped are the ones a
    person went back to re-record -- which is to say the hard ones.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no intended-string file at {path}")
    with open(path) as fh:
        table = json.load(fh)
    by_id = {c.clip_id: c for c in clips}
    unknown = sorted(set(table) - set(by_id))
    if unknown:
        raise SystemExit(f"{path} names clips that are not in the recording: {unknown[:5]}\n"
                         f"known ids look like: {sorted(by_id)[:3]}")
    for cid, text in table.items():
        by_id[cid].intended = "".join(ch for ch in str(text).upper() if ch.isalpha())


def load_model(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            f"no {what} model at {path}.\n"
            "Train it first; there is no number to report without it.")
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if not isinstance(blob, dict) or "model" not in blob:
        raise SystemExit(f"{path} is not a {{'model':..., 'classes':...}} pickle")
    return blob


# ---------------------------------------------------------------- replay

class _Tap(Segmenter):
    """A Segmenter that also records every event span the state machine cut.

    Subclassed rather than patched: segmenter.py is the object under test and must not be
    altered to be measured. The tap records spans that the cheap vetoes discard as well as
    those that reach the classifier, which is the difference between "the model got it wrong"
    and "the model was never asked" -- two failures a confusion matrix reports identically.
    """

    def __init__(self, *a, **kw):
        # Every span the machine cuts, aborted or scored, before any veto: the on_event hook
        # label_events.py uses to build training data, so the spans scored here are the ones
        # the trainer would have labeled.
        self.spans = []
        kw.setdefault("on_event", self.spans.append)
        super().__init__(*a, **kw)
        self.cut = 0

    def _score(self, t, *a, **kw):
        if self._track_from is not None and len(self._since(self._track_from)) >= 3:
            self.cut += 1
        return super()._score(t, *a, **kw)


class _Probe:
    """Wraps the event classifier to capture every feature vector it is actually asked about.

    Only events that survive duration, path-length and straightness vetoes ever reach here,
    so `seen` is exactly the set of events the 3-class metrics are defined over.
    """

    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def predict_proba(self, X):
        p = self.inner.predict_proba(X)
        x = np.asarray(X)[0]
        assert x.shape == (F.EVENT_DIM,), (
            f"event feature is {x.shape}, not {F.EVENT_DIM}-D: features.py's layout moved "
            f"under index {EVENT_PATH_LEN}, which this module reads as path length")
        self.seen.append((x, np.asarray(p)[0]))
        return p


def replay(clip, th, static=None, motion=None, static_classes=None, motion_classes=None,
           static_feature_tag=None):
    """Drive one recorded clip through the state machine. Returns (emissions, tap, probe).

    A fresh Segmenter per clip: state must never carry across recordings, and the buffer,
    cooldown and last_emitted of one take have no business influencing the next.

    `static_feature_tag` is the 'feature' tag the static pickle carries (live_demo.py and
    tests/test_segmenter_replay.py pass the same thing): the Segmenter builds the vector that
    tag names and refuses a forest of another width. Left at None it resolves to static/v3,
    which is right only for pickles written before the tag existed -- the shipped 112-D
    static/v4 forest raised in the width check here until main() threaded the tag through.

    The NaN rows the recorder writes for undetected frames are passed as `None`, which is what
    a live frame with no hand delivers, so the gap-interpolation and gap-reset paths are
    exercised here exactly as they would be on the camera.
    """
    probe = _Probe(motion) if motion is not None else None
    seg = _Tap(th, static_model=static, motion_model=probe,
               static_classes=static_classes, motion_classes=motion_classes,
               static_feature_tag=static_feature_tag)
    emissions = []
    for k, (t, row) in enumerate(zip(clip.times, clip.lm)):
        lms = row if np.isfinite(row).all() else None
        em = seg.step(float(t), lms, clip.hand_at(k), clip.width, clip.height)
        if em is not None:
            emissions.append(em)
    return emissions, seg, probe


# ---------------------------------------------------------------- Levenshtein

def levenshtein(ref, hyp):
    """Edit distance with a backtrace. Returns the op list, most-recent-last.

    Written out rather than imported: the alignment itself is the deliverable, not the
    distance. Ties are broken toward the diagonal, then deletion, then insertion, so the path
    is deterministic; the distance is unique even where the path is not, and the S/I/D
    breakdown of a tied alignment can differ by one op between equally valid paths.
    """
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
                          D[i - 1, j] + 1,
                          D[i, j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------- reporting helpers

def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def print_confusion(M, classes, groups_per_class):
    w = max(6, max(len(c) for c in classes) + 1)
    head = "true|pred"
    print(f"    {head:>{w}}" + "".join(f"{c:>8}" for c in classes)
          + f"{'total':>8}{'groups':>8}")
    for i, c in enumerate(classes):
        print(f"    {c:>{w}}" + "".join(f"{M[i, j]:>8d}" for j in range(len(classes)))
              + f"{M[i].sum():>8d}{groups_per_class.get(c, 0):>8d}")


def print_pr(M, classes, groups_per_class):
    # GROUPS beside every count: an event count is not an evidence count, because one
    # prompted item can yield several events (its false starts and settle are MOVE rows).
    print(f"    {'class':>8}{'n_events':>10}{'groups':>8}{'precision':>11}{'recall':>8}")
    for i, c in enumerate(classes):
        tp = M[i, i]
        support = M[i].sum()
        predicted = M[:, i].sum()
        prec = tp / predicted if predicted else float("nan")
        rec = tp / support if support else float("nan")
        print(f"    {c:>8}{support:>10d}{groups_per_class.get(c, 0):>8d}"
              f"{prec:>11.3f}{rec:>8.3f}")


# ---------------------------------------------------------------- number 1

def _take_events(clip, th, motion, motion_classes):
    """Replay one prompted take with the motion model only and label every event the machine
    cut, by the training rule: one credited gesture per item (the longest runtime-reachable
    span whose onset lies in the item's PARK/GO window under the matching gate), every other
    reachable span MOVE. Returns (rows, probe, tap) with rows aligned to probe.seen."""
    import label_events as LE
    _, tap, probe = replay(clip, th, motion=motion, motion_classes=motion_classes)
    rows = LE.assign_continuous(tap.spans, clip.items, th, other="move")
    reach = [r for r in rows if r["reachable"]]
    # A reachable span is exactly one the vetoes let through to the classifier, in the order
    # the tracks ended, so the probe's records line up with it one for one.
    assert len(reach) == len(probe.seen), (len(reach), len(probe.seen))
    return rows, reach, probe, tap


def event_metrics(test_clips, motion, motion_classes, th):
    """3-class {J,Z,MOVE} metrics over events the state machine cut from held-out clips.

    Ground truth is assigned conservatively. A prompted 2 s clip contains one rep, so if the
    segmenter cuts several scorable events out of it, only the one with the longest tip path
    (event feature [53]) is credited as the rep; the rest are counted as MOVE, because they
    are motion the signer did not intend as a letter. The alternative -- crediting every
    event from a J clip as a J -- would let a spurious cut that happens to score J count as a
    success. A prompted take is labeled by the same rule per item, through
    label_events.assign_continuous, and its REST windows and out-of-item spans are MOVE.

    MOVE truth is mined from negative footage only. The continuous fingerspelling takes also
    produce events, but their intended strings contain real J and Z letters, so an event mined
    from them is not reliably a negative.
    """
    prompted = [c for c in test_clips if c.label in ("J", "Z")]
    negatives = [c for c in test_clips if c.label in NEGATIVE_LABELS]
    takes = [c for c in test_clips if c.is_take]
    if not prompted and not negatives and not takes:
        print("  not computable: the held-out sessions hold no prompted J/Z clips, no prompted "
              "takes and no negative footage.")
        return

    y_true, y_pred, p_win, margins = [], [], [], []
    groups_seen = {c: set() for c in MOTION_CLASSES}
    missed, multi, cut_total, scored_total = [], 0, 0, 0
    n_items = {"J": 0, "Z": 0}

    def record(truth, proba, group):
        order = np.argsort(proba)[::-1]
        y_true.append(truth)
        y_pred.append(motion_classes[int(order[0])])
        p_win.append(float(proba[order[0]]))
        margins.append(float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0)
        groups_seen[truth].add(group)

    for clip in prompted + negatives:
        _, seg, probe = replay(clip, th, motion=motion, motion_classes=motion_classes)
        cut_total += seg.cut
        events = probe.seen
        scored_total += len(events)

        if clip.label in ("J", "Z"):
            n_items[clip.label] += 1
            if not events:
                missed.append(clip.clip_id)
                continue
            if len(events) > 1:
                multi += 1
            primary = int(np.argmax([f[EVENT_PATH_LEN] for f, _ in events]))
            truth = [clip.label if k == primary else "MOVE" for k in range(len(events))]
        else:
            truth = ["MOVE"] * len(events)
        for (f, proba), tr in zip(events, truth):
            record(tr, proba, clip.clip_id)

    for clip in takes:
        rows, reach, probe, tap = _take_events(clip, th, motion, motion_classes)
        cut_total += len(tap.spans)
        scored_total += len(reach)
        credited = {r["gid"] for r in rows if r["credited"]}
        per_item = {}
        for r in reach:
            if r["phase"] == "item":
                per_item[r["gid"]] = per_item.get(r["gid"], 0) + 1
        for I in clip.items:
            if I["label"] in ("J", "Z"):
                n_items[I["label"]] += 1
                if I["index"] not in credited:
                    missed.append(f"{clip.clip_id}#{I['index']}")
                elif per_item.get(I["index"], 0) > 1:
                    multi += 1
        for r, (f, proba) in zip(reach, probe.seen):
            # The same group id train_motion.py splits on: the prompted item the span's onset
            # fell in (label_events.ORPHAN_GID outside every item).
            record(r["label"], proba, f"{clip.clip_id}#{r['gid']}")

    n_prompted = n_items["J"] + n_items["Z"]
    neg_min = (sum(c.duration for c in negatives) + sum(c.negative_seconds for c in takes)) / 60
    print(f"  source: {n_items['J']} prompted J, {n_items['Z']} prompted Z "
          f"({len(prompted)} clips, {len(takes)} takes), {len(negatives)} negative takes; "
          f"{neg_min:.2f} min of negative footage (NONE clips, near-miss items, REST windows, "
          "out-of-item time)")
    print(f"  events cut by the state machine: {cut_total}; reached the classifier: "
          f"{scored_total} ({cut_total - scored_total} aborted or killed by the duration, "
          f"path-length and straightness vetoes before any model ran)")

    if missed:
        print(f"  prompted reps the segmenter produced NO scorable event for: "
              f"{len(missed)} of {n_prompted} "
              f"({len(missed) / max(n_prompted, 1):.1%}) -- these are recall failures that "
              f"the matrix below cannot show, because they never became events")
    else:
        print(f"  prompted reps with no scorable event: 0 of {n_prompted}")
    if multi:
        print(f"  prompted reps yielding more than one scorable event: {multi} "
              f"(only the longest-path event is credited as the rep; the others count as MOVE)")

    if not y_true:
        print("  no events to score.")
        return

    groups = {c: len(g) for c, g in groups_seen.items()}
    print()
    print_confusion(confusion(y_true, y_pred, MOTION_CLASSES), MOTION_CLASSES, groups)
    print()
    print_pr(confusion(y_true, y_pred, MOTION_CLASSES), MOTION_CLASSES, groups)
    print("    (groups: prompted items for J and Z; for MOVE, the clips and the items whose "
          "rest or false starts produced them)")

    # The operating point is part of the design, not a post-hoc knob: an abstention is the
    # correct answer for most motion, and a recall figure obtained by lowering P_EMIT is a
    # different system from the one that ships. A fire is an emission, so it needs the
    # argmax to be J or Z as well as clearing P_EMIT and MARGIN.
    yt, yp = np.array(y_true), np.array(y_pred)
    fires = (np.isin(yp, ["J", "Z"]) & (np.array(p_win) >= th.P_EMIT)
             & (np.array(margins) >= th.MARGIN))
    jz = np.isin(yt, ["J", "Z"])
    print(f"\n    at P_EMIT={th.P_EMIT} / MARGIN={th.MARGIN}: emits on {int(fires.sum())} of "
          f"{len(yt)} events; J+Z emitted as the right letter "
          f"{int((fires & (yp == yt) & jz).sum())} of {int(jz.sum())} credited gestures "
          f"({(fires & (yp == yt) & jz).sum() / max(jz.sum(), 1):.3f}), "
          f"and {n_prompted} prompted overall ({(fires & (yp == yt) & jz).sum() / max(n_prompted, 1):.3f} "
          f"counting the reps that never became events); MOVE events emitted as J: "
          f"{int(((yp == 'J') & fires & (yt == 'MOVE')).sum())}, as Z: "
          f"{int(((yp == 'Z') & fires & (yt == 'MOVE')).sum())} of {int((yt == 'MOVE').sum())}")
    print("    (the matrix above is argmax, ignoring the thresholds; the deployed system "
          "emits nothing for an abstained event)")


# ---------------------------------------------------------------- number 2

def false_fire_rate(test_clips, th, static, static_classes, motion, motion_classes,
                    static_feature_tag=None):
    """False J/Z emissions per minute of held-out negative footage.

    Measured by replaying the whole Segmenter, both models attached, over the recorded frames
    -- not by scoring pre-cut events. What a user experiences is the emission stream, and the
    gates, the rising edge, the vetoes and the cooldown are all part of what produces it.

    Negative footage is every NONE clip, plus every REST window, every near-miss (NONE) item
    and every stretch outside the prompted J/Z items of a CONTINUOUS take (Clip.phase_at /
    negative_windows draw the line, and they draw it where label_events.assign_continuous
    does for the training events). A motion emission belongs to the window its ONSET
    (emission time minus the track's duration) lies in, because a gesture performed in the GO
    window is delivered after its fall-confirm, often inside the REST that follows -- counting
    by delivery time would call the real J's false. A motion emission whose onset lies inside
    an item but names the other letter is reported separately as a wrong-letter fire, and the
    per-item end-to-end tally (hit / miss / double / exact string) is printed for the takes
    because that, not a matrix, is what the signer sees.
    """
    negatives = [c for c in test_clips if c.label in NEGATIVE_LABELS]
    takes = [c for c in test_clips if c.is_take]
    if not negatives and not takes:
        print("  not computable: the held-out sessions hold no negative footage and no "
              "prompted takes.")
        return

    minutes = (sum(c.duration for c in negatives) + sum(c.negative_seconds for c in takes)) / 60
    fires = {"J": 0, "Z": 0}
    wrong_in_item = {"J": 0, "Z": 0}
    other = 0
    per_clip = []
    tally = {"items": 0, "hit": 0, "miss": 0, "double": 0, "exact": 0}
    strings = {}
    for clip in negatives + takes:
        ems, _, _ = replay(clip, th, static=static, motion=motion,
                           static_classes=static_classes, motion_classes=motion_classes,
                           static_feature_tag=static_feature_tag)
        n = 0
        for em in ems:
            if em.kind != "motion" or em.letter not in fires:
                other += 1
                continue
            if not clip.is_take:
                fires[em.letter] += 1
                n += 1
                continue
            onset = em.t - float(em.detail.get("duration", 0.0))
            phase = clip.phase_at(onset)
            if phase in ("rest", "none"):
                fires[em.letter] += 1
                n += 1
            elif em.letter != clip.item_at(onset)["label"]:
                wrong_in_item[em.letter] += 1
        per_clip.append((clip.clip_id, clip.duration if not clip.is_take
                         else clip.negative_seconds, n))
        for I in clip.items:
            L = I["label"]
            if L not in ("J", "Z"):
                continue
            seq = [e for e in ems if I["park"][0] <= e.t < I["rest"][1]]
            s_ = "".join(e.letter for e in seq)
            motion_letters = [e.letter for e in seq if e.kind == "motion"]
            tally["items"] += 1
            tally["hit" if L in motion_letters else "miss"] += 1
            tally["double"] += motion_letters.count(L) >= 2
            tally["exact"] += s_ == L
            strings.setdefault(L, {}).setdefault(s_, 0)
            strings[L][s_] += 1

    tier = "tier 1 only" if not th.TIER2_ENABLED else "tier 1 + tier 2"
    rate = (lambda n: n / minutes) if minutes else (lambda n: float("nan"))
    print(f"  {len(negatives)} negative clips and {len(takes)} prompted takes, "
          f"{minutes:.2f} min of negative footage, {tier}")
    print(f"    false J: {fires['J']:>3d}   {rate(fires['J']):.2f}/min")
    print(f"    false Z: {fires['Z']:>3d}   {rate(fires['Z']):.2f}/min")
    print(f"    total  : {sum(fires.values()):>3d}   {rate(sum(fires.values())):.2f}/min")
    if takes:
        print(f"  motion emissions inside a prompted item naming the OTHER letter: "
              f"J {wrong_in_item['J']}, Z {wrong_in_item['Z']} (not in the rate above; a "
              f"substitution, not a fire on negative footage)")
    print(f"  (static letters also emitted during this footage: {other}; they are not "
          f"counted above, which reports only the motion letters this module is about)")
    if sum(fires.values()):
        worst = max(per_clip, key=lambda r: r[2] / max(r[1], 1e-9))
        print(f"  worst single clip: {worst[0]}, {worst[2]} false motion emissions in "
              f"{worst[1]:.1f} s of negative footage")
    if tally["items"]:
        n = tally["items"]
        print(f"\n  end to end, per prompted item (emissions delivered inside the item's "
              f"PARK..REST span): {n} items, letter emitted {tally['hit']} "
              f"({tally['hit'] / n:.3f}), missed {tally['miss']}, emitted twice "
              f"{tally['double']}, exactly the letter and nothing else {tally['exact']} "
              f"({tally['exact'] / n:.3f})")
        for L in sorted(strings):
            top = sorted(strings[L].items(), key=lambda kv: -kv[1])[:8]
            print(f"    {L} items read as: " + ", ".join(f"{s_ or '(nothing)'} x{k}"
                                                          for s_, k in top))


# ---------------------------------------------------------------- number 3

def letter_error_rate(test_clips, th, static, static_classes, motion, motion_classes,
                      static_feature_tag=None):
    """LER over the continuous held-out takes, by Levenshtein alignment.

    This is the only number that sees segmentation errors, double-fires, missed gestures and
    the deferred-emission rule for I and D at once, because it scores the emitted string
    rather than a set of pre-cut events.
    """
    takes = [c for c in test_clips
             if (c.label in SPELL_LABELS or c.is_take or c.intended) and c.intended]
    if not takes:
        unlabeled = [c for c in test_clips if c.label in SPELL_LABELS or c.is_take]
        if unlabeled:
            print(f"  not computable: {len(unlabeled)} continuous held-out takes are "
                  "recorded, but none\n  carries the string that was actually signed. Nothing "
                  "in the capture path knows it,\n  and it is not inferred from the emissions "
                  "-- that would score the system against its\n  own output -- nor from a "
                  "prompted take's schedule, which is a list of items,\n  not a spelling. "
                  "Write it down and pass --intended with a JSON object keyed by clip id:")
            for c in unlabeled[:10]:
                print(f'      "{c.clip_id}": "..."   ({c.duration:.1f} s)')
            if len(unlabeled) > 10:
                print(f"      ... and {len(unlabeled) - 10} more")
        else:
            print("  not computable: the held-out sessions hold no continuous fingerspelling "
                  "take.")
        return

    tot = {"S": 0, "I": 0, "D": 0, "=": 0}
    jz = {"S": 0, "I": 0, "D": 0}
    n_ref = 0
    print(f"    {'take':>16}{'n_ref':>7}{'sub':>5}{'ins':>5}{'del':>5}{'LER':>8}   emitted")
    for clip in takes:
        ems, _, _ = replay(clip, th, static=static, motion=motion,
                           static_classes=static_classes, motion_classes=motion_classes,
                           static_feature_tag=static_feature_tag)
        hyp = "".join(em.letter for em in ems)
        ops = levenshtein(clip.intended, hyp)
        c = {"S": 0, "I": 0, "D": 0, "=": 0}
        for op, r, h in ops:
            c[op] += 1
            if op != "=" and ((r in ("J", "Z")) or (h in ("J", "Z"))):
                jz[op] += 1
        for k in c:
            tot[k] += c[k]
        n_ref += len(clip.intended)
        ler = (c["S"] + c["I"] + c["D"]) / max(len(clip.intended), 1)
        print(f"    {clip.clip_id[:16]:>16}{len(clip.intended):>7}{c['S']:>5}{c['I']:>5}"
              f"{c['D']:>5}{ler:>8.3f}   {hyp}")

    err = tot["S"] + tot["I"] + tot["D"]
    print(f"\n  independent takes: {len(takes)}   reference letters: {n_ref}")
    print(f"  substitutions {tot['S']}   insertions {tot['I']}   deletions {tot['D']}")
    print(f"  LER = {err}/{n_ref} = {err / max(n_ref, 1):.3f}")
    print(f"  of those errors, {sum(jz.values())} involve J or Z "
          f"(S {jz['S']}, I {jz['I']}, D {jz['D']}); the remaining "
          f"{err - sum(jz.values())} are among the other 24 letters")
    print("  an insertion is usually a spurious motion fire or a re-entered hold; a deletion "
          "a gate\n  or a hold that never fired; a substitution a genuine classifier error")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=CLIPS)
    ap.add_argument("--static-model", default=STATIC_MODEL)
    ap.add_argument("--motion-model", default=MOTION_MODEL)
    ap.add_argument("--thresholds", default=None, help="thresholds JSON; defaults to the "
                                                       "constants in thresholds.py")
    ap.add_argument("--train-session", default="S1")
    ap.add_argument("--test-session", nargs="+", default=None, metavar="S",
                    help="the held-out session(s) to report on; default: every session in "
                         "the recording other than --train-session")
    ap.add_argument("--width", type=int, default=None,
                    help="capture frame width, if the recording does not carry it")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--handedness", default="Right",
                    help="assumed when the recording does not store MediaPipe's label")
    ap.add_argument("--intended", default=None,
                    help="JSON object mapping clip id to the letter string that take actually "
                         "spelled; number 3 is not computable without it")
    args = ap.parse_args()

    th = Thresholds.from_json(args.thresholds) if args.thresholds else DEFAULT
    clips = load_recordings(args.clips, args.width, args.height, args.handedness)
    if args.intended:
        attach_intended(clips, args.intended)

    train_id = _norm_session(args.train_session)
    sessions = sorted({c.session for c in clips})
    if len(sessions) < 2:
        raise SystemExit(
            f"only session(s) {sessions} have been recorded.\n"
            "Refusing to report a generalization number. The headline figure this repository\n"
            "publishes is leave-one-session-out: train on one session, test on another "
            "recorded on a\ndifferent day, in a different room, at a deliberately different "
            "camera distance.\nSplitting one session at random -- over clips, windows or "
            "frames -- measures how well the\nmodel can re-identify that one sitting, which is "
            "exactly the mistake the README documents.\nRecord a second session and run this "
            "again.")
    if args.test_session:
        test_ids_wanted = [_norm_session(x) for x in args.test_session]
        unknown = sorted(set(test_ids_wanted) - set(sessions))
        if unknown:
            raise SystemExit(f"session(s) {unknown} are not in the recording; it holds {sessions}")
        if train_id in test_ids_wanted:
            raise SystemExit(f"{train_id} is the training session; it cannot also be held out")
    else:
        test_ids_wanted = [s_ for s_ in sessions if s_ != train_id]
    if not test_ids_wanted:
        raise SystemExit(f"nothing to hold out: every recorded session is {train_id}")

    train_clips = [c for c in clips if c.session == train_id]
    test_clips = [c for c in clips if c.session in test_ids_wanted]
    train_ids = {c.clip_id for c in train_clips}
    test_ids = {c.clip_id for c in test_clips}

    # Structural, not a matter of discipline: the split is by clip id and the assertion runs
    # every time the numbers are printed, so an edit that lets a clip leak cannot go unnoticed.
    assert not (train_ids & test_ids), \
        f"clip ids in both sessions: {sorted(train_ids & test_ids)[:5]}"
    assert len(train_ids) == len(train_clips) and len(test_ids) == len(test_clips), \
        "duplicate clip ids within a session; ids must identify a clip uniquely"

    static_blob = load_model(args.static_model, "static")
    motion_blob = load_model(args.motion_model, "event")
    # The tag the pickle names decides the vector the Segmenter builds; .get keeps a tagless
    # pre-registry pickle working (None -> static/v3). Numbers 2 and 3 replay with the static
    # forest attached and used to build _Tap without it, which raised on the 112-D forest.
    static_tag = static_blob.get("feature")
    if static_tag is not None and static_tag not in F.STATIC_FEATURES:
        raise SystemExit(f"{args.static_model} was trained on feature {static_tag!r}, which "
                         f"this build cannot compute (known: {sorted(F.STATIC_FEATURES)})")
    motion_classes = list(motion_blob.get("classes", []))
    if set(motion_classes) != set(MOTION_CLASSES):
        raise SystemExit(f"the event model's classes are {motion_classes}; this evaluation is "
                         f"defined over {MOTION_CLASSES} (STATIC is unreachable by "
                         "construction and is not a class)")

    test_label = "+".join(test_ids_wanted)
    print(f"clips: {len(clips)} total, sessions {sessions}")
    print(f"train = {train_id}: {len(train_clips)} clips   "
          f"test = {test_label}: {len(test_clips)} clips   (no clip id crosses)")

    # A number computed over part of a recording without saying which part is the exact error
    # this module exists to correct, so the clips no number below touches are named here. The
    # S2 protocol records a 24-letter static holdout into the same file, and those clips feed
    # the static confusion matrix, which is not one of the three numbers this module owns.
    scored_labels = {"J", "Z"} | NEGATIVE_LABELS | SPELL_LABELS | CONTINUOUS_LABELS
    ignored = [c for c in test_clips if c.label not in scored_labels and not c.intended]
    if ignored:
        counts = {}
        for c in ignored:
            counts[c.label] = counts.get(c.label, 0) + 1
        print(f"NOTE: {len(ignored)} of the {len(test_clips)} {test_label} clips take part in "
              f"none of the three numbers\n      below: {counts}. They are neither prompted "
              "J/Z, nor a prompted take, nor negative footage,\n      nor a continuous take "
              "carrying an intended string.")

    # Held out from WHAT. The static forest is fitted on every static session, so static
    # letters are in-sample everywhere in this file. The motion forest records the sessions
    # it was fitted on (train_motion.py); a test session among them is in-sample for the
    # motion numbers and the header of every block says so.
    trained_on = motion_blob.get("train_sessions")
    if trained_on is None:
        print(f"NOTE: {os.path.basename(args.motion_model)} does not record which sessions it "
              "was fitted on, so\n      this module cannot verify that no held-out clip reached "
              "training. If it was fitted\n      on the whole recording file, nothing here is "
              "held out from the motion forest and the\n      three numbers below are "
              "in-sample -- exactly the figure this repository has published a\n      "
              "correction for. Retrain with train_motion.py (which records train_sessions), "
              "or with\n      --fit-sessions S1 for a forest the other sessions are held out from.")
        status = {s_: "in-sample unless the forest was fitted elsewhere" for s_ in test_ids_wanted}
    else:
        trained_on = {_norm_session(x) for x in trained_on}
        status = {s_: ("IN-SAMPLE for the motion forest" if s_ in trained_on
                       else "held out from the motion forest") for s_ in test_ids_wanted}
        print(f"the motion forest was fitted on sessions {sorted(trained_on)} "
              f"({motion_blob.get('n_events', '?')} events in {motion_blob.get('train_groups', '?')} "
              "groups)")
    for s_ in test_ids_wanted:
        n = sum(1 for c in test_clips if c.session == s_)
        print(f"  {s_}: {n} clip(s), {status[s_]}")
    in_sample = [s_ for s_ in test_ids_wanted if status[s_].startswith("IN-SAMPLE")]
    tag = (f" [{test_label}: "
           + ("IN-SAMPLE for the motion forest" if in_sample == test_ids_wanted else
              "held out from the motion forest" if not in_sample and trained_on is not None else
              f"{'+'.join(in_sample)} in-sample" if in_sample else "provenance unrecorded")
           + "; static letters in-sample]")

    print(f"\n=== 1. event-level 3-class {{J,Z,MOVE}} on {test_label} ==={tag}")
    event_metrics(test_clips, motion_blob["model"], motion_classes, th)

    print(f"\n=== 2. false J/Z emissions per minute of {test_label} negative footage ==={tag}")
    print("    replayed through Segmenter, the object the live demo drives")
    false_fire_rate(test_clips, th, static_blob["model"], list(static_blob.get("classes", [])),
                    motion_blob["model"], motion_classes, static_feature_tag=static_tag)

    print(f"\n=== 3. letter error rate on the {test_label} continuous takes ==={tag}")
    letter_error_rate(test_clips, th, static_blob["model"],
                      list(static_blob.get("classes", [])),
                      motion_blob["model"], motion_classes, static_feature_tag=static_tag)

    n_j = (sum(1 for c in test_clips if c.label == "J")
           + sum(1 for c in test_clips for I in c.items if I["label"] == "J"))
    n_z = (sum(1 for c in test_clips if c.label == "Z")
           + sum(1 for c in test_clips for I in c.items if I["label"] == "Z"))
    who = sorted({c.signer for c in test_clips})
    print("\n--- what these numbers do not cover ---")
    # Counted rather than asserted. The recorder stamps a signer on every clip, and the moment
    # the optional second-signer session is recorded, a hard-coded "one signer" would be a
    # false caveat in the one run where the caveat is the point.
    if who == ["?"]:
        print(f"  {n_j} independent J reps and {n_z} independent Z reps in {test_label}. The "
              "recording names no\n  signer, so how many people these came from is not "
              "something this run can state.")
    elif len(who) == 1:
        print(f"  one signer ({who[0]}). {n_j} independent J reps and {n_z} independent Z reps "
              f"in {test_label},\n  so this measures that one person's J and Z, and nothing "
              "about a second signer.")
    else:
        per = ", ".join(
            f"{s_} ({sum(1 for c in test_clips if c.signer == s_ and c.label == 'J')} J, "
            f"{sum(1 for c in test_clips if c.signer == s_ and c.label == 'Z')} Z)" for s_ in who)
        print(f"  {len(who)} signers in {test_label}: {per}.\n  Every figure above pools them, "
              "so none of them is a per-signer number; and a few dozen reps\n  from a second "
              "person measures that person, not signers in general.")
    print(f"  the separation between {train_id} and {test_label} -- different day, room, "
          "lighting, camera\n  distance -- is what the session tag asserts, not something this "
          "module measured. It reads\n  the tag the recorder was given.")
    print("  nothing about a left-handed signer (the mirroring path is untested code until "
          "someone\n  left-handed uses it), another camera, or co-articulated fluent "
          "fingerspelling. The\n  supported input is deliberate, isolated fingerspelling.")


if __name__ == "__main__":
    main()

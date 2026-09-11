"""The three reported numbers, and nothing flattering.

This repository's README documents, at length, that its old 100.00% and 99.58% figures were
measured on near-duplicate frames from a single capture burst: what they scored was the
forest's ability to re-identify frames it had already seen, not to recognize a handshape.
This module is the correction, so every decision in it is made in the conservative direction
and the count of *independent clips* behind a figure is printed beside the figure itself --
the habit that would have caught the old number, which was computed over 23 of 24 letters.

Three numbers, all on session 2 only:

1. Event-level 3-class {J, Z, MOVE} metrics -- confusion matrix, per-class precision and
   recall. Reported next to them, because they are the part a confusion matrix cannot show:
   how many prompted reps the state machine never cut an event out of at all, and how many
   cut events the cheap vetoes killed before the classifier ever saw them. A recall figure
   computed only over events that reached the model would silently exclude every gesture the
   segmenter missed, which is the more likely failure.

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
this repository already published a correction for.

Data contract. `motion_clips.npz` as written by collect_motion.py:

    clips     object (N,)   each (T,21,3) MediaPipe-normalized landmarks, NaN rows where the
                            hand was not detected
    stamps    object (N,)   each (T,) seconds from the start of the clip
    labels    (N,) str      "J" | "Z" | "NONE" (negative) | "SPELL" (continuous take)

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
                            its own output.

Run:  ../.venv/bin/python temporal/evaluate.py
"""
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

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
    the frame aspect, and it mirrors the whole hand when it is wrong.
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
    intended = _pick(d, ("intended", "intent", "prompts"))
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
        out.append(Clip(
            clip_id=str(ids[i]) if ids is not None else f"{sess}:{lab}:{i}",
            label=lab, session=sess, times=ts, lm=lm, width=w, height=h,
            handedness=_hand_field(hands[i], handedness) if hands is not None else handedness,
            intended="".join(ch for ch in str(intended[i]).upper() if ch.isalpha())
                     if intended is not None else "",
            signer=str(signers[i]) if signers is not None else "?"))
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
        super().__init__(*a, **kw)
        self.cut = 0

    def _score(self, t):
        if self._track_from is not None and len(self._since(self._track_from)) >= 3:
            self.cut += 1
        return super()._score(t)


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


def replay(clip, th, static=None, motion=None, static_classes=None, motion_classes=None):
    """Drive one recorded clip through the state machine. Returns (emissions, tap, probe).

    A fresh Segmenter per clip: state must never carry across recordings, and the buffer,
    cooldown and last_emitted of one take have no business influencing the next.

    The NaN rows the recorder writes for undetected frames are passed as `None`, which is what
    a live frame with no hand delivers, so the gap-interpolation and gap-reset paths are
    exercised here exactly as they would be on the camera.
    """
    probe = _Probe(motion) if motion is not None else None
    seg = _Tap(th, static_model=static, motion_model=probe,
               static_classes=static_classes, motion_classes=motion_classes)
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


def print_confusion(M, classes, clips_per_class):
    w = max(6, max(len(c) for c in classes) + 1)
    head = "true|pred"
    print(f"    {head:>{w}}" + "".join(f"{c:>8}" for c in classes)
          + f"{'total':>8}{'clips':>8}")
    for i, c in enumerate(classes):
        print(f"    {c:>{w}}" + "".join(f"{M[i, j]:>8d}" for j in range(len(classes)))
              + f"{M[i].sum():>8d}{clips_per_class.get(c, 0):>8d}")


def print_pr(M, classes, clips_per_class):
    print(f"    {'class':>8}{'n_events':>10}{'n_independent_clips':>21}"
          f"{'precision':>11}{'recall':>8}")
    for i, c in enumerate(classes):
        tp = M[i, i]
        support = M[i].sum()
        predicted = M[:, i].sum()
        prec = tp / predicted if predicted else float("nan")
        rec = tp / support if support else float("nan")
        print(f"    {c:>8}{support:>10d}{clips_per_class.get(c, 0):>21d}"
              f"{prec:>11.3f}{rec:>8.3f}")


# ---------------------------------------------------------------- number 1

def event_metrics(test_clips, motion, motion_classes, th):
    """3-class {J,Z,MOVE} metrics over events the state machine cut from session-2 clips.

    Ground truth is assigned conservatively. A prompted 2 s clip contains one rep, so if the
    segmenter cuts several scorable events out of it, only the one with the longest tip path
    (event feature [53]) is credited as the rep; the rest are counted as MOVE, because they
    are motion the signer did not intend as a letter. The alternative -- crediting every
    event from a J clip as a J -- would let a spurious cut that happens to score J count as a
    success.

    MOVE truth is mined from negative footage only. The continuous fingerspelling takes also
    produce events, but their intended strings contain real J and Z letters, so an event mined
    from them is not reliably a negative.
    """
    prompted = [c for c in test_clips if c.label in ("J", "Z")]
    negatives = [c for c in test_clips if c.label in NEGATIVE_LABELS]
    if not prompted and not negatives:
        print("  not computable: session 2 holds no prompted J/Z clips and no negative "
              "footage.")
        return

    y_true, y_pred, p_win, margins = [], [], [], []
    clips_seen = {c: set() for c in MOTION_CLASSES}
    missed_clips, multi_clips, cut_total, scored_total = [], 0, 0, 0

    for clip in prompted + negatives:
        _, seg, probe = replay(clip, th, motion=motion, motion_classes=motion_classes)
        cut_total += seg.cut
        events = probe.seen
        scored_total += len(events)

        if clip.label in ("J", "Z"):
            if not events:
                missed_clips.append(clip.clip_id)
                continue
            if len(events) > 1:
                multi_clips += 1
            primary = int(np.argmax([f[EVENT_PATH_LEN] for f, _ in events]))
            truth = [clip.label if k == primary else "MOVE" for k in range(len(events))]
        else:
            truth = ["MOVE"] * len(events)

        for (f, proba), tr in zip(events, truth):
            order = np.argsort(proba)[::-1]
            y_true.append(tr)
            y_pred.append(motion_classes[int(order[0])])
            p_win.append(float(proba[order[0]]))
            margins.append(float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0)
            clips_seen[tr].add(clip.clip_id)

    n_j = sum(1 for c in prompted if c.label == "J")
    n_z = sum(1 for c in prompted if c.label == "Z")
    print(f"  session-2 source clips: {n_j} prompted J, {n_z} prompted Z, "
          f"{len(negatives)} negative takes "
          f"({sum(c.duration for c in negatives) / 60:.2f} min of negatives)")
    print(f"  events cut by the state machine: {cut_total}; reached the classifier: "
          f"{scored_total} ({cut_total - scored_total} killed by the duration, path-length "
          f"and straightness vetoes before any model ran)")

    if missed_clips:
        print(f"  prompted reps the segmenter produced NO scorable event for: "
              f"{len(missed_clips)} of {len(prompted)} "
              f"({len(missed_clips) / len(prompted):.1%}) -- these are recall failures that "
              f"the matrix below cannot show, because they never became events")
    else:
        print(f"  prompted reps with no scorable event: 0 of {len(prompted)}")
    if multi_clips:
        print(f"  prompted clips yielding more than one scorable event: {multi_clips} "
              f"(only the longest-path event is credited as the rep; the others count as MOVE)")

    if not y_true:
        print("  no events to score.")
        return

    clips_per_class = {c: len(s) for c, s in clips_seen.items()}
    print()
    print_confusion(confusion(y_true, y_pred, MOTION_CLASSES), MOTION_CLASSES, clips_per_class)
    print()
    print_pr(confusion(y_true, y_pred, MOTION_CLASSES), MOTION_CLASSES, clips_per_class)

    # The operating point is part of the design, not a post-hoc knob: an abstention is the
    # correct answer for most motion, and a recall figure obtained by lowering P_EMIT is a
    # different system from the one that ships.
    abst = sum(1 for p, m in zip(p_win, margins) if p < th.P_EMIT or m < th.MARGIN)
    print(f"\n    abstained at P_EMIT={th.P_EMIT} / MARGIN={th.MARGIN}: "
          f"{abst} of {len(y_true)} events ({abst / len(y_true):.1%})")
    print("    (the matrix above is argmax, ignoring the thresholds; the deployed system "
          "emits nothing for an abstained event)")


# ---------------------------------------------------------------- number 2

def false_fire_rate(test_clips, th, static, static_classes, motion, motion_classes):
    """False J/Z emissions per minute of session-2 negative footage.

    Measured by replaying the whole Segmenter, both models attached, over the recorded frames
    -- not by scoring pre-cut events. What a user experiences is the emission stream, and the
    gates, the rising edge, the vetoes and the cooldown are all part of what produces it.
    """
    negatives = [c for c in test_clips if c.label in NEGATIVE_LABELS]
    if not negatives:
        print("  not computable: session 2 holds no negative footage.")
        return

    minutes = sum(c.duration for c in negatives) / 60.0
    fires = {"J": 0, "Z": 0}
    other = 0
    per_clip = []
    for clip in negatives:
        ems, _, _ = replay(clip, th, static=static, motion=motion,
                           static_classes=static_classes, motion_classes=motion_classes)
        n = 0
        for em in ems:
            if em.kind == "motion" and em.letter in fires:
                fires[em.letter] += 1
                n += 1
            else:
                other += 1
        per_clip.append((clip.clip_id, clip.duration, n))

    tier = "tier 1 only" if not th.TIER2_ENABLED else "tier 1 + tier 2"
    rate = (lambda n: n / minutes) if minutes else (lambda n: float("nan"))
    print(f"  {len(negatives)} independent negative clips, {minutes:.2f} min total, {tier}")
    print(f"    false J: {fires['J']:>3d}   {rate(fires['J']):.2f}/min")
    print(f"    false Z: {fires['Z']:>3d}   {rate(fires['Z']):.2f}/min")
    print(f"    total  : {sum(fires.values()):>3d}   {rate(sum(fires.values())):.2f}/min")
    print(f"  (static letters also emitted during this footage: {other}; they are not "
          f"counted above, which reports only the motion letters this module is about)")
    if sum(fires.values()):
        worst = max(per_clip, key=lambda r: r[2] / max(r[1], 1e-9))
        print(f"  worst single clip: {worst[0]}, {worst[2]} false motion emissions in "
              f"{worst[1]:.1f} s")


# ---------------------------------------------------------------- number 3

def letter_error_rate(test_clips, th, static, static_classes, motion, motion_classes):
    """LER over the continuous session-2 takes, by Levenshtein alignment.

    This is the only number that sees segmentation errors, double-fires, missed gestures and
    the deferred-emission rule for I and D at once, because it scores the emitted string
    rather than a set of pre-cut events.
    """
    takes = [c for c in test_clips
             if (c.label in SPELL_LABELS or c.intended) and c.intended]
    if not takes:
        unlabeled = [c for c in test_clips if c.label in SPELL_LABELS]
        if unlabeled:
            print(f"  not computable: {len(unlabeled)} continuous session-2 takes are "
                  "recorded, but none\n  carries the string that was actually signed. Nothing "
                  "in the capture path knows it,\n  and it is not inferred from the emissions "
                  "-- that would score the system against its\n  own output. Write it down "
                  "and pass --intended with a JSON object keyed by clip id:")
            for c in unlabeled[:10]:
                print(f'      "{c.clip_id}": "..."   ({c.duration:.1f} s)')
            if len(unlabeled) > 10:
                print(f"      ... and {len(unlabeled) - 10} more")
        else:
            print("  not computable: session 2 holds no continuous fingerspelling take.")
        return

    tot = {"S": 0, "I": 0, "D": 0, "=": 0}
    jz = {"S": 0, "I": 0, "D": 0}
    n_ref = 0
    print(f"    {'take':>16}{'n_ref':>7}{'sub':>5}{'ins':>5}{'del':>5}{'LER':>8}   emitted")
    for clip in takes:
        ems, _, _ = replay(clip, th, static=static, motion=motion,
                           static_classes=static_classes, motion_classes=motion_classes)
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
    print(f"\n  n_independent_clips (takes): {len(takes)}   reference letters: {n_ref}")
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
    ap.add_argument("--test-session", default="S2")
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
    test_id = _norm_session(args.test_session)
    sessions = sorted({c.session for c in clips})
    if len(sessions) < 2 or test_id not in sessions:
        raise SystemExit(
            f"only session(s) {sessions} have been recorded.\n"
            "Refusing to report a generalization number. The headline figure this repository\n"
            "publishes is leave-one-session-out: train on S1, test on S2 recorded on a "
            "different\nday, in a different room, at a deliberately different camera "
            "distance. Splitting one\nsession at random -- over clips, windows or frames -- "
            "measures how well the model can\nre-identify that one sitting, which is exactly "
            "the mistake the README documents. Record\nsession 2 and run this again.")

    train_clips = [c for c in clips if c.session == train_id]
    test_clips = [c for c in clips if c.session == test_id]
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
    motion_classes = list(motion_blob.get("classes", []))
    if set(motion_classes) != set(MOTION_CLASSES):
        raise SystemExit(f"the event model's classes are {motion_classes}; this evaluation is "
                         f"defined over {MOTION_CLASSES} (STATIC is unreachable by "
                         "construction and is not a class)")

    print(f"clips: {len(clips)} total, sessions {sessions}")
    print(f"train = {train_id}: {len(train_clips)} clips   "
          f"test = {test_id}: {len(test_clips)} clips   (no clip id crosses)")

    # A number computed over part of a recording without saying which part is the exact error
    # this module exists to correct, so the clips no number below touches are named here. The
    # S2 protocol records a 24-letter static holdout into the same file, and those clips feed
    # the static confusion matrix, which is not one of the three numbers this module owns.
    scored_labels = {"J", "Z"} | NEGATIVE_LABELS | SPELL_LABELS
    ignored = [c for c in test_clips if c.label not in scored_labels and not c.intended]
    if ignored:
        counts = {}
        for c in ignored:
            counts[c.label] = counts.get(c.label, 0) + 1
        print(f"NOTE: {len(ignored)} of the {len(test_clips)} {test_id} clips take part in "
              f"none of the three numbers\n      below: {counts}. They are neither prompted "
              "J/Z, nor negative footage, nor a\n      continuous take carrying an intended "
              "string.")

    trained_on = motion_blob.get("train_clip_ids")
    if trained_on is None:
        print(f"NOTE: {os.path.basename(args.motion_model)} does not record which clips it was "
              "fitted on, so\n      this module cannot verify that no session-2 clip reached "
              "training. If it was fitted\n      on the whole recording file, session 2 is not "
              "held out and the three numbers below\n      are in-sample -- exactly the figure "
              "this repository has published a correction for.\n      Fit the event model on "
              f"{train_id} clips alone, or have it record its training clip ids.")
    else:
        leaked = set(map(str, trained_on)) & test_ids
        assert not leaked, f"the event model was trained on held-out clips: {sorted(leaked)[:5]}"
        print(f"the event model records {len(set(map(str, trained_on)))} training clip ids, "
              f"none of them in {test_id}")

    print("\n=== 1. event-level 3-class {J,Z,MOVE} on held-out session 2 ===")
    event_metrics(test_clips, motion_blob["model"], motion_classes, th)

    print("\n=== 2. false J/Z emissions per minute of session-2 negative footage ===")
    print("    replayed through Segmenter, the object the live demo drives")
    false_fire_rate(test_clips, th, static_blob["model"], list(static_blob.get("classes", [])),
                    motion_blob["model"], motion_classes)

    print("\n=== 3. letter error rate on the session-2 continuous takes ===")
    letter_error_rate(test_clips, th, static_blob["model"],
                      list(static_blob.get("classes", [])),
                      motion_blob["model"], motion_classes)

    n_j = sum(1 for c in test_clips if c.label == "J")
    n_z = sum(1 for c in test_clips if c.label == "Z")
    who = sorted({c.signer for c in test_clips})
    print("\n--- what these numbers do not cover ---")
    # Counted rather than asserted. The recorder stamps a signer on every clip, and the moment
    # the optional second-signer session is recorded, a hard-coded "one signer" would be a
    # false caveat in the one run where the caveat is the point.
    if who == ["?"]:
        print(f"  {n_j} independent J reps and {n_z} independent Z reps in {test_id}. The "
              "recording names no\n  signer, so how many people these came from is not "
              "something this run can state.")
    elif len(who) == 1:
        print(f"  one signer ({who[0]}). {n_j} independent J reps and {n_z} independent Z reps "
              f"in {test_id},\n  so this measures that one person's J and Z, and nothing about "
              "a second signer.")
    else:
        per = ", ".join(
            f"{s} ({sum(1 for c in test_clips if c.signer == s and c.label == 'J')} J, "
            f"{sum(1 for c in test_clips if c.signer == s and c.label == 'Z')} Z)" for s in who)
        print(f"  {len(who)} signers in {test_id}: {per}.\n  Every figure above pools them, so "
              "none of them is a per-signer number; and a few dozen reps\n  from a second "
              "person measures that person, not signers in general.")
    print(f"  the separation between {train_id} and {test_id} -- different day, room, lighting, "
          "camera\n  distance -- is what the session tag asserts, not something this module "
          "measured. It reads\n  the tag the recorder was given.")
    print("  nothing about a left-handed signer (the mirroring path is untested code until "
          "someone\n  left-handed uses it), another camera, or co-articulated fluent "
          "fingerspelling. The\n  supported input is deliberate, isolated fingerspelling.")


if __name__ == "__main__":
    main()

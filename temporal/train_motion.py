"""Train the 3-class event classifier {J, Z, MOVE} on the 79-D motion feature.

The label space is three classes, not four: an event only exists downstream of a motion
trigger, so a still hand never produces one and a STATIC class would be unreachable at
runtime while inflating every accuracy printed here.

The split is BY CLIP and nothing else. Each recorded take is one independent observation;
the augmented copies of a take are correlated with it and with each other, so a row-level
split would put a time-warped copy of a test gesture into training and report a number that
means "can the forest re-identify this take" rather than "does it recognize a J". That is
the exact mistake the README documents this project already making once, and the structural
defence -- not the disciplinary one -- is that the array handed to the splitter has one row
per clip, and feature rows are materialised only afterwards, inside each side. Augmentation
runs on the training side only, and both facts are asserted at runtime rather than tested.

Every accuracy is printed beside n_independent_clips, because an accuracy over augmented
rows describes a row count that is not an evidence count.

Cross-validation is confined to one recording session, because a number pooled across two
sittings is neither a within-session number nor a generalization one. The headline is
leave-one-session-out and evaluate.py owns it; nothing printed here is that number.

This file writes model_motion.p and prints a report. It trains nothing until
motion_clips.npz exists; no data is synthesised to stand in for the recording.
"""
import argparse
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from thresholds import DEFAULT, Thresholds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "motion_clips.npz")
OUT = os.path.join(HERE, "model_motion.p")

#: The recorder's label for negatives; the classifier's third class is named for what it is.
LABEL_TO_CLASS = {"J": "J", "Z": "Z", "NONE": "MOVE"}


def feature_names():
    """Readable name per index of the 79-D event feature, matching features.event_features.

    Importances over anonymous column numbers are unreadable, and reading them is the point:
    if J is decided by signed net turning, straightness and the thirds code then the model
    learned the geometry of the glyph; if it is decided by duration or peak speed it learned
    this signer's tempo, and the accuracy is measuring the wrong thing.
    """
    names = []
    for i in range(F.K_RESAMPLE):
        names += [f"path{i + 1}.x", f"path{i + 1}.y"]              # [0:32]
    names += [f"turn{k + 1}" for k in range(F.K_RESAMPLE - 2)]     # [32:46]
    names += [f"ext_med.{k}" for k in ("thumb", "index", "mid", "ring", "pinky")]
    names += ["orient.x", "orient.y", "path_len", "duration", "straightness",
              "turn_count", "turn_absmax", "turn_signed_sum", "sigma_max", "sigma_med",
              "scale_ratio"]
    for i in (1, 2, 3):
        names += [f"third{i}.dx", f"third{i}.dy"]                  # [62:68]
    names += ["speed_mean", "speed_peak",
              "art.path_len", "art.net", "art.straightness", "art.dir_x", "art.dir_y",
              "art.turn_abs", "art.std_x", "art.std_y", "arm_flag"]
    assert len(names) == F.EVENT_DIM, f"{len(names)} names for {F.EVENT_DIM} features"
    return names


NAMES = feature_names()

#: Features that describe how fast this signer moved rather than what shape was drawn.
#: Named so the report can say out loud when the forest is leaning on tempo.
TEMPO = {"duration", "speed_mean", "speed_peak"}


@dataclass
class Row:
    x: np.ndarray
    y: str
    clip_id: int
    augmented: bool


@dataclass
class Clip:
    clip_id: int
    times: np.ndarray     # (T,) seconds, monotonic, starting at 0
    P: np.ndarray         # (T,21,2) isotropic, handedness-canonicalized, NOT palm-scaled
    label: str            # "J" | "Z" | "MOVE"
    arm: str              # "J" | "Z" -- which fingertip writes
    session: Optional[str] = None   # the sitting it was recorded in, if it says


# ---------------------------------------------------------------- loading

def _pick(d, names):
    """First key present among `names`. The recorder's metadata keys have had several names."""
    for k in names:
        if k in d.files:
            return d[k]
    return None


def _norm_session(s):
    """'s1', 'S1', '1' and 1 all name the same sitting -- the rule evaluate.py applies too."""
    return "S" + str(s).strip().upper().lstrip("S")


def modal_handedness(entry, default):
    """One label per clip out of what the recorder stores, which is one label per FRAME.

    "None" is written on frames where MediaPipe found no hand, so the mode is taken over the
    real labels only. Handing the array straight to features.canonicalize_handedness instead
    silently does nothing at all -- str(array) begins with '[' -- so a left-handed recording
    would be trained un-mirrored while the live demo mirrors it, which is exactly the
    train/serve skew features.py exists to prevent.
    """
    if entry is None:
        return default
    real = [str(v) for v in np.atleast_1d(entry).ravel()
            if str(v)[:1].lower() in ("l", "r")]
    return max(set(real), key=real.count) if real else default


def interpolate_gaps(lm, times, max_gap):
    """Drop the NaN frames the recorder writes for undetected hands, or reject the clip.

    The recorder keeps a NaN row for every frame MediaPipe lost so the clip's time base stays
    honest. A gap shorter than the segmenter's own GAP_INTERP is interpolated here exactly as
    the segmenter interpolates it at runtime; a longer one aborts the track at runtime, so a
    clip containing one is not a clip the runtime could ever have scored and it is dropped
    rather than repaired. Returns (times, P) or None.
    """
    lm = np.asarray(lm, dtype=np.float64)[:, :, :2]
    times = np.asarray(times, dtype=np.float64)
    ok = np.isfinite(lm).all(axis=(1, 2))
    if ok.sum() < 3:
        return None

    lo, hi = int(np.argmax(ok)), len(ok) - 1 - int(np.argmax(ok[::-1]))
    lm, times, ok = lm[lo:hi + 1], times[lo:hi + 1], ok[lo:hi + 1]

    miss = np.flatnonzero(~ok)
    if len(miss):
        breaks = np.flatnonzero(np.diff(miss) > 1)
        for run in np.split(miss, breaks + 1):
            if times[run[-1] + 1] - times[run[0] - 1] > max_gap:
                return None
        good = np.flatnonzero(ok)
        for j in range(21):
            for d in range(2):
                lm[:, j, d] = np.interp(times, times[good], lm[good, j, d])
    return times, lm


def choose_arm(P, label):
    """Which fingertip is the writing tip: the pinky for J, the index for Z.

    For J and Z the label fixes it. A negative clip has no intended arm, so it is featurised
    as whichever launch pose the hand actually spent more frames in -- that is the arm the
    segmenter would have armed had this footage triggered a track, and a MOVE example is only
    useful if it sits in the same feature space as the false positive it is meant to reject.
    Neither gate firing means the hand was in neither launch pose, in which case the arm is
    arbitrary; Z is the default because its gate is the looser of the two and therefore the
    one more incidental motion reaches.
    """
    if label in ("J", "Z"):
        return label
    j, z = int(F.j_gate(P).sum()), int(F.z_gate(P).sum())
    return "J" if j > z else "Z"


def load_clips_cut(path, aspect, handedness, th):
    """Load clips, but cut each one into events by replaying the segmenter over it.

    This is the default, and whole-clip loading is the fallback, because the boundaries a
    recording has and the boundaries the runtime produces are not the same thing. A recorded
    clip is 2.0 s of lead-in plus gesture plus settle; Segmenter cuts from motion onset to
    motion offset and rejects anything past T_MAX = 1.80 s. Training on whole clips therefore
    builds a classifier whose every example is longer than anything it will ever be asked to
    score -- it would cross-validate beautifully and recognize nothing live.

    Reusing label_events.harvest rather than reimplementing the cut keeps a single definition
    of "where does an event start and stop" across labelling, training and inference.
    """
    import label_events as LE

    d = np.load(path, allow_pickle=True)
    raw_clips = _pick(d, ("clips",))
    raw_stamps = _pick(d, ("stamps", "times"))
    raw_labels = _pick(d, ("labels", "label"))
    raw_handed = _pick(d, ("handed", "handedness", "hands"))
    raw_sess = _pick(d, ("sessions", "session"))
    wh = _pick(d, ("frame_size", "frame_wh", "wh", "size"))
    if wh is None and aspect is None:
        raise SystemExit(f"{path} does not record the capture frame size and --aspect was not "
                         "given; aspect correction cannot be applied without it.")
    # Per-clip when the file carries one size per clip (ingested third-party footage is not
    # one camera); otherwise one size for all of them.
    if aspect:
        sizes = [tuple(aspect)] * len(raw_clips)
    else:
        arr = np.asarray(wh).reshape(-1, 2)
        sizes = ([(int(a), int(b)) for a, b in arr] if len(arr) == len(raw_clips)
                 else [(int(arr[0][0]), int(arr[0][1]))] * len(raw_clips))

    out, stats = [], {}
    for i in range(len(raw_clips)):
        lab = str(raw_labels[i]).strip().upper()
        cls = LABEL_TO_CLASS.get(lab)
        if cls is None:
            continue
        handed_i = raw_handed[i] if raw_handed is not None else handedness
        spans = LE.harvest(np.asarray(raw_clips[i]), np.asarray(raw_stamps[i]),
                           handed_i, th, sizes[i][0], sizes[i][1])
        stats.setdefault(lab, Counter())[len(spans)] += 1
        for s in spans:
            # An event armed under the other gate is a negative: at runtime this same span
            # would be scored under that arm, so the classifier must have seen it there.
            use = cls if (cls == "MOVE" or s["arm"] == cls) else "MOVE"
            out.append(Clip(clip_id=i, times=s["times"], P=s["P"], label=use, arm=s["arm"],
                            session=_norm_session(raw_sess[i]) if raw_sess is not None else None))

    print("events cut by replaying the segmenter (counts are events-per-clip):")
    for lab in sorted(stats):
        note = ""
        if lab in ("J", "Z"):
            missed = stats[lab].get(0, 0)
            if missed:
                note = f"   {missed} clip(s) produced NO event -- the runtime would miss those too"
        print(f"  {lab:5s} {dict(sorted(stats[lab].items()))}{note}")
    return out


def load_clips(path, aspect, handedness):
    d = np.load(path, allow_pickle=True)
    raw_clips = _pick(d, ("clips",))
    raw_stamps = _pick(d, ("stamps", "times"))
    raw_labels = _pick(d, ("labels", "label"))
    if raw_clips is None or raw_stamps is None or raw_labels is None:
        raise SystemExit(f"{path} is missing one of clips/stamps/labels; it was not written "
                         "by collect_motion.py")

    # W/H sets the isotropy correction, and features.py's gates are calibrated on corrected
    # coordinates: featurising 4:3 footage as 16:9 skews every diagonal relative to what the
    # live demo hands the same model. The recorder stores the size, so the file wins and
    # --aspect exists only for footage recorded before it did. There is deliberately no
    # default, because a wrong aspect is invisible in every number printed below.
    wh = _pick(d, ("frame_size", "frame_wh", "wh", "size"))
    if wh is None and aspect is None:
        raise SystemExit(
            f"{path} does not record the capture frame size and --aspect was not given.\n"
            "u = x*(W/H) cannot be applied without it, and every gate threshold in features.py\n"
            "is calibrated on corrected coordinates. Pass the size the footage was shot at,\n"
            "e.g. --aspect 640x480.")
    wh = np.asarray(wh) if wh is not None else None
    if wh is None:
        print(f"frame size {aspect[0]}x{aspect[1]} from --aspect; the recording carries none")
    elif wh.ndim == 1:
        print(f"frame size {int(wh[0])}x{int(wh[1])} read from {os.path.basename(path)}")
    else:
        print(f"per-clip frame size read from {os.path.basename(path)}")

    hands = _pick(d, ("handed", "handedness", "hands"))
    if hands is None:
        print(f"handedness ASSUMED '{handedness}' for every clip; the recording carries none")
    sessions = _pick(d, ("sessions", "session"))
    if sessions is None:
        print("no per-clip session label; the cross-validation below cannot be confined to one "
              "sitting, so read it as a pooled number")

    clips, dropped_gap, dropped_label = [], 0, {}
    for i in range(len(raw_clips)):
        label = LABEL_TO_CLASS.get(str(raw_labels[i]).strip().upper())
        if label is None:
            key = str(raw_labels[i])
            dropped_label[key] = dropped_label.get(key, 0) + 1
            continue
        got = interpolate_gaps(raw_clips[i], raw_stamps[i], DEFAULT.GAP_INTERP)
        if got is None:
            dropped_gap += 1
            continue
        times, lm = got
        if wh is None:
            w, h = aspect
        elif wh.ndim == 1:
            w, h = int(wh[0]), int(wh[1])
        else:
            w, h = int(wh[i][0]), int(wh[i][1])
        P = F.to_isotropic(lm, w, h)
        P = F.canonicalize_handedness(
            P, modal_handedness(hands[i] if hands is not None else None, handedness))
        clips.append(Clip(i, times, P, label, choose_arm(P, label),
                          _norm_session(sessions[i]) if sessions is not None else None))
    if dropped_gap:
        print(f"dropped {dropped_gap} clip(s): a tracking gap longer than GAP_INTERP="
              f"{DEFAULT.GAP_INTERP}s, which aborts the track at runtime too")
    if dropped_label:
        # A continuous fingerspelling take is not one event, and featurising a whole 20 s take
        # as a single gesture would put a label on something the runtime never sees. Its MOVE
        # examples are mined by replaying the segmenter over it, which is evaluate.py's job.
        print(f"dropped {dropped_label}: not single-event labels. This trainer consumes "
              f"{sorted(LABEL_TO_CLASS)} clips only.")
    return clips


# ---------------------------------------------------------------- augmentation

def augment(clip, rng):
    """One correlated variant of a clip. Train side only; it multiplies rows, not information.

    Mirroring is deliberately absent although the spec lists it. These landmarks are already
    canonicalized onto the right-hand convention, so a mirrored copy is a hand the runtime
    would mirror straight back before featurising -- the variant could never occur at
    inference. Left-handed robustness comes from the chirality flip in features.py, not from
    augmenting past it.

    Boundary extension is likewise absent: a prompted take contains only the gesture, so
    there are no frames outside the event to extend into. Only truncation is available, and
    it is what trains the model against the segmenter cutting a fraction of a second early.
    """
    times, P = clip.times.copy(), clip.P.copy()

    trim_s = rng.uniform(0.0, 0.1)
    if trim_s > 0:
        end = 1 if rng.random() < 0.5 else -1
        keep = times <= times[-1] - trim_s if end > 0 else times >= times[0] + trim_s
        if keep.sum() >= 3:
            times, P = times[keep], P[keep]

    times = (times - times[0]) * rng.uniform(0.75, 1.25)   # tempo

    # Amplitude scales the palm-centre TRAJECTORY, not the landmarks. Scaling the landmarks
    # about a global centroid would scale the palm as well, and every feature here is in palm
    # units, so it would change nothing at all.
    a = rng.uniform(0.8, 1.2)
    m = F.palm_centre(P)
    P = P + ((a - 1.0) * (m - m[0]))[:, None, :]

    # 0.01 palm of landmark jitter is the design spec's figure, and it is an assumption, not a
    # measurement: nothing in this repository has yet measured MediaPipe's per-landmark noise
    # on a still hand. It is far below SHAPE_STABLE, so it cannot move a gate either way.
    sigma = 0.01 * F.palm_scale(P)[:, None, None]
    P = P + rng.normal(0.0, 1.0, P.shape) * sigma
    return times, P


def rows_from(clips, n_aug=0, seed=0):
    """Materialise feature rows from clips. Called once per side of a split, never before it."""
    rows = []
    for c in clips:
        rows.append(Row(F.event_features(c.times, c.P, c.arm), c.label, c.clip_id, False))
        rng = np.random.default_rng(seed + c.clip_id)
        for _ in range(n_aug):
            t, P = augment(c, rng)
            rows.append(Row(F.event_features(t, P, c.arm), c.label, c.clip_id, True))
    return rows


def matrix(rows):
    return np.stack([r.x for r in rows]), np.array([r.y for r in rows])


# ---------------------------------------------------------------- reporting

def models():
    """The shipped forest plus two diagnostics fitted on the identical matrix.

    Logistic regression is the floor: if the forest beats a linear model by a wide margin on
    held-out clips the extra capacity is buying something, and if it does not, the geometry
    was already linearly separable and the forest is decoration. ExtraTrees answers the
    narrower question of whether *this* forest matters or any ensemble of trees would do.
    """
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               min_samples_leaf=2, random_state=0),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                           min_samples_leaf=2, random_state=0),
    }


def cross_validate(clips, n_splits, n_aug):
    """GroupKFold over clip ids, with rows built inside each fold. Returns pooled predictions."""
    ids = np.array([c.clip_id for c in clips])
    k = min(n_splits, len(clips))
    if k < 2:
        print("fewer than 2 clips: no cross-validation is possible")
        return None

    folds = {name: [] for name in models()}
    pooled = {"y": [], "p": [], "classes": None, "clip": []}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(ids, groups=ids)):
        train_clips = [clips[i] for i in tr]
        test_clips = [clips[i] for i in te]

        train_ids = {c.clip_id for c in train_clips}
        test_ids = {c.clip_id for c in test_clips}
        assert train_ids & test_ids == set(), \
            f"fold {fold}: clip(s) {sorted(train_ids & test_ids)} on both sides of the split"

        tr_rows = rows_from(train_clips, n_aug=n_aug)
        te_rows = rows_from(test_clips, n_aug=0)
        assert not any(r.augmented for r in te_rows), "augmented row on the test side"
        assert {r.clip_id for r in tr_rows} & {r.clip_id for r in te_rows} == set()

        Xtr, ytr = matrix(tr_rows)
        Xte, yte = matrix(te_rows)
        if len(set(ytr)) < 2:
            continue
        for name, mdl in models().items():
            m = mdl.fit(Xtr, ytr)
            acc = float((m.predict(Xte) == yte).mean())
            folds[name].append((acc, len(test_clips)))
            if name == "RandomForest":
                pooled["y"].append(yte)
                pooled["p"].append(m.predict_proba(Xte))
                pooled["clip"].append(np.array([r.clip_id for r in te_rows]))
                pooled["classes"] = list(m.classes_)

    if pooled["classes"] is None:
        print("no fold contained more than one class; cross-validation says nothing")
        return None

    print(f"\n=== GroupKFold({k}) over clip ids, rows built inside each fold ===")
    print(f"{'model':>20}  {'mean acc':>9}  {'fold accs':>9}")
    for name, res in folds.items():
        accs = [a for a, _ in res]
        detail = " ".join(f"{a:.3f}" for a in accs)
        print(f"{name:>20}  {np.mean(accs):>9.3f}  {detail}")
    print(f"n_independent_clips = {len(clips)} (test clips per fold: "
          f"{[n for _, n in folds['RandomForest']]}); every accuracy above is over "
          f"un-augmented held-out clips only")
    return {"y": np.concatenate(pooled["y"]), "p": np.concatenate(pooled["p"]),
            "classes": pooled["classes"], "n_clips": len(clips)}


def report_pooled(pooled):
    y, p, classes = pooled["y"], pooled["p"], pooled["classes"]
    pred = np.array(classes)[p.argmax(axis=1)]

    print(f"\n=== pooled out-of-fold predictions, n_independent_clips = {pooled['n_clips']} ===")
    print("confusion (rows true, cols predicted):")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in classes))
    for c in classes:
        row = [(int(((y == c) & (pred == d)).sum())) for d in classes]
        print(f"{c:>8}" + "".join(f"{v:>8}" for v in row))

    print(f"\n{'class':>8}  {'prec':>6}  {'recall':>6}  {'support':>8}")
    for c in classes:
        tp = int(((y == c) & (pred == c)).sum())
        prec = tp / max(int((pred == c).sum()), 1)
        rec = tp / max(int((y == c).sum()), 1)
        print(f"{c:>8}  {prec:>6.3f}  {rec:>6.3f}  {int((y == c).sum()):>8}")

    # The state machine does not emit the argmax; it emits only when P_EMIT and MARGIN are both
    # met. An accuracy measured on the argmax therefore describes a decision rule the runtime
    # never uses, so the operating point the runtime does use is reported beside it.
    order = np.sort(p, axis=1)
    fires = (order[:, -1] >= DEFAULT.P_EMIT) & (order[:, -1] - order[:, -2] >= DEFAULT.MARGIN)
    print(f"\nat the runtime operating point P_EMIT={DEFAULT.P_EMIT} MARGIN={DEFAULT.MARGIN}:")
    print(f"  abstention rate {1 - fires.mean():.3f} over {len(y)} held-out events")
    if fires.any():
        print(f"  accuracy when it does fire {float((pred[fires] == y[fires]).mean()):.3f}"
              f"  (n_independent_clips = {pooled['n_clips']})")
    for c in classes:
        if c == "MOVE":
            continue
        false_fire = int(((pred == c) & fires & (y == "MOVE")).sum())
        print(f"  MOVE events emitted as {c}: {false_fire}")


def report_importances(X, y, top=15):
    """Global importances, then one-vs-rest importances per class.

    feature_importances_ on the 3-class forest says which features the model uses, not which
    class each one serves. Re-fitting the same forest one-vs-rest is the cheapest way to get
    that, and it is what makes the geometry-versus-tempo check per class rather than overall.
    """
    def show(title, imp):
        idx = np.argsort(imp)[::-1][:top]
        print(f"\n{title}")
        for r, i in enumerate(idx):
            print(f"  {r + 1:>2}. {NAMES[i]:<18} {imp[i]:.4f}   [{i}]")
        tempo = [NAMES[i] for i in idx[:5] if NAMES[i] in TEMPO]
        if tempo:
            print(f"  NOTE: {', '.join(tempo)} in the top 5 -- that is this signer's tempo, "
                  f"not the shape of the glyph. Treat the accuracy above with suspicion.")

    base = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  min_samples_leaf=2, random_state=0).fit(X, y)
    show(f"top-{top} features, 3-class forest", base.feature_importances_)
    for c in base.classes_:
        ovr = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     min_samples_leaf=2, random_state=0)
        ovr.fit(X, (y == c).astype(int))
        show(f"top-{top} features, {c} vs rest", ovr.feature_importances_)


def report_runtime_reachability(clips):
    """How many recorded clips the segmenter's pre-model vetoes would have thrown away.

    A J the runtime vetoes before the classifier ever sees it is a J the system cannot
    recognise no matter how good this model is, so the count belongs in the training report
    and not in a separate evaluation nobody runs.
    """
    th = DEFAULT
    print("\n=== clips the segmenter's pre-model vetoes would reject ===")
    print(f"{'class':>8}  {'n':>4}  {'dur':>6}  {'len':>6}  {'straight':>9}  {'any':>5}")
    for c in sorted({c.label for c in clips}):
        sub = [k for k in clips if k.label == c]
        bad_t = bad_l = bad_s = bad_any = 0
        for k in sub:
            dur = float(k.times[-1] - k.times[0])
            S = float(np.median(F.palm_scale(k.P)))
            tip = k.P[:, F.TIP_FOR_ARM[k.arm], :]
            L = F.path_length(tip) / S
            straight = float(np.linalg.norm(tip[-1] - tip[0])) / S / L if L > 1e-9 else 1.0
            t_bad = not (th.T_MIN <= dur <= th.T_MAX)
            l_bad = not (th.L_MIN <= L <= th.L_MAX)
            s_bad = straight > th.STRAIGHT_VETO
            bad_t += t_bad
            bad_l += l_bad
            bad_s += s_bad
            bad_any += t_bad or l_bad or s_bad
        print(f"{c:>8}  {len(sub):>4}  {bad_t:>6}  {bad_l:>6}  {bad_s:>9}  {bad_any:>5}")
    print("For J and Z these are unreachable gestures, not rejected noise: a whole-clip event "
          f"longer than T_MAX={th.T_MAX}s can never be scored at runtime. For MOVE they are "
          "the vetoes doing their job.")


# ---------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--aspect", default=None,
                    help="WxH the clips were recorded at, for recordings that do not carry it. "
                         "Only the ratio is used and it is load-bearing (16:9 vs 4:3 skews "
                         "every diagonal), so there is no default")
    ap.add_argument("--handedness", default="Right",
                    help="MediaPipe's label for the signing hand, if the recording lacks it")
    ap.add_argument("--cv-session", default="S1",
                    help="the session cross-validation is confined to; the shipped model is "
                         "still fitted on every clip")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thresholds", default=None,
                    help="a Thresholds json from calibrate.py. Events MUST be cut with the same "
                         "thresholds the runtime will use: cutting training events with the "
                         "defaults while the live demo runs calibrated ones reintroduces exactly "
                         "the train/serve skew this pipeline exists to avoid.")
    ap.add_argument("--whole-clips", action="store_true",
                    help="train on whole recorded clips instead of segmenter-cut events. "
                         "Almost always wrong: recorded clips are longer than T_MAX, so the "
                         "runtime can never produce an example shaped like them.")
    ap.add_argument("--aug", type=int, default=4,
                    help="augmented copies per training clip; 0 disables augmentation")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"no recording at {args.data}")
        print("Nothing to train on, and nothing is invented to stand in for it. Record first:")
        print("    ./.venv/bin/python temporal/collect_motion.py --camera 0")
        print("which writes motion_clips.npz with J, Z and NONE clips, then re-run this.")
        return

    aspect = tuple(int(v) for v in args.aspect.lower().split("x")) if args.aspect else None
    if args.whole_clips:
        print("WARNING: training on whole clips. Their durations exceed T_MAX, so the runtime "
              "cannot produce events of this shape. Use this only to diagnose.\n")
        clips = load_clips(args.data, aspect, args.handedness)
    else:
        th = Thresholds.from_json(args.thresholds) if args.thresholds else DEFAULT
        if args.thresholds:
            print(f"cutting events with thresholds from {args.thresholds}")
        clips = load_clips_cut(args.data, aspect, args.handedness, th)
    if not clips:
        print(f"{args.data} contains no usable clip")
        return

    counts = {c: sum(k.label == c for k in clips) for c in sorted({k.label for k in clips})}
    print(f"\n{len(clips)} independent clips {counts}")
    arms = {c: sum(k.label == c and k.arm == "J" for k in clips) for c in counts}
    print(f"J-armed clips per class: {arms}  (the rest are Z-armed)")
    by_session = {s: sum(k.session == s for k in clips)
                  for s in sorted({k.session for k in clips if k.session})}
    if by_session:
        print(f"sessions: {by_session}")
    if len(counts) < 2:
        print("only one class present; a classifier over one class is not a classifier")
        return

    report_runtime_reachability(clips)

    # Pooling two sittings into one GroupKFold produces a number that is neither the
    # within-session one nor the generalization one, and it is the larger of the two, so it is
    # the one that would get quoted. Confine it to the development session and say so.
    dev = [k for k in clips if k.session == _norm_session(args.cv_session)]
    if len(dev) >= 2 and len({k.label for k in dev}) >= 2:
        print(f"\ncross-validating on session {_norm_session(args.cv_session)} only: "
              f"{len(dev)} of {len(clips)} clips")
    else:
        dev = clips
        print(f"\ncross-validating over all {len(clips)} clips: no usable "
              f"{_norm_session(args.cv_session)} subset in this recording")
    print("This is a within-session number and is not the headline. The generalization number "
          "is leave-one-session-out and evaluate.py owns it.")

    pooled = cross_validate(dev, args.folds, args.aug)
    if pooled is not None:
        report_pooled(pooled)

    # The shipped model is fitted on every clip. The cross-validation above exists to
    # characterise the representation, not to select this model, and its accuracy is not a
    # property of the pickle written here.
    rows = rows_from(clips, n_aug=args.aug)
    X, y = matrix(rows)
    report_importances(X, y)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   min_samples_leaf=2, random_state=0).fit(X, y)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": model, "classes": list(model.classes_),
                     "feature": "event/v1", "n_train": len(X)}, fh)
    print(f"\nwrote {args.out}: {len(X)} rows from {len(clips)} independent clips "
          f"({args.aug} augmented copies per clip), classes {list(model.classes_)}, "
          f"feature event/v1")
    print("n_train counts rows. n_independent_clips is "
          f"{len(clips)}, and that is the number any capacity claim has to quote.")


if __name__ == "__main__":
    main()

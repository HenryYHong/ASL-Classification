"""Cut recorded clips into training events by replaying the segmenter over them.

This module exists to prevent one specific, fatal mistake: training the motion classifier on
whole recorded clips.

A recorded clip is 2.0 s and contains a lead-in, the gesture, and a settle. The runtime never
sees anything like that -- Segmenter cuts an event from motion onset to motion offset, which is
shorter, and rejects anything longer than T_MAX (2.10 s) outright. Train on whole clips and
every training example is a shape the deployed system can never produce. The model would score
well in cross-validation and recognize nothing in front of a camera, which is precisely the
failure this repository already documents for Approach A.

So the boundaries come from the same code path either way. label_events replays each clip
through a real Segmenter and harvests the spans it cuts, via the on_event hook.

The diagnostic this prints is the important output, more than the file it writes: for every
J and Z clip it reports whether the segmenter found exactly one event. A clip that yields zero
events is one the runtime would also have missed -- the gate never armed, or the motion never
crossed the trigger. That is worth knowing before training, not after.

    ../.venv/bin/python temporal/label_events.py
    ../.venv/bin/python temporal/label_events.py --clips other.npz --out other_events.npz

--out is not optional on the second line, and this file refuses to run without it: see
default_out_refusal().
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from segmenter import Segmenter
from thresholds import Thresholds, DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLIPS = os.path.join(HERE, "motion_clips.npz")
DEFAULT_OUT = os.path.join(HERE, "events.npz")


def same_path(a, b):
    """True when two paths name the same file, whatever spelling they arrived in."""
    return os.path.realpath(os.path.abspath(str(a))) == os.path.realpath(os.path.abspath(str(b)))


def default_out_refusal(in_path, out_path, default_in=None, default_out=None,
                        in_flag="--clips", out_flag="--out", suffix="_events"):
    """The refusal message for a run that would write the committed default over foreign input.

    Returns None when the run is safe, and the text to print when it is not.

    The hazard is one line of argparse: `--out` defaults to a committed artifact, so a command
    that changes only the INPUT still writes the DEFAULT OUTPUT. `label_events.py --clips
    third_party.npz` replaces temporal/events.npz -- the motion training set every later run
    trains on -- with the events cut from somebody else's footage, prints the path it wrote as
    if that were routine, and exits 0. This was not hypothetical: it happened here while
    testing an ingest, and the committed file had to be restored from HEAD.

    The rule is narrow on purpose. Default input with default output is the ordinary retrain
    and stays allowed; an explicit `--out`, wherever it points, is a decision someone typed and
    stays allowed. Only the mismatch -- foreign input, default output -- is refused, because
    that is the one combination nobody means.
    """
    default_in = DEFAULT_CLIPS if default_in is None else default_in
    default_out = DEFAULT_OUT if default_out is None else default_out
    if same_path(in_path, default_in) or not same_path(out_path, default_out):
        return None
    suggested = os.path.splitext(str(in_path))[0] + suffix + os.path.splitext(str(default_out))[1]
    return (f"refusing to write {default_out}\n"
            f"{os.path.basename(default_out)} is the committed default, and {in_flag} points at "
            f"{in_path} instead of\n{default_in}, so what this run would write is not what that "
            f"file holds.\nPass {out_flag} explicitly, for instance:\n"
            f"    {out_flag} {suggested}\n"
            f"Nothing has been read or written.")

#: Recorder label -> event class. NONE clips are negatives; whatever the segmenter cuts out of
#: them is, by construction, motion that is not a letter.
LABEL_TO_CLASS = {"J": "J", "Z": "Z", "NONE": "MOVE"}

#: Group ids partition the split, so they must never collide across recordings. Every event
#: takes its recording's index times GROUP_STRIDE plus, for a prompted take, the item index its
#: onset fell in (ORPHAN_GID when it fell in no item). A per-clip recording is one group of its
#: own (item 0). Per-clip ids used to be the bare index, so take 0's items 1..N shared ids with
#: clips 1..N in a file that mixed the two recording modes and GroupKFold treated independent
#: recordings as one group.
GROUP_STRIDE, ORPHAN_GID = 1000, 999


def group_id(recording_index, item_index=0):
    """The clip_id (GroupKFold group) of an event from recording `recording_index`."""
    if not (0 <= item_index < GROUP_STRIDE):
        raise ValueError(f"item index {item_index} does not fit under GROUP_STRIDE={GROUP_STRIDE}")
    return int(recording_index) * GROUP_STRIDE + int(item_index)


def modal_handedness(labels, default="Right"):
    """Reduce the recorder's per-frame labels to one modal real label.

    features.canonicalize_handedness takes a scalar and now raises on a sequence, because
    passing the array used to silently leave mirrored clips unflipped.
    """
    real = [str(x) for x in np.atleast_1d(labels)
            if str(x) not in ("None", "Unknown", "nan", "")]
    return Counter(real).most_common(1)[0][0] if real else default


def frame_sizes(npz, n_clips, override=None):
    """Per-clip (W,H). Returns a list of n_clips pairs, or None if unknown.

    A recording made by collect_motion.py carries one size for the whole file; footage ingested
    from third parties carries one per clip, because it is not one camera. Collapsing the latter
    to the first entry would apply one video's aspect ratio to all of them, and u = x*(W/H) is
    load-bearing -- skipping or mis-applying it triples the Z-gate false-positive rate on this
    repo's own archive.
    """
    if override:
        return [tuple(override)] * n_clips
    for key in ("frame_size", "frame_wh", "wh"):
        if key in npz:
            wh = np.asarray(npz[key]).reshape(-1, 2)
            if len(wh) == n_clips:
                return [(int(a), int(b)) for a, b in wh]
            return [(int(wh[0][0]), int(wh[0][1]))] * n_clips
    return None


def harvest(clip, stamps, handed, th, width, height):
    """Replay one clip and return every motion span the segmenter cuts from it.

    Fed one modal handedness label per recording. The runtime is fed MediaPipe's per-frame
    label, and the Segmenter's HANDEDNESS LATCH (segmenter.py) is what makes the two agree:
    on the committed takes the raw per-frame label lost 6 of 113 gestures to single-frame
    flips that this modal label never saw, and with the latch the per-frame replay credits
    the same 85 items this cut does.
    """
    spans = []
    seg = Segmenter(th, on_event=lambda ev: spans.append(ev))
    lr = modal_handedness(handed)
    clip = np.asarray(clip, dtype=np.float64)
    for i in range(len(clip)):
        lm = clip[i]
        ok = np.all(np.isfinite(lm[:, :2]))
        seg.step(float(stamps[i]), lm if ok else None, lr, width, height)
    # Recorded footage ends; a live stream does not. Score anything still in flight so a clip
    # trimmed tight to the end of the gesture is not silently discarded.
    seg.flush(float(stamps[-1]))
    return spans


#: What to do with every span that is NOT an item's credited gesture.
#:   "drop_unreachable"  (default) spans the runtime could never score -- aborted tracks and
#:           spans the T/L/straightness vetoes reject -- are discarded; every reachable
#:           non-credited span (a false start inside an item, a rest-phase move) is MOVE.
#:           GroupKFold(5) on the S1 takes: J+Z recall at the operating point 0.940 with
#:           2/27 false fires on reachable MOVE, against 0.910 and 2/27 when the unreachable
#:           spans are kept as MOVE -- an aborted J labeled MOVE teaches the forest that a
#:           J-shaped path can be nothing, and the runtime never shows it one.
#:   "move"  every other span is a MOVE example (fragments, rest-phase, aborted, veto-failing)
#:   "drop"  in-item fragments of a J/Z item are discarded as well; they are runtime-reachable,
#:           so this trains against fewer of the negatives the runtime actually presents. A
#:           near-miss (NONE) item's reachable spans are its negatives and stay MOVE under
#:           every policy: they are the "moved, but did not sign the letter" footage the
#:           recorder asks for (collect_motion.py --negatives), not fragments of a gesture.
OTHER_POLICIES = ("move", "drop", "drop_unreachable")


def span_geometry(sp, th):
    """Duration, tip path and straightness of a span, plus whether _score would reach the
    classifier with it (scored, T/L/straightness vetoes all passed)."""
    P, times, arm = sp["P"], sp["times"], sp["arm"]
    dur = float(times[-1] - times[0])
    tip = P[:, F.TIP_FOR_ARM[arm], :]
    S = float(np.median(F.palm_scale(P)))
    L = F.path_length(tip) / S
    net = float(np.linalg.norm(tip[-1] - tip[0])) / S
    straight = net / L if L > 1e-9 else 1.0
    reachable = (str(sp.get("reason", "scored")).startswith("scored")
                 and th.T_MIN <= dur <= th.T_MAX
                 and th.L_MIN <= L <= th.L_MAX
                 and straight <= th.STRAIGHT_VETO)
    return dur, L, straight, reachable


def assign_continuous(spans, items, th, other="drop_unreachable"):
    """Label the spans cut from one prompted take, ONE credited gesture per item.

    A prompted J item routinely yields two or three spans -- a false start, the gesture, a
    settle -- and labeling every span whose onset falls in the item as 'J' taught the
    classifier that a 1.7-palm fragment is a J, which is what made the runtime emit "JJ".
    The credited span is the longest (tip path) runtime-reachable span whose onset lies in
    the item's park/go window and whose armed gate matches the letter. Every other span is
    handled by `other` (see OTHER_POLICIES).

    Returns a list of dicts: span, label, gid (the item index, one GROUP per item; 999 when
    the onset lies in no item), credited, reachable. Rows with label None are dropped.
    """
    if other not in OTHER_POLICIES:
        raise ValueError(f"other={other!r}; expected one of {OTHER_POLICIES}")
    rows = []
    for sp in spans:
        t0 = float(sp["times"][0])
        it = next((I for I in items if I["park"][0] <= t0 < I["rest"][1]), None)
        dur, L, straight, reachable = span_geometry(sp, th)
        if it is None:
            phase, gid, letter = "none", ORPHAN_GID, None
        elif t0 >= it["rest"][0]:
            phase, gid, letter = "rest", it["index"], None
        else:
            phase, gid = "item", it["index"]
            letter = it["label"] if it["label"] in ("J", "Z") else None
        rows.append({"span": sp, "gid": gid, "phase": phase, "letter": letter, "L": L,
                     "dur": dur, "reachable": reachable, "credited": False, "label": "MOVE"})
    best = {}
    for k, r in enumerate(rows):
        if r["letter"] and r["reachable"] and r["span"]["arm"] == r["letter"]:
            if r["gid"] not in best or r["L"] > rows[best[r["gid"]]]["L"]:
                best[r["gid"]] = k
    for k in best.values():
        rows[k]["credited"] = True
        rows[k]["label"] = rows[k]["letter"]
    out = []
    for r in rows:
        if not r["credited"]:
            # A fragment is a non-credited span of a J/Z item (r["letter"] set); a NONE item's
            # spans have no letter to be a fragment of and are kept as MOVE.
            if other == "drop" and ((r["phase"] == "item" and r["letter"]) or not r["reachable"]):
                continue
            if other == "drop_unreachable" and not r["reachable"]:
                continue
        out.append(r)
    return out


def clip_as_item(label, stamps):
    """A per-clip recording (SPACE-driven collect_motion.py, or ingested footage) as a
    one-item schedule covering the whole clip, so the per-clip branch labels through
    assign_continuous exactly as a prompted take does: one credited runtime-reachable gesture
    per clip under the matching gate, every other reachable span MOVE, aborted and
    veto-failing spans dropped by the default policy, and `--other` honored. A NONE clip is
    a near-miss item (letter None), so all of its reachable spans are MOVE. Before this the
    per-clip branch kept every span harvest() returned -- abort:GAP / abort:T_MAX /
    RIGID_VETO tracks and veto-failing spans included -- and labeled each J-armed one of a J
    clip 'J', the population the CONTINUOUS relabel removed from the prompted takes.
    """
    t0, t1 = float(stamps[0]) - 1.0, float(stamps[-1]) + 1.0
    return [{"label": label if label in ("J", "Z") else "NONE", "index": 0,
             "park": [t0, t0], "go": [t0, t1], "rest": [t1, t1]}]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"where the events go (default {DEFAULT_OUT}). REQUIRED whenever "
                         "--clips is not the committed recording: the default is the committed "
                         "training set and this refuses to overwrite it with foreign events")
    ap.add_argument("--thresholds", default=None, help="a Thresholds json from calibrate.py")
    ap.add_argument("--aspect", nargs=2, type=int, metavar=("W", "H"), default=None)
    ap.add_argument("--other", default="drop_unreachable", choices=OTHER_POLICIES,
                    help="what becomes of a prompted item's or a recorded clip's non-credited "
                         "spans")
    args = ap.parse_args()

    # Before anything is read, and before the long MediaPipe-free but still slow replay: a run
    # that would replace the committed events.npz with events cut from other footage stops here.
    refusal = default_out_refusal(args.clips, args.out)
    if refusal:
        print(refusal)
        return 2

    if not os.path.exists(args.clips):
        print(f"no recording at {args.clips}")
        print("Nothing to cut, and nothing is invented to stand in for it. Record first:")
        print("    ./.venv/bin/python temporal/collect_motion.py --camera 0")
        return 0

    npz = np.load(args.clips, allow_pickle=True)
    clips, stamps, labels = list(npz["clips"]), list(npz["stamps"]), list(npz["labels"])
    handed = list(npz["handed"]) if "handed" in npz else [None] * len(clips)
    prompts = list(npz["prompts"]) if "prompts" in npz else [""] * len(clips)
    sessions = list(npz["sessions"]) if "sessions" in npz else ["S1"] * len(clips)
    signers = list(npz["signers"]) if "signers" in npz else ["signer1"] * len(clips)

    wh = frame_sizes(npz, len(clips), args.aspect)
    if wh is None:
        print("the recording carries no frame size and --aspect was not given.")
        print("Aspect correction is load-bearing: without it the Z-gate's false-positive rate")
        print("triples (118 -> 367 frames on the archive). Refusing to guess.")
        return 1
    th = Thresholds.from_json(args.thresholds) if args.thresholds else DEFAULT
    distinct = sorted(set(wh))
    shown = f"{wh[0][0]}x{wh[0][1]}" if len(distinct) == 1 else f"{len(distinct)} distinct sizes"
    print(f"{len(clips)} clips, frame {shown}, thresholds "
          f"{'from ' + args.thresholds if args.thresholds else 'defaults'}\n")

    X, y, clip_ids, sess_out, signer_out = [], [], [], [], []
    per_label = {}
    per_take_frag = {}
    for i, (clip, ts, lab) in enumerate(zip(clips, stamps, labels)):
        lab = str(lab).strip().upper()

        if lab == "CONTINUOUS":
            # One unbroken take, prompted on a rhythm. Every event is assigned to whichever
            # prompted item its ONSET falls inside: an event beginning in a GO window is that
            # item's letter, and one beginning in a REST gap is a genuine negative -- the hand
            # was moving and no letter was being signed, which is precisely a MOVE example.
            items = json.loads(str(prompts[i])) if str(prompts[i]) else []
            spans = harvest(np.asarray(clip), np.asarray(ts), handed[i], th, wh[i][0], wh[i][1])
            rows = assign_continuous(spans, items, th, other=args.other)
            got = Counter(r["gid"] for r in rows if r["credited"])
            # Counted on the unfiltered view (other="move" keeps every span), so the number
            # says what the policy cut rather than what survived it.
            per_take_frag[i] = sum(1 for r in assign_continuous(spans, items, th, other="move")
                                   if r["phase"] == "item" and r["letter"] and not r["credited"])
            for r in rows:
                sp = r["span"]
                try:
                    feat = F.event_features(sp["times"], sp["P"], sp["arm"])
                except ValueError:
                    continue
                X.append(feat); y.append(r["label"])
                clip_ids.append(group_id(i, r["gid"]))
                sess_out.append(str(sessions[i])); signer_out.append(str(signers[i]))
            for I in items:
                if I["label"] in ("J", "Z"):
                    per_label.setdefault(I["label"], Counter())[got.get(I["index"], 0)] += 1
            continue

        cls = LABEL_TO_CLASS.get(lab)
        if cls is None:
            continue
        # One recorded clip is one prompted item spanning the whole clip (clip_as_item), so
        # the same rule labels it: one credited reachable gesture under the matching gate, a
        # span armed under the other gate is MOVE (at runtime it would be scored under that
        # arm, so the classifier had better have seen it), the policy handles the rest.
        spans = harvest(np.asarray(clip), np.asarray(ts), handed[i], th, wh[i][0], wh[i][1])
        rows = assign_continuous(spans, clip_as_item(lab, ts), th, other=args.other)
        rec = per_label.setdefault(lab, Counter())
        rec[sum(r["credited"] for r in rows) if cls in ("J", "Z") else len(rows)] += 1

        for r in rows:
            sp = r["span"]
            try:
                feat = F.event_features(sp["times"], sp["P"], sp["arm"])
            except ValueError:
                continue
            X.append(feat)
            y.append(r["label"])
            clip_ids.append(group_id(i))
            sess_out.append(str(sessions[i]))
            signer_out.append(str(signers[i]))

    print("events cut per clip, by recorded label (J/Z: credited gestures per item or clip; "
          "NONE: MOVE spans kept per clip):")
    for lab in sorted(per_label):
        dist = dict(sorted(per_label[lab].items()))
        n = sum(per_label[lab].values())
        clean = per_label[lab].get(1, 0)
        note = ""
        if lab in ("J", "Z"):
            missed = per_label[lab].get(0, 0)
            note = f"   {clean}/{n} credited with one event"
            if missed:
                note += f"; {missed} gave NONE reachable -- the runtime would have missed those too"
        print(f"  {lab:5s} {dist}{note}")
    if per_take_frag:
        print(f"non-credited in-item spans (fragments) per take: {per_take_frag} "
              f"-> policy --other={args.other}")

    if not X:
        print("\nno events were cut from any clip.")
        print("Every recorded gesture failed to arm a gate or to cross the motion trigger.")
        print("Run calibrate.py --live first: the defaults come from a 1080p 15 fps archive")
        print("and may not fit this camera.")
        return 1

    X = np.stack(X)
    np.savez_compressed(args.out, X=X, y=np.array(y), clip_id=np.array(clip_ids),
                        session=np.array(sess_out), signer=np.array(signer_out),
                        feature=np.array("event/v1"))
    print(f"\nwrote {args.out}: {X.shape[0]} events x {X.shape[1]}-D from "
          f"GROUPS={len(set(clip_ids))} independent prompted items {dict(Counter(y))}")
    print("clip_id is stored so the split can partition by clip; never split these rows randomly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

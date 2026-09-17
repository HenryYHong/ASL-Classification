"""The idle-hand gate: a relaxed hand must not be read as a letter.

The vote floor in thresholds.py (VOTE_PROB_FLOOR) is set against this file, so it has to be a
committed measurement rather than a remembered one. It replays the clean idle holds through
the static vote rule with a given forest and counts the holds and votes that would emit.

The holds come from docs/browser_log.jsonl, one live session of the author on the hosted
page (MediaPipe Tasks landmarks, the page's own segmenter), which is gitignored because the
dev server appends to it. Consecutive hold records with gaps <= 0.2 s (in file order) form
148 holds; the 79 the page emitted a letter on are out, and so are the 26 that overlap
[track start - 2.5 s, track end + 1 s] of a J/Z track record -- those are the I/D launch
poses, read correctly at mean probabilities up to 1.00 and cancelled by D_WAIT or by the
motion emission, so counting them would call a correct read of an I a false emission. The
remaining 43 holds (2,180 records; 2,058 sliding four-record votes plus 6 holds shorter than
four records that are one vote each, 2,064 votes) are the clean idle set: a relaxed hand the
page was right to stay silent on. `--extract` writes those 43 holds' landmarks to
temporal/idle_holds.npz (366 KB), which is what the gate reads, so the measurement survives
the log being rotated and can be re-run from a clean checkout.

Rule, from thresholds.py: any clean idle hold that emits raises VOTE_PROB_FLOOR by 0.05 and
the gate is re-run. History: the one-signer forest at 0.50 emitted on 1 of 43 (max mean
probability 0.509), so it shipped at 0.55. The multi-signer forest reads a relaxed hand as a
loose G at 0.55-0.70 (4 of 43 holds, 40 votes, all G), so it ships at 0.75 -- 0 of 43 with
0.07 of headroom above its most confident idle vote (0.68).

    ./.venv/bin/python temporal/idle_gate.py                          # the committed forest
    ./.venv/bin/python temporal/idle_gate.py --model some_forest.p     # a candidate
    ./.venv/bin/python temporal/idle_gate.py --sweep                   # holds emitting per floor
    ./.venv/bin/python temporal/idle_gate.py --extract                 # rebuild idle_holds.npz
"""
import argparse
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "docs", "browser_log.jsonl")
HOLDS = os.path.join(HERE, "idle_holds.npz")
MODEL = os.path.join(HERE, "model_static.p")
GAP, BEFORE, AFTER = 0.2, 2.5, 1.0


def clean_idle_holds(log=LOG):
    holds, tracks = [], []
    with open(log) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            (holds if r.get("kind") == "hold" else tracks).append(r)
    groups, cur = [], [holds[0]]
    for r in holds[1:]:
        d = r["t"] - cur[-1]["t"]
        if 0 <= d <= GAP:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    spans = [(t["times"][0], t["t"]) for t in tracks]
    emitted = [g for g in groups if any(r.get("why") == "emitted" for r in g)]
    launch, clean = [], []
    for g in groups:
        if g in emitted:
            continue
        t0, t1 = g[0]["t"], g[-1]["t"]
        (launch if any(s0 - BEFORE <= t1 and t0 <= s1 + AFTER for s0, s1 in spans) else clean).append(g)
    return groups, emitted, launch, clean


def extract(log=LOG, out=HOLDS):
    groups, emitted, launch, clean = clean_idle_holds(log)
    P = np.concatenate([np.asarray([r["P"] for r in g], dtype=np.float32) for g in clean])
    hold = np.concatenate([np.full(len(g), i) for i, g in enumerate(clean)])
    t = np.concatenate([np.asarray([r["t"] for r in g], dtype=np.float32) for g in clean])
    np.savez(out, P=P, hold=hold, t=t,
             note=f"{len(clean)} clean idle holds of {len(groups)} ({len(emitted)} page-emitted and "
                  f"{len(launch)} launch-adjacent excluded); P is canonical isotropic (21,2) per record")
    print(f"{len(groups)} holds: {len(emitted)} page-emitted, {len(launch)} launch-adjacent, {len(clean)} clean "
          f"({len(P)} records) -> {out} ({os.path.getsize(out) / 1e3:.0f} KB)")


def load_holds(path=HOLDS):
    d = np.load(path)
    P, hold = d["P"].astype(np.float64), d["hold"]
    return [P[hold == h] for h in np.unique(hold)]


def vote(probs, th):
    """The static vote on one window of per-frame posteriors: (emits, mean winner p, winner)."""
    idx = probs.argmax(1)
    vals, cnt = np.unique(idx, return_counts=True)
    win = vals[cnt.argmax()]
    agree = cnt.max() / len(idx)
    mean = probs.mean(0)
    meanp = mean[win]
    margin = meanp - np.sort(mean)[-2]
    ok = (agree >= th.VOTE_AGREE and margin >= th.VOTE_MARGIN
          and (meanp >= th.VOTE_PROB or (margin >= th.VOTE_MARGIN_CLEAR and meanp >= th.VOTE_PROB_FLOOR)))
    return ok, float(meanp), int(win)


def gate(model, featfn, classes, th=DEFAULT, holds=None, votes=DEFAULT.VOTE_MIN):
    """-> (holds emitting, holds, votes emitting, votes, max mean winner p, letters emitted)"""
    holds = load_holds() if holds is None else holds
    eh = ev = nv = 0
    maxp = 0.0
    letters = {}
    for P in holds:
        pr = model.predict_proba(featfn(P))
        wins = [vote(pr[i:i + votes], th) for i in range(len(P) - votes + 1)] if len(P) >= votes else [vote(pr, th)]
        nv += len(wins)
        em = [w for w in wins if w[0]]
        ev += len(em)
        eh += bool(em)
        maxp = max(maxp, max(w[1] for w in wins))
        for w in em:
            letters[classes[w[2]]] = letters.get(classes[w[2]], 0) + 1
    return eh, len(holds), ev, nv, maxp, letters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--extract", action="store_true", help="rebuild idle_holds.npz from docs/browser_log.jsonl")
    ap.add_argument("--sweep", action="store_true", help="holds emitting at floors 0.50 .. 0.85")
    args = ap.parse_args()
    if args.extract:
        extract()
        return
    blob = pickle.load(open(args.model, "rb"))
    featfn, dim = F.static_feature_for(blob["feature"])
    classes = list(blob["classes"])
    holds = load_holds()
    eh, nh, ev, nv, maxp, letters = gate(blob["model"], featfn, classes, holds=holds)
    verdict = "PASS" if eh == 0 else "FAIL: raise VOTE_PROB_FLOOR by 0.05 and re-run"
    print(f"{os.path.basename(args.model)} at VOTE_PROB_FLOOR {DEFAULT.VOTE_PROB_FLOOR} / VOTE_MARGIN_CLEAR "
          f"{DEFAULT.VOTE_MARGIN_CLEAR}: clean idle holds emitting {eh}/{nh}, votes {ev}/{nv}, "
          f"max mean winner probability {maxp:.3f}" + (f", letters {letters}" if letters else "") + f" -> {verdict}")
    if args.sweep:
        for floor in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            # the confident route (VOTE_PROB) admits any vote above it whatever the floor says,
            # so a floor above VOTE_PROB is only real if VOTE_PROB rises with it
            th = replace(DEFAULT, VOTE_PROB_FLOOR=floor, VOTE_PROB=max(DEFAULT.VOTE_PROB, floor))
            eh, nh, ev, nv, _, letters = gate(blob["model"], featfn, classes, th=th, holds=holds)
            print(f"  floor {floor:.2f}: {eh}/{nh} holds, {ev}/{nv} votes" + (f" {letters}" if letters else ""))


if __name__ == "__main__":
    main()

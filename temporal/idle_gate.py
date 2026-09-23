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

Rule, from thresholds.py: any clean idle hold that emits raises the EMITTING LETTER's floor
by 0.05 and the gate is re-run. History: the one-signer forest at 0.50 emitted on 1 of 43
(max mean probability 0.509), so it shipped at 0.55. The first multi-signer forest read a
relaxed hand as a loose G at 0.55-0.70 (4 of 43 holds, 40 votes, all G), so it shipped at
0.75 flat -- and charged every other letter for G's habit. The forest that ships now is
gated per letter (thresholds.VOTE_PROB_LETTER): its gated idle maxima are G 0.7087, O 0.4660,
A 0.4326, H 0.3585, M 0.3346 at seed 0, so G stands at 0.75 with 0.0413 of headroom and the
other 23 letters at 0.55 with 0.037 (idle_maxima and headroom below print both).

ONE PICKLE IS NOT A RECIPE. Until this release the gate only ever ran on the committed
model_static.p, and a forest's idle behavior turns out to move with the seed by more than the
margin it passes on. Re-measured here: the PREVIOUS release's recipe (the author's four
sessions uncapped + ASLNow + Ankara, no ASL-HG, RF 80 trees / min leaf 5) passes at seed 0
with 0 of 43 holds and a most-confident idle vote of 0.6950 -- the number thresholds.py quotes
-- and the SAME recipe at seed 2 emits: 1 hold, 1 vote, a G, at 0.7540 above the 0.75 floor
(--seeds 3 --author-cap 0 --no-aslhg). So the headroom that was credited to the floor was a
property of one draw. --seeds N trains the shipped recipe at each seed and gates every one;
a candidate that emits at ANY of them is disqualified, which is the rule train_static.py's
docstring now selects the forest by.

    ./.venv/bin/python temporal/idle_gate.py                          # the committed forest
    ./.venv/bin/python temporal/idle_gate.py --model some_forest.p     # a candidate
    ./.venv/bin/python temporal/idle_gate.py --sweep                   # holds emitting per floor
    ./.venv/bin/python temporal/idle_gate.py --seeds 3                 # the recipe, at 3 seeds
    ./.venv/bin/python temporal/idle_gate.py --seeds 3 --trees 60 --leaf 8   # a candidate recipe
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


def vote(probs, th, classes=None):
    """The static vote on one window of per-frame posteriors: (emits, mean winner p, winner).

    `classes` is the class list, needed only because the floor is per letter now
    (thresholds.VOTE_PROB_LETTER); without it the flat VOTE_PROB / VOTE_PROB_FLOOR apply,
    which is what a caller scoring something other than the 24 letters wants.
    """
    idx = probs.argmax(1)
    vals, cnt = np.unique(idx, return_counts=True)
    win = vals[cnt.argmax()]
    agree = cnt.max() / len(idx)
    mean = probs.mean(0)
    meanp = mean[win]
    margin = meanp - np.sort(mean)[-2]
    letter = classes[int(win)] if classes is not None else None
    prob_th = th.vote_prob_for(letter) if letter is not None else th.VOTE_PROB
    floor_th = th.vote_floor_for(letter) if letter is not None else th.VOTE_PROB_FLOOR
    ok = (agree >= th.VOTE_AGREE and margin >= th.VOTE_MARGIN
          and (meanp >= prob_th or (margin >= th.VOTE_MARGIN_CLEAR and meanp >= floor_th)))
    return ok, float(meanp), int(win)


def idle_maxima(model, featfn, classes, holds=None, th=DEFAULT, votes=DEFAULT.VOTE_MIN):
    """{letter: the most confident idle vote it wins that ALREADY clears agree and margin}.

    The headroom that matters. gate()'s `maxp` is the largest mean winner probability over
    every window, including windows the agreement or margin test throws out, so it overstates
    how close a relaxed hand came to emitting. This is the number the floor has to clear, and
    with a per-letter floor (thresholds.VOTE_PROB_LETTER) it has to be read per letter: on
    this forest a relaxed hand is a loose G at up to 0.71 and nothing else above 0.52.
    """
    holds = load_holds() if holds is None else holds
    out = {}
    for P in holds:
        pr = model.predict_proba(featfn(P))
        n = len(pr)
        spans = [(i, i + votes) for i in range(n - votes + 1)] if n >= votes else [(0, n)]
        for a, b in spans:
            w = pr[a:b]
            idx = w.argmax(1)
            vals, cnt = np.unique(idx, return_counts=True)
            win = int(vals[cnt.argmax()])
            mean = w.mean(0)
            meanp, margin = mean[win], mean[win] - np.sort(mean)[-2]
            if cnt.max() / len(idx) >= th.VOTE_AGREE and margin >= th.VOTE_MARGIN:
                out[classes[win]] = max(out.get(classes[win], 0.0), float(meanp))
    return out


def headroom(maxima, th=DEFAULT):
    """(letter, floor - its most confident gated idle vote) for the tightest letter."""
    if not maxima:
        return None, float("inf")
    pairs = [(L, th.vote_prob_for(L) - v) for L, v in maxima.items()]
    return min(pairs, key=lambda kv: kv[1])


def gate(model, featfn, classes, th=DEFAULT, holds=None, votes=DEFAULT.VOTE_MIN):
    """-> (holds emitting, holds, votes emitting, votes, max mean winner p, letters emitted)"""
    holds = load_holds() if holds is None else holds
    eh = ev = nv = 0
    maxp = 0.0
    letters = {}
    for P in holds:
        pr = model.predict_proba(featfn(P))
        wins = ([vote(pr[i:i + votes], th, classes) for i in range(len(P) - votes + 1)]
                if len(P) >= votes else [vote(pr, th, classes)])
        nv += len(wins)
        em = [w for w in wins if w[0]]
        ev += len(em)
        eh += bool(em)
        maxp = max(maxp, max(w[1] for w in wins))
        for w in em:
            letters[classes[w[2]]] = letters.get(classes[w[2]], 0) + 1
    return eh, len(holds), ev, nv, maxp, letters


def gate_recipe(seeds, trees=None, leaf=None, author_cap=None, aslhg_cap=None, strangers=True,
                aslhg=True, n_jobs=4, th=DEFAULT):
    """Train the shipped recipe at seeds 0..N-1 and gate each fit. -> [(seed, gate tuple, nodes)]

    The training block is crossval_static.shipped_training_set, which is the one the shipped
    forest is fitted on, so this measures the RECIPE and not the pickle that happens to be
    committed. Imported here rather than at module scope: the gate itself only needs a model.
    """
    import crossval_static as CV
    import train_static as T
    holds = load_holds()
    out, tights = [], []
    with T.forest_override(n_estimators=trees, min_samples_leaf=leaf) as forest:
        per, label = CV.shipped_training_set(author_cap=author_cap, aslhg_cap=aslhg_cap,
                                             strangers=strangers, aslhg=aslhg)
        print(f"recipe: {label}, jitter {T.JITTER_SIGMA} x{T.JITTER_COPIES}, "
              f"RF({forest['n_estimators']}, min_samples_leaf={forest['min_samples_leaf']})")
        for seed in range(seeds):
            X, y, _ = T.training_rows(per, seed=seed)
            model = T.make_forest(seed, n_jobs=n_jobs).fit(X, y)
            got = gate(model, T.FEATFN, T.LETTERS, th=th, holds=holds)
            out.append((seed, got, T.node_count(model)))
            eh, nh, ev, nv, maxp, letters = got
            maxima = idle_maxima(model, T.FEATFN, T.LETTERS, holds=holds, th=th)
            tights.append(headroom(maxima, th))
            print(f"  seed {seed}: {len(X)} rows, {T.node_count(model)} nodes, clean idle holds "
                  f"emitting {eh}/{nh}, votes {ev}/{nv}, gated idle maxima "
                  + ", ".join(f"{L} {v:.4f}" for L, v in sorted(maxima.items(), key=lambda kv: -kv[1]))
                  + (f", letters {letters}" if letters else ""), flush=True)
    bad = [s for s, g, _ in out if g[0]]
    tight = min(tights, key=lambda kv: kv[1]) if tights else (None, float("inf"))
    print(f"over {seeds} seeds: {'PASS' if not bad else 'FAIL at seed(s) ' + str(bad)}"
          f" -- tightest letter across all seeds {tight[0]} with {tight[1]:+.4f} of headroom "
          f"under its floor {th.vote_prob_for(tight[0]):.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--extract", action="store_true", help="rebuild idle_holds.npz from docs/browser_log.jsonl")
    ap.add_argument("--sweep", action="store_true", help="holds emitting at floors 0.50 .. 0.85")
    ap.add_argument("--seeds", type=int, default=0,
                    help="train the recipe at seeds 0..N-1 and gate each, instead of gating --model")
    ap.add_argument("--trees", type=int, default=None, help="--seeds: forest n_estimators")
    ap.add_argument("--leaf", type=int, default=None, help="--seeds: forest min_samples_leaf")
    ap.add_argument("--author-cap", type=int, default=None,
                    help="--seeds: the author's frames per letter (0 = uncapped, the old recipe)")
    ap.add_argument("--aslhg-cap", type=int, default=None, help="--seeds: ASL-HG per signer per letter")
    ap.add_argument("--no-aslhg", action="store_true", help="--seeds: leave ASL-HG out entirely")
    ap.add_argument("--no-strangers", action="store_true", help="--seeds: the author's frames only")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    if args.extract:
        extract()
        return
    if args.seeds:
        gate_recipe(args.seeds, trees=args.trees, leaf=args.leaf, author_cap=args.author_cap,
                    aslhg_cap=args.aslhg_cap, aslhg=not args.no_aslhg,
                    strangers=not args.no_strangers, n_jobs=args.n_jobs)
        return
    blob = pickle.load(open(args.model, "rb"))
    featfn, dim = F.static_feature_for(blob["feature"])
    classes = list(blob["classes"])
    holds = load_holds()
    eh, nh, ev, nv, maxp, letters = gate(blob["model"], featfn, classes, holds=holds)
    verdict = "PASS" if eh == 0 else "FAIL: raise the emitting letters' floors by 0.05 and re-run"
    maxima = idle_maxima(blob["model"], featfn, classes, holds=holds)
    L, head = headroom(maxima)
    print(f"{os.path.basename(args.model)} at VOTE_PROB {DEFAULT.VOTE_PROB} / VOTE_PROB_FLOOR "
          f"{DEFAULT.VOTE_PROB_FLOOR} / VOTE_MARGIN_CLEAR {DEFAULT.VOTE_MARGIN_CLEAR} / per letter "
          f"{DEFAULT.VOTE_PROB_LETTER}: clean idle holds emitting {eh}/{nh}, votes {ev}/{nv}"
          + (f", letters {letters}" if letters else "") + f" -> {verdict}")
    print("  gated idle maxima (the votes that already clear agree and margin): "
          + ", ".join(f"{k} {v:.4f}" for k, v in sorted(maxima.items(), key=lambda kv: -kv[1])))
    print(f"  tightest letter {L} at {head:+.4f} under its floor {DEFAULT.vote_prob_for(L):.2f}")
    if args.sweep:
        for floor in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            # One knob, as replay_static.py sweeps it: the confident route (VOTE_PROB) admits
            # any vote above it whatever the floor says, so the two have to move together or
            # the sweep is flat above VOTE_PROB and means nothing there.
            th = replace(DEFAULT, VOTE_PROB_FLOOR=floor, VOTE_PROB=floor)
            eh, nh, ev, nv, _, letters = gate(blob["model"], featfn, classes, th=th, holds=holds)
            print(f"  floor {floor:.2f}: {eh}/{nh} holds, {ev}/{nv} votes" + (f" {letters}" if letters else ""))


if __name__ == "__main__":
    main()

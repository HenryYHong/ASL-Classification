"""Replay the author's 71 held signs through the real Segmenter, with held-out forests.

crossval_static.py scores frames; this scores what the runtime would have SHOWN. Every hold
of every session is replayed through Segmenter.step with its recorded timestamps (the S1
archive at 1/15 s, the JPEG burst rate), a forest that never saw that session (the
leave-one-session-out fold model, both shipped caps and the strangers on its training side,
exactly as train_static.py fits the one that ships),
the committed motion forest attached so a false track start counts, and thresholds.DEFAULT.
One fresh Segmenter per hold: the between-hold frames were never stored, so cooldown and
duplicate suppression across holds are not exercised. A deferred I or D is released by
stepping the clock D_WAIT past the last frame, the way the page's timer would.

Per hold: did exactly the letter come out (and nothing else), did the letter come out at all,
was the first emission wrong, was the hold silent, how long from the first frame to the
first emission. Read the pooled row and the cross-day (S1) row; S2-S4 were recorded the same
evening.

The vote floor is what this measures the cost of: a higher floor silences the relaxed hands in
idle_gate.py and, at the same time, the author's least confident real holds. The two scripts
are read together, and together they chose the per-letter profile that ships
(thresholds.VOTE_PROB_LETTER). On this forest, over the 71 holds at each of seeds 0, 1 and 2:

    per letter (G 0.75, rest 0.55)   exact 68/71   silent 2  {S1-G, S2-K}   idle 0/43
    flat 0.55                        exact 69/71   silent 1  {S2-K}         idle 4-5/43, all G
    flat 0.72 and flat 0.75          exact 61/71   silent 9                 idle 0/43

--floor is a FLAT what-if: it moves VOTE_PROB and VOTE_PROB_FLOOR together and clears the
per-letter profile, which is how the middle and bottom rows above were measured.

    ./.venv/bin/python temporal/replay_static.py                  # shipped recipe, seed 0
    ./.venv/bin/python temporal/replay_static.py --henry-only     # the one-signer forest
    ./.venv/bin/python temporal/replay_static.py --floor 0.55     # a flat-floor what-if
"""
import argparse
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import static_aug as A  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from segmenter import Segmenter  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = T.LETTERS
NC = len(LETTERS)
S1_FPS = 15.0
EXTRA = (("S2", "static_s2.npz"), ("S3", "static_s3.npz"), ("S4", "static_s4.npz"))


def raw_holds():
    """{session: [(letter, raw landmarks (n,21,3), stamps (n,), handed labels (n,), W, H)]}"""
    out = {}
    d = np.load(os.path.join(HERE, "static_sequences.npz"))
    lm, found = d["lm"], d["found"]
    handed = d["handed"] if "handed" in d else None
    W, H = T.W, T.H
    s1 = []
    for c in range(NC):
        idx = np.where(found[c])[0]
        labs = handed[c][idx] if handed is not None else np.full(len(idx), "Left")
        s1.append((LETTERS[c], lm[c][idx].astype(np.float64), idx / S1_FPS, [str(x) for x in labs], W, H))
    out["S1"] = s1
    for tag, name in EXTRA:
        d = np.load(os.path.join(HERE, name), allow_pickle=True)
        lm, st = d["lm"].astype(np.float64), np.asarray(d["stamps"], dtype=np.float64)
        letters = [str(x) for x in d["letters"]]
        labs = [str(x) for x in d["handed"]] if "handed" in d else ["Unknown"] * len(lm)
        w, h = [int(v) for v in np.asarray(d["frame_size"]).ravel()[:2]]
        holds, i = [], 0
        while i < len(lm):
            j = i + 1
            while j < len(lm) and letters[j] == letters[i] and st[j] - st[j - 1] <= T.HOLD_GAP:
                j += 1
            if letters[i] in LETTERS:
                holds.append((letters[i], lm[i:j], st[i:j], labs[i:j], w, h))
            i = j
        out[tag] = holds
    return out


def fold_models(src, seed, henry_only, n_jobs, author_cap=None, aslhg=True):
    """One forest per held-out session, on the SHIPPED training recipe minus that session.

    The caps are part of that recipe (train_static.AUTHOR_CAP on the author's pooled training
    sessions, strangers.ASLHG_CAP inside load_strangers), so they are applied here too --
    otherwise this replay would be measuring what a different forest would have shown.
    """
    author_cap = T.AUTHOR_CAP if author_cap is None else author_cap
    strangers = None if henry_only else ST.load_strangers(aslhg=aslhg)[0]
    models = {}
    for held in src:
        train = ST.merge(*[src[s][0] for s in src if s != held])
        if author_cap:
            train = A.cap_per_letter(train, author_cap)
        if strangers is not None:
            train = ST.merge(train, strangers)
        X, y, _ = T.training_rows(train, seed=seed)
        models[held] = T.make_forest(seed, n_jobs=n_jobs).fit(X, y)
    return models


def replay_hold(model, motion, letter, lm, st, labs, w, h, th):
    seg = Segmenter(th, static_model=model, motion_model=motion["model"], static_classes=LETTERS,
                    motion_classes=list(motion["classes"]), static_feature_tag=T.FEATURE_TAG)
    out, tracks = [], 0
    for k in range(len(lm)):
        em = seg.step(float(st[k]), lm[k], labs[k], w, h)
        if em is not None:
            out.append((em.letter, float(st[k]) - float(st[0])))
    em = seg.step(float(st[-1]) + th.D_WAIT + 0.01, None, None, w, h)   # release a parked I/D
    if em is not None:
        out.append((em.letter, float(st[-1]) + th.D_WAIT + 0.01 - float(st[0])))
    return out


def run(seed=0, henry_only=False, floor=None, n_jobs=4, author_cap=None, models=None, src=None,
        aslhg=True):
    import crossval_static as CV
    # One knob: VOTE_PROB and VOTE_PROB_FLOOR both move to `floor`. They are equal in DEFAULT
    # on purpose (thresholds.py), so this is the shipped rule at 0.75 and a monotone sweep
    # everywhere else; moving the floor alone would leave the confident route at 0.75 and the
    # sweep would flatten above it.
    th = DEFAULT if floor is None else replace(DEFAULT, VOTE_PROB_FLOOR=floor, VOTE_PROB=floor,
                                               VOTE_PROB_LETTER={})
    src = CV.sessions() if src is None else src
    models = fold_models(src, seed, henry_only, n_jobs, author_cap, aslhg) if models is None else models
    motion = pickle.load(open(os.path.join(HERE, "model_motion.p"), "rb"))
    holds = raw_holds()
    rows = []
    for tag, hs in holds.items():
        for (letter, lm, st, labs, w, h) in hs:
            ems = replay_hold(models[tag], motion, letter, lm, st, labs, w, h, th)
            letters_out = [e[0] for e in ems]
            rows.append({"session": tag, "letter": letter, "emitted": letters_out,
                         "exact": letters_out == [letter], "among": letter in letters_out,
                         "first_wrong": bool(letters_out) and letters_out[0] != letter,
                         "silent": not letters_out, "wrong": sum(1 for l in letters_out if l != letter),
                         "latency": ems[0][1] if ems else float("nan"),
                         "tracks": sum(1 for l in letters_out if l in ("J", "Z"))})
    return rows, th


def summarize(rows, label):
    n = len(rows)
    lat = [r["latency"] for r in rows if not np.isnan(r["latency"])]
    print(f"{label:<10} holds {n:>3}  exact {sum(r['exact'] for r in rows):>3}/{n} ({np.mean([r['exact'] for r in rows]):.3f})"
          f"  among {sum(r['among'] for r in rows):>3}  first wrong {sum(r['first_wrong'] for r in rows):>2}"
          f"  silent {sum(r['silent'] for r in rows):>2}  wrong letters {sum(r['wrong'] for r in rows):>2}"
          f"  latency median {np.median(lat) if lat else float('nan'):.2f} s / p90 {np.percentile(lat, 90) if lat else float('nan'):.2f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--henry-only", action="store_true", help="fold models without the strangers")
    ap.add_argument("--floor", type=float, default=None,
                    help="a FLAT floor what-if: VOTE_PROB and VOTE_PROB_FLOOR both move here and "
                         "VOTE_PROB_LETTER is cleared (default: thresholds.DEFAULT, which is per letter)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print every hold that is not exact")
    args = ap.parse_args()
    t0 = time.time()
    rows, th = run(args.seed, args.henry_only, args.floor, args.n_jobs)
    print(f"{'strangers on the training side' if not args.henry_only else 'author only'}, seed {args.seed}, "
          f"VOTE_PROB_FLOOR {th.VOTE_PROB_FLOOR} / VOTE_MARGIN_CLEAR {th.VOTE_MARGIN_CLEAR}")
    for tag in ("S1", "S2", "S3", "S4"):
        summarize([r for r in rows if r["session"] == tag], tag + (" (cross-day)" if tag == "S1" else ""))
    summarize(rows, "pooled")
    print(f"track starts across all holds: {sum(r['tracks'] for r in rows)}   ({time.time() - t0:.0f}s)")
    if args.verbose:
        for r in rows:
            if not r["exact"]:
                print(f"  {r['session']} {r['letter']}: {''.join(r['emitted']) or '-'}")


if __name__ == "__main__":
    main()

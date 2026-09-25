"""A 25th class for "not a letter": it fixes the idle gate, and it is not what ships.

This file exists so the next person to have this idea starts from the measurement instead of
the reasoning. The reasoning is good, which is the problem.

THE IDEA. With 24 classes a resting hand has to be assigned SOME letter -- there is no other
answer available -- and the letter it gets is the one a loose fist most resembles. That is why
G, Q and A are exactly the three that breach a flat floor on the idle holds. Give the forest a
25th class, REST, trained on hands that are not making a letter, and it has somewhere else to
put them. `strangers.load_idle_strangers()` is the training data: 530 frames cut from the head
and tail of 123 OpenHands clips by `ingest_idle_strangers.py`, other people's hands, tracked and
demonstrably not holding the letter. Deliberately NOT the author's own idle holds, because those
43 holds are the gate and a gate trained on its own test set measures memorization.

IT WORKS. Wrapped so the 25th column is dropped at the model boundary -- the 24 letters then do
not sum to 1, and the missing mass is what stops a resting hand clearing any floor -- one
jittered copy of those 530 frames takes `idle_gate.py --seeds 16` from failing at 4 of 8 seeds
to passing all 16, and leaves the 71-hold replay at 68 exact. Two copies costs a D hold; eight
FAILS, because the REST mass grows large enough to reshape the letter boundaries and idle G
climbs back to 0.7927. Filtering the negatives to drop frames that sit near a real letter also
breaks it: the near-misses are what do the work.

IT IS STILL NOT WORTH IT, and this is the part that took longest to find. The per-letter floors
set from the same 16-seed idle distribution (G 0.85, Q 0.62, A 0.60) pass the gate on their own,
with no new class at all. So REST buys margin, not correctness -- and it charges for it on
strangers' video. Replaying the 266 OpenHands clips through the real segmenter:

    24 classes, the shipped floors      94 right,  0 WRONG   G headroom +0.0067
    + REST, seed 0                      91 right,  2 WRONG   G headroom +0.0361   (O->E, U->R)
    + REST, seed 1                      92 right,  1 WRONG
    + REST, seed 2                      93 right,  1 WRONG

Consistent across seeds, so it is a property and not a draw. And the two costs are not the same
kind of thing: the gate runs BEFORE anything ships, so a thin margin rejects future candidates,
which is the gate working. A wrong letter reaches a person using the page. Friction in the build
is cheaper than a defect on the page, so the floors ship and the class does not.

What would change the answer: a way to keep the gate margin without the video cost. The U->R in
the table is the structural blindness crossing_probe.py describes, not something REST caused, so
fixing the feature would remove one of the two. A REST class trained on more than 123 clips of
one dataset might also behave differently; 530 frames is not much to describe "not a letter".

    ./.venv/bin/python temporal/rest_probe.py            # the gate, with and without
    ./.venv/bin/python temporal/rest_probe.py --seeds 16 # the full sixteen, slow
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossval_static as CV  # noqa: E402
import idle_gate as G  # noqa: E402
import static_aug as A  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

#: The setting that passed: one jittered copy. See the docstring for 2 and 8.
REST_COPIES = 1


def fit_with_rest(per_class, seed, copies=REST_COPIES):
    """The shipped recipe plus a REST class, presented as a 24-column letter model."""
    X, y = T.training_rows(per_class, seed=seed)[:2]
    P, _ = ST.load_idle_strangers()
    blocks = A.jitter_frames(P, T.JITTER_SIGMA, copies, A.content_rng(seed, P))
    Xr = np.concatenate([T.FEATFN(b) for b in blocks])
    m = T.make_forest(seed, n_jobs=4).fit(
        np.concatenate([X, Xr]), np.concatenate([y, np.full(len(Xr), len(T.LETTERS))]))
    keep = [i for i, c in enumerate(m.classes_) if int(c) < len(T.LETTERS)]

    class LetterView:
        """The 24 letter columns, NOT renormalized: the missing mass is the mechanism."""
        classes_ = m.classes_[keep]

        def predict_proba(self, Xq):
            return m.predict_proba(Xq)[:, keep]

    return LetterView()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4, help="seeds 0..N-1 (16 is the gate's bar)")
    ap.add_argument("--copies", type=int, default=REST_COPIES)
    args = ap.parse_args()

    holds = G.load_holds()
    per, label = CV.shipped_training_set()
    print(f"recipe: {label}\nfloors: {DEFAULT.VOTE_PROB_LETTER}, flat {DEFAULT.VOTE_PROB}\n")
    for name, build in (("24 classes (what ships)", lambda s: T.fit_letters(*T.training_rows(per, seed=s)[:2], s, 4)),
                        (f"+ REST x{args.copies}", lambda s: fit_with_rest(per, s, args.copies))):
        bad, worst = [], {}
        for seed in range(args.seeds):
            m = build(seed)
            eh, nh, ev, nv, _, letters = G.gate(m, T.FEATFN, T.LETTERS, th=DEFAULT, holds=holds)
            for L, p in G.idle_maxima(m, T.FEATFN, T.LETTERS, holds=holds, th=DEFAULT).items():
                worst[L] = max(worst.get(L, 0.0), p)
            if eh:
                bad.append(seed)
            print(f"  {name:24s} seed {seed}: holds {eh}/{nh} votes {ev}/{nv}"
                  + (f"   FAILS {letters}" if eh else ""), flush=True)
        tight = min((DEFAULT.vote_prob_for(L) - v, L) for L, v in worst.items())
        print(f"  -> {name}: {'PASS' if not bad else 'FAIL at ' + str(bad)} over {args.seeds} seeds; "
              f"tightest {tight[1]} {tight[0]:+.4f}\n")
    print("The video cost is in the docstring: replay_strangers.py is what measures it, and it "
          "needs the OpenHands clips, which are not committed.")


if __name__ == "__main__":
    main()

"""Write the leave-one-session-out out-of-fold posteriors of the shipped static recipe.

    ./.venv/bin/python temporal/make_oof.py oof.npz              # shipped recipe, seed 0
    ./.venv/bin/python temporal/make_oof.py oof.npz --seed 1
    ./.venv/bin/python temporal/make_oof.py oof_legacy.npz --legacy

This is `crossval_static.py --oof PATH` with nothing else printed: the same folds, the same
filter and jitter on the training side only, the same forest. Every frame's posterior comes
from the model that never saw its session, so simulate_words.py can spell words out of them
without any in-sample optimism. Schema: probs (N,24), y (N,), session (N,), hold (N,) one id
per held sign (S1 has one hold per letter; S2-S4 holds split at stamp gaps > 0.5 s; 71 in
all), order (N,) frame index within its hold, letters (24,).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossval_static as CV


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="npz to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy", action="store_true", help="the previous release's recipe")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    res = CV.run(seed=args.seed, legacy=args.legacy, n_jobs=args.n_jobs, verbose=True)
    CV.report(res)
    CV.write_oof(res, args.out)


if __name__ == "__main__":
    main()

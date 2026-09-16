#!/usr/bin/env python3
"""Measure the word layer offline: spell words out of held-out letter posteriors, score them
with a Python port of docs/words.js, and count what the page would have shown.

    ./.venv/bin/python temporal/make_oof.py oof.npz                 # step 1: out-of-fold posteriors
    ./.venv/bin/python temporal/simulate_words.py --oof oof.npz    # step 2: the README numbers
    ./.venv/bin/python temporal/simulate_words.py --oof oof.npz --sampling independent
    ./.venv/bin/python temporal/simulate_words.py --oof oof.npz --sweep     # the 112-cell grid WORD_* came from (~7 min)
    ./.venv/bin/python temporal/simulate_words.py --oof oof.npz --legacy    # the previous layer

This is a SIMULATOR of the segmenter and the hint, not a recording of spelled words: no such
recording exists (thresholds.py NEEDS_WORD_DATA says so), and the two retry models below
bracket what a live signer does. Run both and quote both.

Inputs
  --oof   npz as crossval_static.py --oof / make_oof.py write it: probs (N,24) leave-one-
          session-out predict_proba per frame, y (N,) class index, session (N,), hold (N,) id
          of the held sign each frame belongs to, order (N,), letters (24,).
  --list  words.txt, one lowercase word per line, in frequency order (line = rank), as
          docs/build_words.py writes it (default docs/words.txt). --legacy reads it as an
          unordered list instead.
  --freq  Norvig's google-books-common-words.txt (WORD<TAB>COUNT), only used to draw targets;
          default docs/google-books-common-words.txt, which docs/build_words.py's docstring
          says how to fetch (1.5 MB, not committed).
  --thresholds  by default the VOTE_* gate and the WORD_* constants are read from
          temporal/thresholds.py DEFAULT -- the values the segmenter and the page run. A path
          to docs/models.json (its "thresholds" block) or to any JSON dict of the same names
          overrides them, so the numbers can be re-derived for an exported page.

How a word is spelled (per target word, per seed)
  Each static letter is one hold of that letter drawn at random from the OOF frames (a hold the
  model never trained on). The segmenter's vote is replayed on consecutive VOTE_MIN-frame windows
  starting at a random frame of the hold: mean of the window's posteriors, the winner must hold
  >= VOTE_AGREE of the per-frame argmaxes, margin >= VOTE_MARGIN, and meanp >= VOTE_PROB or
  (margin >= VOTE_MARGIN_CLEAR and meanp >= VOTE_PROB_FLOOR). Up to --tries windows are tried;
  if none passes the letter is dropped (the signer gave up). A passing window emits its mean as
  the letter's 26-vector (posteriorsFor on a static emission) and the argmax as the letter.
  J and Z go through the motion branch: abstains with probability 1/6, otherwise emits the
  letter with confidence p ~ U(0.55, 0.97) spread as posteriorsFor does. Consecutive identical
  static emissions collapse to one, as segmenter.lastEmitted does: HELLO is emitted as HELO
  (motion letters are not suppressed: JAZZ can arrive as JAZZ).

Target sets (fixed seeds, so every run scores the same words; deliberately independent of the
list, so lists of different sizes -- the 30k/40k/50k comparison that chose the top-40k list,
and the Webster baseline -- score the same words)
  common  2000 words drawn from the first 5000 eligible rows of the source table with
          probability proportional to count
  names   600 entries of /usr/share/dict/propernames, uniform without replacement
  rare    1000 words drawn uniformly from eligible rows 5001..40000 of the source table, which
          reach source rank ~46,200 because eligibility skips rows; on the committed top-40k
          list about a sixth of them are absent (the printed in-list fraction, 0.833: 150
          past the list's rank-40,000 cut, 17 non-curated two-letter entries such as 'de'),
          so rare 'recovered' is capped there and measures list coverage as well as the layer
          (over the listed rare targets alone it reads about 0.21 / 0.41).
Eligible means a-z, 2..10 letters (build_words.py's length rule only; its two-letter curation
and repeated-letter rule are NOT applied here, hence the few targets the list never carries),
minus the one-letter words A and I, which the hint never fires on: HINT_MIN_LEN is 2. Common
loses 7 of 2000 and names 5 of 600 the same way.

Reported per set: read-ok (the emitted string equals the target), exact ok/wrong, hint ok/wrong,
silent-wrong (misread and nothing shown), recovered = (exact ok + hint ok)/n, shown-wrong =
(exact wrong + hint wrong)/n, hint precision and recall (recall over misread words).

Measured on the shipped forest's out-of-fold posteriors (crossval_static.py seed 0) at the
shipped gate (VOTE_MARGIN_CLEAR 0.20 / VOTE_PROB_FLOOR 0.55, i.e. thresholds.py DEFAULT, the
gate printed on the '# gate' line) and constants (WORD_MIN_RATIO 0.10, WORD_DOMINANCE 10,
WORD_PRIOR 2.5), 5 seeds: common recovered / shown wrong 0.439 / 0.074 with consecutive-window
retries and 0.625 / 0.057 with fresh-hold retries (names 0.307/0.068 and 0.494/0.044, rare
0.181/0.047 and 0.344/0.034) -- the numbers the README quotes, re-derived by the two commands
above. The constants were swept at 0.20 / 0.50, the floor before the idle-gate re-measurement
raised it (thresholds.py VOTE_PROB_FLOOR); at that gate the same OOF gives common 0.477/0.072
and 0.681/0.053 (--thresholds with a {"VOTE_PROB_FLOOR": 0.50} dict reproduces them). The
floor raise costs the simulator 0.04-0.06 recovered because its retries are random 4-frame
windows, while the real-segmenter replay of the same 71 holds is unchanged at 65/71 exact.
The previous layer (--legacy: uniform prior, raw index, 0.02/10) on the previous forest and
gate scored common 0.445/0.158 (consecutive) and 0.612/0.091 (fresh-hold).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from thresholds import DEFAULT  # noqa: E402

FLOOR = 1e-4                     # words.js FLOOR: a letter the static branch cannot emit still costs something finite
GATE_KEYS = ("VOTE_MIN", "VOTE_AGREE", "VOTE_PROB", "VOTE_MARGIN", "VOTE_MARGIN_CLEAR", "VOTE_PROB_FLOOR")
WORD_KEYS = ("WORD_MIN_RATIO", "WORD_DOMINANCE", "WORD_PRIOR")
MIN_LEN, MAX_LEN = 2, 10          # targets; the list itself also carries A and I
TARGET_SEEDS = dict(common=0, names=1, rare=2)


# ------------------------------------------------------------------ words.js, ported

def collapse(s):
    """Run-length collapse: 'hello' -> 'helo'. What the segmenter emits for a doubled letter."""
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


class Index:
    """Port of buildIndex. Entries keep file order; rank = 1-based line number; key = collapsed
    spelling when collapsed=True, else the word itself; log_rank = ln(rank) (0 with no prior)."""

    def __init__(self, lines, collapsed=True):
        """lines: the file's lines in order; rank = 1-based line number (blank and duplicate
        lines are skipped but still count, exactly as buildIndex does)."""
        self.words = []
        seen = set()
        self.collapsed = collapsed
        self.by_len = {}
        for i, w in enumerate(lines):
            w = w.strip()
            if not w or w in seen:
                continue
            seen.add(w)
            self.words.append(w)
            key = collapse(w) if collapsed else w
            self.by_len.setdefault(len(key), []).append((w, key, math.log(i + 1)))
        self.has = seen
        # vectorized form of each bucket
        self.mat, self.logrank, self.keys, self.names = {}, {}, {}, {}
        for n, entries in self.by_len.items():
            self.names[n] = [e[0] for e in entries]
            self.keys[n] = [e[1] for e in entries]
            self.mat[n] = np.array([[ord(c) - 97 for c in e[1]] for e in entries], dtype=np.int64)
            self.logrank[n] = np.array([e[2] for e in entries])
        self.size = len(self.words)


def collapse_reading(reading, posts):
    """Runs of one letter merge into one position whose posterior is the run's element-wise mean."""
    key, out = [], []
    for ch, p in zip(reading, posts):
        if key and key[-1] == ch:
            out[-1].append(p)
        else:
            key.append(ch)
            out.append([p])
    return "".join(key), [np.mean(ps, axis=0) if len(ps) > 1 else ps[0] for ps in out]


HINT_MIN_LEN = 2                 # a hint that CHANGES a letter needs at least this many collapsed positions:
                                 # the one-letter bucket is {a, i}, and a stray N or T would otherwise be
                                 # "corrected" to A on one posterior. The gate sits after the own-candidate
                                 # return (rule R), so a reading whose own collapsed spelling is listed
                                 # (AA -> a, like HELO -> hello) is still hinted at one position: that hint
                                 # repeats what was read, it does not overrule it. docs/words.js agrees.


def closest_word(reading, posts, idx, min_ratio, dominance, lam, unified=True, hint_min_len=HINT_MIN_LEN):
    """Port of closestWord. Returns (word or None, reason).

    score(c) = sum_i log max(p_i[key_i], FLOOR) - lam * ln(rank_c)
    own      = the entry spelled exactly as the reading, else the lowest-rank entry whose collapsed
               key equals the collapsed reading, if any
    best     = highest score, ties to the lower rank (first in file order); when the reading
               carries a deliberate double and is itself the listed word, other entries with the
               same key are excluded (the frames cannot tell TOO from TO; the hand drop can)
    second   = highest score among the other entries (an equal score counts)
    Verdict (unified):
      own exists and no rival beats it by dominance -> 'exact' if own.word == reading
                                                       else 'hint' own.word (HELO -> hello)
      n < hint_min_len -> 'too-short'
      otherwise best must clear min_ratio against the reading and dominance against second -> 'hint'
    With unified=False (the previous release's layer): a reading in the list is 'exact' outright.
    """
    reading = reading.lower()
    if idx is None:
        return None, "no-list"
    if idx.collapsed:
        key, posts = collapse_reading(reading, posts)
    else:
        key = reading
    n = len(key)
    if n == 0:
        return None, "too-long"
    if not unified and reading in idx.has:
        return reading, "exact"
    if n not in idx.mat:
        return None, "too-long"
    logp = np.log(np.maximum(np.stack(posts), FLOOR))                  # (n, 26)
    lik = logp[np.arange(n)[None, :], idx.mat[n]].sum(1)                # (entries,)
    score = lik - lam * idx.logrank[n]
    read_lik = float(logp[np.arange(n), [ord(c) - 97 for c in key]].sum())
    names = idx.names[n]
    keys = idx.keys[n]
    r = -1
    if unified:
        for i, k in enumerate(keys):                                    # own: raw match beats key match
            if names[i] == reading:
                r = i
                break
            if r < 0 and k == key:
                r = i
        if r >= 0 and names[r] == reading and reading != key:           # a deliberate double: same-key
            for i, k in enumerate(keys):                                # rivals cannot outrank it
                if i != r and k == key:
                    score[i] = -math.inf
    b = int(np.argmax(score))                                           # first max = lowest rank
    best_score = score[b]
    if len(score) > 1:
        rest = np.delete(score, b)
        second = float(rest.max())
    else:
        second = -math.inf
    if unified:
        if r >= 0 and (b == r or best_score - score[r] < math.log(dominance)):
            return names[r], ("exact" if names[r] == reading else "hint")
    if n < hint_min_len:
        return None, "too-short"
    if lik[b] - read_lik < math.log(min_ratio):
        return None, "unlikely"
    if best_score - second < math.log(dominance):
        return None, "ambiguous"
    return names[b], "hint"


# ------------------------------------------------------------------ the segmenter, replayed

class Speller:
    def __init__(self, oof, gate, k=4, tries=3, sampling="consecutive"):
        d = np.load(oof)
        self.P, self.y = d["probs"], d["y"]
        self.letters = [str(x) for x in d["letters"]]
        self.col = np.array([ord(c.lower()) - 97 for c in self.letters])
        hold = d["hold"] if "hold" in d.files else None
        if hold is None:                                     # fall back to (session, class) runs
            sess = d["session"]
            hold = np.zeros(len(self.y), int)
            h = 0
            for i in range(1, len(self.y)):
                if self.y[i] != self.y[i - 1] or sess[i] != sess[i - 1]:
                    h += 1
                hold[i] = h
        self.holds = {c: [] for c in range(len(self.letters))}
        i = 0
        while i < len(self.y):
            j = i
            while j < len(self.y) and hold[j] == hold[i]:
                j += 1
            self.holds[int(self.y[i])].append((i, j))
            i = j
        self.gate, self.k, self.tries, self.sampling = gate, k, tries, sampling

    def vote(self, a, b):
        """One completed vote window on frames [a, b): (passes gate, winner class, mean probs)."""
        W = self.P[a:b]
        mean = W.mean(0)
        votes = W.argmax(1)
        counts = np.bincount(votes, minlength=mean.shape[0])
        win = int(counts.argmax())                           # ties -> lowest class index, as segmenter.js
        agree = counts[win] / len(votes)
        meanp = mean[win]
        runner = np.sort(mean)[-2]                           # secondLargest(mean), whoever won
        margin = meanp - runner
        g = self.gate
        confident = meanp >= g["VOTE_PROB"]
        decisive = margin >= g["VOTE_MARGIN_CLEAR"] and meanp >= g["VOTE_PROB_FLOOR"]
        ok = agree >= g["VOTE_AGREE"] and margin >= g["VOTE_MARGIN"] and (confident or decisive)
        return ok, win, mean

    def static(self, c, rng):
        """Consecutive k-frame windows from a random start in a random hold; None if all tries fail.
        sampling="independent" is the audit's looser model: every try is a fresh random hold and
        start, as if the signer re-formed the letter from scratch."""
        a, b = rng.choice(self.holds[c])
        k = self.k
        s = rng.randrange(a, max(a + 1, b - k + 1))
        for _ in range(self.tries):
            if self.sampling == "independent":
                a, b = rng.choice(self.holds[c])
                s = rng.randrange(a, max(a + 1, b - k + 1))
            elif s + k > b:                                  # ran off the end of the hold: restart
                s = rng.randrange(a, max(a + 1, b - k + 1))
            ok, win, mean = self.vote(s, min(s + k, b))
            if ok:
                return win, mean
            s += k
        return None

    def spell(self, word, rng):
        """-> (reading, [26-vectors], dropped). Consecutive duplicate emissions are suppressed."""
        out, posts, dropped, last = [], [], 0, None
        for ch in word:
            if ch in "jz":
                if rng.random() < 1 / 6:
                    dropped += 1
                    continue
                p = rng.uniform(0.55, 0.97)
                v = np.full(26, (1 - p) / 25)
                v[ord(ch) - 97] = p
                em = ch
            else:
                if ch.upper() not in self.letters:
                    dropped += 1
                    continue
                r = self.static(self.letters.index(ch.upper()), rng)
                if r is None:
                    dropped += 1
                    continue
                win, mean = r
                v = np.zeros(26)
                v[self.col] = mean
                em = self.letters[win].lower()
            if em == last and ch not in "jz":                  # lastEmitted suppresses static repeats
                continue                                          # only; the motion branch re-emits
            last = em
            out.append(em)
            posts.append(v)
        return "".join(out), posts, dropped


# ------------------------------------------------------------------ targets

def eligible(w):
    return w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN


def load_freq(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w, n = line.rstrip("\n").split("\t")
            w = w.lower()
            if eligible(w):
                rows.append((w, int(n)))
    return rows


def load_names(path):
    out = set()
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            w = line.strip().lower()
            if eligible(w):
                out.add(w)
    return sorted(out)


def targets(freq, names, n_common=2000, n_names=600, n_rare=1000):
    ws = [w for w, _ in freq[:5000]]
    p = np.array([c for _, c in freq[:5000]], float)
    p /= p.sum()
    common = list(np.random.default_rng(TARGET_SEEDS["common"]).choice(ws, size=n_common, p=p))
    nm = list(np.random.default_rng(TARGET_SEEDS["names"]).choice(names, size=n_names, replace=False))
    rare = list(np.random.default_rng(TARGET_SEEDS["rare"]).choice([w for w, _ in freq[5000:40000]],
                                                                   size=n_rare, replace=False))
    return {"common": [str(w) for w in common], "names": [str(w) for w in nm], "rare": [str(w) for w in rare]}


# ------------------------------------------------------------------ scoring

def tally(spelled, idx, min_ratio, dominance, lam, unified):
    t = dict(n=0, read_ok=0, exact_ok=0, exact_wrong=0, hint_ok=0, hint_wrong=0, silent_wrong=0,
             dropped=0)
    for target, (reading, posts, dropped) in spelled:
        t["n"] += 1
        t["dropped"] += dropped
        if reading == target:
            t["read_ok"] += 1
        word, reason = closest_word(reading, posts, idx, min_ratio, dominance, lam, unified)
        if reason == "exact":
            t["exact_ok" if word == target else "exact_wrong"] += 1
        elif reason == "hint":
            t["hint_ok" if word == target else "hint_wrong"] += 1
        elif reading != target:
            t["silent_wrong"] += 1
    n = t["n"]
    misread = n - t["read_ok"]
    t.update(
        recovered=(t["exact_ok"] + t["hint_ok"]) / n,
        shown_wrong=(t["exact_wrong"] + t["hint_wrong"]) / n,
        hint_p=t["hint_ok"] / max(1, t["hint_ok"] + t["hint_wrong"]),
        hint_r=t["hint_ok"] / max(1, misread),
        read_ok_rate=t["read_ok"] / n,
    )
    return t


def summarize(per_seed):
    """{metric: (mean, sd)} over seeds."""
    keys = per_seed[0].keys()
    return {k: (float(np.mean([t[k] for t in per_seed])), float(np.std([t[k] for t in per_seed])))
            for k in keys}


def fmt_row(name, S):
    cells = []
    for tset in ("common", "names", "rare"):
        s = S[tset]
        cells.append(f"{tset} rec {s['recovered'][0]:.3f}±{s['recovered'][1]:.3f} "
                     f"wrong {s['shown_wrong'][0]:.3f}±{s['shown_wrong'][1]:.3f} "
                     f"P {s['hint_p'][0]:.2f} R {s['hint_r'][0]:.2f}")
    return f"{name:<44} " + " | ".join(cells)


def load_constants(path):
    """The VOTE_* gate and WORD_* constants: thresholds.py DEFAULT, overridden by a JSON file
    (docs/models.json's "thresholds" block, or a bare dict) when one is given."""
    th = {k: getattr(DEFAULT, k) for k in GATE_KEYS + WORD_KEYS}
    if path:
        given = json.load(open(path))
        given = given.get("thresholds", given)
        for k in th:
            if k in given:
                th[k] = given[k]
    return th


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--list", default=os.path.join(ROOT, "docs", "words.txt"),
                    help="words.txt (frequency order unless --legacy)")
    ap.add_argument("--freq", default=os.path.join(ROOT, "docs", "google-books-common-words.txt"))
    ap.add_argument("--names", default="/usr/share/dict/propernames")
    ap.add_argument("--thresholds", default=None,
                    help="docs/models.json or a JSON dict overriding the VOTE_* / WORD_* constants")
    ap.add_argument("--min-ratio", type=float, default=None, help="WORD_MIN_RATIO (default: thresholds)")
    ap.add_argument("--dominance", type=float, default=None, help="WORD_DOMINANCE (default: thresholds)")
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=None,
                    help="WORD_PRIOR, the rank-prior weight (default: thresholds)")
    ap.add_argument("--no-collapse", action="store_true", help="index raw spellings")
    ap.add_argument("--no-unified", action="store_true", help="a listed reading is 'exact' outright")
    ap.add_argument("--legacy", action="store_true",
                    help="the previous release's layer: uniform prior, raw index, exact-first, 0.02/10")
    ap.add_argument("--k", type=int, default=None, help="vote window frames (default VOTE_MIN)")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--sampling", choices=("consecutive", "independent"), default="consecutive",
                    help="retry on the next window of the same hold (default) or on a fresh hold")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true")
    # The defaults ARE the 112-cell grid the shipped WORD_* were chosen from (4 x 4 x 7 at the
    # 0.20 / 0.50 gate; thresholds.py WORD_MIN_RATIO says how the cell was picked), so the
    # documented --sweep reproduces that table and contains the shipped (0.10, 10, 2.5) cell.
    ap.add_argument("--sweep-min-ratio", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--sweep-dominance", default="3,5,10,20")
    ap.add_argument("--sweep-lam", default="0,1.1,1.4,1.7,2.0,2.5,3.0")
    ap.add_argument("--out", default=None, help="write a TSV of every row here")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    th = load_constants(a.thresholds)
    gate = {k: th[k] for k in GATE_KEYS}
    k = a.k or int(gate["VOTE_MIN"])
    if a.min_ratio is None:
        a.min_ratio = float(th["WORD_MIN_RATIO"])
    if a.dominance is None:
        a.dominance = float(th["WORD_DOMINANCE"])
    if a.lam is None:
        a.lam = float(th["WORD_PRIOR"])
    if a.legacy:
        a.min_ratio, a.dominance, a.lam = 0.02, 10.0, 0.0
        a.no_collapse = a.no_unified = True
    if not os.path.exists(a.freq):
        raise SystemExit(f"{a.freq} is missing; it draws the target words. Fetch it as "
                         "docs/build_words.py's docstring says: curl -L -O "
                         "https://norvig.com/google-books-common-words.txt (into docs/; -L "
                         "because norvig.com redirects to www.norvig.com, and without it curl "
                         "saves a 795-byte '301 Moved Permanently' page)")
    if not os.path.exists(a.list):
        raise SystemExit(f"{a.list} is missing; docs/build_words.py writes it")

    freq = load_freq(a.freq)
    names = load_names(a.names)
    T = targets(freq, names)
    idx = Index(open(a.list).read().split("\n"), collapsed=not a.no_collapse)
    sp = Speller(a.oof, gate, k=k, tries=a.tries, sampling=a.sampling)
    t0 = time.time()
    # spell every target once per seed, then score under every configuration
    spelled = {tset: [] for tset in T}
    for seed in range(a.seeds):
        rng = random.Random(seed)
        for tset, ws in T.items():
            spelled[tset].append([(w, sp.spell(w, rng)) for w in ws])
    doubled = {tset: np.mean([collapse(w) != w for w in ws]) for tset, ws in T.items()}
    cover = {tset: np.mean([w in idx.has for w in ws]) for tset, ws in T.items()}
    print(f"# oof={os.path.basename(a.oof)} list={os.path.basename(a.list)} entries={idx.size} "
          f"collapsed={not a.no_collapse} unified={not a.no_unified} k={k} tries={a.tries} "
          f"sampling={a.sampling} seeds={a.seeds}")
    print(f"# gate {gate}")
    print("# targets: " + ", ".join(f"{t} n={len(ws)} doubled={doubled[t]:.3f} in-list={cover[t]:.3f}"
                                    for t, ws in T.items()))
    for tset in T:
        ro = np.mean([np.mean([r == w for w, (r, _, _) in sp_]) for sp_ in spelled[tset]])
        dr = np.mean([np.mean([d for w, (_, _, d) in sp_]) for sp_ in spelled[tset]])
        print(f"#   {tset}: read-ok {ro:.3f}, dropped letters/word {dr:.3f}")

    rows = []

    def run(name, mr, dom, lam, unified):
        S = {}
        for tset in T:
            per = [tally(sp_, idx, mr, dom, lam, unified) for sp_ in spelled[tset]]
            S[tset] = summarize(per)
        print(fmt_row(name, S), flush=True)
        for tset, s in S.items():
            rows.append(dict(tag=a.tag, name=name, oof=os.path.basename(a.oof), list=os.path.basename(a.list),
                             min_ratio=mr, dominance=dom, lam=lam, unified=unified,
                             collapsed=not a.no_collapse, sampling=a.sampling, set=tset,
                             **{kk: v[0] for kk, v in s.items()}, **{kk + "_sd": v[1] for kk, v in s.items()}))
        return S

    if a.sweep:
        for lam in [float(x) for x in a.sweep_lam.split(",")]:
            for mr in [float(x) for x in a.sweep_min_ratio.split(",")]:
                for dom in [float(x) for x in a.sweep_dominance.split(",")]:
                    run(f"mr={mr} dom={dom} lam={lam}", mr, dom, lam, not a.no_unified)
    else:
        run(f"mr={a.min_ratio} dom={a.dominance} lam={a.lam}", a.min_ratio, a.dominance, a.lam, not a.no_unified)
    print(f"# {time.time() - t0:.0f}s")
    if a.out:
        cols = list(rows[0].keys())
        with open(a.out, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
        print(f"# wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

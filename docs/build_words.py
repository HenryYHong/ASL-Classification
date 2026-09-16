#!/usr/bin/env python3
"""Build words.txt, the frequency-ordered list the page scores a spelled word against.

Source: Peter Norvig's distillation of the Google Books Ngrams English 1-grams (version
20120701) -- https://norvig.com/google-books-common-words.txt, described at
https://norvig.com/mayzner.html: the 97,565 distinct a-z words with at least 100,000 mentions,
one "WORD<TAB>COUNT" per line, sorted by count. The Google Books Ngram data is published under
the Creative Commons Attribution 3.0 Unported license (CC BY 3.0); Norvig's file is a derived
table of it. Download it next to this script (1.5 MB, not committed):

    curl -L -O https://norvig.com/google-books-common-words.txt
    python3 docs/build_words.py            # -> docs/words.txt

(-L because norvig.com redirects to www.norvig.com; without it curl saves the 795-byte
"301 Moved Permanently" page under the file's name and the build reads an empty list.)

The file's ORDER is the prior. Line i (1-based) is rank i, and words.js scores a candidate as
sum_i log p_i(letter_i) - WORD_PRIOR * ln(rank). Nothing else is carried per entry, which is
why the list must be written in frequency order and never sorted.

Rules, in the order applied:
  1. take the top N entries of the source (N = --top, default 40000);
  2. keep a-z only, MIN_LEN..MAX_LEN letters (2..10), plus the two one-letter words A and I;
  3. two-letter entries are replaced by the curated set TWO_LETTER: the source's two-letter
     tail is abbreviations (de, et, al, cm, mr, ...) that turn any misread pair of letters into
     "a word";
  4. drop entries that are one letter repeated (ii, mm, xxx, ...): the segmenter emits a held
     letter once, so their collapsed spelling is a single letter and they would shadow A and I;
  5. insert macOS's /usr/share/dict/propernames (what puts HENRY in the list) at the count of
     the source's rank-NAME_RANK entry (1000), unless the source already ranks the name higher;
  6. write in descending count order; ties keep source order, names after source words.

Reproducibility: the build is deterministic, so re-running it over the same source file and the
same propernames must give a byte-identical words.txt (34,702 entries, 273,091 B,
md5 bd45d3bfbdc81ccba57b476a18271334 for the committed list). A different md5 means the source
table or the macOS names file changed, and every rank the page's prior reads moved with it.
"""
import argparse
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_URL = "https://norvig.com/google-books-common-words.txt"
SOURCE = os.path.join(HERE, "google-books-common-words.txt")
NAMES = "/usr/share/dict/propernames"
OUT = os.path.join(HERE, "words.txt")
MIN_LEN, MAX_LEN = 2, 10
ONE_LETTER = ("a", "i")
NAME_RANK = 1000
# Every two-letter entry the list carries. Real words a person might spell, not abbreviations.
TWO_LETTER = frozenset("""
    ad ah am an as at ax be by do ex go he hi id if in is it ma me my no of oh ok on or ox pa so
    to up us we
""".split())


def load_source(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w, n = line.rstrip("\n").split("\t")
            rows.append((w.lower(), int(n)))
    return rows


def load_names(path):
    out = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            w = line.strip().lower()
            if w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN:
                out.append(w)
    return sorted(set(out))


def keep(w):
    if not (w.isascii() and w.isalpha()):
        return False
    if len(w) == 1:
        return w in ONE_LETTER
    if len(w) == 2:
        return w in TWO_LETTER
    if len(w) > MAX_LEN:
        return False
    if len(set(w)) == 1:          # ii, mm, xxx: one letter repeated
        return False
    return True


def build(rows, names, top=40000, name_rank=NAME_RANK):
    src = rows[:top]
    count = {}
    for w, n in src:
        if keep(w) and w not in count:
            count[w] = n
    floor = src[min(name_rank, len(src)) - 1][1]
    added = 0
    for w in names:
        if not keep(w):
            continue
        if count.get(w, 0) < floor:
            count[w] = floor
            added += 1
    # stable sort: source order among equal counts, names (appended last) after source words
    ordered = sorted(count, key=lambda w: -count[w])
    return ordered, added


def stats(ordered):
    text = "\n".join(ordered) + "\n"
    raw = len(text.encode())
    gz = len(gzip.compress(text.encode(), 9))
    by_len = {}
    for w in ordered:
        by_len[len(w)] = by_len.get(len(w), 0) + 1
    return text, raw, gz, by_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--names", default=NAMES)
    ap.add_argument("--top", type=int, default=40000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", type=int, nargs="*", default=None,
                    help="only print entries/bytes for these N (no file written)")
    a = ap.parse_args()
    if not os.path.exists(a.source):
        print(f"missing {a.source}: curl -L -O {SOURCE_URL} (-L: norvig.com redirects)",
              file=sys.stderr)
        return 1
    if not os.path.exists(a.names):
        print(f"missing {a.names}; on a non-macOS machine point --names at a list of names",
              file=sys.stderr)
        return 1
    rows = load_source(a.source)
    names = load_names(a.names)
    if a.report is not None:
        print(f"{'top-N':>7} {'entries':>8} {'names+':>7} {'raw B':>9} {'gzip-9 B':>9}  by length")
        for N in a.report:
            ordered, added = build(rows, names, N)
            _, raw, gz, by_len = stats(ordered)
            print(f"{N:>7} {len(ordered):>8} {added:>7} {raw:>9} {gz:>9}  "
                  + " ".join(f"{k}:{v}" for k, v in sorted(by_len.items())))
        return 0
    ordered, added = build(rows, names, a.top)
    text, raw, gz, by_len = stats(ordered)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(f"wrote {a.out}: {len(ordered)} entries ({added} from propernames), {raw} B raw, "
          f"{gz} B gzip -9; top-{a.top} of {SOURCE_URL}")
    print("by length: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_len.items())))
    print("first 10: " + " ".join(ordered[:10]) + f"; henry at rank {ordered.index('henry') + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

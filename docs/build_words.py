"""Build words.txt, the list the page checks a spelled word against.

Source is macOS's own /usr/share/dict: `web2`, which is Webster's Second International (1934,
public domain), plus `propernames`, which is what puts HENRY in the list. Neither carries word
frequencies, so the page cannot prefer a common word over an archaic one -- that is the reason
the hint abstains unless one candidate dominates rather than quietly correcting toward whatever
scores highest.

Filtered to a-z only and 2-10 letters: single letters are not words worth hinting at, and a
fingerspelled word longer than ten letters is rare enough that carrying 30k more entries for it
is not worth the bytes.

    python3 docs/build_words.py
"""
import os
import sys

SOURCES = ("/usr/share/dict/words", "/usr/share/dict/propernames")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "words.txt")
MIN_LEN, MAX_LEN = 2, 10


def main():
    words = set()
    for path in SOURCES:
        if not os.path.exists(path):
            print(f"missing {path}; on a non-macOS machine point SOURCES at a word list",
                  file=sys.stderr)
            return 1
        with open(path, encoding="latin-1") as fh:
            for line in fh:
                w = line.strip().lower()
                if w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN:
                    words.add(w)

    ordered = sorted(words)
    with open(OUT, "w") as fh:
        fh.write("\n".join(ordered) + "\n")
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}: {len(ordered)} words, {size / 1e6:.2f} MB "
          f"({size / len(ordered):.1f} bytes each)")
    by_len = {}
    for w in ordered:
        by_len[len(w)] = by_len.get(len(w), 0) + 1
    print("by length: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_len.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

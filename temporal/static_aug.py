"""Training-side augmentation and filtering for the static-letter forest.

Three things happen to a static training frame before it is featurized, and all three happen
to TRAINING frames only -- a held-out fold or a live frame never goes through any of them:

  1. RULE FILTER. Each letter has a defining-geometry rule (RULES below): the conditions a
     hand must satisfy to be that letter at all, written as thresholds on the per-frame
     quantities of temporal/features.py. A training frame that violates its OWN letter's rule
     is dropped for the letters in the "strong" set {X, G, Q, U, V, K, R, P, D}; every other
     letter is never filtered. Measured over the four sessions (4,878 frames): the X rule
     alone drops 34 frames (S1 23, S2 11), all X held with a near-straight index that the
     forest would otherwise learn as "X" and then confuse with D; the strong set drops 137 of
     4,878. The filter deliberately un-learns the archive's straight-index X and its
     together-fingers V, so in-sample recall on those two S1 bursts falls; on the held-out
     folds D stops being read as X.

  2. JITTER. Every kept frame gets `copies` Gaussian-perturbed twins: x,y += N(0, (sigma *
     palm_scale(frame))^2) independently per landmark coordinate, in the canonical isotropic
     coordinates the feature is computed from, so sigma is in palm units and scales with the
     hand's distance from the camera. The originals are kept and everything is re-featurized.
     Measured leave-one-session-out on the author's sessions alone, with the 112-D static/v4
     feature and RF(100, min leaf 5), seed 0 (3-seed means within 0.01), pooled / cross-day
     (the November archive held out): with no filter, jitter sigma 0.12 x4 lifts 0.781 / 0.650
     to 0.874 / 0.790 (+0.09 / +0.14); stacked on the strong filter, it lifts 0.830 / 0.690 to
     0.861 / 0.763 (+0.03 / +0.07). The filter shipped for one release and is retired now
     (train_static.py says why: its thresholds are one hand's, and with other people's hands
     in the training set it costs on every axis); RULESETS and filter_training stay for the
     ablation. The four-angle rotation augmentation that used to ship was retired because it LOWERED
     leave-one-session-out accuracy, 0.782 without it to 0.759 with it (S3 0.823 -> 0.661),
     having only ever been justified by within-session confidence.

  3. PER-LETTER CAP. cap_per_letter trims the author's own pooled block to at most
     train_static.AUTHOR_CAP frames per letter, evenly spaced. The comment on that constant
     carries the measurement; the function's docstring carries why "evenly spaced" and why
     "pooled".

All three functions work on (n, 21, >=2) arrays whose [..., :2] are isotropic, handedness-
canonicalized x,y (what train_static.load / load_extra return). Extra trailing channels are
carried through untouched.
"""
import zlib

import numpy as np

import features as F

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# ---------------------------------------------------------------- defining-geometry rules
# Thresholds were fixed by inspecting the archive BEFORE any leave-one-session-out run, so the
# filter is not tuned to the folds it is measured on. Units: extension ratios and distances in
# palm units (features.palm_scale), straightness in [0, 1], angles in degrees.

EXT_ST, EXT_E, CURL_E = 0.90, 1.60, 1.50    # "extended" = straight >= 0.90 and tip >= 1.60 palm
                                            # from the wrist; "curled" = tip < 1.50 palm


def _wrap(d):
    return (d + 180) % 360 - 180


def quantities(P):
    """Per-frame geometry the rules read. P: (n,21,>=2) canonical isotropic coordinates."""
    P = np.asarray(P)[..., :2]
    S = F.palm_scale(P)
    e = F.extension_ratios(P)
    q = {f"ext_{k}": v for k, v in e.items()}
    for n in ("index", "mid", "ring", "pinky"):
        q[f"st_{n}"] = F.finger_straightness(P, n)
    d = lambda i, j: np.linalg.norm(P[:, i] - P[:, j], axis=-1) / S          # noqa: E731
    q["gap_im"] = d(8, 12); q["gap_mr"] = d(12, 16)
    q["tt_index"] = d(4, 8); q["tt_mid"] = d(4, 12); q["th_midpip"] = d(4, 10)
    q["tpm"] = F.thumb_pinkymcp(P)
    ang = lambda a, b: np.degrees(np.arctan2(P[:, b, 1] - P[:, a, 1], P[:, b, 0] - P[:, a, 0]))  # noqa: E731
    q["idir"] = ang(5, 8)                                   # index direction, image y down
    q["th_ang"] = _wrap(ang(2, 4) - ang(0, 9))              # thumb relative to the palm axis
    q["it_ang"] = np.abs(_wrap(ang(2, 4) - ang(5, 8)))      # thumb-index opening angle
    a, b = P[:, 5], P[:, 17]; row = b - a                   # thumb tip along the knuckle row:
    q["t"] = ((P[:, 4] - a) * row).sum(-1) / (row ** 2).sum(-1)   # 0 = index MCP, 1 = pinky MCP
    q["dy_pip"] = (P[:, 4, 1] - P[:, 6, 1]) / S             # thumb tip below the index PIP
    q["cross"] = np.sign((P[:, 8, 0] - P[:, 12, 0]) * (P[:, 5, 0] - P[:, 9, 0]))   # R crosses
    return q


def ext(q, f):   return (q[f"st_{f}"] >= EXT_ST) & (q[f"ext_{f}"] >= EXT_E)   # noqa: E704
def curl(q, f):  return q[f"ext_{f}"] < CURL_E                                # noqa: E704
def up(q):       return (q["idir"] >= -135) & (q["idir"] <= -60)              # noqa: E704
def side(q, deg): return np.abs(q["idir"]) >= deg                             # noqa: E704
def fist(q):     return curl(q, "index") & curl(q, "mid") & curl(q, "ring") & curl(q, "pinky")   # noqa: E704


RULES = {
 "A": {"fist": fist, "thumb_beside_index(t<=0)": lambda q: q["t"] <= 0.0, "thumb_up(th_ang<=22)": lambda q: q["th_ang"] <= 22},
 "B": {"4_extended": lambda q: ext(q,"index")&ext(q,"mid")&ext(q,"ring")&ext(q,"pinky"), "thumb_across(tpm<1.0)": lambda q: q["tpm"] < 1.0},
 "C": {"4_open(ext>=1.5)": lambda q: (q["ext_index"]>=1.5)&(q["ext_mid"]>=1.5)&(q["ext_ring"]>=1.5)&(q["ext_pinky"]>=1.5),
       "fingers_curved(st_index<=0.92)": lambda q: q["st_index"] <= 0.92, "c_opening(0.6<=tt_index<=1.4)": lambda q: (q["tt_index"]>=0.6)&(q["tt_index"]<=1.4)},
 "D": {"index_ext": lambda q: ext(q,"index"), "mid_ring_pinky_curled": lambda q: curl(q,"mid")&curl(q,"ring")&curl(q,"pinky"),
       "thumb_touches_mid(tt_mid<0.9)": lambda q: q["tt_mid"] < 0.9, "index_up": up},
 "E": {"fist": fist, "thumb_under_tips(dy_pip>=0.35)": lambda q: q["dy_pip"] >= 0.35, "thumb_across(th_ang>=55)": lambda q: q["th_ang"] >= 55},
 "F": {"thumb_index_touch(tt_index<0.4)": lambda q: q["tt_index"] < 0.4, "mid_ring_pinky_ext": lambda q: ext(q,"mid")&ext(q,"ring")&ext(q,"pinky")},
 "G": {"index_ext": lambda q: ext(q,"index"), "mid_curled(st_mid<0.70)": lambda q: q["st_mid"] < 0.70, "mid_ring_pinky_curled": lambda q: curl(q,"mid")&curl(q,"ring")&curl(q,"pinky"),
       "thumb_ext(ext_thumb>=1.3)": lambda q: q["ext_thumb"] >= 1.3, "index_horizontal(|idir|>=160)": lambda q: side(q, 160)},
 "H": {"index_mid_ext": lambda q: ext(q,"index")&ext(q,"mid"), "together(gap_im<0.5)": lambda q: q["gap_im"] < 0.5, "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"),
       "sideways(|idir|>=150)": lambda q: side(q, 150)},
 "I": {"pinky_ext": lambda q: ext(q,"pinky"), "index_mid_ring_curled": lambda q: curl(q,"index")&curl(q,"mid")&curl(q,"ring"), "thumb_in(tpm<1.25)": lambda q: q["tpm"] < 1.25},
 "K": {"index_mid_ext": lambda q: ext(q,"index")&ext(q,"mid"), "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"), "index_up": up,
       "thumb_at_mid_pip(th_midpip<=0.45)": lambda q: q["th_midpip"] <= 0.45, "apart(gap_im>=0.35)": lambda q: q["gap_im"] >= 0.35},
 "L": {"index_ext": lambda q: ext(q,"index"), "mid_ring_pinky_curled": lambda q: curl(q,"mid")&curl(q,"ring")&curl(q,"pinky"), "index_up": up,
       "thumb_out(tpm>=1.4)": lambda q: q["tpm"] >= 1.4, "thumb_index_angle>=50": lambda q: q["it_ang"] >= 50},
 "M": {"fist": fist, "thumb_between_ring_pinky(t>=0.6)": lambda q: q["t"] >= 0.6, "thumb_tip_at_pip(dy_pip<=0.15)": lambda q: q["dy_pip"] <= 0.15, "thumb_across(40<=th_ang<=75)": lambda q: (q["th_ang"]>=40)&(q["th_ang"]<=75)},
 "N": {"fist": fist, "thumb_between_mid_ring(0.3<=t<=0.62)": lambda q: (q["t"]>=0.3)&(q["t"]<=0.62), "thumb_tip_at_pip(dy_pip<=0.12)": lambda q: q["dy_pip"] <= 0.12, "thumb_across(35<=th_ang<=65)": lambda q: (q["th_ang"]>=35)&(q["th_ang"]<=65)},
 "O": {"circle_closed(tt_index<0.5)": lambda q: q["tt_index"] < 0.5, "fingers_curved(ext_index>=1.3,st_index<=0.95)": lambda q: (q["ext_index"]>=1.3)&(q["st_index"]<=0.95),
       "4_curved_open(ext_mid,ring,pinky>=1.3)": lambda q: (q["ext_mid"]>=1.3)&(q["ext_ring"]>=1.3)&(q["ext_pinky"]>=1.3)},
 "P": {"index_ext": lambda q: ext(q,"index"), "mid_ext(st_mid>=0.85,ext>=1.3)": lambda q: (q["st_mid"]>=0.85)&(q["ext_mid"]>=1.3), "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"),
       "pointing_down_or_side(|idir|>=140)": lambda q: side(q, 140), "thumb_at_mid_pip(th_midpip<=0.5)": lambda q: q["th_midpip"] <= 0.5},
 "Q": {"index_ext": lambda q: ext(q,"index"), "mid_curled(st_mid<0.70)": lambda q: q["st_mid"] < 0.70, "mid_ring_pinky_curled": lambda q: curl(q,"mid")&curl(q,"ring")&curl(q,"pinky"),
       "thumb_ext(ext_thumb>=1.3)": lambda q: q["ext_thumb"] >= 1.3, "index_down(100<=idir<=160)": lambda q: (q["idir"]>=100)&(q["idir"]<=160)},
 "R": {"index_mid_ext": lambda q: ext(q,"index")&ext(q,"mid"), "crossed(cross<0)": lambda q: q["cross"] < 0, "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"), "index_up": up},
 "S": {"fist": fist, "thumb_over_fingers(0.1<=t<=0.65)": lambda q: (q["t"]>=0.1)&(q["t"]<=0.65), "thumb_tip_below_pip(dy_pip>=0.12)": lambda q: q["dy_pip"] >= 0.12, "thumb_across(35<=th_ang<=70)": lambda q: (q["th_ang"]>=35)&(q["th_ang"]<=70)},
 "T": {"fist": fist, "thumb_between_index_mid(-0.15<=t<=0.30)": lambda q: (q["t"]>=-0.15)&(q["t"]<=0.30), "thumb_tip_at_pip(dy_pip<=0.12)": lambda q: q["dy_pip"] <= 0.12, "thumb_tilt(15<=th_ang<=42)": lambda q: (q["th_ang"]>=15)&(q["th_ang"]<=42)},
 "U": {"index_mid_ext": lambda q: ext(q,"index")&ext(q,"mid"), "together(gap_im<0.45)": lambda q: q["gap_im"] < 0.45, "not_crossed": lambda q: q["cross"] > 0, "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"), "index_up": up},
 "V": {"index_mid_ext": lambda q: ext(q,"index")&ext(q,"mid"), "apart(gap_im>=0.45)": lambda q: q["gap_im"] >= 0.45, "ring_pinky_curled": lambda q: curl(q,"ring")&curl(q,"pinky"), "index_up": up},
 "W": {"index_mid_ring_ext": lambda q: ext(q,"index")&ext(q,"mid")&ext(q,"ring"), "pinky_curled": lambda q: curl(q,"pinky"), "apart(gap_im>=0.3)": lambda q: q["gap_im"] >= 0.3, "index_up": up},
 "X": {"index_hooked(0.5<=st_index<=0.93)": lambda q: (q["st_index"]>=0.5)&(q["st_index"]<=0.93), "index_raised(ext_index>=1.3)": lambda q: q["ext_index"] >= 1.3,
       "mid_ring_pinky_curled": lambda q: curl(q,"mid")&curl(q,"ring")&curl(q,"pinky"), "thumb_near_mid(tt_mid<0.9)": lambda q: q["tt_mid"] < 0.9},
 "Y": {"pinky_ext": lambda q: ext(q,"pinky"), "index_mid_ring_curled": lambda q: curl(q,"index")&curl(q,"mid")&curl(q,"ring"), "thumb_out(tpm>=1.25)": lambda q: q["tpm"] >= 1.25},
}
assert set(RULES) == set(LETTERS)

#: Which letters have their TRAINING frames filtered by RULES[letter]. "strong" is what ships:
#: X, the G/Q pair, the U/V/K/R family, P and D -- the letters whose off-rule frames sit inside
#: a neighbor's shape (a straight-index X is a D; a G held with the index up is a Q or an H;
#: U/V/K/R differ only in finger gap and thumb, P is a K pointed down) and whose exclusion the
#: audit measured as a gain on the held-out folds. The fists (A, E, M, N, S, T) are NOT in it
#: on purpose: the archive's M, N and S violate their own rules on 47-100% of frames, so
#: filtering them would delete the letters rather than clean them, and the audit measured the
#: pooled number falling (0.759 -> 0.752) when it tried. "X" is the single-letter set the
#: first audit used; "none" disables the filter.
RULESETS = {
    "none": (),
    "X": ("X",),
    "strong": ("X", "G", "Q", "U", "V", "K", "R", "P", "D"),
}


def satisfies(P, letter):
    """Boolean (n,) mask: the frame satisfies every condition of RULES[letter]."""
    q = quantities(P)
    ok = np.ones(len(q["t"]), bool)
    for fn in RULES[letter].values():
        ok &= fn(q)
    return ok


def keep_mask(P, letter, ruleset="strong"):
    """(n,) True = keep this TRAINING frame. Letters outside the rule set are always kept."""
    n = len(P)
    if letter not in RULESETS[ruleset] or n == 0:
        return np.ones(n, bool)
    return satisfies(P, letter)


def filter_training(per_class, ruleset="strong"):
    """[per-class (n,21,>=2)] -> the same list with off-rule TRAINING frames removed.

    Test data never goes through this: a held-out frame that violates its letter's rule is
    still a frame the signer meant as that letter, and the runtime will see such frames.
    """
    return [P[keep_mask(P, LETTERS[c], ruleset)] if len(P) else P for c, P in enumerate(per_class)]


def cap_per_letter(per_class, k):
    """[per-class (n,21,>=2)] -> at most `k` frames per letter, evenly spaced through the block.

    A TRAINING-side cap, like filter_training above; a test fold never goes through it.

    Evenly spaced and not the first k, and not a random draw either. The block handed in is one
    signer's sessions concatenated in recording order, so np.linspace over it keeps frames from
    every session and from the whole length of every hold, which is exactly the variety a cap
    is meant to preserve. The first k would be the first seconds of the oldest session, and a
    random draw would move with the seed while the point of the cap is to be part of the recipe.

    Why cap at all, and why on the POOLED per-letter block rather than per session: the author's
    four sessions hold 400 frames of T and 308 of C against 161 of E and 100 of most letters,
    and nothing exceeds 160 within any ONE session, so a per-session cap is a no-op (the first
    attempt at this measured identical to no cap for exactly that reason). Pooled, the cap is
    what stops the letters the author re-recorded most from outvoting the rest -- and, with ten
    other signers now on the training side, from outvoting them too.
    """
    out = []
    for P in per_class:
        out.append(P[np.linspace(0, len(P) - 1, k).round().astype(int)] if len(P) > k else P)
    return out


# ---------------------------------------------------------------- jitter

def jitter_once(P, sigma, rng):
    """One jittered copy of (n,21,>=2): x,y += N(0, (sigma*palm_scale(frame))^2), per frame."""
    P = np.asarray(P)
    S = F.palm_scale(P[..., :2])[..., None, None]
    Q = P.copy()
    Q[..., :2] = P[..., :2] + rng.normal(0.0, sigma, size=P[..., :2].shape) * S
    return Q


def jitter_frames(P, sigma, copies, rng):
    """[P] + `copies` independent jittered copies of P, drawn in order from `rng`.

    The originals come first so a caller that concatenates the list keeps train_static.py's
    row order (all originals, then each copy as a block) -- row order changes the forest's
    bootstrap draws, so the order is part of what makes a run reproducible.
    """
    P = np.asarray(P)
    if copies <= 0 or sigma <= 0.0 or len(P) == 0:
        return [P]
    return [P] + [jitter_once(P, sigma, rng) for _ in range(copies)]


def content_rng(seed, P):
    """A Generator seeded by (seed, len(P), crc32 of P's x,y bytes).

    Seeding by content rather than by call order means a given training block gets the same
    jitter wherever it is drawn -- inside a leave-one-session-out fold, in a nested run, or in
    the final all-session fit -- so two runs that share a block share its augmented rows.
    """
    P = np.asarray(P)
    return np.random.default_rng([int(seed), len(P),
                                  zlib.crc32(np.ascontiguousarray(P[..., :2]).tobytes())])

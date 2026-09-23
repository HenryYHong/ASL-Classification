"""The launch-pose gates and the letter forest, on other people's hands.

features.j_gate and features.z_gate decide whether a single frame is the pose a J or a Z is
launched from. They are hand-written geometry, and every constant in them (J_THUMB_MAX, the
pinky and index tests, the curled-finger ceilings) was tuned on ONE hand: the author's archive
and the author's prompted takes. thresholds.NEEDS_GESTURE_DATA lists what is still reasoned
rather than measured; these two were measured, but on a sample of one signer.

That is the gap this file closes. temporal/aslnow.npz carries 248 J and Z stills that no
training set, no fold and no published number has ever used -- a still of a J is an I and a
still of a Z is a D, so they cannot train or score a motion classifier -- and 1,874 records of
the 24 static letters from the same strangers. Between them they can score exactly the two
things that read a single frame: the gates, and the forest's behavior on the poses J and Z are
launched from. Nothing here downloads anything; every number below is a function of files
already committed.

The reference, measured on the author's own S1 archive (temporal/static_sequences.npz, 100
detected frames per letter, 2,378 in all) by the run that set the bands here:

    j_gate   I 100/100     13 of the other 2,278 frames, every one a Y
    z_gate   D 100/100     91 of the other 2,278: X 34/100, L 23/100, G 16/100, P 16/100, K 2

Read that beside test_gates() below. j_gate transfers almost intact and z_gate does not, and
the second of those is this release's most useful negative result: it is the measured root
cause of the Z transfer failure, and it lives one layer below the motion forest, in geometry
nobody had ever scored on a hand that was not the author's.

Plain functions with a main(), like the other files here; pytest collects them as well.
"""
import os
import pickle
import sys
import traceback
from collections import Counter
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import features as F  # noqa: E402
import idle_gate as G  # noqa: E402
import strangers as ST  # noqa: E402
import train_static as T  # noqa: E402
from thresholds import DEFAULT  # noqa: E402

ASLNOW = os.path.join(ROOT, "aslnow.npz")
STATIC_MODEL = os.path.join(ROOT, "model_static.p")


def all_records():
    """Every ASLNow record, J and Z included, canonicalized the way strangers.load_aslnow does.

    load_aslnow drops J and Z, which is right for a training loader and useless here, so the
    four canonicalizing lines are repeated. test_canonicalization_matches_the_shipped_loader
    is what keeps that repetition honest: if load_aslnow ever orients a record differently,
    this file stops agreeing with it and says so, instead of quietly scoring a mirrored hand.
    """
    d = np.load(ASLNOW, allow_pickle=True)
    lm = d["lm"].astype(np.float64)
    letters = np.asarray([str(x) for x in d["letters"]])
    mirror = np.asarray(d["mirror"])
    w, h = [int(v) for v in np.asarray(d["aspect"]).ravel()]
    P = F.to_isotropic(lm[..., :2], w, h)
    P[mirror, :, 0] = -P[mirror, :, 0]
    return P, letters


def test_canonicalization_matches_the_shipped_loader():
    P, letters = all_records()
    assert P.shape == (2122, 21, 2) and len(letters) == 2122
    static = np.isin(letters, ST.LETTERS)
    assert int(static.sum()) == 1874 and int((~static).sum()) == 248
    assert int((letters == "J").sum()) == 93 and int((letters == "Z").sum()) == 155
    # the same frames, in the same order, as the loader the training set is built from
    _, Pl, letters_l, _ = ST.load_aslnow()
    assert np.array_equal(P[static], Pl) and np.array_equal(letters[static], letters_l)


def test_j_gate_transfers_to_other_peoples_hands():
    """j_gate is the one piece of J/Z geometry that survives leaving the author's hand."""
    P, letters = all_records()
    jg = np.asarray(F.j_gate(P)).astype(bool)
    static = np.isin(letters, ST.LETTERS)
    isI = letters == "I"
    others = static & ~isI

    # 65 of ASLNow's 68 I records, 0.9559, against 100/100 on the author's own archive. The
    # I handshape is the J launch pose, so this is the recall of arming: a J the gate never
    # sees is a J the motion branch never gets a chance at. 0.90 is a floor with 0.056 of
    # room; anything that drops below it has broken arming for strangers.
    recall = float(jg[isI].mean())
    assert recall >= 0.90, f"j_gate fires on only {jg[isI].sum()}/{isI.sum()} = {recall:.4f} of ASLNow's I"

    # 1 of the 1,806 records of the other 23 letters, 0.00055, and that one is a Y -- the same
    # letter, and only that letter, that costs 13 of 2,278 on the author's archive. Y is the
    # pinky-out handshape with the thumb out too, which is precisely what J_THUMB_MAX (1.30)
    # separates, so the failure mode is identical on both hands. A false arm cannot by itself
    # produce a false J: a track also needs a rising edge, the vetoes and the motion forest.
    fp = float(jg[others].mean())
    assert fp <= 0.01, (f"j_gate fires on {jg[others].sum()}/{others.sum()} = {fp:.5f} of the other "
                        f"23 letters: {dict(Counter(letters[others & jg]))}")
    assert set(letters[others & jg]) <= {"Y"}, dict(Counter(letters[others & jg]))


def test_z_gate_does_not_transfer_and_that_is_the_z_failure():
    """THE ROOT CAUSE OF THE Z TRANSFER FAILURE, measured rather than guessed.

    z_gate wants a straight index finger (finger_straightness > 0.90), three curled fingers
    (max extension ratio < 1.20) and a tucked thumb (< 1.45). On the author's archive all three
    clauses pass on 100 of 100 D frames. On ASLNow's D records -- the same handshape, other
    people's hands -- the gate fires on 35 of 72. So on a stranger about half of every Z is
    unreachable before any model runs: no arm, no rising edge, no track, no motion vote,
    silence. That is one layer BELOW the motion forest, and no quantity of motion training data
    can move it.

    It is NOT the straightness test, which is the clause I would have blamed. Per clause, on
    those 72 records (a run of mine; the author's own 100 in brackets):

        index straight > 0.90    71/72   [100/100]
        others curled < 1.20     37/72   [100/100]    <- this is the whole failure
        thumb < 1.45             58/72   [100/100]

    22 of the 37 misses fail the curled-finger clause and nothing else. The author curls the
    three idle fingers to a median extension ratio of 0.663; the strangers sit at 1.194, right
    on the 1.20 line, so half of them fall the wrong side of a constant fitted to one hand.

    The obvious fix looks cheap on this data and I did not take it, because features.py is not
    this file's to change and the price has only been measured here. For the next person:
    moving the curled ceiling 1.20 -> 1.40 takes D from 35/72 to 47/72 while the other 23
    letters go 70/1802 -> 74/1802, and moving the thumb ceiling 1.45 -> 1.70 takes D to 37/72
    and the false positives to 158/1802. One is nearly free and one is not. Neither has been
    checked against the author's archive, the prompted takes or a segmenter replay, and all
    three have to agree before a gate constant moves.
    """
    P, letters = all_records()
    zg = np.asarray(F.z_gate(P)).astype(bool)
    static = np.isin(letters, ST.LETTERS)
    isD = letters == "D"
    others = static & ~isD

    # 35/72 = 0.4861 against the author's 100/100. The band is two-sided on purpose: the lower
    # bound catches a gate that stops firing at all, and the UPPER bound is the one that has to
    # be argued with. Raising it means someone made z_gate transfer, which is a change to
    # README.md's Z story as much as to this number, and it must not happen silently.
    recall = float(zg[isD].mean())
    assert 0.40 <= recall <= 0.60, (f"z_gate fires on {zg[isD].sum()}/{isD.sum()} = {recall:.4f} of "
                                    "ASLNow's D; the author's archive reads 100/100")

    # The clause decomposition above, asserted, so the docstring cannot rot into a story about
    # the wrong constant. Straightness is nearly free on strangers; the curled ceiling is not.
    e = F.extension_ratios(P[isD])
    curled = np.maximum(np.maximum(e["mid"], e["ring"]), e["pinky"])
    straight_ok = np.asarray(F.finger_straightness(P[isD], "index")) > 0.90
    assert straight_ok.mean() >= 0.95, f"index straightness passes only {straight_ok.mean():.3f}"
    assert float((np.asarray(curled) < 1.20).mean()) < straight_ok.mean(), (
        "the curled-finger clause is supposed to be the binding one on strangers' D")

    # 70 of the 1,802 records of the other 23 letters, 0.0388 -- almost exactly the author's own
    # 91/2,278 = 0.0399, so the gate is no LOOSER on strangers, only deafer. It fails closed.
    fp = float(zg[others].mean())
    assert fp <= 0.05, (f"z_gate fires on {zg[others].sum()}/{others.sum()} = {fp:.5f} of the other "
                        f"23 letters: {dict(Counter(letters[others & zg]).most_common(6))}")

    # X leads it: 39 of 106, 0.368, against 34 of 100 on the author's archive. X is an index
    # finger with a hooked tip, and the hook is the only thing between it and a D. X is priced
    # entirely by the straightness clause and not at all by the other two -- it sits at 0.368
    # for every curled ceiling from 1.20 to 1.60, and moves to 0.528 at straightness 0.70 and
    # 0.783 at 0.50. That is the measured reason the cheap fix for D's recall is the curled
    # ceiling and not the straightness one. 0.45 leaves 0.08 of room.
    isX = letters == "X"
    x_rate = float(zg[isX].mean())
    assert x_rate <= 0.45, f"z_gate fires on {zg[isX].sum()}/{isX.sum()} = {x_rate:.4f} of ASLNow's X"


def test_static_forest_on_the_j_and_z_stills():
    """The forest has no J and no Z class, so every letter it emits on these stills is a false one.

    93 J stills and 155 Z stills, each scored as the runtime's vote gate scores a settled hold:
    idle_gate.vote with the shipped class list, so thresholds.VOTE_PROB_LETTER applies. This is
    the vote gate alone, not the whole segmenter -- a still cannot produce a rising edge, so
    nothing here exercises arming, the vetoes or D_WAIT.
    """
    P, letters = all_records()
    blob = pickle.load(open(STATIC_MODEL, "rb"))
    featfn, _ = F.static_feature_for(blob["feature"])
    classes = list(blob["classes"])
    assert classes == list(T.LETTERS) and "J" not in classes and "Z" not in classes

    def emissions(sel, th):
        pr = blob["model"].predict_proba(featfn(P[sel]))
        top = np.array([classes[i] for i in pr.argmax(1)])
        em = np.array([G.vote(pr[i:i + 1], th, classes)[0] for i in range(len(pr))])
        return em, top

    # J stills: 0 of 93 emit, at the shipped per-letter floor and at a flat 0.75 alike. The
    # forest does not agree with itself about what an I-with-a-thumb-out is (top-1 Y 24, I 18,
    # O 16, H 13, and the launch pose I on only 0.19 of them), the posterior is spread thin
    # (mean max p 0.228), and spread posteriors are exactly what VOTE_MARGIN throws out. A J
    # held in front of this page produces nothing until it moves. That is the design working.
    em_j, _ = emissions(letters == "J", DEFAULT)
    assert int(em_j.sum()) == 0, f"{em_j.sum()} of 93 J stills emit a letter"

    # Z stills: 39 of 155, 0.252 -- and this is NOT the silence the J stills give. A Z launch
    # pose is a D, and this forest reads strangers' D as X (53), P (37) and D (19); X and P are
    # confident enough to clear 0.55. The band's floor is what makes the row real: assert only
    # an upper bound and a forest that stopped predicting anything would pass.
    #
    # The handed-down expectation for this row was "<= 10 of 155", and it is the PREVIOUS
    # release's number, not this one. I re-measured it on this same pickle at the previous flat
    # 0.75 floor: 12 of 155 (X 10, P 1, D 1). The whole move from 12 to 39 is the floor drop to
    # 0.55 for 23 of the 24 letters, which replay_static.py bought 68/71 exact holds with. A
    # relaxed hand still never emits (idle_gate.py, 0 of 43 holds) -- a stranger's D does.
    em_z, top_z = emissions(letters == "Z", DEFAULT)
    n_z = int(em_z.sum())
    assert 25 <= n_z <= 50, f"{n_z} of 155 Z stills emit a letter at the shipped floor"
    assert set(top_z[em_z]) <= {"X", "P", "D", "G", "N", "Q"}, dict(Counter(top_z[em_z]))

    flat75 = replace(DEFAULT, VOTE_PROB=0.75, VOTE_PROB_FLOOR=0.75, VOTE_PROB_LETTER={})
    em_z75, _ = emissions(letters == "Z", flat75)
    assert int(em_z75.sum()) < n_z, ("the previous release's flat 0.75 must be the stricter of the "
                                     f"two on these stills; got {em_z75.sum()} vs {n_z}")


def test_the_stills_are_the_unused_part_of_aslnow():
    """These 248 records are in no training set and no published fold. If that ever changes,
    every number in this file becomes a training score and stops meaning anything."""
    merged, sources = ST.load_strangers()
    assert sum(len(p) for p in merged) > 0 and len(merged) == len(ST.LETTERS)
    P, letters = all_records()
    jz = P[~np.isin(letters, ST.LETTERS)]
    assert len(jz) == 248
    # the training block is 24 per-class arrays of the 24 static letters; a J or Z still can
    # only get in through a mislabeled row, so compare the frames themselves
    train = np.concatenate([p for p in merged if len(p)], axis=0)
    keys = {r.tobytes() for r in np.round(train, 6)}
    leaked = sum(1 for r in np.round(jz, 6) if r.tobytes() in keys)
    assert leaked == 0, f"{leaked} of the 248 J/Z stills are in the strangers training block"
    assert "aslnow" in sources


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

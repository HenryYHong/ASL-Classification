"""Re-extract landmarks from RandomForest/data/ preserving frame order.

The original notebook iterated os.listdir(), which yields 0.jpg, 1.jpg, 10.jpg, 11.jpg, ...
That is fine for a bag of independent frames but destroys the temporal order, and every
motion feature depends on it. Each class folder holds 100 consecutive frames of one held
handshape, so in numeric order it is a real short video clip of a static sign.

Writes temporal/static_sequences.npz:
    lm      float32 (24, 100, 21, 3)  landmarks, NaN where no hand was found
    found   bool    (24, 100)         whether MediaPipe returned a hand
    handed  <U8     (24, 100)         MediaPipe's Left/Right label per frame

The handedness label is recorded because the live pipeline mirrors left hands onto a single
convention, and training must apply the identical transform. It did not, which meant the static
classifier was trained on unmirrored landmarks and fed mirrored ones at inference -- every
chiral letter failed while near-symmetric ones like B kept working.

DETECTOR PATH vs TRACKING PATH. The committed static_sequences.npz was extracted with
static_image_mode=True (min_detection_confidence 0.3): every frame goes through the palm
detector and a fresh landmark fit, the way the original notebook processed a bag of JPEGs.
Every other source of landmarks in this project runs MediaPipe in TRACKING mode
(static_image_mode=False, min_detection_confidence 0.5, min_tracking_confidence 0.3-0.5):
collect_motion.py for static_s2/s3/s4.npz and the motion takes, live_demo.py, calibrate.py
--live and ingest_external.py. In tracking mode the landmarker re-fits from the previous
frame's hand instead of re-detecting, and its landmarks are temporally smoother and differ
slightly in per-joint statistics. So 2,378 of the 4,878 static training frames (48.7%)
carry detector-path landmarks while the runtime serves tracking-path ones -- a train/serve
difference that the jitter augmentation in train_static.py is measured to absorb (the
cross-day fold is within +-0.03 of the same frames re-extracted with the browser's Tasks
landmarker), but that a re-extraction would remove outright. --tracking reproduces the
runtime path (tracking mode, detection 0.5, tracking 0.3, one Hands instance per class
burst so no track carries across letters) and writes to --out; it has NOT been run for the
committed file, because the archive's cross-day numbers, the S1 golden cases and the vote
thresholds were all measured on the detector-path landmarks and would have to be re-derived
together. Run it, retrain, and re-measure as one change.
"""
import argparse
import os
import numpy as np
import cv2
import mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "RandomForest", "data")
OUT = os.path.join(os.path.dirname(__file__), "static_sequences.npz")
N_CLASSES, N_FRAMES, N_LM = 24, 100, 21


def hands_for(tracking):
    """The MediaPipe configuration of each path. The tracking one matches live_demo.py and
    ingest_external.py exactly; the detector one is what the committed file was built with."""
    if tracking:
        return dict(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.3)
    return dict(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", action="store_true",
                    help="extract in the runtime's tracking mode instead of the detector path "
                         "the committed file used (see the docstring before running this)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    lm = np.full((N_CLASSES, N_FRAMES, N_LM, 3), np.nan, dtype=np.float32)
    found = np.zeros((N_CLASSES, N_FRAMES), dtype=bool)
    handed = np.full((N_CLASSES, N_FRAMES), "", dtype="<U8")

    # One Hands instance for the whole run, not one per image: the notebook rebuilt the
    # graph 2,400 times, which is why its output carries 2,400 gl_context lines. In tracking
    # mode it is one instance per class burst instead, so the tracker never carries a hand
    # from the last frame of one letter into the first frame of the next.
    def extract(hands, c):
        for f in range(N_FRAMES):
            path = os.path.join(DATA_DIR, str(c), f"{f}.jpg")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            pts = res.multi_hand_landmarks[0].landmark
            lm[c, f] = [(p.x, p.y, p.z) for p in pts]
            found[c, f] = True
            if res.multi_handedness:
                handed[c, f] = res.multi_handedness[0].classification[0].label
        print(f"class {c:2d}: {found[c].sum():3d}/100 frames with a hand", flush=True)

    cfg = hands_for(args.tracking)
    print(f"extracting with {cfg} ({'tracking' if args.tracking else 'detector'} path)")
    if args.tracking:
        for c in range(N_CLASSES):
            with mp.solutions.hands.Hands(**cfg) as hands:
                extract(hands, c)
    else:
        with mp.solutions.hands.Hands(**cfg) as hands:
            for c in range(N_CLASSES):
                extract(hands, c)

    np.savez_compressed(args.out, lm=lm, found=found, handed=handed,
                        mode=np.array("tracking" if args.tracking else "static_image"))
    print(f"\nwrote {args.out}  ({found.sum()} of {N_CLASSES * N_FRAMES} frames usable)")


if __name__ == "__main__":
    main()

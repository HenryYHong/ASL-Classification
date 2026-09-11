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
"""
import os
import numpy as np
import cv2
import mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "RandomForest", "data")
OUT = os.path.join(os.path.dirname(__file__), "static_sequences.npz")
N_CLASSES, N_FRAMES, N_LM = 24, 100, 21


def main():
    lm = np.full((N_CLASSES, N_FRAMES, N_LM, 3), np.nan, dtype=np.float32)
    found = np.zeros((N_CLASSES, N_FRAMES), dtype=bool)
    handed = np.full((N_CLASSES, N_FRAMES), "", dtype="<U8")

    # One Hands instance for the whole run, not one per image: the notebook rebuilt the
    # graph 2,400 times, which is why its output carries 2,400 gl_context lines.
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=0.3) as hands:
        for c in range(N_CLASSES):
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

    np.savez_compressed(OUT, lm=lm, found=found, handed=handed)
    print(f"\nwrote {OUT}  ({found.sum()} of {N_CLASSES * N_FRAMES} frames usable)")


if __name__ == "__main__":
    main()

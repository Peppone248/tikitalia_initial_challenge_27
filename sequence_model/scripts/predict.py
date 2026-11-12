"""Per-frame probabilities and decoded labels for one match.
Usage: python sequence_model/scripts/predict.py --weights W.npz --match M(.npz IDSSE | .xml ATD) --out DIR
Writes DIR/probs.npz (p_h, p_a, p_d, p_change, float32, one row per video frame) and DIR/labels.csv
(frame,label; H/A/D on the graded frames, X elsewhere)."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from seqmodel.decode import decode  # noqa: E402
from seqmodel.infer import NumpyTCN  # noqa: E402
from seqmodel.match import CFG, load_atd, load_idsse  # noqa: E402

_LETTER = {CFG["labels"][k]: k for k in "HADX"}


def load_match(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_idsse(path)
    if ext == ".xml":
        return load_atd(path, os.path.splitext(path)[0] + ".txt")
    raise ValueError(f"unsupported match file {path!r}: expected .npz (IDSSE) or .xml (ATD)")


def run(weights, match_path, out, beta=0.0, w=1.0, trans=None, prior=None):
    m = load_match(match_path)
    p3, pc = NumpyTCN.load(weights).predict_match(m)
    lab = decode(p3, pc, m.period, w, beta, trans, prior).copy()
    lab[~np.isin(m.period, m.graded_periods)] = CFG["labels"]["X"]
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "probs.npz"), p_h=p3[:, 0].astype(np.float32),
             p_a=p3[:, 1].astype(np.float32), p_d=p3[:, 2].astype(np.float32),
             p_change=pc.astype(np.float32))
    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        f.write("frame,label\n")
        f.write("".join(f"{i},{_LETTER[int(c)]}\n" for i, c in enumerate(lab)))
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--w", type=float, default=1.0)
    a = ap.parse_args()
    run(a.weights, a.match, a.out, a.beta, a.w)


if __name__ == "__main__":
    main()

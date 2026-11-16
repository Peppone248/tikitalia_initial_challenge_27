"""Hybrid heads from two saved LOMO runs, no retraining: P(home) from run B (e.g. the GRU), P(dead)
and the boundary probabilities from run A (the boosted trees), decoded with FIXED decoder
settings given on the command line (chosen before looking: those of run A's final model).

    python changepoint_model/scripts/eval_hybrid.py --dir outputs/cpm_lomo \
        --a v2_all18 --b v3_gru_all18 --gamma 8 --w 0.25

Prior of the emissions: the class shares of each held-out file's own decoded labels are NOT used;
a fixed prior (H, A, D) = (0.3, 0.3, 0.4) is used for all, the same for both rows, so the
comparison A vs hybrid is paired and decided only by the swapped head.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cpm import decode as dec  # noqa: E402
from cpm.evaluate import score  # noqa: E402
from cpm.match import X  # noqa: E402

PRIOR = (0.3, 0.3, 0.4)


def periods_from_gt(gt):
    g = gt != X
    edges = np.flatnonzero(np.diff(np.concatenate(([0], g.astype(int), [0]))))
    period = np.zeros(len(gt), dtype=np.int16)
    for k, (s, e) in enumerate(zip(edges[0::2], edges[1::2]), 1):
        period[s:e] = k
    return period


def run(p_home, src, gt, gamma, w, k=12):
    period = periods_from_gt(gt)
    logE = dec.emissions(p_home, src["p_dead"], PRIOR, w=w)
    hc, hs, hr = dec.hazards_from_boundaries(src["p_change"], src["p_stop"], src["p_restart"],
                                             k=k, gamma=gamma)
    return dec.decode_match(logE, hc, hs, hr, period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--scenario", default="degraded")
    ap.add_argument("--seed", default="2027")
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--w", type=float, default=0.25)
    args = ap.parse_args()
    rows = []
    prefix_b = "pred_{}_{}_{}_".format(args.b, args.scenario, args.seed)
    for pb in sorted(glob.glob(os.path.join(args.dir, prefix_b + "*.npz"))):
        name = os.path.basename(pb)[len(prefix_b):-4]
        pa = os.path.join(args.dir, "pred_{}_{}_{}_{}.npz".format(args.a, args.scenario, args.seed, name))
        if not os.path.isfile(pa):
            continue
        A, B = np.load(pa), np.load(pb)
        gt = A["gt"]
        ra = score(gt, run(A["p_home"], A, gt, args.gamma, args.w))
        rh = score(gt, run(B["p_home"], A, gt, args.gamma, args.w))
        rb = score(gt, B["labels"])
        rows.append((name, ra, rh, rb))
        print("{:<12} A {:.3f} (team {:.3f}) | hybrid {:.3f} (team {:.3f}) | B as decoded {:.3f}".format(
            name, ra["spellF1"], ra["team_acc"], rh["spellF1"], rh["team_acc"], rb["spellF1"]), flush=True)
    d = [r[2]["spellF1"] - r[1]["spellF1"] for r in rows]
    print("mean A {:.4f} | hybrid {:.4f} | B {:.4f} | hybrid - A {:+.4f} ({}/{})".format(
        np.mean([r[1]["spellF1"] for r in rows]), np.mean([r[2]["spellF1"] for r in rows]),
        np.mean([r[3]["spellF1"] for r in rows]), np.mean(d), sum(x > 0 for x in d), sum(x < 0 for x in d)))


if __name__ == "__main__":
    main()

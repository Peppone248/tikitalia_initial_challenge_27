"""Where are the changes lost? Diagnostics on saved LOMO predictions (pred_*.npz).

    python changepoint_model/scripts/diag_changes.py outputs/cpm_lomo/pred_v1_degraded_2027_*.npz
"""

import glob
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from cpm.match import H, A, D, X, spells  # noqa: E402


def runs_of(gt):
    g = gt != X
    e = np.flatnonzero(np.diff(np.concatenate(([0], g.astype(int), [0]))))
    return list(zip(e[0::2], e[1::2]))


def main(paths):
    agg = {"true_changes": 0, "found_1s": 0, "found_2s": 0, "pred_changes": 0, "false_2s": 0,
           "next_lt1": [0, 0], "next_1_2": [0, 0], "next_2_5": [0, 0], "next_ge5": [0, 0],
           "unseen": [0, 0], "seen": [0, 0], "poss_true": 0, "poss_pred": 0,
           "stops_true": 0, "stops_found_2s": 0, "restarts_true": 0, "restarts_found_2s": 0}
    for path in paths:
        z = np.load(path)
        gt, pred = z["gt"], z["labels"]
        pch = z["p_change"] if "p_change" in z else None
        for s, e in runs_of(gt):
            g, p = gt[s:e], pred[s:e]
            per = np.ones(e - s, dtype=int)
            tc = [t for t in np.flatnonzero((g[1:] != g[:-1]) & (g[1:] < D) & (g[:-1] < D)) + 1]
            pc = [t for t in np.flatnonzero((p[1:] != p[:-1]) & (p[1:] < D) & (p[:-1] < D)) + 1]
            pc_arr = np.array(pc) if pc else np.array([-10 ** 9])
            tc_arr = np.array(tc) if tc else np.array([-10 ** 9])
            sp = spells(g, per)
            agg["poss_true"] += sum(1 for x in sp if x[0] < D)
            agg["poss_pred"] += sum(1 for x in spells(p, per) if x[0] < D)
            for t in tc:
                agg["true_changes"] += 1
                # a predicted change to the same new team within tolerance
                same = [q for q in pc if abs(q - t) <= 50 and p[min(q + 1, e - s - 1)] == g[t]]
                f1 = any(abs(q - t) <= 25 for q in same)
                f2 = len(same) > 0
                agg["found_1s"] += f1
                agg["found_2s"] += f2
                # length of the next possession
                nxt = [x for x in sp if x[1] == t]
                L = (nxt[0][2] - nxt[0][1]) / 25.0 if nxt else 0
                key = "next_lt1" if L < 1 else "next_1_2" if L < 2 else "next_2_5" if L < 5 else "next_ge5"
                agg[key][0] += 1
                agg[key][1] += f2
                # ball seen around the change?
                seen = z["p_home"] is not None
            for q in pc:
                agg["pred_changes"] += 1
                if not np.any(np.abs(tc_arr - q) <= 50):
                    agg["false_2s"] += 1
            ts = np.flatnonzero((g[1:] == D) & (g[:-1] < D)) + 1
            ps = np.flatnonzero((p[1:] == D) & (p[:-1] < D)) + 1
            agg["stops_true"] += len(ts)
            agg["stops_found_2s"] += sum(1 for t in ts if len(ps) and np.min(np.abs(ps - t)) <= 50)
            tr = np.flatnonzero((g[1:] < D) & (g[:-1] == D)) + 1
            pr = np.flatnonzero((p[1:] < D) & (p[:-1] == D)) + 1
            agg["restarts_true"] += len(tr)
            agg["restarts_found_2s"] += sum(1 for t in tr if len(pr) and np.min(np.abs(pr - t)) <= 50)
    n = agg["true_changes"]
    print("true H<->A changes {}  found within 1 s {:.0%}  within 2 s {:.0%}".format(
        n, agg["found_1s"] / n, agg["found_2s"] / n))
    print("predicted changes {}  of which not within 2 s of a true one {:.0%}".format(
        agg["pred_changes"], agg["false_2s"] / max(1, agg["pred_changes"])))
    for k in ("next_lt1", "next_1_2", "next_2_5", "next_ge5"):
        a, b = agg[k]
        print("  next possession {:<8} {:5d} ({:.0%} of changes)  found {:.0%}".format(
            k[5:], a, a / n, b / max(1, a)))
    print("possessions true {} pred {}".format(agg["poss_true"], agg["poss_pred"]))
    print("stops true {} found within 2 s {:.0%}; restarts true {} found {:.0%}".format(
        agg["stops_true"], agg["stops_found_2s"] / max(1, agg["stops_true"]),
        agg["restarts_true"], agg["restarts_found_2s"] / max(1, agg["restarts_true"])))


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(a)
    main(sorted(paths))

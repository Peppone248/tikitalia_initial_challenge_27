"""Where do the missing possessions come from? (point 2) -- IDSSE degraded, controlled labels.

A possession (H/A spell) ends either at a CHANGE (H<->A, no D between) or at a DEAD BALL. For each
version: how many true changes are found within +/-1 s and +/-2 s, how many true dead spells are
found (any predicted D overlapping), split by length, and how many predicted changes are false.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa
from hmm_v0 import paths  # noqa
from hmm_v0 import validation as V  # noqa
from hmm_v0.baseline_b import boosted_baseline, slow_dead  # noqa
import yaml  # noqa
from tikitalia.preprocessing import runs  # noqa


def changes(lab):
    i = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ("H", "A")) & np.isin(lab[:-1], ("H", "A")))
    return i + 1


def near(a, b, tol):
    if not len(b):
        return np.zeros(len(a), bool)
    j = np.searchsorted(b, a)
    d = np.full(len(a), np.inf)
    for k in (j - 1, j):
        ok = (k >= 0) & (k < len(b))
        d[ok] = np.minimum(d[ok], np.abs(b[k[ok]] - a[ok]))
    return d <= tol


def main(variant="degraded", ldir="outputs/validation_b"):
    vc = V.load_config(); stats = V.barath_stats(vc["degrade"]); ddir = paths.resolve(vc["data_dir"])
    tot = {}
    for i, m in enumerate(vc["matches"]):
        d = V.load_idsse(os.path.join(ddir, m + ".npz"))
        if variant == "degraded":
            d = V.degrade(d, vc["degrade"], stats, i)
        gt = V.controlled_labels(V.labels_of(d), vc["controlled_min_s"])
        c3 = np.load(os.path.join(paths.resolve(ldir), "labels_c_v3_{}_{}.npy".format(variant, m)))
        g = np.isin(d["period"], (1, 2))
        cb = yaml.safe_load(open(os.path.join(paths.HMM_DIR, "config_b", "baseline_b.yaml")))
        half = np.where(g, d["period"], 0).astype(np.int8)
        sd = slow_dead(d["ball_x"], d["px"], d["py"], half, cb["slow_dead"])
        B = boosted_baseline(d["ball_x"], d["ball_y"], d["px"], d["py"], d["player_team"], g,
                             (c3 == "D") | sd, window=cb["window"], min_poss_s=cb["min_poss_s"],
                             unseen=cb["unseen"], spread_smooth=cb["spread_smooth"])
        for name, p in (("c_v3", c3), ("B", B)):
            t = tot.setdefault(name, {})
            tc, pc = changes(gt), changes(p)
            add = lambda k, v: t.__setitem__(k, t.get(k, 0) + v)
            add("true_changes", len(tc)); add("pred_changes", len(pc))
            add("found_1s", near(tc, pc, 25).sum()); add("found_2s", near(tc, pc, 50).sum())
            add("pred_false_2s", (~near(pc, tc, 50)).sum())
            # true change seen as a D in pred (pred D within +/-1 s): a split, not a change
            pdead = np.flatnonzero(p == "D")
            add("change_as_D_1s", (near(tc, pdead, 25) & ~near(tc, pc, 50)).sum())
            gd = [(s, e) for s, e in runs(gt == "D")]
            for lo, hi, key in ((0, 5, "D<5s"), (5, 15, "D5-15s"), (15, 1e9, "D>=15s")):
                sel = [(s, e) for s, e in gd if lo * 25 <= e - s < hi * 25]
                add("true_" + key, len(sel))
                add("found_" + key, sum((p[s:e] == "D").any() for s, e in sel))
            add("true_spells", len(runs(np.isin(gt, ("H", "A")))) if False else int(sum(1 for _ in runs(gt == "H")) + sum(1 for _ in runs(gt == "A"))))
            add("pred_spells", int(sum(1 for _ in runs(p == "H")) + sum(1 for _ in runs(p == "A"))))
            add("pred_D_spells", len(runs(p == "D")))
    for name, t in tot.items():
        print("\n==", name, variant)
        print("possessions pred/true: {}/{} | dead spells pred/true: {}/{}".format(
            t["pred_spells"], t["true_spells"], t["pred_D_spells"],
            t["true_D<5s"] + t["true_D5-15s"] + t["true_D>=15s"]))
        print("true changes H<->A: {} | found +-1 s {:.0%}, +-2 s {:.0%} | seen as D instead {:.0%} | pred changes {} ({:.0%} false at 2 s)".format(
            t["true_changes"], t["found_1s"] / t["true_changes"], t["found_2s"] / t["true_changes"],
            t["change_as_D_1s"] / t["true_changes"], t["pred_changes"], t["pred_false_2s"] / max(1, t["pred_changes"])))
        for k in ("D<5s", "D5-15s", "D>=15s"):
            print("true dead {}: {} found {:.0%}".format(k, t["true_" + k], t["found_" + k] / max(1, t["true_" + k])))


if __name__ == "__main__":
    main(*sys.argv[1:])

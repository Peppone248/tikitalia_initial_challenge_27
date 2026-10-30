"""Approach B on the IDSSE validation: boosted baseline (nearest player + vote + our D + minimum
possession), scored with the official scorer against the same degraded/clean tracking and labels
as scripts/run_validation.py.

    python HMM_first_imp/scripts/eval_baseline_b.py --labels-dir outputs/validation_b

Needs the c_v3 labels of a previous run_validation.py run with --st-sim (labels_c_v3_<variant>_<m>.npy):
their D is the dead-ball mask written on top of the baseline. Writes <labels-dir>/results_b.json/md.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0 import validation as V  # noqa: E402
from hmm_v0 import baseline_b as BB  # noqa: E402
from hmm_v0.baseline_b import boosted_baseline  # noqa: E402

KEYS = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1", "D_f1",
        "team_acc_gt_alive", "n_pred_spells", "n_gt_spells", "mean_spell_duration_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="outputs/validation_b")
    ap.add_argument("--variants", nargs="+", default=["degraded", "clean"])
    ap.add_argument("--dead-from", default="c_v3")
    ap.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args()
    vc = V.load_config()
    with open(os.path.join(paths.HMM_DIR, "config_b", "baseline_b.yaml")) as h:
        cb = yaml.safe_load(h)
    ldir = paths.resolve(args.labels_dir)
    stats = V.barath_stats(vc["degrade"])
    data_dir = paths.resolve(vc["data_dir"])
    out = {}
    for variant in args.variants:
        acc = {}
        for i, m in enumerate(vc["matches"]):
            f = os.path.join(ldir, "labels_{}_{}_{}.npy".format(args.dead_from, variant, m))
            if not os.path.isfile(f):
                print("missing", f)
                continue
            ref = np.load(f)
            d = V.load_idsse(os.path.join(data_dir, m + ".npz"))
            if variant == "degraded":
                d = V.degrade(d, vc["degrade"], stats, i)
            gt = V.controlled_labels(V.labels_of(d), vc["controlled_min_s"])
            graded = np.isin(d["period"], (1, 2))
            half = np.where(graded, d["period"], 0).astype(np.int8)
            sd = BB.slow_dead(d["ball_x"], d["px"], d["py"], half, cb["slow_dead"])
            dead3 = ref == "D"
            kw = dict(window=cb["window"], min_poss_s=cb["min_poss_s"], spread_smooth=cb["spread_smooth"])
            args_ = (d["ball_x"], d["ball_y"], d["px"], d["py"], d["player_team"], graded)
            preds = {
                "baseline50 (official)": V.baseline_labels(d, 50),
                "c_v3": ref,
                "B0: baseline50 + our D": boosted_baseline(*args_, dead3, window=50, min_poss_s=0, fill_unseen=False),
                "B1: + last team, w25, min 2 s": boosted_baseline(*args_, dead3, unseen="last", **kw),
                "B2: + spread when ball unseen": boosted_baseline(*args_, dead3, unseen="spread", **kw),
                "B (final): + D-SLOW": boosted_baseline(*args_, dead3 | sd, unseen="spread", **kw),
            }
            for k, p in preds.items():
                a = acc.setdefault(k, {"gt": [], "pred": [], "per": {}})
                a["gt"].append(gt)
                a["pred"].append(p)
                a["per"][m] = V.score(gt, p)[0]["spell_f1_mean"]
            print(variant, m, "done", flush=True)
        res = {}
        base = acc[args.dead_from]["per"]
        for k, a in acc.items():
            pool, _ = V.pooled(a["gt"], a["pred"], args.bootstrap)
            diffs = [a["per"][m] - base[m] for m in base]
            res[k] = {kk: pool.get(kk) for kk in KEYS}
            res[k]["paired_vs_" + args.dead_from] = float(np.mean(diffs))
            res[k]["wins"] = int(sum(x > 0 for x in diffs))
        out[variant] = res
        print("\n## {} (labels: controlled)".format(variant))
        print("| version | spellF1 [90% CI] | macroF1 | D F1 | team acc | poss pred/true | "
              "mean s | vs {} (wins/7) |".format(args.dead_from))
        print("|---|---|---|---|---|---|---|---|")
        for k, r in res.items():
            print("| {} | {:.3f} [{:.3f}–{:.3f}] | {:.3f} | {:.3f} | {:.3f} | {}/{} | {:.1f} | "
                  "{:+.3f} ({}) |".format(k, r["spell_f1_mean"], r["spell_bootstrap_low"],
                                         r["spell_bootstrap_high"], r["macro_f1"],
                                         r["D_f1"] or 0, r["team_acc_gt_alive"],
                                         r["n_pred_spells"], r["n_gt_spells"],
                                         r["mean_spell_duration_s"],
                                         r["paired_vs_" + args.dead_from], r["wins"]))
    with open(os.path.join(ldir, "results_b.json"), "w") as h:
        json.dump(out, h, indent=2, default=float)
    lines = ["# Approach B on IDSSE (7 matches, controlled labels, simulated Smart Tagging for the c_v3 D)", ""]
    for variant, res in out.items():
        lines += ["## tracking: " + variant, "",
                  "| version | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions pred / true | mean s | paired vs c_v3 (wins/7) |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, r in res.items():
            lines.append("| {} | {:.3f} [{:.3f}–{:.3f}] | {:.3f} | {:.3f} | {:.3f} | {} / {} | {:.1f} | {:+.3f} ({}) |".format(
                k, r["spell_f1_mean"], r["spell_bootstrap_low"], r["spell_bootstrap_high"], r["macro_f1"],
                r["D_f1"] or 0, r["team_acc_gt_alive"], r["n_pred_spells"], r["n_gt_spells"],
                r["mean_spell_duration_s"], r["paired_vs_" + args.dead_from], r["wins"]))
        lines.append("")
    with open(os.path.join(ldir, "results_b.md"), "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

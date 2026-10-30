"""Validation of pipeline versions on the IDSSE matches (other games, labels from the DFL tracking).

    python HMM_first_imp/scripts/run_validation.py                       # versions/variants of validation.yaml
    python HMM_first_imp/scripts/run_validation.py --matches J03WPY --variants degraded
    python HMM_first_imp/scripts/run_validation.py --versions v1 --set v1:hmm.hmm.team_change.closed_factor=5

For every variant (degraded = BAR-ATH-like tracking, clean = IDSSE as it is) and every match, the
SAME degraded tracking is given to every version, so the comparison is paired. Scores come from
evaluation/possession_metrics.evaluate (the official scorer): per match and pooled over all
matches, with a bootstrap interval on the pooled spellF1.

Writes outputs/validation/results.json, results.md and, per version/variant/match, the predicted
labels (labels_<version>_<variant>_<match>.npy) for later inspection. Reads no BAR-ATH ground truth.
"""

import argparse
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0 import validation as V  # noqa: E402
from hmm_v0.config import Config  # noqa: E402
from hmm_v0.diagnostics import fragmentation  # noqa: E402
from hmm_v0.run import run_match  # noqa: E402

LABEL_SETS = ("dfl", "controlled")
SOURCES = ["P-DRIB", "P-FREE", "P-PASS", "T-INTERCEPT", "T-REV", "R-TOUCH", "R-KO"]


def fmt(x, nd=3):
    return "–" if x is None else ("{:.%df}" % nd).format(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="validation.yaml (default: config/)")
    ap.add_argument("--matches", nargs="+", default=None)
    ap.add_argument("--versions", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="VERSION:KEY=VALUE")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--st-sim", action="store_true",
                    help="attach a simulated Smart Tagging feed to every IDSSE match "
                         "(validation.yaml smart_tagging_sim)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse labels_<version>_<variant>_<match>.npy of an earlier run")
    args = ap.parse_args()

    vc = V.load_config(args.config)
    matches = args.matches or vc["matches"]
    versions = {k: dict(v) for k, v in vc["versions"].items()
                if not args.versions or k in args.versions}
    for item in args.set:
        ver, kv = item.split(":", 1)
        key, value = kv.split("=", 1)
        versions[ver].setdefault("set", {})[key] = yaml.safe_load(value)
    variants = args.variants or vc["variants"]
    out_dir = paths.resolve(args.out_dir or vc["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    data_dir = paths.resolve(vc["data_dir"])
    missing = [m for m in matches if not os.path.isfile(os.path.join(data_dir, m + ".npz"))]
    if missing:
        print("missing in {}: {} -- run validation/fetch_idsse.py".format(data_dir, missing))
        matches = [m for m in matches if m not in missing]
        if not matches:
            return 1

    stats = V.barath_stats(vc["degrade"])
    results = {"matches": matches, "versions": versions, "variants": variants,
               "degrade": vc["degrade"], "barath_stats_used": stats is not None, "runs": {}}
    t0 = time.time()
    for variant in variants:
        per = {(v, ls): {"gt": [], "pred": [], "rows": {}} for v in versions
               for ls in LABEL_SETS}
        for i, m in enumerate(matches):
            d = V.load_idsse(os.path.join(data_dir, m + ".npz"))
            st = V.simulate_smart_tagging(d, vc["smart_tagging_sim"], i) if args.st_sim else []
            if variant == "degraded":
                d = V.degrade(d, vc["degrade"], stats, i)
            gts = {"dfl": V.labels_of(d)}
            gts["controlled"] = V.controlled_labels(gts["dfl"], vc["controlled_min_s"])
            miss = V.ball_missing_share(d)
            for ver, spec in versions.items():
                cache = os.path.join(out_dir, "labels_{}_{}_{}.npy".format(ver, variant, m))
                if "baseline_window" in spec:
                    pred, frag = V.baseline_labels(d, int(spec["baseline_window"])), None
                elif args.reuse and os.path.isfile(cache):
                    pred, frag = np.load(cache), None
                else:
                    cfg = Config(paths.resolve(spec["config_dir"]), spec.get("set") or {})
                    match = V.match_from(d, cfg.params)
                    match.smart_tagging = st
                    res = run_match(match, V.pitch_for(d), cfg)
                    pred = res["labels"]
                    frag = fragmentation(pred, match, res["outputs"], SOURCES, 25)
                    np.save(cache, pred)
                for ls, gt in gts.items():
                    row, _ = V.score(gt, pred)
                    row["ball_missing_share"] = round(miss, 3)
                    row["pred_D_share"] = round(float((pred[gt != "X"] == "D").mean()), 3)
                    row["gt_D_share"] = round(float((gt[gt != "X"] == "D").mean()), 3)
                    row["fragmentation"] = frag
                    per[(ver, ls)]["gt"].append(gt)
                    per[(ver, ls)]["pred"].append(pred)
                    per[(ver, ls)]["rows"][m] = row
                    print("[{:6.0f}s] {:8s} {:4s} {} labels={:10s}: spellF1 {}  macroF1 {}  "
                          "poss {}/{}".format(time.time() - t0, variant, ver, m, ls,
                                              fmt(row["spell_f1_mean"]), fmt(row["macro_f1"]),
                                              row["n_pred_spells"], row["n_gt_spells"]),
                          flush=True)
        for (ver, ls), acc in per.items():
            pool, _ = V.pooled(acc["gt"], acc["pred"], int(vc["bootstrap"]))
            results["runs"].setdefault(variant, {}).setdefault(ls, {})[ver] = {
                "pooled": pool, "per_match": acc["rows"]}
        # paired difference per match (same tracking for every version)
        if len(versions) > 1:
            names = list(versions)
            base = names[0]
            for ls in LABEL_SETS:
                for ver in names[1:]:
                    diffs = [per[(ver, ls)]["rows"][m]["spell_f1_mean"]
                             - per[(base, ls)]["rows"][m]["spell_f1_mean"] for m in matches]
                    results["runs"][variant][ls][ver]["paired_spell_diff_vs_" + base] = {
                        "mean": float(np.mean(diffs)), "per_match": dict(zip(matches, diffs)),
                        "wins": int(sum(x > 0 for x in diffs)), "n": len(diffs)}

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2, default=float)
    lines = ["# Validation on IDSSE ({} matches)".format(len(matches)), "",
             "labels: dfl = DFL BallStatus/BallPossession as provided; controlled = the same with "
             "possession flips shorter than {} s between two spells of the other team removed "
             "(CHALLENGE.md: a touch without control is not a change)".format(vc["controlled_min_s"]),
             ""]
    for variant in variants:
        for ls in LABEL_SETS:
            block = results["runs"][variant][ls]
            lines += ["## tracking: {} · labels: {}".format(variant, ls), "",
                      "| version | spellF1 pooled [90% CI] | macroF1 | D F1 | alive/dead F1 | "
                      "team acc | possessions pred / true | mean poss s pred / true | "
                      "D share pred / true |", "|---|---|---|---|---|---|---|---|---|"]
            for ver in versions:
                p = block[ver]["pooled"]
                rows = block[ver]["per_match"].values()
                lines.append("| {} | {} [{} – {}] | {} | {} | {} | {} | {} / {} | {} / {} | "
                             "{} / {} |".format(
                                 ver, fmt(p["spell_f1_mean"]), fmt(p["spell_bootstrap_low"]),
                                 fmt(p["spell_bootstrap_high"]), fmt(p["macro_f1"]),
                                 fmt(p.get("D_f1")), fmt(p["alive_dead_f1"]),
                                 fmt(p["team_acc_gt_alive"]), p["n_pred_spells"], p["n_gt_spells"],
                                 fmt(p["mean_spell_duration_s"], 1),
                                 fmt(p["gt_mean_spell_duration_s"], 1),
                                 fmt(np.mean([r["pred_D_share"] for r in rows]), 2),
                                 fmt(np.mean([r["gt_D_share"] for r in rows]), 2)))
            for ver, r in block.items():
                keys = [k for k in r if k.startswith("paired")]
                if keys:
                    k = keys[0]
                    lines += ["", "paired spellF1 difference {} - {}: mean {:+.3f}, better on "
                              "{}/{} matches".format(ver, k.split("_vs_")[1], r[k]["mean"],
                                                     r[k]["wins"], r[k]["n"])]
            lines += ["", "| match | " + " | ".join("{} spellF1 / macroF1".format(v)
                                                    for v in versions) + " | ball missing |",
                      "|---" * (len(versions) + 2) + "|"]
            for m in matches:
                cells = ["{} / {}".format(fmt(block[v]["per_match"][m]["spell_f1_mean"]),
                                          fmt(block[v]["per_match"][m]["macro_f1"]))
                         for v in versions]
                miss = block[list(versions)[0]]["per_match"][m]["ball_missing_share"]
                lines.append("| {} | {} | {:.0%} |".format(m, " | ".join(cells), miss))
            lines.append("")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwritten: {}".format(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

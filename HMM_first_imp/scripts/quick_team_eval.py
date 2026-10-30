"""Fast A/B of the TEAM part of approach C on IDSSE: the dead ball is taken from cached labels
of an earlier validation run (same D for every variant), only the possession logic is re-run.

    python HMM_first_imp/scripts/quick_team_eval.py --dead-from outputs/validation_clean \
        --dead-version c_v2 --variants degraded clean \
        --try v2=config_c v3=config_c3 "v3_nogk=config_c3:segment.goalkeeper.enabled=false"

Each --try is NAME=CONFIG_DIR[:key=value[;key=value]]. Scores with the official scorer, pooled
over the matches, paired per match against the first variant. IDSSE only (other games).
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hmm_v0 import paths, segment_c, votes                          # noqa: E402
from hmm_v0 import validation as V                                   # noqa: E402
from hmm_v0.config import Config                                     # noqa: E402
from hmm_v0.run import build_context, apply_rules                    # noqa: E402


def parse_try(spec):
    name, rest = spec.split("=", 1)
    cdir, _, sets = rest.partition(":")
    over = {}
    for kv in filter(None, sets.split(";")):
        k, v = kv.split("=", 1)
        over[k] = yaml.safe_load(v)
    return name, cdir, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead-from", default="outputs/validation_clean")
    ap.add_argument("--dead-version", default="c_v2")
    ap.add_argument("--variants", nargs="+", default=["degraded"])
    ap.add_argument("--matches", nargs="+")
    ap.add_argument("--try", dest="tries", nargs="+", required=True)
    ap.add_argument("--labels", default="controlled")
    ap.add_argument("--st-sim", action="store_true",
                    help="attach a simulated Smart Tagging feed (validation.yaml smart_tagging_sim)")
    args = ap.parse_args()
    vc = V.load_config()
    matches = args.matches or vc["matches"]
    stats = V.barath_stats(vc["degrade"])
    tries = [parse_try(t) for t in args.tries]
    t0 = time.time()
    for variant in args.variants:
        acc = {n: {"gt": [], "pred": [], "rows": {}} for n, _, _ in tries}
        for i, m in enumerate(matches):
            d = V.load_idsse(os.path.join(paths.resolve(vc["data_dir"]), m + ".npz"))
            st = V.simulate_smart_tagging(d, vc["smart_tagging_sim"], i) if args.st_sim else []
            if variant == "degraded":
                d = V.degrade(d, vc["degrade"], stats, i)
            gt = V.labels_of(d)
            if args.labels == "controlled":
                gt = V.controlled_labels(gt, vc["controlled_min_s"])
            dead_lab = np.load(os.path.join(paths.resolve(args.dead_from), "labels_{}_{}_{}.npy"
                                            .format(args.dead_version, variant, m)))
            pitch = V.pitch_for(d)
            base_cache = {}
            for name, cdir, over in tries:
                cfg = Config(paths.resolve(cdir), over)
                key = cdir
                if key not in base_cache:
                    match = V.match_from(d, cfg.params)
                    match.smart_tagging = st
                    ctx = build_context(match, pitch, cfg)
                    rules, outs = apply_rules(match, pitch, ctx, cfg)
                    E, _dead, _w = votes.collect(rules, outs, match.n, cfg.hmm["votes"])
                    base_cache[key] = (match, ctx, rules, outs, E)
                match, ctx, rules, outs, E = base_cache[key]
                seg = dict(cfg.segment)
                seg["dead"] = dict(seg["dead"], source="hmm")
                lab, info = segment_c.decide(match, ctx, rules, outs, E, seg, dead_lab,
                                             p=cfg.params, pitch=pitch)
                row, _ = V.score(gt, lab)
                acc[name]["gt"].append(gt)
                acc[name]["pred"].append(lab)
                acc[name]["rows"][m] = row
                print("[{:5.0f}s] {} {} {:10s} spellF1 {:.3f} macroF1 {:.3f} poss {}/{}".format(
                    time.time() - t0, variant, m, name, row["spell_f1_mean"], row["macro_f1"],
                    row["n_pred_spells"], row["n_gt_spells"]), flush=True)
        print("\n== {} · labels {} ==".format(variant, args.labels))
        first = tries[0][0]
        for name, _, _ in tries:
            pool, _ = V.pooled(acc[name]["gt"], acc[name]["pred"], 0)
            diffs = [acc[name]["rows"][m]["spell_f1_mean"] - acc[first]["rows"][m]["spell_f1_mean"]
                     for m in matches]
            print("{:12s} D F1 {:.3f} (P {:.2f} R {:.2f})  spellF1 {:.3f}  macroF1 {:.3f}  team acc {:.3f}  poss {}/{}  "
                  "dur {:.1f}/{:.1f}  vs {}: {:+.3f} ({}/{})".format(
                      name, pool["D_f1"], pool["D_precision"], pool["D_recall"], pool["spell_f1_mean"], pool["macro_f1"], pool["team_acc_gt_alive"],
                      pool["n_pred_spells"], pool["n_gt_spells"], pool["mean_spell_duration_s"],
                      pool["gt_mean_spell_duration_s"], first, np.mean(diffs),
                      sum(x > 0 for x in diffs), len(diffs)), flush=True)


if __name__ == "__main__":
    main()

"""Leave-one-match-out evaluation on external matches, official scorer.

    python changepoint_model/scripts/eval_lomo.py --metrica /path/to/metrica/data \
        --idsse data/idsse --scenarios degraded clean --seeds 2027 --out outputs/cpm_lomo

For every scenario, seed and held-out match: the models are fitted on the other matches (degraded
the same way), (gamma, w) of the decoder are chosen on the training matches, the held-out match is
scored once. Rows: baseline50, constant-hazard HMM (same emissions, a priori rates), the
boundary-driven HMM, and ablations.
"""

import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cpm.loaders import load_metrica_csv, load_metrica_epts, load_idsse  # noqa: E402
from cpm import pipeline  # noqa: E402
from cpm.evaluate import score, baseline  # noqa: E402
from cpm.match import X  # noqa: E402

GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WS = (0.25, 0.5, 1.0, 2.0)


def load_matches(args):
    ms = []
    if args.metrica:
        cache = os.path.join(args.out, "cache")
        os.makedirs(cache, exist_ok=True)
        ms.append(load_metrica_csv(args.metrica, 1))
        ms.append(load_metrica_csv(args.metrica, 2))
        ms.append(load_metrica_epts(args.metrica, 3, cache=os.path.join(cache, "metrica_g3.npz")))
    if args.idsse:
        for p in sorted(glob.glob(os.path.join(args.idsse, "*.npz"))):
            ms.append(load_idsse(p))
    if args.pff:
        for p in sorted(glob.glob(os.path.join(args.pff, "*.npz"))):
            ms.append(load_idsse(p))
    if args.only:
        ms = [m for m in ms if m.name in args.only]
    return ms


def gt_codes(m):
    g = m.labels.copy()
    g[~m.graded()] = X
    return g


def choose_decoder(models, preps, gammas, ws, hazard, min_frames=None, gamma_sr=None):
    """(gamma, w) with the best mean spellF1 over the training matches (in-sample)."""
    best, best_v = None, -1
    probs = [pipeline.predict(models, P) for P in preps]
    for gamma, w in itertools.product(gammas, ws):
        vals = []
        for P, pr in zip(preps, probs):
            lab = pipeline.decode(pr, P.period, models, gamma=gamma, w=w, hazard=hazard,
                                  min_frames=min_frames, gamma_sr=gamma_sr)
            vals.append(score(gt_codes(P.match), lab)["spellF1"])
        v = float(np.mean(vals))
        if v > best_v:
            best_v, best = v, (gamma, w)
    return best, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica")
    ap.add_argument("--idsse")
    ap.add_argument("--pff")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--scenarios", nargs="+", default=["degraded", "clean"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2027])
    ap.add_argument("--backend", default="hgb")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default="outputs/cpm_lomo")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--min-ha", type=int, default=20, help="minimum possession, frames")
    ap.add_argument("--min-d", type=int, default=50, help="minimum stoppage, frames")
    ap.add_argument("--gamma-sr", type=float, default=None, help="fixed gamma for stop / restart")
    ap.add_argument("--skip-constant", action="store_true")
    ap.add_argument("--heads", default="frame", choices=["frame", "seq"])
    ap.add_argument("--tune-matches", type=int, default=0,
                    help="choose (gamma, w) on this many training matches only (0 = all)")
    ap.add_argument("--gammas", nargs="*", type=float, default=list(GAMMAS))
    ap.add_argument("--ws", nargs="*", type=float, default=list(WS))
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="only these matches are held out (the others are training only)")
    ap.add_argument("--seq-epochs", type=int, default=30)
    ap.add_argument("--seq-hidden", type=int, default=64)
    args = ap.parse_args()
    min_frames = {0: args.min_ha, 1: args.min_ha, 2: args.min_d}
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    matches = load_matches(args)
    print("matches:", [m.name for m in matches], "({:.0f} s)".format(time.time() - t0), flush=True)
    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            t1 = time.time()
            preps = [pipeline.prepare(m, scenario, seed + i) for i, m in enumerate(matches)]
            print("[{} seed {}] features {:.0f} s".format(scenario, seed, time.time() - t1), flush=True)
            for i, P in enumerate(preps):
                if args.holdout and P.match.name not in args.holdout:
                    continue
                gt = gt_codes(P.match)
                train = [Q for j, Q in enumerate(preps) if j != i]
                t2 = time.time()
                models = pipeline.fit(train, {"backend": args.backend, "k": args.k,
                                              "stack": not args.no_stack, "min_frames": min_frames,
                                              "gamma_sr": args.gamma_sr, "heads": args.heads,
                                              "subsample": args.subsample,
                                              "seq": {"epochs": args.seq_epochs,
                                                      "hidden": args.seq_hidden}},
                                      verbose=True, log=lambda *a: print(*a, flush=True))
                tf = time.time() - t2
                pr = pipeline.predict(models, P)
                tune = train[:args.tune_matches] if args.tune_matches else train
                GAM, WSS = tuple(args.gammas), tuple(args.ws)
                # baseline
                r = score(gt, baseline(P.match, 50))
                rows.append(dict(scenario=scenario, seed=seed, match=P.match.name, model="baseline50", **r))
                # constant hazards (no boundary heads)
                if not args.skip_constant:
                    (g_c, w_c), _ = choose_decoder(models, tune, GAM, WSS, "constant")
                    lab = pipeline.decode(pr, P.period, models, gamma=g_c, w=w_c, hazard="constant")
                    r = score(gt, lab)
                    rows.append(dict(scenario=scenario, seed=seed, match=P.match.name,
                                     model="hmm_constant", gamma=g_c, w=w_c, **r))
                else:
                    g_c = w_c = None
                    rows.append(dict(rows[-1], model="hmm_constant"))
                # boundary hazards
                (g_b, w_b), _ = choose_decoder(models, tune, GAM, WSS, "boundary")
                lab_b = pipeline.decode(pr, P.period, models, gamma=g_b, w=w_b, hazard="boundary")
                r = score(gt, lab_b)
                rows.append(dict(scenario=scenario, seed=seed, match=P.match.name, model="hmm_boundary",
                                 gamma=g_b, w=w_b, **r))
                print("  held-out {:<12} fit {:4.0f} s | base {:.3f} | const {:.3f} (g {} w {}) | "
                      "boundary {:.3f} (g {} w {}) poss {}/{} D-F1 {:.3f} team {:.3f}".format(
                          P.match.name, tf, rows[-3]["spellF1"], rows[-2]["spellF1"], g_c, w_c,
                          r["spellF1"], g_b, w_b, r["n_pred"], r["n_gt"], r["D_f1"], r["team_acc"]),
                      flush=True)
                np.savez_compressed(os.path.join(args.out, "pred_{}_{}_{}_{}.npz".format(
                    args.tag, scenario, seed, P.match.name)), labels=lab_b, gt=gt, **pr)
    path = os.path.join(args.out, "lomo_{}.json".format(args.tag))
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(summary(rows))
    with open(os.path.join(args.out, "lomo_{}.md".format(args.tag)), "w") as fh:
        fh.write(summary(rows))


def summary(rows):
    import collections
    out = ["| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |",
           "|---|---|---|---|---|---|---|---|---|"]
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["scenario"], r["model"]), []).append(r)
    for (sc, mo), rs in groups.items():
        f = lambda k: np.mean([r[k] for r in rs])
        out.append("| {} | {} | {:.4f} | {:.4f} | {:.0f} / {:.0f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            sc, mo, f("spellF1"), f("macroF1"), f("n_pred"), f("n_gt"), f("D_f1"), f("team_acc"),
            f("transF1@25") if "transF1@25" in rs[0] else float("nan"), len(rs)))
    # paired comparison boundary - constant
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["scenario"], r["seed"], r["match"])][r["model"]] = r["spellF1"]
    for sc in sorted(set(r["scenario"] for r in rows)):
        d = [v["hmm_boundary"] - v["hmm_constant"] for k, v in by.items()
             if k[0] == sc and "hmm_boundary" in v and "hmm_constant" in v]
        if d:
            out.append("\n{}: boundary - constant = {:+.4f} ({}/{})".format(
                sc, np.mean(d), sum(x > 0 for x in d), sum(x < 0 for x in d)))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()

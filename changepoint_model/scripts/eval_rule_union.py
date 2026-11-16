"""v4 (and v2) against "+ the precise rule layers" on the held-out external matches, paired.

    python changepoint_model/scripts/eval_rule_union.py --dir outputs/cpm_lomo \
        --metrica ext/metrica/data --idsse data/idsse --pff data/pff

Uses the saved LOMO predictions (pred_<tag>_degraded_2027_<match>.npz) and rebuilds each match's
tracking with the same degradation and seed as eval_lomo.py (match i of the list gets 2027 + i;
PFF as given), so the layers see exactly the tracking the model saw.

Variants (decided before looking; the decoder settings are v4's, gamma 8, w 0.25):
    hard       decoded labels, then every layer frame -> D, then R6 islands -> D
    soft       P(dead) raised to 1 - (1 - P(dead)) * 0.05 on layer frames, then decoded
Decision rule fixed in advance: adopt a variant if the mean paired spellF1 gain over the 6 v4 folds
is > +0.005, at least 5 / 6 folds are better, and neither PFF fold is worse.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from cpm import decode as dec  # noqa: E402
from cpm import pipeline  # noqa: E402
from cpm.evaluate import score  # noqa: E402
from cpm.match import D, X  # noqa: E402
from cpm.rule_layers import rule_layers, islands, union, LAYERS  # noqa: E402
from eval_lomo import load_matches  # noqa: E402
from eval_hybrid import PRIOR, periods_from_gt  # noqa: E402

GAMMA, W = 8.0, 0.25
SOFT_Q = 0.95


def decode_with(p_home, src, p_dead, gt):
    period = periods_from_gt(gt)
    logE = dec.emissions(p_home, p_dead, PRIOR, w=W)
    hc, hs, hr = dec.hazards_from_boundaries(src["p_change"], src["p_stop"], src["p_restart"],
                                             k=12, gamma=GAMMA)
    return dec.decode_match(logE, hc, hs, hr, period)


def hard(labels, lay, gt):
    out = labels.copy()
    g = gt != X
    out[union(lay) & g] = D
    out[islands(out == D, lay) & g] = D
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metrica")
    ap.add_argument("--idsse")
    ap.add_argument("--pff")
    ap.add_argument("--only")
    ap.add_argument("--pidx", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out_dir = args.dir
    args.only = None
    matches = load_matches(argparse.Namespace(metrica=args.metrica, idsse=args.idsse, pff=args.pff,
                                              only=None, out=args.dir))
    rows = []
    layer_stats = {n: [0, 0, 0, 0] for n in LAYERS}   # frames, truly D, v4-alive, v4-alive & truly D
    for i, m in enumerate(matches):
        pa = os.path.join(args.dir, "pred_v2_all18_degraded_2027_{}.npz".format(m.name))
        pb = os.path.join(args.dir, "pred_v3_gru_all18_degraded_2027_{}.npz".format(m.name))
        if not os.path.isfile(pa):
            continue
        P = pipeline.prepare(m, "degraded", 2027 + i)
        lay = rule_layers(P.match)
        A = np.load(pa)
        gt = A["gt"]
        g = gt != X
        r = {"match": m.name, "source": "PFF" if m.name.isdigit() else
             ("IDSSE" if m.name.startswith("J03") else "Metrica")}
        # v2 (all 18 folds)
        v2 = decode_with(A["p_home"], A, A["p_dead"], gt)
        pd_soft = np.where(union(lay), 1 - (1 - A["p_dead"]) * (1 - SOFT_Q), A["p_dead"])
        r["v2"] = score(gt, v2)
        r["v2_hard"] = score(gt, hard(v2, lay, gt))
        r["v2_soft"] = score(gt, decode_with(A["p_home"], A, pd_soft, gt))
        if os.path.isfile(pb):
            B = np.load(pb)
            v4 = decode_with(B["p_home"], A, A["p_dead"], gt)
            r["v4"] = score(gt, v4)
            r["v4_hard"] = score(gt, hard(v4, lay, gt))
            r["v4_soft"] = score(gt, decode_with(B["p_home"], A, pd_soft, gt))
            ref = v4
        else:
            ref = v2
        for n in LAYERS:
            f = lay[n] & g
            s = layer_stats[n]
            s[0] += int(f.sum())
            s[1] += int((f & (gt == D)).sum())
            s[2] += int((f & (ref < D)).sum())
            s[3] += int((f & (ref < D) & (gt == D)).sum())
        rows.append(r)
        msg = "{:<11} v2 {:.3f} hard {:+.3f} soft {:+.3f}".format(
            m.name, r["v2"]["spellF1"], r["v2_hard"]["spellF1"] - r["v2"]["spellF1"],
            r["v2_soft"]["spellF1"] - r["v2"]["spellF1"])
        if "v4" in r:
            msg += " | v4 {:.3f} hard {:+.3f} soft {:+.3f} | D rec {:.2f} -> {:.2f} / {:.2f}".format(
                r["v4"]["spellF1"], r["v4_hard"]["spellF1"] - r["v4"]["spellF1"],
                r["v4_soft"]["spellF1"] - r["v4"]["spellF1"], r["v4"]["D_rec"],
                r["v4_hard"]["D_rec"], r["v4_soft"]["D_rec"])
        print(msg, flush=True)
    lines = ["# Rule-layer union on external held-out matches", ""]
    lines += ["## Layer precision (graded frames; 'added' = frames the model called alive)", "",
              "| layer | frames | truly dead | added | added and truly dead |", "|---|---|---|---|---|"]
    for n, (a, b, c, d) in layer_stats.items():
        lines.append("| {} | {} | {:.1%} | {} | {:.1%} |".format(n, a, b / max(a, 1), c, d / max(c, 1)))
    for base in ("v2", "v4"):
        rs = [r for r in rows if base in r]
        if not rs:
            continue
        lines += ["", "## {} ({} held-out matches)".format(base, len(rs)), "",
                  "| variant | spellF1 | macroF1 | D F1 | D rec | D prec | paired vs {} | better / worse | PFF better / worse |".format(base),
                  "|---|---|---|---|---|---|---|---|---|"]
        for v in (base, base + "_hard", base + "_soft"):
            d = [r[v]["spellF1"] - r[base]["spellF1"] for r in rs]
            dp = [r[v]["spellF1"] - r[base]["spellF1"] for r in rs if r["source"] == "PFF"]
            f = lambda k: np.mean([r[v][k] for r in rs])
            lines.append("| {} | {:.4f} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:+.4f} | {} / {} | {} / {} |".format(
                v, f("spellF1"), f("macroF1"), f("D_f1"), f("D_rec"), f("D_prec"), np.mean(d),
                sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x > 0 for x in dp), sum(x < 0 for x in dp)))
        for src in ("Metrica", "IDSSE", "PFF"):
            rr = [r for r in rs if r["source"] == src]
            if rr:
                lines.append("- {}: {} {:.3f}, hard {:+.4f}, soft {:+.4f}".format(
                    src, base, np.mean([r[base]["spellF1"] for r in rr]),
                    np.mean([r[base + "_hard"]["spellF1"] - r[base]["spellF1"] for r in rr]),
                    np.mean([r[base + "_soft"]["spellF1"] - r[base]["spellF1"] for r in rr])))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(args.out or os.path.join(args.dir, "rule_union.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()

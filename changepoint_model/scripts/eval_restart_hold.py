"""v4 (6 held-out folds) and v2 (18) against "+ stoppage held until the restart kick", paired.

    python changepoint_model/scripts/eval_restart_hold.py --dir outputs/cpm_lomo \
        --metrica ext/metrica/data --idsse data/idsse --pff data/pff

Same protocol as eval_rule_union.py: saved LOMO predictions, each match rebuilt with the same
degradation and seed, v4's decoder settings. Decision rule fixed before the run: adopt if the mean
paired spellF1 gain over the 6 v4 folds is > +0.005, >= 5 / 6 folds better, no PFF fold worse.
Sensitivity rows (max hold 5 / 20 s, kick speed 6 / 10 m/s) are reported, not used to choose.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from cpm import pipeline  # noqa: E402
from cpm.evaluate import score  # noqa: E402
from cpm.restart_hold import apply_restart_hold  # noqa: E402
from eval_lomo import load_matches  # noqa: E402
from eval_rule_union import decode_with  # noqa: E402

VARIANTS = {"hold": {}, "hold_5s": {"max_hold_s": 5.0}, "hold_20s": {"max_hold_s": 20.0},
            "kick_6": {"kick_speed": 6.0}, "kick_10": {"kick_speed": 10.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metrica")
    ap.add_argument("--idsse")
    ap.add_argument("--pff")
    args = ap.parse_args()
    matches = load_matches(argparse.Namespace(metrica=args.metrica, idsse=args.idsse, pff=args.pff,
                                              only=None, out=args.dir))
    rows = []
    for i, m in enumerate(matches):
        pa = os.path.join(args.dir, "pred_v2_all18_degraded_2027_{}.npz".format(m.name))
        pb = os.path.join(args.dir, "pred_v3_gru_all18_degraded_2027_{}.npz".format(m.name))
        if not os.path.isfile(pa):
            continue
        P = pipeline.prepare(m, "degraded", 2027 + i)
        A = np.load(pa)
        gt = A["gt"]
        src = "PFF" if m.name.isdigit() else ("IDSSE" if m.name.startswith("J03") else "Metrica")
        bases = {"v2": decode_with(A["p_home"], A, A["p_dead"], gt)}
        if os.path.isfile(pb):
            B = np.load(pb)
            bases["v4"] = decode_with(B["p_home"], A, A["p_dead"], gt)
        r = {"match": m.name, "source": src}
        for b, lab in bases.items():
            r[b] = score(gt, lab)
            for vn, kw in VARIANTS.items():
                out, info = apply_restart_hold(lab, P.match, **kw)
                r["{}_{}".format(b, vn)] = score(gt, out)
                if vn == "hold":
                    r["{}_info".format(b)] = info
        rows.append(r)
        msg = "{:<11} v2 {:.3f} hold {:+.4f}".format(m.name, r["v2"]["spellF1"],
                                                     r["v2_hold"]["spellF1"] - r["v2"]["spellF1"])
        if "v4" in r:
            msg += " | v4 {:.3f} hold {:+.4f} D-F1 {:.3f}->{:.3f} {}".format(
                r["v4"]["spellF1"], r["v4_hold"]["spellF1"] - r["v4"]["spellF1"], r["v4"]["D_f1"],
                r["v4_hold"]["D_f1"], r["v4_info"])
        print(msg, flush=True)
    lines = ["# Stoppage held until the restart kick, external held-out matches", ""]
    for b in ("v4", "v2"):
        rs = [r for r in rows if b in r]
        lines += ["## {} ({} held-out matches)".format(b, len(rs)), "",
                  "| variant | spellF1 | macroF1 | D F1 | D rec | D prec | paired | better / worse | PFF better / worse |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for vn in [b] + ["{}_{}".format(b, v) for v in VARIANTS]:
            d = [r[vn]["spellF1"] - r[b]["spellF1"] for r in rs]
            dp = [x for x, r in zip(d, rs) if r["source"] == "PFF"]
            f = lambda k: np.mean([r[vn][k] for r in rs])
            lines.append("| {} | {:.4f} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:+.4f} | {} / {} | {} / {} |".format(
                vn, f("spellF1"), f("macroF1"), f("D_f1"), f("D_rec"), f("D_prec"), np.mean(d),
                sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x > 0 for x in dp), sum(x < 0 for x in dp)))
        for s in ("Metrica", "IDSSE", "PFF"):
            rr = [r for r in rs if r["source"] == s]
            if rr:
                lines.append("- {}: {} {:.3f}, hold {:+.4f}".format(s, b, np.mean([r[b]["spellF1"] for r in rr]),
                             np.mean([r[b + "_hold"]["spellF1"] - r[b]["spellF1"] for r in rr])))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.dir, "restart_hold.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()

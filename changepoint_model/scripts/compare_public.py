"""Score several submissions on the public window and show WHERE each one wins or loses.

    python changepoint_model/scripts/compare_public.py --ref outputs/cpm_v4/tikitalia.csv
    python changepoint_model/scripts/compare_public.py --ref outputs/cpm_v4/tikitalia.csv \
        --extra v2=outputs/cpm_v2/tikitalia.csv rules=HMM_first_imp/outputs/run_b3/tikitalia.csv

Without --extra it scores the reference, the official baselines in evaluation/results/ and every
tikitalia.csv found under the repository (venv excluded), named by folder. Output: a markdown
report on screen and in outputs/cpm_public_compare.md (outputs/ is not committed). It contains
scores and counts only, no label.

THE PUBLIC WINDOW IS NOT A TUNING SET (RULES.md 2, mail of 03/11): run it once, after the choice,
and change nothing because of it. 17 possessions: spellF1 noise about +-0.39.

Per true possession (IoU 0.5, the scorer's own greedy match) the failure of an unmatched one is:
    merged    the best same-team predicted spell also covers most of another true possession
    split     two or more same-team predicted spells share it, none reaches IoU 0.5
    limits    one predicted spell overlaps it well but starts / ends too far away
    lost      no same-team predicted spell covers half of it (wrong team or dead ball)
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
GT = os.path.join(REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
H, A, D, X = 0, 1, 2, 3


def codes_of(path):
    return np.asarray(pm.label_codes(pm.read_label_csv(path)), dtype=np.int64)


def discover(ref):
    found = {}
    for p in sorted(glob.glob(os.path.join(REPO, "evaluation", "results", "baseline_*.csv"))):
        found[os.path.splitext(os.path.basename(p))[0]] = p
    for p in sorted(glob.glob(os.path.join(REPO, "**", "tikitalia.csv"), recursive=True)):
        rel = os.path.relpath(p, REPO)
        if rel.split(os.sep)[0] in ("venv", "venv_yolo", ".git"):
            continue
        if os.path.abspath(p) == os.path.abspath(ref):
            continue
        name = os.path.relpath(os.path.dirname(p), REPO).replace(os.sep, "/")
        name = name.replace("HMM_first_imp/outputs/", "").replace("outputs/", "")
        found[name] = p
    return found


def change_stats(gt, pred, runs):
    """Direct H<->A changes: true ones found within 1 s / 2 s (same new team), false ones."""
    def direct(codes):
        return [(f, b) for f, a, b in pm.transitions(codes, runs) if a in (H, A) and b in (H, A)]
    tg, tp = direct(gt), direct(pred)
    pf = np.array([f for f, _ in tp]) if tp else np.array([-10 ** 9])
    found1 = sum(1 for f, b in tg if any(abs(q - f) <= FPS and pb == b for q, pb in tp))
    found2 = sum(1 for f, b in tg if any(abs(q - f) <= 2 * FPS and pb == b for q, pb in tp))
    tgf = np.array([f for f, _ in tg]) if tg else np.array([-10 ** 9])
    false2 = sum(1 for q, _ in tp if np.min(np.abs(tgf - q)) > 2 * FPS)
    return len(tg), found1, found2, len(tp), false2


def spell_diag(gt, pred, runs, thr=0.5):
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = pm.matched_gt_flags(ov, len(gs), thr)
    kinds = []
    for i, (s, e, k) in enumerate(gs):
        if flags[i]:
            kinds.append("ok")
            continue
        same = [(j, ps_) for j, ps_ in enumerate(ps) if ps_[2] == k and ps_[1] > s and ps_[0] < e]
        inter = [(min(e, b) - max(s, a), j, (a, b)) for j, (a, b, _) in same]
        cover = sum(x[0] for x in inter) / float(e - s)
        if cover < 0.5:
            kinds.append("lost")
            continue
        best = max(inter)
        a, b = best[2]
        merged = any(i2 != i and k2 == k and min(b, e2) - max(a, s2) > 0.5 * (e2 - s2)
                     for i2, (s2, e2, k2) in enumerate(gs))
        if merged:
            kinds.append("merged")
        elif len([x for x in inter if x[0] > 0.2 * (e - s)]) >= 2:
            kinds.append("split")
        else:
            kinds.append("limits")
    n_spur = len(ps) - int(flags.sum())
    return gs, kinds, n_spur


def frame_errors(gt, pred):
    g = gt != X
    dead_as_alive = np.sum(g & (gt == D) & (pred < D)) / FPS
    alive_as_dead = np.sum(g & (gt < D) & (pred == D)) / FPS
    wrong_team = np.sum(g & (gt < D) & (pred < D) & (gt != pred)) / FPS
    return dead_as_alive, alive_as_dead, wrong_team


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the submission under test (v4)")
    ap.add_argument("--extra", nargs="*", default=None, help="name=path; replaces auto-discovery")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "cpm_public_compare.md"))
    args = ap.parse_args()
    gt = codes_of(GT)
    runs = pm.valid_runs(gt)
    subs = {"v4 (ref)": args.ref}
    if args.extra:
        for kv in args.extra:
            k, v = kv.split("=", 1)
            subs[k] = v
    else:
        subs.update(discover(args.ref))
    rows, diags = [], {}
    for name, path in subs.items():
        try:
            pred = codes_of(path)
            if len(pred) != len(gt):
                print("skip {}: {} rows".format(name, len(pred)))
                continue
        except Exception as e:  # noqa: BLE001
            print("skip {}: {}".format(name, e))
            continue
        bs = args.bootstrap if name in ("v4 (ref)", "baseline_smooth50") else 0
        r = pm.evaluate(gt, pred, bootstrap_n=bs)
        filled = pm.fill_abstentions(pred, runs)
        gs, kinds, n_spur = spell_diag(gt, filled, runs)
        ch = change_stats(gt, filled, runs)
        fe = frame_errors(gt, pred)
        rows.append((name, r, ch, fe, kinds, n_spur))
        diags[name] = kinds
    ref = rows[0]
    base = next((x for x in rows if x[0] == "baseline_smooth50"), None)
    out = ["# Public window: v4 against the baseline and the earlier models", "",
           "Official scorer, 5 minutes, {} true possessions. Structure check only: nothing is "
           "chosen or changed because of it.".format(len(gs)), ""]
    r0 = ref[1]
    out.append("v4 spellF1 {:.3f} [{:.3f}-{:.3f}], macroF1 {:.3f} [{:.3f}-{:.3f}]".format(
        r0["spell_f1_mean"], r0["spell_bootstrap_low"], r0["spell_bootstrap_high"], r0["macro_f1"],
        r0["bootstrap_low"], r0["bootstrap_high"]))
    if base:
        rb = base[1]
        out.append("baseline_smooth50 spellF1 {:.3f} [{:.3f}-{:.3f}], macroF1 {:.3f}: v4 is {:+.3f} "
                   "spellF1 ({:.2f}x), {:+.3f} macroF1".format(
                       rb["spell_f1_mean"], rb["spell_bootstrap_low"], rb["spell_bootstrap_high"],
                       rb["macro_f1"], r0["spell_f1_mean"] - rb["spell_f1_mean"],
                       r0["spell_f1_mean"] / max(rb["spell_f1_mean"], 1e-9),
                       r0["macro_f1"] - rb["macro_f1"]))
    out += ["", "## 1. Scores", "",
            "| model | spellF1 | macroF1 | IoU .3 / .5 / .7 | possessions (true {}) | D F1 (prec / rec) | team acc | transF1@25 | latency |".format(len(gs)),
            "|---|---|---|---|---|---|---|---|---|"]
    for name, r, ch, fe, kinds, n_spur in rows:
        sf = r["spell_f1"]
        t25 = r["transitions"].get(25, {})
        out.append("| {} | {:.3f} | {:.3f} | {:.2f} / {:.2f} / {:.2f} | {} | {:.3f} ({:.2f} / {:.2f}) | {:.3f} | {:.3f} | {:+.1f} |".format(
            name, r["spell_f1_mean"], r["macro_f1"], sf[0.3]["f1"], sf[0.5]["f1"], sf[0.7]["f1"],
            r["n_pred_spells"], r["per_class"]["D"]["f1"], r["per_class"]["D"]["precision"],
            r["per_class"]["D"]["recall"], r["team_acc_both_alive"], t25.get("f1", float("nan")),
            t25.get("mean_latency", float("nan")) if t25.get("mean_latency") is not None else float("nan")))
    out += ["", "## 2. Where the errors are", "",
            "| model | true possessions ok / merged / split / limits / lost (IoU 0.5) | spurious predicted | H<->A changes found 1 s / 2 s (of {}) | predicted changes, false at 2 s | s dead called alive | s alive called dead | s wrong team |".format(ref[2][0]),
            "|---|---|---|---|---|---|---|---|"]
    for name, r, ch, fe, kinds, n_spur in rows:
        c = {k: kinds.count(k) for k in ("ok", "merged", "split", "limits", "lost")}
        out.append("| {} | {ok} / {merged} / {split} / {limits} / {lost} | {} | {} / {} | {} / {} | {:.0f} | {:.0f} | {:.0f} |".format(
            name, n_spur, ch[1], ch[2], ch[3], ch[4], fe[0], fe[1], fe[2], **c))
    out += ["", "## 3. Possession by possession", "",
            "Columns: the true possessions in time order (team, seconds). ok = matched at IoU 0.5; "
            "M merged, S split, B limits, L lost.", ""]
    head = ["model"] + ["{}{}".format("H" if k == H else "A", i + 1) for i, (s, e, k) in enumerate(gs)]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    out.append("| length s | " + " | ".join("{:.0f}".format((e - s) / FPS) for s, e, k in gs) + " |")
    short = {"ok": "ok", "merged": "M", "split": "S", "limits": "B", "lost": "L"}
    for name, kinds in diags.items():
        out.append("| {} | ".format(name) + " | ".join(short[k] for k in kinds) + " |")
    out += ["", "## 4. v4 against each model, possession by possession", "",
            "| model | v4 ok, model not | model ok, v4 not | both ok | neither |", "|---|---|---|---|---|"]
    kr = diags["v4 (ref)"]
    for name, kinds in diags.items():
        if name == "v4 (ref)":
            continue
        a = sum(1 for x, y in zip(kr, kinds) if x == "ok" and y != "ok")
        b = sum(1 for x, y in zip(kr, kinds) if x != "ok" and y == "ok")
        c = sum(1 for x, y in zip(kr, kinds) if x == "ok" and y == "ok")
        d = sum(1 for x, y in zip(kr, kinds) if x != "ok" and y != "ok")
        out.append("| {} | {} | {} | {} | {} |".format(name, a, b, c, d))
    text = "\n".join(out) + "\n"
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()

"""Track the official public-window evaluation of every version, in one Markdown document.

    python HMM_first_imp/scripts/track_public_eval.py            # -> HMM_first_imp/EVAL_TRACKING.md

Scores each CSV listed in config/eval_tracking.yaml with the OFFICIAL scorer
(evaluation/possession_metrics.evaluate, exactly what run_evaluation.py --bootstrap 1000 calls),
adds the full-match diagnostics of each run (run_summary.json, no ground truth involved) and
writes the facets side by side: headline, dead ball, team, whole possessions, changes, full match,
and the change from one version to the next.

The public 5 minutes are a sanity check (17 possessions): nothing here is used to choose
parameters. Choices are made on IDSSE (scripts/run_validation.py).
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # HMM_first_imp
REPO = os.path.dirname(ROOT)
EVAL_DIR = os.path.join(REPO, "evaluation")
sys.path.insert(0, EVAL_DIR)

import possession_metrics as pm                    # noqa: E402

FPS = 25.0
H, A, D = 0, 1, 2


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gt_path(name):
    if name == "public":
        return os.path.join(EVAL_DIR, "gt", "BAR-ATH_possession_gt_public.csv")
    return name


def score(csv, gt_codes, n_boot):
    table = pm.read_label_csv(csv)
    reason = pm.validate_submission(table, len(gt_codes))
    if reason:
        raise ValueError("{}: {}".format(csv, reason))
    r = pm.evaluate(gt_codes, pm.label_codes(table), bootstrap_n=n_boot)
    cm = r["confusion"]
    t25 = r["transitions"][25]
    t12 = r["transitions"][12]
    s = r["spell_f1"]
    out = {
        "macro_f1": r["macro_f1"], "macro_lo": r["bootstrap_low"], "macro_hi": r["bootstrap_high"],
        "spell_f1": r["spell_f1_mean"], "spell_lo": r["spell_bootstrap_low"],
        "spell_hi": r["spell_bootstrap_high"],
        "poss": r["n_pred_spells"], "poss_gt": r["n_gt_spells"],
        "dur": r["mean_spell_duration_s"], "dur_gt": r["gt_mean_spell_duration_s"],
        "acc": r["accuracy"], "kappa": r["kappa"],
        "f1": {k: r["per_class"][k]["f1"] for k in "HAD"},
        "prec": {k: r["per_class"][k]["precision"] for k in "HAD"},
        "rec": {k: r["per_class"][k]["recall"] for k in "HAD"},
        "support": {k: r["per_class"][k]["support"] for k in "HAD"},
        "alive_dead_f1": r["alive_dead_f1"],
        "team_gt_alive": r["team_acc_gt_alive"], "team_both_alive": r["team_acc_both_alive"],
        "home_gt": r["gt_home_share"], "home_pred": r["pred_home_share"],
        "share_err": r["share_err"],
        "frag": r["fragmentation"], "seg_pred": r["n_pred_segments"], "seg_gt": r["n_gt_segments"],
        # confusion on graded frames (rows = truth, cols = prediction)
        "H_as_A": cm[H][A], "A_as_H": cm[A][H],
        "D_missed": cm[D][H] + cm[D][A],                 # true dead ball called H/A
        "D_false": cm[H][D] + cm[A][D],                  # alive called D
        "D_true": cm[D][D],
        "abstain": r["n_abstain_on_graded"],
        "spell_iou": {str(k): v["f1"] for k, v in s.items()},
        "iou05": {k: s[0.5][k] for k in ("n_matched", "n_missed", "n_spurious")},
        "trans_gt": t25["n_gt"], "trans_pred": t25["n_pred"],
        "t12": {k: t12[k] for k in ("f1", "n_matched")},
        "t25": {k: t25[k] for k in ("f1", "precision", "recall", "n_matched", "n_missed",
                                    "n_spurious")},
        "lat_mean": t25["mean_latency"], "lat_median": t25["median_latency"],
    }
    return out


def full_match(summary_path):
    if not summary_path or not os.path.isfile(summary_path):
        return None
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    o = s.get("output", {})
    minutes = o.get("graded_frames", 0) / FPS / 60.0
    fr = s.get("fragmentation", {}) or {}
    return {
        "minutes": minutes, "share": o.get("label_share_%", {}),
        "poss": o.get("possessions"), "poss_per_min": (o.get("possessions") or 0) / minutes
        if minutes else None,
        "poss_s": o.get("possession_s", {}), "dead": o.get("dead_spells"),
        "dead_s": o.get("dead_s", {}), "short1": o.get("possessions_shorter_than_1s"),
        "short3": fr.get("shorter_than_3s"), "short5": fr.get("shorter_than_5s"),
        "flips": fr.get("flips_H-A-H_middle<5s"),
        "open_changes": o.get("changes_H<->A_without_D"),
        "split_by_short_D": o.get("same_team_split_by_D<2s"),
    }


# ----------------------------------------------------------------------------- formatting
def f3(x):
    return "–" if x is None else "{:.3f}".format(x)


def f1d(x):
    return "–" if x is None else "{:.1f}".format(x)


def pct(x):
    return "–" if x is None else "{:.0f}%".format(100 * x)


def sec(frames):
    return "{:.0f} s".format(frames / FPS)


def delta(a, b, fmt="{:+.3f}"):
    if a is None or b is None:
        return "–"
    return fmt.format(a - b)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(cfg, versions, refs, noise, gt_file):
    L = []
    last = versions[-1]
    L.append("# Tracking della valutazione pubblica — v0 → v1 → C\n")
    L.append("Generato da `scripts/track_public_eval.py` il {} · ground truth `{}` · scorer ufficiale "
             "`evaluation/possession_metrics.evaluate` (lo stesso di `run_evaluation.py --bootstrap {}`)."
             .format(datetime.date.today().isoformat(), os.path.basename(gt_file), cfg["bootstrap"]))
    L.append("")
    L.append("```powershell\n.\\venv\\Scripts\\python.exe HMM_first_imp\\scripts\\track_public_eval.py\n```")
    L.append("Per aggiungere una versione: una voce in `config/eval_tracking.yaml`, poi rilancia.\n")
    g = last["eval"]
    L.append("> **Come leggerlo.** La finestra pubblica è 5 minuti: {} frame giudicati, **{} possessi veri** "
             "di {:.1f} s in media. Differenze di spellF1 sotto **~{:.2f}** (l'intervallo bootstrap più "
             "largo) possono essere fortuna. Serve a controllare che l'output abbia senso, **non a "
             "scegliere i parametri**: quello si fa su IDSSE (ultima colonna della tabella 1)."
             .format(sum(g["support"].values()), g["poss_gt"], g["dur_gt"], noise))
    L.append("")

    allv = versions + refs
    # 1 headline
    L.append("## 1. Il quadro in una tabella\n")
    rows = []
    for v in allv:
        e = v["eval"]
        rows.append(["**{}**".format(v["name"]) if v is last else v["name"],
                     "{} [{}–{}]".format(f3(e["spell_f1"]), f3(e["spell_lo"]), f3(e["spell_hi"])),
                     "{} [{}–{}]".format(f3(e["macro_f1"]), f3(e["macro_lo"]), f3(e["macro_hi"])),
                     "{} / {}".format(e["poss"], e["poss_gt"]),
                     "{} / {}".format(f1d(e["dur"]), f1d(e["dur_gt"])),
                     "{:.2f}".format(e["frag"]),
                     f3(v.get("idsse_spellf1")) if v.get("idsse_spellf1") else "–"])
    L.append(table(["versione", "spellF1 [5–95%]", "macroF1 [5–95%]", "possessi (pred / veri)",
                    "durata media s (pred / vera)", "frammentazione (1 = giusta)",
                    "spellF1 IDSSE (7 partite)"], rows))
    L.append("\n- **spellF1** (classifica): possessi indovinati come pezzi interi (inizio, fine, squadra).\n"
             "- **macroF1**: etichetta giusta frame per frame, media di H, A, D.\n"
             "- **Frammentazione** = segmenti previsti / veri: sopra 1 spezza, sotto 1 unisce.\n")

    # 2 dead ball
    L.append("## 2. Palla morta (D)\n")
    rows = []
    for v in allv:
        e = v["eval"]
        rows.append([v["name"], f3(e["f1"]["D"]), pct(e["prec"]["D"]), pct(e["rec"]["D"]),
                     "{} ({})".format(e["D_missed"], sec(e["D_missed"])),
                     "{} ({})".format(e["D_false"], sec(e["D_false"])), f3(e["alive_dead_f1"])])
    L.append(table(["versione", "F1 D", "precision (D detta e giusta)", "recall (D vera trovata)",
                    "D vera chiamata H/A", "palla viva chiamata D", "F1 viva/morta"], rows))
    L.append("\nD vera nella finestra: {} frame ({}). La **recall** dice quanta palla morta troviamo; "
             "la **precision** quanto ci si può fidare quando diciamo D.\n"
             .format(g["support"]["D"], sec(g["support"]["D"])))

    # 3 team
    L.append("## 3. Squadra (H / A)\n")
    rows = []
    for v in allv:
        e = v["eval"]
        rows.append([v["name"], f3(e["f1"]["H"]), f3(e["f1"]["A"]), pct(e["team_gt_alive"]),
                     "{} ({})".format(e["H_as_A"], sec(e["H_as_A"])),
                     "{} ({})".format(e["A_as_H"], sec(e["A_as_H"])),
                     "{} vs {} (err {:.3f})".format(pct(e["home_pred"]), pct(e["home_gt"]),
                                                    e["share_err"])])
    L.append(table(["versione", "F1 H", "F1 A", "squadra giusta a palla viva", "H chiamato A",
                    "A chiamato H", "quota possesso casa (pred vs vera)"], rows))
    L.append("")

    # 4 whole possessions
    L.append("## 4. Possessi interi (spellF1 per soglia di sovrapposizione)\n")
    rows = []
    for v in allv:
        e = v["eval"]
        si = e["spell_iou"]
        rows.append([v["name"]] + [f3(si[k]) for k in ("0.3", "0.4", "0.5", "0.6", "0.7")] +
                    ["{} / {} / {}".format(e["iou05"]["n_matched"], e["iou05"]["n_missed"],
                                           e["iou05"]["n_spurious"])])
    L.append(table(["versione", "IoU 0.3", "0.4", "0.5", "0.6", "0.7",
                    "@0.5 presi / mancati / inventati"], rows))
    L.append("\nIoU = quanto il possesso previsto si sovrappone a quello vero. Se lo spellF1 cala molto "
             "da 0.3 a 0.7 i possessi sono quelli giusti ma con i **confini imprecisi**.\n")

    # 5 changes
    L.append("## 5. Momenti di cambio\n")
    rows = []
    for v in allv:
        e = v["eval"]
        rows.append([v["name"], "{} / {}".format(e["trans_pred"], e["trans_gt"]),
                     "{} ({})".format(f3(e["t12"]["f1"]), e["t12"]["n_matched"]),
                     "{} ({})".format(f3(e["t25"]["f1"]), e["t25"]["n_matched"]),
                     "{} / {}".format(e["t25"]["n_missed"], e["t25"]["n_spurious"]),
                     "{} / {}".format(f1d(e["lat_mean"]), f1d(e["lat_median"]))])
    L.append(table(["versione", "cambi pred / veri", "F1 ±0.5 s (presi)", "F1 ±1 s (presi)",
                    "±1 s mancati / inventati", "latenza media / mediana (frame; < 0 = in anticipo)"],
                   rows))
    L.append("")

    # 6 full match
    L.append("## 6. Partita intera (BAR-ATH, nessuna ground truth)\n")
    L.append("Dal `run_summary.json` di ogni run: controlli di plausibilità, non punteggi. Riferimento "
             "della finestra pubblica: {:.1f} possessi/min, palla morta {} dei frame.\n"
             .format(g["poss_gt"] / 5.0, pct(g["support"]["D"] / float(sum(g["support"].values())))))
    rows = []
    for v in versions:
        m = v.get("full")
        if not m:
            rows.append([v["name"]] + ["–"] * 9)
            continue
        sh = m["share"]
        ps, ds = m["poss_s"], m["dead_s"]
        rows.append([v["name"],
                     "{:.1f} / {:.1f} / {:.1f}".format(sh.get("H", 0), sh.get("A", 0), sh.get("D", 0)),
                     "{} ({:.1f}/min)".format(m["poss"], m["poss_per_min"]),
                     "{} / {} / {}".format(f1d(ps.get("mean")), f1d(ps.get("median")),
                                           f1d(ps.get("p90"))),
                     m["dead"],
                     "{} / {} / {}".format(f1d(ds.get("mean")), f1d(ds.get("median")),
                                           f1d(ds.get("p90"))),
                     "{} / {} / {}".format(m["short1"], m["short3"], m["short5"]),
                     m["flips"], m["open_changes"], m["split_by_short_D"]])
    L.append(table(["versione", "H / A / D %", "possessi", "durata poss. media / mediana / p90 s",
                    "interruzioni", "durata D media / mediana / p90 s", "possessi < 1 / 3 / 5 s",
                    "flip H-A-H < 5 s", "cambi H↔A a palla viva", "stessa squadra spezzata da D < 2 s"],
                   rows))
    L.append("")

    # 7 deltas
    L.append("## 7. Cosa cambia da una versione alla successiva\n")
    keys = [("spellF1", lambda e: e["spell_f1"]), ("macroF1", lambda e: e["macro_f1"]),
            ("F1 D", lambda e: e["f1"]["D"]), ("recall D", lambda e: e["rec"]["D"]),
            ("squadra a palla viva", lambda e: e["team_gt_alive"]),
            ("spellF1 @IoU 0.7", lambda e: e["spell_iou"]["0.7"]),
            ("F1 cambi ±1 s", lambda e: e["t25"]["f1"])]
    header = ["metrica"] + ["{} → {}".format(versions[i - 1]["name"], versions[i]["name"])
                            for i in range(1, len(versions))]
    rows = []
    for label, get in keys:
        rows.append([label] + [delta(get(versions[i]["eval"]), get(versions[i - 1]["eval"]))
                               for i in range(1, len(versions))])
    rows.append(["possessi (veri {})".format(g["poss_gt"])] +
                ["{} → {}".format(versions[i - 1]["eval"]["poss"], versions[i]["eval"]["poss"])
                 for i in range(1, len(versions))])
    # idsse_delta_prev: the paired difference measured in the same conditions, when the two
    # IDSSE numbers of the table come from different runs (e.g. with / without simulated tagging)
    rows.append(["spellF1 IDSSE (appaiato)"] +
                ["{:+.3f}".format(versions[i]["idsse_delta_prev"])
                 if versions[i].get("idsse_delta_prev") is not None else
                 delta(versions[i].get("idsse_spellf1"), versions[i - 1].get("idsse_spellf1"))
                 for i in range(1, len(versions))])
    L.append(table(header, rows))
    L.append("\nUna variazione pubblica sotto ~{:.2f} di spellF1 non è una prova da sola: va letta "
             "insieme alla riga IDSSE.\n".format(noise))

    # 8 versions
    L.append("## 8. Versioni e file\n")
    rows = [[v["name"], v.get("date", ""), v.get("what", ""), "`{}`".format(v["csv"]),
             "`{}…`".format(v["sha"][:12])] for v in versions]
    L.append(table(["versione", "data", "cosa", "CSV", "SHA-256"], rows))
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "eval_tracking.yaml"))
    ap.add_argument("--readings", default=os.path.join(ROOT, "EVAL_TRACKING_notes.md"),
                    help="hand-written reading appended at the end, if the file exists")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gt_file = gt_path(cfg.get("gt", "public"))
    gt_codes = pm.label_codes(pm.read_label_csv(gt_file))
    n_boot = int(cfg.get("bootstrap", 1000))

    versions = []
    for v in cfg["versions"]:
        csv = os.path.join(ROOT, v["csv"])
        print("scoring", v["name"], flush=True)
        v = dict(v, eval=score(csv, gt_codes, n_boot), sha=sha(csv),
                 full=full_match(os.path.join(ROOT, v["summary"]) if v.get("summary") else None))
        versions.append(v)
    refs = []
    for name in cfg.get("references", []):
        csv = os.path.join(EVAL_DIR, "results", name + ".csv")
        if os.path.isfile(csv):
            print("scoring", name, flush=True)
            refs.append({"name": name, "eval": score(csv, gt_codes, n_boot)})

    noise = max(v["eval"]["spell_hi"] - v["eval"]["spell_lo"] for v in versions + refs)
    md = build(cfg, versions, refs, noise, gt_file)
    if os.path.isfile(args.readings):
        with open(args.readings, encoding="utf-8") as f:
            md += "\n" + f.read()
    out_md = os.path.join(ROOT, cfg["out_md"])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    out_json = os.path.join(ROOT, cfg["out_json"])
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"versions": [{k: v[k] for k in v} for v in versions],
                   "references": refs, "noise_band": noise}, f, indent=1, default=str)
    print("wrote", out_md)
    print("wrote", out_json)


if __name__ == "__main__":
    main()

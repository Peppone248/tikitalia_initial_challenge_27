"""Comparison table of the LOMO runs and the keep/drop criterion for the TCN."""
import numpy as np

COLS = ["spellF1", "macroF1", "n_pred_poss", "n_true_poss", "transF1_25"]
MIN_MEAN, MIN_POS = 0.01, 5


def criterion(deltas):
    """deltas {seed: [per-match spellF1 delta vs frame]}: True iff in EVERY seed the mean delta is
    >= 0.01 and at least 5 matches have a strictly positive delta."""
    if not deltas:
        return False
    for v in deltas.values():
        v = np.asarray(v, dtype=float)
        if len(v) == 0 or not (v.mean() >= MIN_MEAN - 1e-12 and (v > 0).sum() >= MIN_POS):
            return False
    return True


def render(df, ref="frame", cand="tcn_change"):
    """df columns: model, seed, match, COLS (+ beta, w for tcn_change). Returns markdown."""
    L = ["## Per model and seed (mean over matches)", "",
         "| model | seed | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for (name, seed), g in df.groupby(["model", "seed"], sort=False):
        L.append(f"| {name} | {seed} | " + " | ".join(
            f"{g[c].mean():.3f}" if c in ("spellF1", "macroF1", "transF1_25")
            else f"{g[c].mean():.1f}" for c in COLS) + " |")
    L.append("")
    if "beta" in df.columns:
        sel = df[df.beta.notna()]
        if len(sel):
            L += ["## Chosen (beta, w) per fold", "", "| model | seed | match | beta | w |",
                  "|---|---|---|---|---|"]
            L += [f"| {r.model} | {r.seed} | {r.match} | {r.beta:g} | {r.w:g} |"
                  for r in sel.itertuples()]
            L.append("")
    verdict = None
    if ref in set(df.model):
        L += [f"## Paired spellF1 delta vs {ref} (per match)", ""]
        for name in [m for m in dict.fromkeys(df.model) if m != ref]:
            deltas = {}
            for seed in dict.fromkeys(df[df.model == name].seed):
                a = df[(df.model == name) & (df.seed == seed)].set_index("match").spellF1
                b = df[(df.model == ref) & (df.seed == seed)].set_index("match").spellF1
                d = (a - b.reindex(a.index)).dropna()
                deltas[seed] = d
                L += [f"### {name}, seed {seed}", "", "| match | delta |", "|---|---|"]
                L += [f"| {k} | {v:+.3f} |" for k, v in d.items()]
                L.append(f"| **mean** | {d.mean():+.3f} |")
                L.append("")
                L.append(f"better {int((d > 0).sum())}/{len(d)}, worse {int((d < 0).sum())}/{len(d)}")
                L.append("")
            if name == cand:
                verdict = criterion({s: list(d.values) for s, d in deltas.items()})
    L += ["Note: for tcn_change the (beta, w) selection uses the 6 training matches of each fold "
          "predicted by the network trained on them (in-sample, no inner retraining, to bound CPU "
          "time); the held-out match is never used.", ""]
    if verdict is not None:
        L.append("criterion: " + ("SODDISFATTO" if verdict else "NON SODDISFATTO"))
    return "\n".join(L) + "\n"

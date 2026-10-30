"""From rule outputs to evidence per frame (MOTORE.md 4.1, NOMENCLATURA 4).

    w_r(f)  = q_r * s_r(f)                          the vote weight (s from the MAD z-scores)
    p_r(f)  = 1/K + (1 - 1/K) * w_r(f)              w = 0 -> chance, w = 1 -> certain (capped)
    e_r(f)  = ln((K - 1) p / (1 - p))               optimal log-odds weight of a K-class voter
    E[f, c] = sum over rules voting c on f of e_r(f)

    e_r(f) *= kind_scale[kind]                      SOFT votes are continuous and autocorrelated
                                                    over seconds: scaled down so that they fill
                                                    gaps but never outvote an ANCHOR (REGOLE 6)

Context: DEAD where a D rule votes with w >= conf_dead_context; there OPEN_PLAY rules do not vote
(REGOLE 6: H/A rules only in open play). Classes: H, A, D, X.
"""

import numpy as np

CLASSES = ("H", "A", "D", "X")
K = len(CLASSES)


def weight_to_evidence(w, p_max):
    p = 1.0 / K + (1.0 - 1.0 / K) * np.clip(w, 0.0, 1.0)
    p = np.minimum(p, p_max)
    return np.log((K - 1) * p / (1.0 - p))


def collect(rules, outputs, n, cfg_votes):
    """Returns (E (n, 4), dead_mask (n,), per_rule_weight {id: (n,)})."""
    by_id = {r.id: r for r in rules}
    weights = {rid: by_id[rid].prior_q * outputs[rid].strength * (outputs[rid].label != "")
               for rid in outputs}
    dead = np.zeros(n, dtype=bool)
    for rid, out in outputs.items():
        r = by_id[rid]
        if "D" in r.emits:
            dead |= (out.label == "D") & (weights[rid] >= cfg_votes["conf_dead_context"])
    E = np.zeros((n, K))
    for rid, out in outputs.items():
        r = by_id[rid]
        e = weight_to_evidence(weights[rid], cfg_votes["p_max"]) * cfg_votes["kind_scale"][r.kind]
        e[weights[rid] <= 0] = 0.0
        if r.scope == "OPEN_PLAY":
            e = np.where(dead, 0.0, e)
        for k, c in enumerate(CLASSES):
            m = out.label == c
            E[m, k] += e[m]
    return E, dead, weights


def restart_frames(outputs, rules, context):
    """Frames where D -> H/A is allowed (detected restarts and kick-offs)."""
    frames = set(r["frame"] for r in context["restarts"])
    by_id = {r.id: r for r in rules}
    for rid, out in outputs.items():
        if by_id[rid].kind == "RESTART":
            frames.update(int(ep["start"]) for ep in out.episodes if "start" in ep)
    return np.array(sorted(frames), dtype=int)


def summary(rules, outputs, weights, dead, graded):
    rows = []
    for r in rules:
        out = outputs[r.id]
        fired = out.label != ""
        g = fired & graded
        rows.append({
            "rule": r.id, "kind": r.kind, "scope": r.scope, "prior_q": r.prior_q,
            "episodes": sum(1 for e in out.episodes if "start" in e and e.get("label")),
            "coverage_graded_%": round(100.0 * g.sum() / max(1, graded.sum()), 2),
            "mean_s": round(float(out.strength[g].mean()), 3) if g.any() else None,
            "share_s_saturated_%": round(100.0 * float((out.strength[g] >= 0.999).mean()), 1)
            if g.any() else None,
            "mean_w": round(float(weights[r.id][g].mean()), 3) if g.any() else None,
            "muted_by_dead_%": round(100.0 * float((g & dead).sum()) / max(1, g.sum()), 1)
            if r.scope == "OPEN_PLAY" else None,
            "overlap_frames": getattr(out, "overlap_frames", 0),
        })
    return rows


def change_points(outputs, sources):
    """{'H': frames, 'A': frames}: first frames of the episodes in which a rule of `sources`
    says that team took the ball (a control, a pass chain, a turnover, a restart). These are the
    only places where the HMM lets the team change at its base rate (hmm.py team gates)."""
    pts = {"H": [], "A": []}
    for rid in sources:
        out = outputs.get(rid)
        if out is None:
            continue
        for ep in out.episodes:
            if "start" in ep and ep.get("label") in pts:
                pts[ep["label"]].append(int(ep["start"]))
    return {t: np.array(sorted(set(v)), dtype=int) for t, v in pts.items()}

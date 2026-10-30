"""Checks of the output WITHOUT any ground truth: how many possessions, how long, how many tiny
spells, how often the final label contradicts an anchor rule. The numbers to look at before
trusting a file (FAQ: "why is my submission full of tiny possessions?")."""

import numpy as np

from tikitalia.preprocessing import runs


def spells(labels, graded, half):
    """Maximal runs of one label on graded frames, per half: [(label, start, end, half)]."""
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(graded & (half == h))
        if idx.size == 0:
            continue
        lab = labels[idx]
        cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], cut))
        ends = np.concatenate((cut, [lab.size]))
        out += [(lab[s], int(idx[s]), int(idx[e - 1]) + 1, h) for s, e in zip(starts, ends)]
    return out


def output_summary(labels, match, fps):
    g = match.graded
    sp = spells(labels, g, match.half)
    poss = [(e - s) / fps for l, s, e, h in sp if l in ("H", "A")]
    dead = [(e - s) / fps for l, s, e, h in sp if l == "D"]
    seq = [l for l, *_ in sp]
    # possessions of the SAME team separated only by a D shorter than 2 s: H D H patterns
    split_by_short_d = sum(1 for i in range(1, len(sp) - 1)
                           if sp[i][0] == "D" and (sp[i][2] - sp[i][1]) / fps < 2.0
                           and sp[i - 1][0] == sp[i + 1][0] and sp[i - 1][0] in ("H", "A"))
    return {
        "graded_frames": int(g.sum()),
        "label_share_%": {c: round(100.0 * float((labels[g] == c).mean()), 2) for c in "HADX"},
        "possessions": len(poss),
        "possessions_per_half": {h: sum(1 for l, s, e, hh in sp if hh == h and l in ("H", "A"))
                                 for h in (1, 2)},
        "dead_spells": len(dead),
        "possession_s": _quant(poss),
        "dead_s": _quant(dead),
        "possessions_shorter_than_1s": int(sum(d < 1.0 for d in poss)),
        "changes_H<->A_without_D": int(sum(1 for a, b in zip(seq, seq[1:])
                                           if {a, b} == {"H", "A"})),
        "same_team_split_by_D<2s": split_by_short_d,
        "sequence_head": "".join(seq[:60]),
    }


def _quant(values):
    if not values:
        return {}
    v = np.array(values)
    return {"n": int(v.size), "mean": round(float(v.mean()), 1),
            "median": round(float(np.median(v)), 1),
            "p10": round(float(np.percentile(v, 10)), 1),
            "p90": round(float(np.percentile(v, 90)), 1)}


def anchor_agreement(labels, rules, outputs, weights, graded, conf):
    """Share of the frames of each ANCHOR/RESTART vote (w >= conf) where the output agrees."""
    rows = {}
    for r in rules:
        out = outputs[r.id]
        m = graded & (out.label != "") & (weights[r.id] >= conf)
        if not m.any():
            continue
        lab = out.label[m]
        ok = labels[m] == lab
        if "X" in r.emits:
            ok = np.ones_like(ok)                     # X is never an output label
        rows[r.id] = {"frames": int(m.sum()), "agree_%": round(100.0 * float(ok.mean()), 1)}
    return rows


def fragmentation(labels, match, outputs, sources, fps):
    """Signs of fragmented possessions (no ground truth): short possessions, H-A-H flips with a
    short middle spell, and short possessions with no control / turnover / restart rule of that
    team inside them (a change nothing explains)."""
    sp = spells(labels, match.graded, match.half)
    poss = [(l, s, e) for l, s, e, h in sp if l in ("H", "A")]
    short = [(l, s, e) for l, s, e in poss if (e - s) / fps < 5.0]
    unsupported = 0
    for l, s, e in short:
        if not any(rid in outputs and (outputs[rid].label[s:e] == l).any() for rid in sources):
            unsupported += 1
    seq = [(l, s, e) for l, s, e, h in sp]
    flips = sum(1 for i in range(1, len(seq) - 1)
                if seq[i][0] in ("H", "A") and seq[i - 1][0] == seq[i + 1][0]
                and seq[i - 1][0] in ("H", "A") and seq[i][0] != seq[i - 1][0]
                and (seq[i][2] - seq[i][1]) / fps < 5.0)
    return {"possessions": len(poss), "shorter_than_3s": sum(1 for l, s, e in poss if (e - s) / fps < 3),
            "shorter_than_5s": len(short), "short_without_rule_support": unsupported,
            "flips_H-A-H_middle<5s": flips}

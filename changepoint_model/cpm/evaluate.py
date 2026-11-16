"""The official scorer (evaluation/possession_metrics.py) applied to a Match, plus the baseline."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

from .match import H, A, D, X


def score(gt, pred, bootstrap_n=0):
    """Dict of the official metrics. gt / pred: int codes (H=0, A=1, D=2, X=3)."""
    r = pm.evaluate(np.asarray(gt, dtype=np.int64), np.asarray(pred, dtype=np.int64),
                    bootstrap_n=bootstrap_n)
    out = {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
           "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
           "D_f1": r["per_class"]["D"]["f1"], "D_prec": r["per_class"]["D"]["precision"],
           "D_rec": r["per_class"]["D"]["recall"],
           "team_acc": r["team_acc_both_alive"], "alive_dead_f1": r["alive_dead_f1"],
           "mean_s": r["mean_spell_duration_s"], "gt_mean_s": r["gt_mean_spell_duration_s"]}
    for tau, v in r["transitions"].items():
        out["transF1@{}".format(tau)] = v.get("f1", np.nan)
        out["latency@{}".format(tau)] = v.get("mean_latency", np.nan)
    return out


def baseline(m, window=50):
    """The official baseline: team of the nearest tracked player where the ball is seen, D where
    it is not, majority vote over +-window frames inside each period."""
    T = m.T
    codes = np.full(T, -1, dtype=np.int8)
    seen = ~np.isnan(m.ball[:, 0])
    dx = m.px - m.ball[:, 0:1]
    dy = m.py - m.ball[:, 1:2]
    d = np.where(np.isnan(dx), np.inf, dx ** 2 + dy ** 2)
    j = np.argmin(d, axis=1)
    has = np.isfinite(d[np.arange(T), j]) & seen
    codes[has] = m.team[j[has]]
    out = np.full(T, X, dtype=np.int8)
    for p, s, e in m.period_slices():
        c = codes[s:e]
        hc = np.concatenate(([0], np.cumsum(c == 0)))
        ac = np.concatenate(([0], np.cumsum(c == 1)))
        idx = np.arange(e - s)
        lo = np.maximum(0, idx - window)
        hi = np.minimum(e - s, idx + window + 1)
        h = hc[hi] - hc[lo]
        a = ac[hi] - ac[lo]
        lab = np.full(e - s, D, dtype=np.int8)
        vis = (h + a) > 0
        lab[vis] = np.where(h[vis] >= a[vis], H, A)
        out[s:e] = lab
    return out

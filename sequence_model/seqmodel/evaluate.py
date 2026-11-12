"""Leave-one-match-out evaluation of any predictor with the official scorer, and the Baseline50
reference (nearest tracked player + majority vote)."""
import os
import sys

import numpy as np
import pandas as pd

from .decode import count_transitions, decode
from .match import CFG, controlled_labels
from .degrade import degrade

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402


class Baseline50:
    """Team of the tracked player nearest to the ball (D where the ball is unseen), then a centred
    majority vote over +-50 frames within each period. One-hot output, no decoding."""
    decode_output = False
    HALF = CFG["baseline50"]["half_window"]

    def fit(self, train):
        return self

    def predict(self, m):
        T = len(m.label)
        bx, by = np.asarray(m.ball_x, dtype=float), np.asarray(m.ball_y, dtype=float)
        seen = np.isfinite(bx) & np.isfinite(by)
        d = np.hypot(m.px - bx[:, None], m.py - by[:, None])
        d = np.where(np.isfinite(d), d, np.inf)
        ok = seen & np.isfinite(d.min(axis=1))
        raw = np.full(T, 2, dtype=np.int8)
        raw[ok] = m.team[np.argmin(d[ok], axis=1)]
        out = np.empty(T, dtype=np.int8)
        for per in np.unique(m.period):
            idx = np.flatnonzero(m.period == per)
            oh = np.eye(3)[raw[idx]]
            cs = np.vstack([np.zeros((1, 3)), np.cumsum(oh, axis=0)])
            n = len(idx)
            lo = np.clip(np.arange(n) - self.HALF, 0, n)
            hi = np.clip(np.arange(n) + self.HALF + 1, 0, n)
            out[idx] = np.argmax(cs[hi] - cs[lo], axis=1)
        return np.eye(3)[out], None


def lomo(matches, make_predictor, seed, cfg, use_change):
    rows = []
    deg = {j: degrade(m, seed + j, cfg) for j, m in enumerate(matches)}
    for i, m in enumerate(matches):
        tr_idx = [j for j in range(len(matches)) if j != i]
        train = [deg[j] for j in tr_idx]
        test = deg[i]
        pred = make_predictor().fit(train)
        p3, pc = pred.predict(test)
        if not np.isfinite(p3).all():
            raise ValueError(f"non-finite probabilities for held-out match {m.name}")
        if pred.decode_output:
            labs = [controlled_labels(t.label) for t in train]
            periods = [t.period for t in train]
            allv = np.concatenate([l[l <= 2] for l in labs])
            prior = np.bincount(allv, minlength=3)[:3] / len(allv)
            out = decode(p3, pc if use_change else None, test.period,
                         trans=count_transitions(labs, periods), prior=prior)
        else:
            out = np.argmax(p3, axis=1)
        gt = controlled_labels(m.label)
        r = pm.evaluate(gt, np.asarray(out, dtype=gt.dtype))
        rows.append(dict(match=m.name, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                         n_pred_poss=r["n_pred_spells"], n_true_poss=r["n_gt_spells"],
                         transF1_25=r["transitions"][25]["f1"]))
    return pd.DataFrame(rows)

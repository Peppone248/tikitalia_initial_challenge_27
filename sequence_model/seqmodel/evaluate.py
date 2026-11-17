"""Leave-one-match-out evaluation of any predictor with the official scorer, and the Baseline50
reference (nearest tracked player + majority vote)."""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .decode import count_transitions, decode
from .kickoff import apply_kickoff, kickoff_rows
from .match import CFG, controlled_labels
from .clean import clean_if_enabled
from .degrade import degrade

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


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


def plain_decoder(use_change):
    """Default decoding: argmax for one-hot predictors, otherwise decode() with beta=0 (or the
    default beta when use_change). Returns (labels, info dict)."""
    def f(c):
        if not c.pred.decode_output:
            return np.argmax(c.p3, axis=1), {}
        return decode(c.p3, c.pc if use_change else None, c.test.period,
                      trans=c.trans, prior=c.prior), {}
    return f


def select_decoder(betas, ws):
    """Pick (beta, w) by mean spellF1 on the training matches, predicted by the already fitted
    network (in-sample, no retraining), then decode the held-out match with it."""
    def f(c):
        t0 = time.time()
        if c.inner is None:         # shared by every select_decoder of the fold: the network is the same
            cache = []
            for t in c.train:
                p3, pc = c.pred.predict(t)
                if not np.isfinite(p3).all() or (pc is not None and not np.isfinite(pc).all()):
                    raise ValueError(f"non-finite probabilities for training match {t.name}")
                cache.append((t, p3, pc, controlled_labels(t.label)))
            c.inner = cache
            _log(f"  [select_decoder] inner predictions on {len(c.train)} training matches: "
                 f"{time.time() - t0:.1f}s")
        cache = c.inner
        t0 = time.time()
        best, best_s = None, -np.inf
        for beta in betas:
            for w in ws:
                s = np.mean([pm.evaluate(gt, np.asarray(
                    decode(p3, pc, t.period, w=w, beta=beta, trans=c.trans, prior=c.prior),
                    dtype=gt.dtype))["spell_f1_mean"] for t, p3, pc, gt in cache])
                if np.isfinite(s) and s > best_s:
                    best, best_s = (beta, w), s
        if best is None:
            raise ValueError("select_decoder: every inner spellF1 score is NaN")
        beta, w = best
        _log(f"  [select_decoder] grid {len(betas)}x{len(ws)} over {len(cache)} matches: "
             f"{time.time() - t0:.1f}s -> beta={beta:g} w={w:g}")
        return decode(c.p3, c.pc, c.test.period, w=w, beta=beta, trans=c.trans,
                      prior=c.prior), dict(beta=float(beta), w=float(w))
    return f


def lomo_multi(matches, make_predictor, seed, cfg, decoders, progress=None):
    """Leave-one-match-out where each fold's predictor is fitted once and decoded by every entry of
    decoders {name: f(ctx) -> (labels, info)}. Returns {name: DataFrame}. progress(match, seconds)
    is called after every fold."""
    rows = {k: [] for k in decoders}
    # the observed tracking goes through the same cleaning as the provided match (scripts/predict.py)
    deg = {j: clean_if_enabled(degrade(m, seed + j, cfg)) for j, m in enumerate(matches)}
    for i, m in enumerate(matches):
        t0 = time.time()
        train = [deg[j] for j in range(len(matches)) if j != i]
        test = deg[i]
        pred = make_predictor().fit(train)
        t1 = time.time()
        _log(f"  [fold {m.name}] fit: {t1 - t0:.1f}s")
        p3, pc = pred.predict(test)
        if not np.isfinite(p3).all():
            raise ValueError(f"non-finite probabilities for held-out match {m.name}")
        if pc is not None and not np.isfinite(pc).all():
            raise ValueError(f"non-finite p_change for held-out match {m.name}")
        trans = prior = None
        if pred.decode_output:
            labs = [controlled_labels(t.label) for t in train]
            allv = np.concatenate([l[l <= 2] for l in labs])
            prior = np.bincount(allv, minlength=3)[:3] / len(allv)
            if getattr(pred, "prior_corrected", False):
                prior = None        # the net was trained with inverse-frequency class weights: its
                #                     output is already roughly prior-free, do not correct it twice
            trans = count_transitions(labs, [t.period for t in train])
        ctx = SimpleNamespace(pred=pred, train=train, test=test, p3=p3, pc=pc, trans=trans,
                              prior=prior, inner=None)
        gt = controlled_labels(m.label)
        t2 = time.time()
        _log(f"  [fold {m.name}] test-match predict: {t2 - t1:.1f}s")
        ko_rows = kickoff_rows(test, test.graded_periods) if CFG["kickoff"]["enabled"] else None
        for k, dec in decoders.items():
            out, info = dec(ctx)
            if ko_rows is not None:             # same final step as scripts/predict.py
                out = apply_kickoff(out, test, test.graded_periods, ko_rows)
            r = pm.evaluate(gt, np.asarray(out, dtype=gt.dtype))
            rows[k].append(dict(match=m.name, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                                n_pred_poss=r["n_pred_spells"], n_true_poss=r["n_gt_spells"],
                                transF1_25=r["transitions"][25]["f1"], **info))
        _log(f"  [fold {m.name}] test-match decode(s): {time.time() - t2:.1f}s")
        if progress is not None:
            progress(m.name, time.time() - t0)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def lomo(matches, make_predictor, seed, cfg, use_change, progress=None):
    return lomo_multi(matches, make_predictor, seed, cfg, {"x": plain_decoder(use_change)},
                      progress)["x"]

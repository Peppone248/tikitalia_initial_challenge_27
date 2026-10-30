"""NN targets from the rule evidence only. No ground truth, no person decides a label (RULES 1:
"pseudo-labels produced by an algorithm, provided no person decided any of them").

Two steps, so that the NN's X means "live play, ball not seen" and not just "ball not seen":
    1. among H, A, D: target = argmax if E_best >= min_evidence and E_best - E_second >= min_margin
    2. otherwise X, if the X evidence (X-NOBALL) >= min_evidence and no H/A/D class reaches
       min_evidence: the ball is not seen and nothing says stoppage or team
    otherwise -1 (unlabelled). Only graded frames are labelled.
An occluded stoppage with D evidence (D-HOLD, D-GOAL, ...) is therefore a D target, and the NN
can learn what an unseen stoppage looks like from the players (collective slowdown, shape).
"""

import numpy as np

from .votes import CLASSES


def make(E, graded, cfg):
    had = E[:, :3]
    order = np.argsort(-had, axis=1)
    rows = np.arange(len(E))
    best, second = had[rows, order[:, 0]], had[rows, order[:, 1]]
    ok = graded & (best >= cfg["min_evidence"]) & (best - second >= cfg["min_margin"])
    target = np.where(ok, order[:, 0], -1).astype(np.int8)
    x = graded & ~ok & (E[:, 3] >= cfg["min_evidence"]) & (best < cfg["min_evidence"])
    target[x] = 3
    counts = {c: int((target == k).sum()) for k, c in enumerate(CLASSES)}
    counts["unlabelled_graded"] = int((graded & (target < 0)).sum())
    return target, counts

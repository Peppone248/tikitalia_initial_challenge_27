"""Frame-level classifiers.

TeamModel      P(home has the ball | alive frame) from the odd features: logistic regression with
               no intercept, scaled by the training standard deviations only (no centring), so
               that P(home | swapped teams) = 1 - P(home) exactly. Newton steps with an L2 penalty
               (MAP with a Gaussian prior), deterministic, numpy only.
EvenModel      a classifier on the even features (+ |odd| features), for the dead ball and for the
               three boundary heads. Two backends: 'logistic' (numpy, same solver) and 'hgb'
               (sklearn HistGradientBoostingClassifier, fixed seed). Even inputs make the output
               invariant under the team swap by construction.

Targets for the boundary heads (within a period, from the labels):
    change   a direct H<->A change of possession within +-k frames
    stop     an alive -> D transition within +-k frames
    restart  a D -> alive transition within +-k frames
"""

import json

import numpy as np

from .match import H, A, D, X

BOUNDARY_K = 12


def boundary_targets(labels, period, k=BOUNDARY_K):
    """Three boolean arrays (T,): change, stop, restart; a frame is positive within +-k frames of
    the transition frame (the first frame of the new state)."""
    T = len(labels)
    change = np.zeros(T, dtype=bool)
    stop = np.zeros(T, dtype=bool)
    restart = np.zeros(T, dtype=bool)
    same_period = period[1:] == period[:-1]
    prev, cur = labels[:-1], labels[1:]
    t_change = np.flatnonzero(same_period & (((prev == H) & (cur == A)) | ((prev == A) & (cur == H)))) + 1
    t_stop = np.flatnonzero(same_period & (prev < D) & (cur == D)) + 1
    t_restart = np.flatnonzero(same_period & (prev == D) & (cur < D)) + 1
    for arr, ts in ((change, t_change), (stop, t_stop), (restart, t_restart)):
        for t in ts:
            arr[max(0, t - k):min(T, t + k + 1)] = True
    return change, stop, restart


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _newton_logistic(Xm, y, l2, intercept, max_iter=50, tol=1e-8, sample_weight=None):
    n, d = Xm.shape
    if intercept:
        Xm = np.concatenate([Xm, np.ones((n, 1))], axis=1)
    w = np.zeros(Xm.shape[1])
    sw = np.ones(n) if sample_weight is None else sample_weight
    reg = np.full(Xm.shape[1], l2)
    if intercept:
        reg[-1] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(Xm @ w)
        g = Xm.T @ (sw * (p - y)) + reg * w
        s = sw * p * (1 - p) + 1e-9
        Hm = (Xm * s[:, None]).T @ Xm + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        # damping: halve the step while the penalised loss does not decrease
        loss0 = _loss(Xm, y, w, reg, sw)
        alpha = 1.0
        while alpha > 1e-3:
            w_new = w - alpha * step
            if _loss(Xm, y, w_new, reg, sw) <= loss0 + 1e-12:
                break
            alpha *= 0.5
        w = w_new
        if np.max(np.abs(alpha * step)) < tol:
            break
    return w


def _loss(Xm, y, w, reg, sw):
    z = Xm @ w
    ll = np.sum(sw * (np.logaddexp(0, z) - y * z))
    return ll + 0.5 * np.sum(reg * w * w)


class TeamModel:
    """Antisymmetric logistic regression on the odd features."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.scale = None
        self.w = None
        self.names = None

    def fit(self, odd, y_home, names=None, sample_weight=None):
        self.names = names
        self.scale = odd.std(axis=0) + 1e-6
        Xm = odd / self.scale
        self.w = _newton_logistic(Xm, y_home.astype(float), self.l2, intercept=False,
                                  sample_weight=sample_weight)
        return self

    def predict_proba(self, odd):
        return _sigmoid((odd / self.scale) @ self.w)

    def to_dict(self):
        return {"kind": "team", "l2": self.l2, "scale": self.scale.tolist(), "w": self.w.tolist(),
                "names": list(self.names) if self.names else None}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"])
        m.scale = np.array(d["scale"])
        m.w = np.array(d["w"])
        m.names = d.get("names")
        return m


def even_inputs(even, odd):
    """Inputs of the even heads: the even features and the absolute values of the odd ones."""
    return np.concatenate([even, np.abs(odd)], axis=1)


class EvenModel:
    def __init__(self, backend="hgb", l2=1.0, seed=2027, max_iter=200, learning_rate=0.1,
                 max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0):
        self.backend = backend
        self.l2 = l2
        self.seed = seed
        self.hgb_params = dict(max_iter=max_iter, learning_rate=learning_rate,
                               max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                               l2_regularization=l2_regularization)
        self.model = None
        self.mean = None
        self.scale = None
        self.w = None

    def fit(self, Xe, y, sample_weight=None):
        y = y.astype(int)
        if self.backend == "logistic":
            self.mean = Xe.mean(axis=0)
            self.scale = Xe.std(axis=0) + 1e-6
            Xm = (Xe - self.mean) / self.scale
            self.w = _newton_logistic(Xm, y.astype(float), self.l2, intercept=True,
                                      sample_weight=sample_weight)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(random_state=self.seed,
                                                        early_stopping=False, **self.hgb_params)
            self.model.fit(Xe, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, Xe):
        if self.backend == "logistic":
            Xm = (Xe - self.mean) / self.scale
            return _sigmoid(Xm @ self.w[:-1] + self.w[-1])
        return self.model.predict_proba(Xe)[:, 1]

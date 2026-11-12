"""Per-frame classifier (the fair reference for sequence models): multinomial logistic regression on
frame_features, with exact team-swap symmetry. Implements the predictor protocol
(fit(list[Match]) -> self, predict(Match) -> (p3, p_change|None), decode_output)."""
import numpy as np

from .features import ODD, frame_features
from .match import CFG, controlled_labels

_C = CFG["frame_model"]
L2, STEP, MAX_ITER, TOL = _C["l2"], _C["step"], _C["max_iter"], _C["tol"]


def fit_standardiser(X):
    """(mu, sd) of the feature columns: odd columns scale only (no centring), so negating them
    stays an exact team swap; even columns are centred and scaled. Constant columns get sd 1."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    mu[ODD] = 0.0
    sd = np.sqrt(((X - mu) ** 2).mean(axis=0))
    return mu, np.where(sd > 1e-9, sd, 1.0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class FrameModel:
    decode_output = True

    def __init__(self, l2=L2, step=STEP):
        self.l2, self.step = l2, step

    # ---- low-level API on raw feature matrices ----
    def _standardise(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def _fit_xy(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.mu, self.sd = fit_standardiser(X)
        Z = self._standardise(X)
        Zs = Z.copy()
        Zs[:, ODD] *= -1
        ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
        Z = np.vstack([Z, Zs])
        y = np.concatenate([y, ys]).astype(int)
        n, f = Z.shape
        Zb = np.hstack([Z, np.ones((n, 1))])
        Y = np.eye(3)[y]
        reg = np.full(f + 1, self.l2)
        reg[-1] = 0.0
        R = np.diag(np.repeat(reg, 3))      # parameter order: feature-major, class-minor
        W = np.zeros((f + 1, 3))
        for _ in range(MAX_ITER):
            P = _softmax(Zb @ W)
            g = (Zb.T @ (P - Y) + reg[:, None] * W).ravel()
            H = np.zeros((3 * (f + 1), 3 * (f + 1)))
            for a in range(3):
                for b in range(a, 3):
                    wgt = P[:, a] * ((a == b) - P[:, b])
                    blk = (Zb * wgt[:, None]).T @ Zb
                    H[a::3, b::3] = blk
                    if a != b:
                        H[b::3, a::3] = blk
            H += R + 1e-9 * np.eye(len(H))
            step = np.linalg.solve(H, g)
            W -= step.reshape(f + 1, 3)
            if np.abs(step).max() < TOL:
                break
        self.W = W
        return self

    def _proba(self, Z):
        return _softmax(np.hstack([Z, np.ones((len(Z), 1))]) @ self.W)

    def predict_proba(self, X):
        Z = self._standardise(X)
        Zs = Z.copy()
        Zs[:, ODD] *= -1
        return 0.5 * (self._proba(Z) + self._proba(Zs)[:, [1, 0, 2]])

    # ---- protocol ----
    def fit(self, train, y=None):
        if y is not None:
            return self._fit_xy(train, y)
        Xs, ys = [], []
        for m in train:
            X, lab = frame_features(m), controlled_labels(m.label)
            keep = np.flatnonzero(lab <= 2)
            keep = keep[::self.step]
            Xs.append(X[keep])
            ys.append(lab[keep])
        return self._fit_xy(np.vstack(Xs), np.concatenate(ys))

    def predict(self, m):
        p = self.predict_proba(frame_features(m))
        if not np.isfinite(p).all():
            raise ValueError("FrameModel.predict: non-finite probabilities")
        return p, None

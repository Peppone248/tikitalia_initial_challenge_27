"""Deterministic plain-numpy inference of the exported TCN. float64 throughout, no BLAS (einsum
without `optimize` runs fixed C loops), inputs rounded to a 2^-16 grid and outputs to 3 decimals,
so the submission regenerates byte-identically on any machine. Reads the npz of export_weights."""
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .features import FEATURES, ODD, frame_features
from .match import CFG
from .tcn import TCNPredictor, _period_slices, export_weights

_IN_GRID = 2.0 ** 16
_OUT_DECIMALS = 3


def _conv(x, w, b, dilation):
    """x (Cin,T), w (Cout,Cin,K) -> (Cout,T); zero 'same' padding dilation*(K-1)//2."""
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    win = sliding_window_view(xp, dilation * (k - 1) + 1, axis=1)[:, :, ::dilation]   # (Cin,T,K)
    return np.einsum("itk,oik->ot", win, w) + b[:, None]


def _softmax(z):
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


class NumpyTCN:
    def __init__(self, z):
        self.levels, self.kernel = int(z["levels"]), int(z["kernel"])
        self.mu, self.sd = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
        self.odd = z["odd"].astype(bool)
        n_in = int(z["n_in"])
        if n_in != len(FEATURES) or len(self.odd) != len(FEATURES):
            raise ValueError(f"weights expect {n_in} input features but this code produces "
                             f"{len(FEATURES)}: stale weights, retrain")
        f = lambda k: z[k].astype(np.float64)          # noqa: E731
        self.inp = (f("inp.w"), f("inp.b"))
        self.blocks = [(f(f"l{i}.conv.w"), f(f"l{i}.conv.b"), f(f"l{i}.pw.w"), f(f"l{i}.pw.b"))
                       for i in range(self.levels)]
        self.head3 = (f("head3.w"), f("head3.b"))
        self.headc = (f("headc.w"), f("headc.b"))

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(z)

    def _logits(self, x):
        h = _conv(x, *self.inp, 1)
        for i, (cw, cb, pw, pb) in enumerate(self.blocks):
            h = h + _conv(np.maximum(_conv(h, cw, cb, 2 ** i), 0.0), pw, pb, 1)
        return _conv(h, *self.head3, 1), _conv(h, *self.headc, 1)[0]

    def predict(self, X, odd_mask=None, round_outputs=True):
        """X (T,F) raw features -> (p3 (T,3), p_change (T,)); mean of f(x) and swap(f(swap(x)))."""
        odd = self.odd if odd_mask is None else np.asarray(odd_mask, dtype=bool)
        x = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        if not np.isfinite(x).all():
            raise ValueError("NumpyTCN.predict: non-finite features")
        x = (np.round(x * _IN_GRID) / _IN_GRID).T
        xs = x.copy()
        xs[odd] *= -1
        l3, lc = self._logits(x)
        s3, sc = self._logits(xs)
        p3 = 0.5 * (_softmax(l3) + _softmax(s3)[[1, 0, 2]])
        pc = 0.5 * (1 / (1 + np.exp(-lc)) + 1 / (1 + np.exp(-sc)))
        p3, pc = p3.T, pc
        if round_outputs:
            p3, pc = np.round(p3, _OUT_DECIMALS), np.round(pc, _OUT_DECIMALS)
        return p3, pc

    def predict_match(self, m, round_outputs=True):
        """Same contract as TCNPredictor.predict: runs per period, never across a boundary."""
        X = frame_features(m)
        p3, pc = np.zeros((len(X), 3)), np.zeros(len(X))
        for a, b in _period_slices(m.period):
            p3[a:b], pc[a:b] = self.predict(X[a:b], self.odd, round_outputs)
        return p3, pc


class NumpyTCNPredictor:
    """Predictor protocol; trains with train_tcn but predicts through NumpyTCN (the submission path)."""
    decode_output = True
    prior_corrected = True      # see TCNPredictor

    def __init__(self, seed, cfg=None):
        self.seed, self.cfg = seed, CFG["tcn"] if cfg is None else cfg

    def fit(self, train):
        tp = TCNPredictor(self.seed, self.cfg).fit(train)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "w.npz")
            export_weights(tp.model, path)
            self.net = NumpyTCN.load(path)
        return self

    def predict(self, m):
        return self.net.predict_match(m)

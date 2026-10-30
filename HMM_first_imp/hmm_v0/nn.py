"""Light NN in plain numpy: P(H, A, D, X | window of frames). No framework, so the run is
reproducible on any machine with numpy (RULES 3: the repository must regenerate the same file).

    input   (2k+1) * F features (frame_features.window), standardised on the training set
    hidden  ReLU layers (hmm.yaml nn.hidden)
    output  softmax over the 4 classes
    loss    class-weighted cross-entropy + L2, Adam, fixed seed

Trained on the pseudo-labels of the rules (pseudo_labels.py), never on any ground truth.

Reproducibility across machines (RULES 3), nn.exact_arithmetic: true (default):
    inputs quantised to a 2^-16 grid, standardisation from exact integer sums, every matrix
    product in int64 fixed point (exact.qmatmul: independent of CPU / BLAS summation order),
    softmax with a polynomial exp (exact.det_exp), uniform init from integers, Adam bias
    corrections by running products. v0 (float BLAS training) differed on 1.3% of the frames
    between two PCs; with exact arithmetic the training trajectory is the same everywhere. With
swap_augment every training frame is also shown with home and away exchanged (and H <-> A in
the target): the network cannot learn "home is more likely", only the geometry.
"""

import numpy as np

from . import exact
from .frame_features import window, window_swap

INPUT_BITS = 16

SWAP_CLASS = np.array([1, 0, 2, 3])      # H <-> A, D and X unchanged


class MLP(object):

    def __init__(self, n_in, hidden, n_out, seed, exact_arith=True):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [n_out]
        self.exact = bool(exact_arith)
        # He-uniform init from integers (no libm in the random draws)
        self.W = [(rng.integers(-2 ** 24, 2 ** 24, size=(a, b)) / 2.0 ** 24) * np.sqrt(6.0 / a)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def _mm(self, a, b):
        return exact.qmatmul(a, b) if self.exact else a @ b

    def forward(self, x, keep=False):
        h = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        acts = [h]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = self._mm(h, W) + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        z = h - h.max(axis=1, keepdims=True)
        p = exact.det_exp(z) if self.exact else np.exp(z)
        p = p / (p[:, 0] + p[:, 1] + p[:, 2] + p[:, 3])[:, None]          # fixed summation order
        return (p, acts) if keep else p

    def step(self, x, y, cw, lr, l2, state):
        p, acts = self.forward(x, keep=True)
        n = x.shape[0]
        w = cw[y]
        wsum = float(exact.quantize(w, 20).sum())                         # grid values: exact sum
        loss = float(-(w * np.log(p[np.arange(n), y] + 1e-9)).sum() / wsum)  # reporting only
        g = p.copy()
        g[np.arange(n), y] -= 1.0
        g *= (w / wsum)[:, None]
        ones = np.ones((1, n))
        grads_W, grads_b = [], []
        for i in range(len(self.W) - 1, -1, -1):
            grads_W.append(self._mm(acts[i].T, g) + l2 * self.W[i])
            grads_b.append(self._mm(ones, g)[0])
            if i > 0:
                g = self._mm(g, self.W[i].T) * (acts[i] > 0)
        grads_W, grads_b = grads_W[::-1], grads_b[::-1]
        b1, b2, eps = 0.9, 0.999, 1e-8
        state["b1t"] *= b1                                               # b1 ** t, no pow()
        state["b2t"] *= b2
        for i, (gw, gb) in enumerate(zip(grads_W, grads_b)):
            for key, param, grad in (("W", self.W, gw), ("b", self.b, gb)):
                m = state[key + "m"][i] = b1 * state[key + "m"][i] + (1 - b1) * grad
                v = state[key + "v"][i] = b2 * state[key + "v"][i] + (1 - b2) * grad * grad
                mh, vh = m / (1 - state["b1t"]), v / (1 - state["b2t"])
                param[i] = param[i] - lr * mh / (np.sqrt(vh) + eps)
        return loss


class RandomFeatureNet(object):
    """One hidden layer with FIXED random weights (ReLU), trained output layer = class-weighted
    softmax regression with L2 (an extreme-learning-machine / random-feature network).

    Why (RULES 3): the training problem is convex, solved by full-batch gradient descent with a
    fixed step <= 1/L. On a convex L-smooth problem every gradient step is non-expansive (with
    L2: contracting), so a tiny difference in the inputs (another CPU's float32 features, another
    BLAS) stays tiny in the output and disappears in the rounding of the probabilities. Minibatch
    SGD on the MLP is chaotic instead: 10 inputs moved by 2^-16 changed 1,896 output frames.
    """

    def __init__(self, n_in, n_hidden, seed):
        rng = np.random.default_rng(seed)
        self.R = (rng.integers(-2 ** 24, 2 ** 24, size=(n_in, n_hidden)) / 2.0 ** 24) \
            * np.sqrt(3.0 / n_in)
        self.c = (rng.integers(-2 ** 24, 2 ** 24, size=n_hidden) / 2.0 ** 24) * 0.5
        self.V = np.zeros((n_hidden + 1, 4))
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)
        self.h_mu = np.zeros(n_hidden)
        self.h_sd = np.ones(n_hidden)

    def hidden(self, x, set_scale=False):
        z = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        h = np.maximum(z @ self.R + self.c, 0.0)
        if set_scale:                                  # standardised hidden units: better
            self.h_mu, self.h_sd = h.mean(axis=0), h.std(axis=0) + 1e-3   # conditioning
        h = (h - self.h_mu) / self.h_sd
        return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def forward(self, x):
        return self._softmax(self.hidden(x) @ self.V)

    def fit(self, Phi, y, cw, l2, iters, log=print):
        n = Phi.shape[0]
        w = cw[y] / cw[y].sum()                       # weights sum to 1
        Y = np.zeros((n, 4))
        Y[np.arange(n), y] = 1.0
        # L of the weighted softmax loss <= 0.5 * lambda_max(Phi^T diag(w) Phi) + l2
        v = np.ones(Phi.shape[1]) / np.sqrt(Phi.shape[1])
        for _ in range(30):                            # power iteration
            v = Phi.T @ (w * (Phi @ v))
            v /= np.linalg.norm(v)
        L = 0.5 * float(v @ (Phi.T @ (w * (Phi @ v)))) + l2
        eta = 1.0 / L
        hist = []
        # Nesterov accelerated gradient, fixed step and fixed momentum schedule: a deterministic,
        # Lipschitz-continuous function of the data (no line search, no restarts, no minibatch)
        V_prev = self.V.copy()
        for it in range(int(iters)):
            mom = it / (it + 3.0)
            Yk = self.V + mom * (self.V - V_prev)
            P = self._softmax(Phi @ Yk)
            G = Phi.T @ ((P - Y) * w[:, None]) + l2 * Yk
            V_prev = self.V
            self.V = Yk - eta * G
            if (it + 1) % max(1, int(iters) // 5) == 0:
                P = self._softmax(Phi @ self.V)
                loss = float(-(w * np.log(P[np.arange(n), y] + 1e-12)).sum())
                hist.append(round(loss, 4))
                log("  rf-net iter {}: loss {}".format(it + 1, hist[-1]))
        return hist, eta


def _stratified(target, idx, max_n, rng):
    """At most max_n frames, the same share removed from every class (keeps rare classes)."""
    if len(idx) <= max_n:
        return idx
    keep = []
    for c in np.unique(target[idx]):
        ic = idx[target[idx] == c]
        k = max(1, int(round(len(ic) * max_n / float(len(idx)))))
        keep.append(rng.choice(ic, size=min(k, len(ic)), replace=False))
    return np.sort(np.concatenate(keep))


def train(X, swap, target, cfg, log=print):
    """Train on frames with target >= 0. Returns (model, info)."""
    k = int(cfg["context_frames"])
    rng = np.random.default_rng(int(cfg["seed"]))
    idx = np.flatnonzero(target >= 0)
    idx = _stratified(target, idx, int(cfg["max_train_frames"]), rng)
    exact_arith = bool(cfg.get("exact_arithmetic", True))
    Xw = window(X, idx, k).astype(np.float64)
    y = target[idx].astype(int)
    if cfg.get("swap_augment", True):
        perm = window_swap(swap, k)
        Xw = np.concatenate([Xw, Xw[:, perm]])
        y = np.concatenate([y, SWAP_CLASS[y]])
    counts = np.bincount(y, minlength=4).astype(float)
    cw = np.where(counts > 0, counts.sum() / (4.0 * np.maximum(counts, 1)), 0.0) \
        if cfg.get("class_balance", True) else np.ones(4)
    cw = exact.quantize(cw, 20)

    if exact_arith:
        mu, sd = exact.exact_mean_std(Xw, INPUT_BITS)
    else:
        mu, sd = Xw.mean(axis=0), Xw.std(axis=0)
    prior = counts / counts.sum()
    if cfg.get("model", "mlp") == "rf":
        model = RandomFeatureNet(Xw.shape[1], int(cfg["rf_hidden"]), int(cfg["seed"]))
        model.mu, model.sd = mu, sd + 1e-3
        Phi = model.hidden(Xw, set_scale=True)
        del Xw
        hist, eta = model.fit(Phi, y, cw, float(cfg["rf_l2"]), int(cfg["rf_iters"]), log=log)
        pred = model._softmax(Phi @ model.V).argmax(axis=1)
        info = {"model": "rf", "train_frames": int(len(idx)), "train_rows_with_swap": int(len(y)),
                "class_counts": counts.astype(int).tolist(), "class_weights": cw.round(3).tolist(),
                "prior": prior.round(4).tolist(), "loss_trace": hist, "step": eta,
                "train_agreement_with_pseudo_labels": round(float((pred == y).mean()), 4),
                "n_inputs": int(mu.size), "n_hidden": int(cfg["rf_hidden"])}
        return model, prior, info
    model = MLP(Xw.shape[1], cfg["hidden"], 4, int(cfg["seed"]), exact_arith)
    model.mu, model.sd = mu, sd + 1e-3
    state = {"t": 0, "b1t": 1.0, "b2t": 1.0, "Wm": [np.zeros_like(w) for w in model.W],
             "Wv": [np.zeros_like(w) for w in model.W],
             "bm": [np.zeros_like(b) for b in model.b], "bv": [np.zeros_like(b) for b in model.b]}
    bs = int(cfg["batch_size"])
    history = []
    for epoch in range(int(cfg["epochs"])):
        order = rng.permutation(len(y))
        losses = [model.step(Xw[order[i:i + bs]], y[order[i:i + bs]], cw,
                             float(cfg["learning_rate"]), float(cfg["l2"]), state)
                  for i in range(0, len(y), bs)]
        history.append(round(float(np.mean(losses)), 4))
        log("  nn epoch {}: loss {}".format(epoch + 1, history[-1]))
    pred = model.forward(Xw).argmax(axis=1)
    info = {"model": "mlp", "train_frames": int(len(idx)), "train_rows_with_swap": int(len(y)),
            "class_counts": counts.astype(int).tolist(), "class_weights": cw.round(3).tolist(),
            "prior": prior.round(4).tolist(), "loss_per_epoch": history,
            "train_agreement_with_pseudo_labels": round(float((pred == y).mean()), 4),
            "n_inputs": int(Xw.shape[1])}
    return model, prior, info


def predict(model, X, k, batch=20000, symmetric=True, swap=None, decimals=None):
    """P (n, 4) for every frame. symmetric: average with the swapped view (exact H/A symmetry).
    decimals: round the probabilities (cross-machine reproducibility), floor 10^-decimals."""
    n = X.shape[0]
    out = np.zeros((n, 4))
    perm = window_swap(swap, k) if symmetric else None
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        xw = window(X, idx, k)
        p = model.forward(xw)
        if symmetric:
            ps = model.forward(xw[:, perm])[:, SWAP_CLASS]
            p = 0.5 * (p + ps)
        out[idx] = p
    if decimals is not None:
        out = np.maximum(np.round(out, int(decimals)), 10.0 ** -int(decimals))
    return out

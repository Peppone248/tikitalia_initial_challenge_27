"""A small bidirectional GRU over the per-frame features: the sequence version of the heads.

Why: the per-frame classifiers see a fixed set of windows; a recurrent network sees the whole
neighbourhood of a frame and learns the shape of a change of possession (the votes stop, the
teams swap roles, a new run of control starts) instead of being told which windows to look at.
It outputs, per frame: P(H), P(A), P(D) and the three boundary probabilities (change, stop,
restart, within +-k frames), exactly the quantities the decoder of decode.py consumes.

Team-swap symmetry: training uses the swap augmentation (odd features negated, H <-> A); at
inference the network is run on x and on swap(x) and the two are averaged (with H / A exchanged
in the second), so the output is exactly symmetric.

Training needs torch; prediction does not: the weights are exported to a .npz and run by the numpy
forward pass below (same arithmetic, so the same labels on a machine without torch).
"""

import numpy as np

from .match import H, A, D, X
from .models import boundary_targets, BOUNDARY_K

SEQ_CFG = dict(hidden=64, layers=2, window=1500, batch=16, lr=1e-3, epochs=30,
               windows_per_epoch=100, dropout=0.1, seed=2027, threads=2, k=BOUNDARY_K,
               boundary_weight=1.0)


# ------------------------------------------------------------------------------------------------
# data

def stack_inputs(P, mean, scale, swap=False):
    odd = -P.odd if swap else P.odd
    x = np.concatenate([P.even, odd], axis=1).astype(np.float32)
    return (x - mean) / scale


def fit_scaler(preps):
    xs = np.concatenate([np.concatenate([P.even, P.odd], axis=1)[P.graded][::10] for P in preps])
    mean = xs.mean(axis=0)
    mean[preps[0].even.shape[1]:] = 0.0     # odd features keep their centre at 0 (swap symmetry)
    scale = xs.std(axis=0) + 1e-6
    return mean.astype(np.float32), scale.astype(np.float32)


def targets(P, k):
    lab = P.labels.astype(np.int64).copy()
    ch, st, re = boundary_targets(P.labels, P.period, k)
    b = np.stack([ch, st, re], axis=1).astype(np.float32)
    return lab, b


# ------------------------------------------------------------------------------------------------
# torch model

def build_model(n_in, cfg):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, cfg["hidden"]), nn.ReLU())
            self.gru = nn.GRU(cfg["hidden"], cfg["hidden"], num_layers=cfg["layers"],
                              batch_first=True, bidirectional=True,
                              dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0)
            self.head_cls = nn.Linear(2 * cfg["hidden"], 3)
            self.head_bnd = nn.Linear(2 * cfg["hidden"], 3)

        def forward(self, x):
            h = self.inp(x)
            h, _ = self.gru(h)
            return self.head_cls(h), self.head_bnd(h)
    return Net()


def train_seq(preps, cfg=None, verbose=True, log=print):
    import torch
    import torch.nn.functional as F
    cfg = dict(SEQ_CFG, **(cfg or {}))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.RandomState(cfg["seed"])
    mean, scale = fit_scaler(preps)
    data = []
    n_even = preps[0].even.shape[1]
    for P in preps:
        x = stack_inputs(P, mean, scale)          # one float32 copy per match; swap made per window
        lab, b = targets(P, cfg["k"])
        for p, s, e in P.match.period_slices():
            if e - s > cfg["window"]:
                data.append((x[s:e], lab[s:e], b[s:e]))
    n_in = data[0][0].shape[1]
    net = build_model(n_in, cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    W = cfg["window"]
    # class weights: inverse frequency over the graded frames, so that D and the teams weigh alike
    labs_all = np.concatenate([d[1] for d in data])
    labs_all = labs_all[labs_all != X]
    freq = np.bincount(labs_all, minlength=3)[:3] / len(labs_all)
    cw = torch.tensor((1.0 / np.maximum(freq, 1e-3)) / 3.0, dtype=torch.float32)
    for epoch in range(cfg["epochs"]):
        net.train()
        tot, nb = 0.0, 0
        for _ in range(cfg["windows_per_epoch"] // cfg["batch"]):
            xb, yb, bb = [], [], []
            for _ in range(cfg["batch"]):
                x, lab, b = data[rng.randint(len(data))]
                s = rng.randint(0, len(x) - W)
                xw = x[s:s + W].copy()
                yw = lab[s:s + W].copy()
                if rng.rand() < 0.5:                  # team swap: odd inputs negated, H <-> A
                    xw[:, n_even:] *= -1.0
                    h, a = yw == H, yw == A
                    yw[h], yw[a] = A, H
                xb.append(xw)
                yb.append(yw)
                bb.append(b[s:s + W])
            xb = torch.from_numpy(np.stack(xb))
            yb = torch.from_numpy(np.stack(yb))
            bb = torch.from_numpy(np.stack(bb))
            logits, bnd = net(xb)
            mask = yb != X
            loss_c = F.cross_entropy(logits[mask], yb[mask], weight=cw)
            loss_b = F.binary_cross_entropy_with_logits(bnd[mask], bb[mask])
            loss = loss_c + cfg["boundary_weight"] * loss_b
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if verbose:
            log("[seq] epoch {}/{} loss {:.4f}".format(epoch + 1, cfg["epochs"], tot / max(nb, 1)))
    return export_weights(net, cfg, mean, scale)


def export_weights(net, cfg, mean, scale):
    import torch
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    sd["_mean"] = mean
    sd["_scale"] = scale
    sd["_cfg"] = np.array([cfg["hidden"], cfg["layers"], cfg["k"]], dtype=np.int64)
    return sd


# ------------------------------------------------------------------------------------------------
# numpy forward pass

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _gru_dir(x, w_ih, w_hh, b_ih, b_hh, reverse):
    """One direction of a GRU over x (T, n_in) -> (T, hidden). PyTorch gate order r, z, n."""
    T = x.shape[0]
    hid = w_hh.shape[1]
    hs = np.zeros((T, hid), dtype=np.float32)
    h = np.zeros(hid, dtype=np.float32)
    gi = x @ w_ih.T + b_ih                      # (T, 3h)
    order = range(T - 1, -1, -1) if reverse else range(T)
    for t in order:
        gh = w_hh @ h + b_hh
        r = _sigmoid(gi[t, :hid] + gh[:hid])
        z = _sigmoid(gi[t, hid:2 * hid] + gh[hid:2 * hid])
        n = np.tanh(gi[t, 2 * hid:] + r * gh[2 * hid:])
        h = (1 - z) * n + z * h
        hs[t] = h
    return hs


def forward_numpy(sd, x):
    """x (T, n_in) standardised -> (logits_cls (T, 3), logits_bnd (T, 3))."""
    hidden, layers, _ = [int(v) for v in sd["_cfg"]]
    h = np.maximum(x @ sd["inp.0.weight"].T + sd["inp.0.bias"], 0.0).astype(np.float32)
    for l in range(layers):
        fwd = _gru_dir(h, sd["gru.weight_ih_l{}".format(l)], sd["gru.weight_hh_l{}".format(l)],
                       sd["gru.bias_ih_l{}".format(l)], sd["gru.bias_hh_l{}".format(l)], False)
        bwd = _gru_dir(h, sd["gru.weight_ih_l{}_reverse".format(l)],
                       sd["gru.weight_hh_l{}_reverse".format(l)],
                       sd["gru.bias_ih_l{}_reverse".format(l)],
                       sd["gru.bias_hh_l{}_reverse".format(l)], True)
        h = np.concatenate([fwd, bwd], axis=1)
    return h @ sd["head_cls.weight"].T + sd["head_cls.bias"], h @ sd["head_bnd.weight"].T + sd["head_bnd.bias"]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_seq(sd, P, use_torch=False):
    """Symmetrised probabilities: dict p_home, p_dead, p_change, p_stop, p_restart (T,)."""
    mean, scale = sd["_mean"], sd["_scale"]
    T = P.match.T
    cls = np.zeros((T, 3), dtype=np.float64)
    bnd = np.zeros((T, 3), dtype=np.float64)
    runner = _torch_forward(sd) if use_torch else forward_numpy
    for p, s, e in P.match.period_slices():
        x = stack_inputs(P, mean, scale)[s:e]
        xs = stack_inputs(P, mean, scale, swap=True)[s:e]
        lc, lb = runner(sd, x)
        lcs, lbs = runner(sd, xs)
        pc = _softmax(lc)
        pcs = _softmax(lcs)[:, [1, 0, 2]]          # H <-> A exchanged back
        cls[s:e] = 0.5 * (pc + pcs)
        bnd[s:e] = 0.5 * (_sigmoid(lb) + _sigmoid(lbs))
    alive = np.clip(cls[:, 0] + cls[:, 1], 1e-6, 1.0)
    return {"p_home": cls[:, 0] / alive, "p_dead": cls[:, 2], "p_change": bnd[:, 0],
            "p_stop": bnd[:, 1], "p_restart": bnd[:, 2], "p_cls": cls}


def _torch_forward(sd):
    import torch
    hidden, layers, k = [int(v) for v in sd["_cfg"]]
    n_in = sd["inp.0.weight"].shape[1]
    net = build_model(n_in, dict(SEQ_CFG, hidden=hidden, layers=layers))
    net.load_state_dict({k_: torch.from_numpy(v) for k_, v in sd.items() if not k_.startswith("_")})
    net.eval()

    def run(_sd, x):
        with torch.no_grad():
            lc, lb = net(torch.from_numpy(np.ascontiguousarray(x))[None])
        return lc[0].numpy(), lb[0].numpy()
    return run

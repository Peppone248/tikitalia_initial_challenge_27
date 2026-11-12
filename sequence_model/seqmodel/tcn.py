"""Bidirectional (non-causal) temporal convolutional network over per-frame features: outputs
logits for H/A/D and a logit for "possession change within +-k frames". Training side only; see
config.yaml `tcn` for every constant. Each period is its own sequence: no window or forward pass
crosses a period boundary. Features are standardised like FrameModel (odd columns scale only), so
a team swap is an exact negation of the odd input channels."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .features import ODD, frame_features
from .frame_model import fit_standardiser
from .match import CFG, controlled_labels

_C = CFG["tcn"]


def change_targets(label, period, k=None):
    """1 within +-k frames of a direct H<->A change (D and X in between do not count), else 0.
    The frame of the change is the first frame of the new team; changes never span periods."""
    k = _C["change_k"] if k is None else k
    label, period = np.asarray(label), np.asarray(period)
    out = np.zeros(len(label), dtype=np.float32)
    flip = np.flatnonzero((label[1:] <= 1) & (label[:-1] <= 1) & (label[1:] != label[:-1]) &
                          (period[1:] == period[:-1])) + 1
    for t in flip:
        out[max(0, t - k):t + k + 1] = 1.0
    return out


class _Block(nn.Module):
    def __init__(self, ch, kernel, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation,
                              padding=dilation * (kernel - 1) // 2)
        self.pw = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.pw(torch.relu(self.conv(x)))


class TCN(nn.Module):
    def __init__(self, n_in, channels=32, levels=6, kernel=5):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd for symmetric 'same' padding")
        self.hp = dict(n_in=n_in, channels=channels, levels=levels, kernel=kernel)
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList(_Block(channels, kernel, 2 ** i) for i in range(levels))
        self.head3 = nn.Conv1d(channels, 3, 1)
        self.headc = nn.Conv1d(channels, 1, 1)
        self.mu = self.sd = None            # feature standardisation, set by train_tcn

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head3(h), self.headc(h)


def _swap_x(x, odd_mask):
    s = x.clone()
    s[:, torch.as_tensor(np.asarray(odd_mask)), :] *= -1
    return s


def symmetric_forward(model, x, odd_mask):
    """Mean of f(x) and swap(f(swap(x))): probabilities (B,3,T) and (B,1,T), exactly equivariant."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            l3, lc = model(x)
            s3, sc = model(_swap_x(x, odd_mask))
            p3 = 0.5 * (torch.softmax(l3, 1) + torch.softmax(s3, 1)[:, [1, 0, 2]])
            pc = 0.5 * (torch.sigmoid(lc) + torch.sigmoid(sc))
    finally:
        model.train(was)
    return p3, pc


def _loss(model, x, y, c, w3, pos_weight, cw):
    """Weighted CE on H/A/D + cw * BCE on the change head; X frames (y == 3) are masked out."""
    l3, lc = model(x)
    valid = (y <= 2).float()
    yy = y.clamp(max=2)
    wt = w3[yy] * valid
    ce = (Fn.cross_entropy(l3, yy, reduction="none") * wt).sum() / wt.sum().clamp(min=1e-9)
    bce = Fn.binary_cross_entropy_with_logits(lc[:, 0], c, reduction="none")
    bce = bce * (1 + (pos_weight - 1) * c)
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    return ce + cw * bce


def _period_slices(period):
    cut = np.flatnonzero(np.diff(period)) + 1
    st = np.concatenate([[0], cut])
    return list(zip(st.tolist(), np.concatenate([cut, [len(period)]]).tolist()))


def train_tcn(matches, cfg, seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cfg["threads"])
    rng = np.random.default_rng(seed)
    data = []
    for m in matches:
        lab = controlled_labels(m.label)
        data.append((frame_features(m), lab, change_targets(lab, m.period, cfg["change_k"]),
                     _period_slices(m.period)))
    mu, sd = fit_standardiser(np.vstack([d[0] for d in data]))
    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        raise ValueError("train_tcn: non-finite features")
    prep = [((d[0] - mu) / sd).astype(np.float32) for d in data]
    # the last training match is held out from the gradient (early stopping); with one match it is reused
    tr_ids = list(range(len(data) - 1)) or [0]
    val_id = len(data) - 1
    lab_all = np.concatenate([data[i][1] for i in tr_ids])
    cnt = np.bincount(lab_all[lab_all <= 2], minlength=3).astype(np.float64) + 1.0
    w3 = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32)
    c_all = np.concatenate([data[i][2][data[i][1] <= 2] for i in tr_ids])
    pos_weight = float((len(c_all) - c_all.sum()) / max(c_all.sum(), 1.0))
    # windows: (match, start, end) inside a single period
    periods = [(i, a, b) for i in tr_ids for a, b in data[i][3]]
    win = min(cfg["window"], min(b - a for _, a, b in periods))
    sizes = np.array([b - a for _, a, b in periods], dtype=np.float64)

    def batch(n):
        xs, ys, cs = [], [], []
        for j in rng.choice(len(periods), size=n, p=sizes / sizes.sum()):
            i, a, b = periods[j]
            s = int(rng.integers(a, b - win + 1))
            xs.append(prep[i][s:s + win].T)
            ys.append(data[i][1][s:s + win])
            cs.append(data[i][2][s:s + win])
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.int64)),
                torch.from_numpy(np.stack(cs)))

    model = TCN(prep[0].shape[1], cfg["channels"], cfg["levels"], cfg["kernel"])
    model.mu, model.sd = mu, sd
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for a, b in data[val_id][3]:
                x = torch.from_numpy(prep[val_id][a:b].T[None].copy())
                y = torch.from_numpy(data[val_id][1][a:b][None].astype(np.int64))
                c = torch.from_numpy(data[val_id][2][a:b][None])
                tot += float(_loss(model, x, y, c, w3, pos_weight, cfg["change_weight"])) * (b - a)
                n += b - a
        model.train()
        return tot / n

    odd = torch.as_tensor(ODD)
    best, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    steps = max(1, cfg["windows_per_epoch"] // cfg["batch"])
    model.train()
    for _ in range(cfg["max_epochs"]):
        for _ in range(steps):
            x, y, c = batch(cfg["batch"])
            if rng.random() < cfg["swap_p"]:
                x = x.clone()
                x[:, odd, :] *= -1
                y = torch.where(y == 0, 1, torch.where(y == 1, 0, y))
            opt.zero_grad()
            _loss(model, x, y, c, w3, pos_weight, cfg["change_weight"]).backward()
            opt.step()
        v = val_loss()
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval()


def export_weights(model, path):
    """npz: inp.w/.b (1x1 conv, (C,F,1)); l{i}.conv.w/.b ((C,C,K), dilation 2**i, zero 'same'
    padding) and l{i}.pw.w/.b ((C,C,1)), block i: h = h + pw(relu(conv(h))); head3.w/.b ((3,C,1)),
    headc.w/.b ((1,C,1)); plus scalars channels, levels, kernel, n_in and arrays mu, sd, odd."""
    out = {}
    for k, v in model.state_dict().items():
        name = k.replace("blocks.", "l").replace(".weight", ".w").replace(".bias", ".b")
        out[name] = v.detach().cpu().numpy()
    out.update({k: np.array(v) for k, v in model.hp.items()})
    out["mu"], out["sd"], out["odd"] = np.asarray(model.mu), np.asarray(model.sd), ODD
    np.savez(path, **out)


class TCNPredictor:
    """Predictor protocol (fit(list[Match]) -> self, predict(Match) -> (p3, p_change))."""
    decode_output = True

    def __init__(self, seed, cfg=None):
        self.seed, self.cfg = seed, _C if cfg is None else cfg

    def fit(self, train):
        self.model = train_tcn(train, self.cfg, self.seed)
        return self

    def predict(self, m):
        X = ((frame_features(m) - self.model.mu) / self.model.sd).astype(np.float32)
        p3, pc = np.zeros((len(X), 3)), np.zeros(len(X))
        for a, b in _period_slices(m.period):
            x = torch.from_numpy(X[a:b].T[None].copy())
            q3, qc = symmetric_forward(self.model, x, ODD)
            p3[a:b], pc[a:b] = q3[0].numpy().T, qc[0, 0].numpy()
        if not np.isfinite(p3).all():
            raise ValueError("TCNPredictor.predict: non-finite probabilities")
        return p3, pc

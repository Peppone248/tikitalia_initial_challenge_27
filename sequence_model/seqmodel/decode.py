"""3-state HMM decoder (H, A, D) with minimum durations and possession changes driven by P(change).
Each state is a chain of sub-states (Bakis): non-last sub-states advance with probability 1, which
enforces the minimum duration; only the last sub-state self-loops and exits (to the first sub-state
of another chain). Viterbi in log space, run separately for each period."""
import numpy as np

from .match import CFG

_C = CFG["decode"]
_FPS = CFG["fps"]
_MIN = np.array([_C["min_frames"][k] for k in "HAD"])
_START = np.r_[0, np.cumsum(_MIN)[:-1]]           # first sub-state of each chain
_LAST = _START + _MIN - 1                         # last sub-state of each chain
_CHAIN = np.repeat(np.arange(3), _MIN)            # chain of each expanded state
_NS = int(_MIN.sum())
_KEYS = ("HA", "HD", "AH", "AD", "DH", "DA")


def default_trans():
    ch = 1.0 / (_C["change_every_s"] * _FPS)
    st = 1.0 / (_C["stop_every_s"] * _FPS)
    rs = 0.5 / (_C["stop_extra_s"] * _FPS)
    return {"HA": ch, "HD": st, "AH": ch, "AD": st, "DH": rs, "DA": rs}


def _viterbi(logem, ha, ah, tr):
    """logem (T,3); ha, ah (T,) exit rates H->A and A->H used for the step into each frame."""
    T = len(logem)
    em = logem[:, _CHAIN]
    lo = 1e-12
    hd, ad, dh, da = tr["HD"], tr["AD"], tr["DH"], tr["DA"]
    ha = np.clip(ha, lo, 1 - hd - 1e-9)
    ah = np.clip(ah, lo, 1 - ad - 1e-9)
    l_ha, l_ah = np.log(ha), np.log(ah)
    l_hs, l_as = np.log1p(-(ha + hd)), np.log1p(-(ah + ad))
    l_hd, l_ad, l_dh, l_da = np.log(hd), np.log(ad), np.log(dh), np.log(da)
    l_ds = np.log1p(-(dh + da))
    h0, a0, d0 = _START
    hl, al, dl = _LAST
    idx = np.arange(_NS)
    delta = np.full(_NS, -np.inf)
    delta[_LAST] = em[0, _LAST] - np.log(3)
    bp = np.empty((T, _NS), dtype=np.int16)
    bp[0] = idx
    new = np.empty(_NS)
    cur = np.empty(_NS, dtype=np.int16)
    for t in range(1, T):
        new[1:] = delta[:-1]
        cur[1:] = idx[:-1]
        # last sub-states: advance from previous sub-state or stay
        for c, (last, ls) in enumerate(((hl, l_hs[t]), (al, l_as[t]), (dl, l_ds))):
            stay = delta[last] + ls
            if stay > new[last]:
                new[last] = stay
                cur[last] = last
        # first sub-states: entered from the last sub-state of another chain
        for first, (s1, c1, s2, c2) in (
            (h0, (al, l_ah[t], dl, l_dh)),
            (a0, (hl, l_ha[t], dl, l_da)),
            (d0, (hl, l_hd, al, l_ad)),
        ):
            v1, v2 = delta[s1] + c1, delta[s2] + c2
            if v1 >= v2:
                new[first], cur[first] = v1, s1
            else:
                new[first], cur[first] = v2, s2
        delta = new + em[t]
        bp[t] = cur
    s = int(np.argmax(delta))
    out = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = _CHAIN[s]
        s = bp[t, s]
    return out


def decode(p, p_change, period, w=1.0, beta=0.0, trans=None, prior=None):
    """p (T,3) probabilities of H, A, D; p_change (T,) or None; period (T,). Returns (T,) int8 in 0..2."""
    p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=1 / 3)
    T = len(p)
    tr = default_trans() if trans is None else trans
    pr = np.full(3, 1 / 3) if prior is None else np.asarray(prior, dtype=np.float64)
    logem = w * (np.log(np.clip(p, 0, None) + _C["eps"]) - np.log(pr))
    if p_change is None or beta == 0:
        mult = np.ones(T)
    else:
        mult = 1 + beta * np.clip(np.nan_to_num(np.asarray(p_change, dtype=np.float64)), 0, None)
    period = np.asarray(period)
    out = np.full(T, 2, dtype=np.int8)
    for v in np.unique(period):
        m = period == v
        out[m] = _viterbi(logem[m], tr["HA"] * mult[m], tr["AH"] * mult[m], tr)
    return out


def count_transitions(labels, periods):
    """Per-frame exit rates of H->A, H->D, A->H, A->D, D->H, D->A with pseudo-count 1:
    (n_ab + 1) / (frames in a beyond its minimum + 1). X frames split runs; periods never mix."""
    n = {k: 0 for k in _KEYS}
    beyond = np.zeros(3)
    for lab, per in zip(labels, periods):
        lab, per = np.asarray(lab), np.asarray(per)
        if len(lab) == 0:
            continue
        b = np.flatnonzero((np.diff(lab) != 0) | (np.diff(per) != 0)) + 1
        s = np.r_[0, b]
        e = np.r_[b, len(lab)]
        cls = lab[s]
        for i in range(len(s)):
            c = int(cls[i])
            if c > 2:
                continue
            beyond[c] += max(0, int(e[i] - s[i]) - _MIN[c])
            if i + 1 < len(s) and per[s[i + 1]] == per[s[i]] and int(cls[i + 1]) <= 2:
                n["HAD"[c] + "HAD"[int(cls[i + 1])]] += 1
    return {k: (n[k] + 1) / (beyond["HAD".index(k[0])] + 1) for k in _KEYS}

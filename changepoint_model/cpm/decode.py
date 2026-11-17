"""Three-state HMM (H, A, D) with minimum durations and *time-varying* transition hazards.

A usual HMM keeps one constant rate per transition. Here the rate at frame t comes from the
boundary classifiers: where they see a change of possession the hazard rises and a cut becomes
cheap; where they see none the state is held. That is how a possession of one or two seconds can be
cut out of a long one, which a constant-rate HMM never does (raising a constant rate everywhere only
adds wrong cuts).

Emissions: the frame posteriors of the classifiers divided by the class priors (the hybrid
HMM / classifier convention), to the power w.

States: a chain of `min_frames` sub-states per label; a chain can be left only from its last
sub-state. Decoding: Viterbi in the log domain (numba when installed, plain Python otherwise,
same arithmetic).
"""

import numpy as np

from .match import H, A, D

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


MIN_FRAMES = {H: 20, A: 20, D: 50}
EPS = 1e-6


def hazards_from_boundaries(p_change, p_stop, p_restart, k=12, gamma=1.0, h_floor=1e-5,
                            h_ceil=0.5):
    """Per-frame hazards of each transition, from the 'within +-k frames' probabilities."""
    width = 2 * k + 1.0
    hc = np.clip(gamma * p_change / width, h_floor, h_ceil)
    hs = np.clip(gamma * p_stop / width, h_floor, h_ceil)
    hr = np.clip(gamma * p_restart / width, h_floor, h_ceil)
    return hc, hs, hr


def constant_hazards(T, change_every_s=20.0, stop_every_s=30.0, dead_extra_s=12.0, fps=25):
    hc = np.full(T, 1.0 / (change_every_s * fps))
    hs = np.full(T, 1.0 / (stop_every_s * fps))
    hr = np.full(T, 1.0 / (dead_extra_s * fps))
    return hc, hs, hr


def emissions(p_home, p_dead, prior, w=1.0, eps=EPS):
    """log emission (T, 3) for H, A, D from P(home | alive) and P(dead)."""
    pH = (1 - p_dead) * p_home
    pA = (1 - p_dead) * (1 - p_home)
    pD = p_dead
    post = np.stack([pH, pA, pD], axis=1)
    post = np.clip(post, eps, 1.0)
    prior = np.clip(np.asarray(prior, dtype=float), 1e-4, 1.0)
    return w * (np.log(post) - np.log(prior)[None, :])


def _layout(min_frames):
    nH, nA, nD = min_frames[H], min_frames[A], min_frames[D]
    return nH, nA, nD


@njit(cache=True)
def _viterbi(logE, log_hc, log_hs, log_hr, log_keep_HA, log_keep_D, nH, nA, nD):
    T = logE.shape[0]
    S = nH + nA + nD
    NEG = -1e18
    delta = np.full(S, NEG)
    back = np.zeros((T, S), dtype=np.int16)
    # start: first sub-state of any chain
    delta[0] = logE[0, 0]
    delta[nH] = logE[0, 1]
    delta[nH + nA] = logE[0, 2]
    new = np.full(S, NEG)
    lastH, lastA, lastD = nH - 1, nH + nA - 1, S - 1
    for t in range(1, T):
        for s in range(S):
            new[s] = NEG
        # inside the chains: forced advance
        for i in range(1, nH):
            new[i] = delta[i - 1] + logE[t, 0]
            back[t, i] = i - 1
        for i in range(1, nA):
            new[nH + i] = delta[nH + i - 1] + logE[t, 1]
            back[t, nH + i] = nH + i - 1
        for i in range(1, nD):
            new[nH + nA + i] = delta[nH + nA + i - 1] + logE[t, 2]
            back[t, nH + nA + i] = nH + nA + i - 1
        # staying in the last sub-state
        vH = delta[lastH] + log_keep_HA[t] + logE[t, 0]
        vA = delta[lastA] + log_keep_HA[t] + logE[t, 1]
        vD = delta[lastD] + log_keep_D[t] + logE[t, 2]
        if vH > new[lastH]:
            new[lastH] = vH
            back[t, lastH] = lastH
        if vA > new[lastA]:
            new[lastA] = vA
            back[t, lastA] = lastA
        if vD > new[lastD]:
            new[lastD] = vD
            back[t, lastD] = lastD
        # entering a chain (first sub-state)
        # -> H from A (change) or from D (restart)
        cand = delta[lastA] + log_hc[t]
        bp = lastA
        v2 = delta[lastD] + log_hr[t]
        if v2 > cand:
            cand = v2
            bp = lastD
        new[0] = cand + logE[t, 0]
        back[t, 0] = bp
        # -> A from H (change) or from D (restart)
        cand = delta[lastH] + log_hc[t]
        bp = lastH
        v2 = delta[lastD] + log_hr[t]
        if v2 > cand:
            cand = v2
            bp = lastD
        new[nH] = cand + logE[t, 1]
        back[t, nH] = bp
        # -> D from H or A (stop)
        cand = delta[lastH] + log_hs[t]
        bp = lastH
        v2 = delta[lastA] + log_hs[t]
        if v2 > cand:
            cand = v2
            bp = lastA
        new[nH + nA] = cand + logE[t, 2]
        back[t, nH + nA] = bp
        for s in range(S):
            delta[s] = new[s]
    # backtrack
    best = 0
    for s in range(1, S):
        if delta[s] > delta[best]:
            best = s
    path = np.zeros(T, dtype=np.int16)
    s = best
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def viterbi(logE, hc, hs, hr, min_frames=None):
    """logE (T, 3); hazards (T,) each: P(H->A or A->H), P(alive->D), P(D->alive) at frame t
    (the new state starts at t). Returns labels (T,) in {H, A, D}."""
    min_frames = min_frames or MIN_FRAMES
    nH, nA, nD = _layout(min_frames)
    hc = np.clip(hc, 1e-12, 0.5)
    hs = np.clip(hs, 1e-12, 0.5)
    hr = np.clip(hr, 1e-12, 0.98)
    # from H / A the exits are change + stop; from D the exit is restart to either team
    keep_HA = np.clip(1.0 - hc - hs, 1e-12, 1.0)
    keep_D = np.clip(1.0 - hr, 1e-12, 1.0)
    path = _viterbi(np.ascontiguousarray(logE, dtype=np.float64),
                    np.log(hc), np.log(hs), np.log(hr / 2.0), np.log(keep_HA), np.log(keep_D),
                    nH, nA, nD)
    labels = np.where(path < nH, H, np.where(path < nH + nA, A, D)).astype(np.int8)
    return labels


def decode_match(logE, hc, hs, hr, period, min_frames=None, fps=25):
    """Decode each period separately; X outside the periods."""
    from .match import X
    T = logE.shape[0]
    out = np.full(T, X, dtype=np.int8)
    for p in np.unique(period):
        if p <= 0:
            continue
        idx = np.flatnonzero(period == p)
        s, e = idx[0], idx[-1] + 1
        mf = dict(min_frames or MIN_FRAMES)
        if fps != 25:
            mf = {k: int(round(v * fps / 25.0)) for k, v in mf.items()}
        out[s:e] = viterbi(logE[s:e], hc[s:e], hs[s:e], hr[s:e], mf)
    return out

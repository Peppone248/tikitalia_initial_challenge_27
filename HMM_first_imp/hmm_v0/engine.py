"""The MOTORE: one label per frame from the NN probabilities and the weighted rule votes.

Observation model: every hidden state m emits one of the 4 observation classes c (H, A, D, X)
with probability omega[m, c] (hmm.yaml emission). One matrix per source, because X does not
mean the same thing for the two:
    rules  X = X-NOBALL, "the ball is not tracked": a fact, uninformative between an unseen
           possession (XH, XA) and an unseen stoppage (D) -> same weight in the three rows
    NN     X = learnt from the pseudo-labels, "ball not seen AND no sign of a stoppage or of a
           team" (pseudo_labels.py) -> points to XH / XA, not to D
Hybrid HMM/ANN combination:

    log b_m(f) = alpha * log sum_c omega_nn[m, c]    * P_nn(c | f) / prior(c)   scaled likelihood
               + beta  * log sum_c omega_rules[m, c] * exp(E_rules(c, f))       w = q * s, log-odds

With omega[XH] = {X: .8, H: .2}, team evidence (S-PHASE, P-SHAPE) still tells WHO has the ball
while nobody sees it.
    path       = Viterbi of the Bakis HMM, once per half, on the graded frames of that half
                 (the memory never crosses half-time, MOTORE.md 4.3)
    label      = output_label[path]  (XH -> H, XA -> A): never X on a graded frame (A1);
                 X on every non-graded frame (never scored)

The engine knows nothing about football: it reads probabilities, evidence and the frames where a
restart was detected. Changes of possession are dated by the evidence itself (a control episode
votes from its first frame), not by when the engine became sure (A2).
"""

import numpy as np

from . import exact
from .hmm import BakisHMM
from .votes import CLASSES


def omega_matrix(cfg_hmm, source="omega_rules"):
    states = cfg_hmm["states"]
    om = np.zeros((len(states), len(CLASSES)))
    for j, m in enumerate(states):
        row = cfg_hmm["emission"][source][m]
        for c, v in row.items():
            om[j, CLASSES.index(c)] = float(v)
        om[j] /= om[j].sum()
    return om


def _log_mix(log_terms, om):
    """log sum_c om[m, c] * exp(log_terms[:, c]) for every state m -> (n, M), stable."""
    mx = log_terms.max(axis=1, keepdims=True)
    return np.log(np.exp(log_terms - mx) @ om.T + 1e-300) + mx


def emissions(P_nn, prior, E, cfg_hmm):
    em = cfg_hmm["emission"]
    log_c = np.log(np.maximum(P_nn, 1e-6)) - np.log(np.maximum(prior, 1e-3))
    log_b = em["alpha_nn"] * _log_mix(log_c, omega_matrix(cfg_hmm, "omega_nn")) + \
        em["beta_rules"] * _log_mix(E, omega_matrix(cfg_hmm, "omega_rules"))
    bits = cfg_hmm.get("log_grid_bits")
    # on a 2^-bits grid the Viterbi sums are exact: ulp differences of log/exp across machines
    # cannot change a comparison (exact.py)
    return exact.quantize(log_b, bits) if bits else log_b


def resolve_auto_times(cfg_hmm, match):
    """Replace 'auto_visible' / 'auto_missing' in mean_time_s by the mean length (s) of the
    stretches where the ball is / is not tracked, on graded frames (no labels involved)."""
    from tikitalia.preprocessing import runs
    g = match.graded
    auto = {}
    for key, mask in (("auto_visible", match.ball_visible & g), ("auto_missing", ~match.ball_visible & g)):
        lengths = [e - s for s, e in runs(mask)]
        auto[key] = float(np.mean(lengths)) / match.fps if lengths else 10.0
        # floor: on a clean track the only gaps are 1-frame jumps removed by V_JUMP, and a
        # mean time below a second would make the exits of a state sum to more than 1
        auto[key] = max(auto[key], 1.0)
    cfg = dict(cfg_hmm)
    cfg["mean_time_s"] = {a: {b: (auto[t] if isinstance(t, str) else t) for b, t in row.items()}
                          for a, row in cfg_hmm["mean_time_s"].items()}
    return cfg, {k: round(v, 2) for k, v in auto.items()}


def _window_mask(n, frames, pre, post):
    m = np.zeros(n, dtype=bool)
    for f in frames:
        m[max(0, f - pre):min(n, f + post + 1)] = True
    return m


def regimes(n, restart_frames, change_pts, cfg_hmm, fps):
    """(T,) regime index of hmm.BakisHMM: 4*restart + 2*open_H + open_A."""
    w = int(round(cfg_hmm["restart"]["window_s"] * fps))
    boost = _window_mask(n, restart_frames, w, w)
    tc = cfg_hmm.get("team_change", {}) or {}
    if tc.get("gate", False) and change_pts is not None:
        pre = int(round(tc["window_pre_s"] * fps))
        post = int(round(tc["window_post_s"] * fps))
        open_h = _window_mask(n, change_pts["H"], pre, post)
        open_a = _window_mask(n, change_pts["A"], pre, post)
    else:
        open_h = open_a = np.ones(n, dtype=bool)
    return 4 * boost + 2 * open_h + open_a, boost, open_h, open_a


def decide(match, P_nn, prior, E, restart_frames, cfg_hmm, log=print, change_pts=None):
    fps = match.fps
    cfg_hmm, auto = resolve_auto_times(cfg_hmm, match)
    hmm = BakisHMM(cfg_hmm, fps)
    log_b = emissions(P_nn, prior, E, cfg_hmm)
    regime, boost, open_h, open_a = regimes(match.n, restart_frames, change_pts, cfg_hmm, fps)

    labels = np.full(match.n, "X", dtype="<U1")
    macro = np.full(match.n, -1, dtype=int)
    out_map = np.array([cfg_hmm["output_label"][m] for m in hmm.macros])
    g = match.graded
    info = {"n_substates": dict(zip(hmm.macros, hmm.n_sub)), "S": hmm.S, "halves": {},
            "auto_mean_time_s": auto,
            "team_gate": bool(hmm.team_gate),
            "gate_open_share_%": {"H": round(100.0 * float(open_h[g].mean()), 1),
                                  "A": round(100.0 * float(open_a[g].mean()), 1)},
            "restart_window_share_%": round(100.0 * float(boost[g].mean()), 2)}
    for h in (1, 2):
        idx = np.flatnonzero(match.half == h)
        if idx.size == 0:
            continue
        if np.any(np.diff(idx) != 1):
            raise ValueError("half {} is not one contiguous block".format(h))
        path, score = hmm.viterbi(log_b[idx], regime=regime[idx])
        macro[idx] = path
        labels[idx] = out_map[path]
        info["halves"][h] = {"frames": int(idx.size), "log_score": round(score, 1)}
        log("  viterbi half {}: {} frames".format(h, idx.size))
    info["macro_share_%"] = {m: round(100.0 * float((macro[match.graded] == i).mean()), 2)
                             for i, m in enumerate(hmm.macros)}
    return labels, macro, hmm, info

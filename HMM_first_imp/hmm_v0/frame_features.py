"""Per-frame input vector of the NN, built from the features (tracking + Smart Tagging).

Only measurements: no rule output enters the NN input (the rules are its TARGETS, pseudo_labels.py;
feeding them in too would teach the NN to copy them). Every value is scaled to O(1), NaN -> 0
with an explicit "missing" flag where the absence itself is informative.

Team columns come in (home, away) pairs and every team quantity is relative to that team's attack
direction, so swapping the pairs is the same frame seen with the teams exchanged: SWAP gives
that permutation (data augmentation and H/A symmetry, nn.py).

The NN sees a window of 2k+1 frames (t-k .. t+k, clamped at the edges): `window()`.
"""

import numpy as np

from tikitalia.preprocessing import HOME, AWAY


def _c(x, lo, hi):
    return np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), lo, hi)


def build(match, feats, pitch, p, kickoff_mask=None, set_piece_mask=None, st_team=None):
    """(X (n, F) float32, names, swap index array)."""
    n = match.n
    cols, names, pairs = [], [], []

    def add(name, values):
        cols.append(np.asarray(values, dtype=np.float32))
        names.append(name)
        return len(cols) - 1

    def add_pair(name, values_h, values_a):
        i = add(name + "_H", values_h)
        j = add(name + "_A", values_a)
        pairs.append((i, j))

    vis = match.ball_visible.astype(float)
    speed = match.ball_speed
    add("ball_visible", vis)
    add("ball_missing_log", np.log1p(_c(feats.ball_missing_for, 0, 600)))
    add("ball_speed", _c(speed, 0, 40) / 10.0)
    add("ball_fast", (np.nan_to_num(speed) > p.V_BALL_FAST).astype(float))   # > human max
    add("ball_still_log", np.log1p(_c(match.ball_still_for, 0, 120)))
    add("ball_static", (match.ball_still_for >= p.T_STILL_RESTART).astype(float))
    add("ball_acc", _c(feats.ball_acc, 0, 200) / 20.0)
    d_out = pitch.dist_outside(match.ball_x, match.ball_y)
    add("ball_out_dist", _c(d_out, -15, 10) / 5.0)
    add("ball_outside", (np.nan_to_num(d_out, nan=-1) > 0).astype(float))
    cx = np.abs(np.nan_to_num(match.ball_x, nan=pitch.length / 2) - pitch.length / 2) / (pitch.length / 2)
    add("ball_x_abs", cx * vis)
    add("slowdown_all", _c(feats.slowdown[:, 2], 0, 4))
    add("slowdown_missing", np.isnan(feats.slowdown[:, 2]).astype(float))
    add("speed_med_all", _c(feats.speed_med[:, 2], 0, 10) / 3.0)
    gap = feats.nearest_dist[:, HOME] - feats.nearest_dist[:, AWAY]
    add("nearest_gap_abs", _c(np.abs(gap), 0, 20) / 5.0)
    add("kickoff_formation", np.zeros(n) if kickoff_mask is None else kickoff_mask)
    add("st_set_piece", np.zeros(n) if set_piece_mask is None else set_piece_mask)

    near = feats.nearest_team
    add_pair("is_nearest", (near == HOME).astype(float), (near == AWAY).astype(float))
    add_pair("nearest_log", *[np.log1p(_c(feats.nearest_dist[:, t], 0, 60)) for t in (HOME, AWAY)])
    add_pair("nearest_missing", *[np.isnan(feats.nearest_dist[:, t]).astype(float)
                                  for t in (HOME, AWAY)])
    add_pair("rel_speed", *[_c(feats.rel_speed[:, t], 0, 30) / 5.0 for t in (HOME, AWAY)])
    add_pair("align_cos", *[_c(feats.align_cos[:, t], -1, 1) for t in (HOME, AWAY)])
    add_pair("closing", *[_c(feats.closing_speed[:, t], -15, 15) / 5.0 for t in (HOME, AWAY)])
    add_pair("chase", *[_c(feats.chase_speed[:, t], -15, 15) / 5.0 for t in (HOME, AWAY)])
    add_pair("pressure_log", *[np.log1p(_c(feats.pressure_dist[:, t], 0, 60)) for t in (HOME, AWAY)])
    add_pair("n_within3", *[feats.n_within(t, 3.0) / 3.0 for t in (HOME, AWAY)])
    add_pair("n_within8", *[feats.n_within(t, 8.0) / 5.0 for t in (HOME, AWAY)])
    add_pair("centroid_fwd", *[_c(feats.centroid_fwd[:, t], -50, 50) / 20.0 for t in (HOME, AWAY)])
    add_pair("spread", *[_c(feats.spread[:, t], 0, 80) / 30.0 for t in (HOME, AWAY)])
    add_pair("n_opp_half", *[feats.n_opp_half[:, t] / 11.0 for t in (HOME, AWAY)])
    add_pair("n_opp_box", *[feats.n_opp_box[:, t] / 5.0 for t in (HOME, AWAY)])
    add_pair("n_own_box", *[feats.n_own_box[:, t] / 5.0 for t in (HOME, AWAY)])
    add_pair("v_forward", *[_c(feats.v_forward[:, t], -8, 8) / 3.0 for t in (HOME, AWAY)])
    add_pair("slowdown", *[_c(feats.slowdown[:, t], 0, 4) for t in (HOME, AWAY)])
    st_h = np.zeros(n) if st_team is None else (st_team == HOME).astype(float)
    st_a = np.zeros(n) if st_team is None else (st_team == AWAY).astype(float)
    add_pair("st_phase_team", st_h, st_a)

    X = np.stack(cols, axis=1).astype(np.float32)
    swap = np.arange(X.shape[1])
    for i, j in pairs:
        swap[i], swap[j] = j, i
    return X, names, swap


def window(X, idx, k):
    """Rows idx of the (2k+1)-frame context matrix: [X[t-k], ..., X[t+k]], clamped."""
    idx = np.asarray(idx)
    n = X.shape[0]
    parts = [X[np.clip(idx + o, 0, n - 1)] for o in range(-k, k + 1)]
    return np.concatenate(parts, axis=1)


def window_swap(swap, k):
    """Column permutation of the windowed matrix that swaps home and away."""
    F = len(swap)
    return np.concatenate([swap + F * j for j in range(2 * k + 1)])


def st_masks(match, codes_poss, codes_def):
    """(set_piece_mask, st_team) from Smart Tagging: team in possession per the phase codes
    (-1 = none). Inputs of the NN only."""
    n = match.n
    sp = np.zeros(n, dtype=np.float32)
    team = np.full(n, -1, dtype=np.int8)
    poss = set(c.upper() for c in codes_poss)
    dfn = set(c.upper() for c in codes_def)
    for t in match.smart_tagging:
        lo, hi = max(0, t["start_frame"]), min(n, t["end_frame"])
        if t["code"] == "SET PIECES":
            sp[lo:hi] = 1.0
        elif t["team"] in (HOME, AWAY):
            if t["code"] in poss:
                team[lo:hi] = t["team"]
            elif t["code"] in dfn:
                team[lo:hi] = 1 - t["team"]
    return sp, team

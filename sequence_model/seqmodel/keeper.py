"""Goalkeeper proxy shared by degrade, features and clean. The data carry no roles, so a team's
keeper is its deepest player in its own attacking frame (x multiplied by +1 when the team attacks
towards +x, -1 otherwise), counted only while he is inside his own penalty area. Constants:
config.yaml `keeper`."""
import numpy as np

from .match import CFG

KP = CFG["keeper"]


def team_signs(m):
    """(T,) sign of the home attacking frame per frame (+1 home attacks +x); away is the negative."""
    return np.array([1.0 if m.home_attacks_right[int(p)] else -1.0 for p in m.period])


def deepest(px, py, sg):
    """Deepest tracked player of one team per frame. px, py (T, P) of that team's columns, sg (T,)
    that team's sign. Returns (x, y, col): raw coordinates (NaN when nobody is tracked) and the
    column index into px (-1 when nobody is tracked)."""
    ok = np.isfinite(px) & np.isfinite(py)
    xa = np.where(ok, px * sg[:, None], np.inf)
    T = len(px)
    if px.shape[1] == 0:
        return np.full(T, np.nan), np.full(T, np.nan), np.full(T, -1)
    col = np.argmin(xa, axis=1)
    has = ok.any(axis=1)
    r = np.arange(T)
    x = np.where(has, px[r, col], np.nan)
    y = np.where(has, py[r, col], np.nan)
    return x, y, np.where(has, col, -1)


def in_own_box(x, y, sg, length):
    """True where the point is inside the penalty area of the team with sign sg (its own goal at
    x_att = -length/2). NaN -> False."""
    with np.errstate(invalid="ignore"):
        xa = x * sg
        return (xa <= -length / 2 + KP["box_depth"]) & (np.abs(y) <= KP["box_half_width"])


def in_opp_box(x, y, sg, length):
    """True where the point is inside the penalty area the team with sign sg attacks."""
    return in_own_box(x, y, -sg, length)


def runs(mask):
    """[(start, end)) of the True runs of a boolean array."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))

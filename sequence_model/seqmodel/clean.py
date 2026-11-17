"""Tracking cleaning: obvious tracker errors fixed by a general rule, after the teammate's rule 1 of
REPORT_B section 10.2 (keeper identity), reimplemented without roles.

The provided tracker sometimes gives one keeper's track the other keeper's identity: the away keeper
slot then sits inside the HOME penalty area for minutes. Rule, per graded period and team T:
- T's keeper slot is the column most often T's deepest player (own attacking frame); it must be the
  deepest on >= min_keeper_share of its tracked frames, else no slot is trusted and nothing moves;
- the slot's track is cut into stretches at gaps and at jumps > jump_m between two frames (another
  person);
- a stretch that spends >= min_box_s in a row inside the OPPONENT's penalty area and never touches
  T's own half did not run there: it is the opponent's keeper. It is moved to a new column of the
  other team; inside it, points frozen (identical coordinates) for >= frozen_s are a ghost and are
  dropped instead. A keeper who goes up for a corner runs there from his own half inside one
  continuous stretch and is left alone (conservative: any doubt, no move).
No frame, slot or player identity of any match appears; constants in config.yaml `clean`."""
import dataclasses

import numpy as np

from .keeper import deepest, in_opp_box, runs, team_signs
from .match import CFG

CC = CFG["clean"]
FPS = CFG["fps"]


def _stretches(x, y, jump_m):
    """[(start, end)) of continuous track: cut at NaN and at a jump > jump_m between two frames."""
    ok = np.isfinite(x) & np.isfinite(y)
    cut = np.zeros(len(x), dtype=bool)
    with np.errstate(invalid="ignore"):
        cut[1:] = np.hypot(np.diff(x), np.diff(y)) > jump_m
    out = []
    for s, e in runs(ok):
        starts = [s] + [int(i) for i in np.flatnonzero(cut[s + 1:e]) + s + 1]
        out += list(zip(starts, starts[1:] + [e]))
    return out


def _frozen(x, y, min_frames):
    """True on frames inside runs of identical consecutive coordinates lasting >= min_frames."""
    same = np.zeros(len(x), dtype=bool)
    same[1:] = (np.diff(x) == 0) & (np.diff(y) == 0)
    out = np.zeros(len(x), dtype=bool)
    # a run of k "same as previous" flags covers k + 1 identical frames
    for s, e in runs(same):
        if e - s + 1 >= min_frames:
            out[s - 1:e] = True
    return out


def keeper_slot(px, py, sg, min_share):
    """Column (into px) of the team's keeper slot in this period, or None when not trusted."""
    _, _, col = deepest(px, py, sg)
    have = col >= 0
    if not have.any():
        return None
    cnt = np.bincount(col[have], minlength=px.shape[1])
    k = int(np.argmax(cnt))
    tracked = int((np.isfinite(px[:, k]) & np.isfinite(py[:, k])).sum())
    return k if tracked and cnt[k] / tracked >= min_share else None


def clean_keepers(m, cfg=None, periods=None):
    """Returns (cleaned Match copy, list of fixes). Each fix: dict(team, col, period, start, end,
    moved, dropped) with team = the slot's original team, rows [start, end) of the stretch, moved /
    dropped frame counts. Without fixes the arrays are returned unchanged (same shape)."""
    cfg = CC if cfg is None else cfg
    periods = m.graded_periods if periods is None else periods
    px, py = m.px.copy(), m.py.copy()
    sg_h = team_signs(m)
    T = len(m.period)
    new = {}                      # receiving team -> (x column, y column)
    fixes = []
    min_box = int(round(cfg["min_box_s"] * FPS))
    min_frozen = int(round(cfg["frozen_s"] * FPS))
    for per in periods:
        rows = np.flatnonzero(m.period == per)
        if len(rows) == 0:
            continue
        for tid in (0, 1):
            cols = np.flatnonzero(m.team == tid)
            sg = (sg_h if tid == 0 else -sg_h)[rows]
            k = keeper_slot(m.px[rows][:, cols].astype(float), m.py[rows][:, cols].astype(float), sg,
                            cfg["min_keeper_share"])
            if k is None:
                continue
            j = int(cols[k])
            x, y = m.px[rows, j].astype(float), m.py[rows, j].astype(float)
            for s, e in _stretches(x, y, cfg["jump_m"]):
                xs, ys, ss = x[s:e], y[s:e], sg[s:e]
                if (xs * ss < 0).any():                    # touches his own half: he ran there
                    continue
                box = in_opp_box(xs, ys, ss, m.length)
                if max((b - a for a, b in runs(box)), default=0) < min_box:
                    continue
                frozen = _frozen(xs, ys, min_frozen)
                other = 1 - tid
                if other not in new:
                    new[other] = (np.full(T, np.nan, dtype=px.dtype), np.full(T, np.nan, dtype=py.dtype))
                nx, ny = new[other]
                r = rows[s:e]
                nx[r[~frozen]], ny[r[~frozen]] = xs[~frozen], ys[~frozen]
                px[r, j], py[r, j] = np.nan, np.nan
                fixes.append(dict(team=tid, col=j, period=int(per), start=int(r[0]), end=int(r[-1]) + 1,
                                  moved=int((~frozen).sum()), dropped=int(frozen.sum())))
    if not fixes:
        return m, fixes
    order = sorted(new)
    px = np.column_stack([px] + [new[t][0] for t in order]).astype(m.px.dtype)
    py = np.column_stack([py] + [new[t][1] for t in order]).astype(m.py.dtype)
    team = np.concatenate([m.team, np.array(order, dtype=m.team.dtype)])
    return dataclasses.replace(m, px=px, py=py, team=team,
                               home_attacks_right=dict(m.home_attacks_right)), fixes


def clean_if_enabled(m):
    """clean_keepers when config.yaml clean.enabled, else m unchanged; the fixes are dropped."""
    return clean_keepers(m)[0] if CC["enabled"] else m

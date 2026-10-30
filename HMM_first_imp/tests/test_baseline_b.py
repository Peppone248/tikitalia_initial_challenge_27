"""Tests of approach B and of the broadcast alignment (synthetic data, no match data needed).

    python -m pytest HMM_first_imp/tests/test_baseline_b.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0 import baseline_b as BB  # noqa: E402
from hmm_v0 import broadcast as BC  # noqa: E402


def _scene(n=500):
    team = np.array([0, 0, 1, 1])
    px = np.tile(np.array([10.0, 20.0, 60.0, 70.0]), (n, 1))
    py = np.tile(np.array([30.0, 40.0, 30.0, 40.0]), (n, 1))
    bx = np.full(n, 10.5)
    by = np.full(n, 30.0)
    return bx, by, px, py, team


def test_min_possession_merges_flicker():
    lab = np.array(list("HHHHHHAAHHHHHH"))
    out = BB._min_possession(lab, np.ones(lab.size, bool), 3)
    assert "".join(out) == "HHHHHHHHHHHHHH"


def test_min_possession_keeps_long_change():
    lab = np.array(list("HHHHHAAAAAA"))
    out = BB._min_possession(lab, np.ones(lab.size, bool), 3)
    assert "".join(out) == "HHHHHAAAAAA"


def test_dead_mask_on_top_and_last_team_fill():
    bx, by, px, py, team = _scene()
    bx[200:300] = np.nan                      # ball unseen: no vote, keeps the last team
    dead = np.zeros(bx.size, bool)
    dead[400:450] = True
    g = np.ones(bx.size, bool)
    lab = BB.boosted_baseline(bx, by, px, py, team, g, dead, window=10, min_poss_s=0)
    assert (lab[400:450] == "D").all()
    assert (lab[200:300] == "H").all()


def test_spread_fill_names_the_open_team():
    bx, by, px, py, team = _scene()
    bx[:] = np.nan
    px[:, 2:] = [40.0, 90.0]                  # away players far apart -> away is in possession
    g = np.ones(bx.size, bool)
    lab = BB.boosted_baseline(bx, by, px, py, team, g, np.zeros(bx.size, bool), window=10,
                              unseen="spread")
    assert (lab == "A").all()


def test_slow_dead_needs_the_ball_unseen():
    n = 500
    px = np.cumsum(np.full((n, 4), 0.2), axis=0)    # 5 m/s
    px[200:350] = px[200]                           # everybody stops for 6 s
    py = np.zeros((n, 4))
    half = np.ones(n, np.int8)
    cfg = {"v_max_ms": 1.1, "smooth_frames": 1, "min_s": 3.0, "close_s": 1.0,
           "unseen_share": 0.5}
    seen = np.zeros(n)
    unseen = np.full(n, np.nan)
    assert not BB.slow_dead(seen, px, py, half, cfg).any()
    sd = BB.slow_dead(unseen, px, py, half, cfg)
    assert sd[220:340].all() and not sd[:190].any()


def test_alignment_recovers_offsets_and_switch():
    nT, team_t = 6000, np.array([0] * 4 + [1] * 4)
    t = np.arange(nT)
    T = {"px": (np.sin(t[:, None] / 50.0 + np.arange(8)) + 1) * 50,
         "py": (np.cos(t[:, None] / 70.0 + 2 * np.arange(8)) + 1) * 30,
         "bx": (np.sin(t / 37.0) + 1) * 50, "by": (np.cos(t / 23.0) + 1) * 30}
    # broadcast frames 0..1999 = tactical +300, frames 2000..3999 = tactical +1500
    b_idx = np.r_[np.arange(2000) + 300, np.arange(2000, 4000) + 1500]
    B = {k: T[k][b_idx].copy() for k in ("px", "py", "bx", "by")}
    B["team"] = team_t
    B["valid"] = np.ones(b_idx.size, bool)
    al = BC.align(B, T, team_t, {"window_s": 8, "every_s": 8, "margin_m": 0.3, "min_share": 0.2})
    assert sorted(o for _lo, _hi, o, _c in al["segments"]) == [300, 1500]
    assert abs(al["segments"][1][0] - 2000) <= 25
    assert al["live"].mean() > 0.95

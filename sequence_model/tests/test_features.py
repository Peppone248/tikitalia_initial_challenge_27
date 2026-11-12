import warnings

import numpy as np
import pytest

from seqmodel.features import FEATURES, ODD, frame_features, swap_teams

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def test_shapes(synthetic_match):
    m = synthetic_match(0, 1000)
    f = frame_features(m)
    assert f.shape == (1000, len(FEATURES)) and f.dtype == np.float32
    assert ODD.shape == (len(FEATURES),) and ODD.dtype == bool
    assert np.isfinite(f).all()


def test_swap_antisymmetry(synthetic_match):
    m = synthetic_match(1, 1200)
    m.ball_x[300:500] = np.nan
    m.ball_y[300:500] = np.nan
    f, g = frame_features(m), frame_features(swap_teams(m))
    assert np.allclose(g[:, ODD], -f[:, ODD], atol=1e-5)
    assert np.allclose(g[:, ~ODD], f[:, ~ODD], atol=1e-5)
    assert ODD.any() and (~ODD).any()


def test_swap_teams_semantics(synthetic_match):
    m = synthetic_match(0, 300)
    s = swap_teams(m)
    assert (s.team == 1 - m.team).all()
    assert s.home_attacks_right == {1: False, 2: True}
    assert m.home_attacks_right == {1: True, 2: False}
    swapped = np.where(m.label == 0, 1, np.where(m.label == 1, 0, m.label))
    assert (s.label == swapped).all()


def test_no_nan_with_missing_players(synthetic_match):
    m = synthetic_match(2, 1000)
    m.px[:, [0, 5, 7]] = np.nan
    m.py[:, [0, 5, 7]] = np.nan
    m.px[100:300, :11] = np.nan
    m.py[100:300, :11] = np.nan
    m.px[600:650, 11:] = np.nan
    m.py[600:650, 11:] = np.nan
    assert np.isfinite(frame_features(m)).all()


def test_no_nan_all_players_missing(synthetic_match):
    m = synthetic_match(2, 400)
    m.px[:] = np.nan
    m.py[:] = np.nan
    assert np.isfinite(frame_features(m)).all()


def test_long_ball_gap(synthetic_match):
    m = synthetic_match(3, 4000)     # gap inside period 1 (frames 0-1999)
    m.ball_x[100:1600] = np.nan
    m.ball_y[100:1600] = np.nan
    f = frame_features(m)
    seen, uns = f[:, FEATURES.index("ball_seen")], f[:, FEATURES.index("ball_unseen_s")]
    assert (seen[100:1600] == 0).all() and (seen[:100] == 1).all()
    assert (np.diff(uns[100:1600]) >= 0).all() and uns[1599] == 60
    assert uns[101] == pytest.approx(2 / 25) and uns[99] == 0


def test_ball_unseen_uses_last_position(synthetic_match):
    m = synthetic_match(3, 600)
    bx = m.ball_x.copy()
    m.ball_x[200:400] = np.nan
    m.ball_y[200:400] = np.nan
    f = frame_features(m)
    i = FEATURES.index("ball_x_att")
    assert np.allclose(f[200:300, i], f[199, i])
    assert np.allclose(f[300:400, i], 0.0)     # period 2: nothing seen yet -> centre spot, no memory
    assert np.allclose(f[300:400, FEATURES.index('ball_unseen_s')][0], 1 / 25)
    assert (f[200:400, FEATURES.index("ball_speed")] == 0).all()
    assert np.isclose(f[199, i], bx[199])


def test_attack_dir_used(synthetic_match):
    m = synthetic_match(4, 1000)
    i = FEATURES.index("ball_x_att")
    f = frame_features(m)
    m2 = synthetic_match(4, 1000)
    m2.home_attacks_right = {1: False, 2: False}
    g = frame_features(m2)
    p1 = m.period == 1
    assert np.allclose(g[p1, i], -f[p1, i])
    assert np.allclose(g[~p1, i], f[~p1, i])


def test_definitions_simple():
    from seqmodel.match import Match
    T = 50
    # home: x = -40,-30,-10 (attacks right); away: 40,30,20
    px = np.tile(np.array([-40, -30, -10, 40, 30, 20], np.float32), (T, 1))
    py = np.tile(np.array([-10, 0, 10, -10, 0, 10], np.float32), (T, 1))
    m = Match("t", px, py, np.array([0, 0, 0, 1, 1, 1], np.int8), np.full(T, 0.0),
              np.full(T, 0.0), np.ones(T, np.int8), np.zeros(T, np.int8), 105.0, 68.0, {1: True})
    f = dict(zip(FEATURES, frame_features(m)[10]))
    assert f["def_line"] == pytest.approx(0.0)          # H -30 minus A (-30)
    assert f["front_line"] == pytest.approx(-10 - (-20))
    assert f["centroid_x"] == pytest.approx(-80 / 3 - (-90 / 3))
    assert f["ball_line_dist"] == pytest.approx(34.0)
    assert f["mean_speed_all"] == 0 and f["slow_unseen"] == 0 and f["ball_seen"] == 1


def test_period_boundary_resets(synthetic_match):
    m = synthetic_match(5, 600)           # period 2 starts at frame 300
    m.px[:300] += 10.0                    # a jump at the boundary must not leak into speeds
    m.ball_x[:], m.ball_y[:] = np.nan, np.nan
    m.ball_x[290:300], m.ball_y[290:300] = 5.0, 5.0
    f = frame_features(m)
    i = FEATURES.index("ball_unseen_s")
    assert f[299, i] == 0 and f[300, i] == pytest.approx(1 / 25)
    assert f[299, FEATURES.index("ball_seen")] == 1 and f[300, FEATURES.index("ball_x_att")] == 0
    # first frames of period 2 equal the features of period 2 computed alone
    m2 = synthetic_match(5, 600)
    m2.ball_x[:], m2.ball_y[:] = np.nan, np.nan
    m2.ball_x[290:300], m2.ball_y[290:300] = 5.0, 5.0
    g = frame_features(m2)
    assert np.allclose(f[300:, FEATURES.index("mean_speed_all")], g[300:, FEATURES.index("mean_speed_all")])


def test_impossible_speeds_are_excluded(synthetic_match):
    m = synthetic_match(6, 400)
    m.px[100, 3] += 500.0                 # one-frame teleport of one player
    m.ball_x[:150] = np.where(np.arange(150) == 50, m.ball_x[50] + 300, m.ball_x[:150])
    f = frame_features(m)
    assert f[:, FEATURES.index("ball_speed")].max() <= 40
    assert f[:, FEATURES.index("mean_speed_all")].max() <= 12
    assert f[:, FEATURES.index("mean_speed")].max() <= 12

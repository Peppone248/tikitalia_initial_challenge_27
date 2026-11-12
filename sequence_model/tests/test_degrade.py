import numpy as np

from seqmodel.degrade import degrade
from seqmodel.match import CFG, D

DC = CFG["degrade"]


def test_degrade_is_deterministic(synthetic_match):
    m = synthetic_match()
    a, b, c = degrade(m, 1, DC), degrade(m, 1, DC), degrade(m, 2, DC)
    assert np.array_equal(a.ball_x, b.ball_x, equal_nan=True)
    assert np.array_equal(a.px, b.px, equal_nan=True)
    assert not np.array_equal(a.ball_x, c.ball_x, equal_nan=True)


def test_labels_untouched(synthetic_match):
    m = synthetic_match()
    before = m.label.copy(), m.ball_x.copy(), m.px.copy()
    d = degrade(m, 0, DC)
    assert np.array_equal(d.label, m.label)
    assert np.array_equal(m.label, before[0]) and np.array_equal(m.ball_x, before[1])
    assert np.array_equal(m.px, before[2])
    assert d.px is not m.px and d.label is not m.label


def test_dead_hidden_more_than_live(synthetic_match):
    m = synthetic_match(n_frames=6000)
    d = degrade(m, 0, DC)
    seen = np.isfinite(d.ball_x)
    dead = m.label == D
    assert seen[dead].mean() < seen[~dead].mean()


def test_live_seen_share(synthetic_match):
    for seed in range(3):
        m = synthetic_match(n_frames=6000)
        d = degrade(m, seed, DC)
        live = m.label != D
        share = np.isfinite(d.ball_x)[live].mean()
        assert 0.5 <= share <= 0.7
        assert abs(share - DC["live_seen_target"]) <= 0.02


def test_player_drop_fraction(synthetic_match):
    m = synthetic_match(n_frames=6000)
    d = degrade(m, 0, DC)
    frac = np.isnan(d.px).mean(axis=0)
    assert np.all(np.abs(frac - DC["player_drop_frac"]) < 0.02)
    assert np.array_equal(np.isnan(d.px), np.isnan(d.py))

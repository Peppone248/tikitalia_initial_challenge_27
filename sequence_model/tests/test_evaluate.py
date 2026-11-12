import dataclasses

import numpy as np
import pytest

from seqmodel.evaluate import Baseline50, lomo
from seqmodel.features import ODD, frame_features
from seqmodel.frame_model import FrameModel
from seqmodel.match import CFG, Match, controlled_labels


def _three(synthetic_match, n=1500):
    return [dataclasses.replace(synthetic_match(s, n), name=f"m{s}") for s in range(3)]


def test_lomo_runs(synthetic_match):
    ms = _three(synthetic_match)
    for make in (FrameModel, Baseline50):
        df = lomo(ms, make, seed=1, cfg=CFG["degrade"], use_change=False)
        assert len(df) == 3
        assert list(df.columns) == ["match", "spellF1", "macroF1", "n_pred_poss", "n_true_poss",
                                    "transF1_25"]
        assert df.spellF1.between(0, 1).all()


def test_x_frames_ignored(synthetic_match):
    m = synthetic_match(0, 1500)
    lab = m.label.copy()
    lab[::10] = 3
    m = dataclasses.replace(m, label=lab)
    X, y = frame_features(m), controlled_labels(m.label)
    keep = y != 3
    assert (~keep).any()
    by_protocol = FrameModel(step=1).fit([m])
    by_hand = FrameModel(step=1).fit(X[keep], y[keep])
    assert np.allclose(by_protocol.predict_proba(X), by_hand.predict_proba(X))
    # relabelling X as H must change the fit, so the equality above can fail
    y2 = np.where(y == 3, 0, y)
    other = FrameModel(step=1).fit(X, y2)
    assert not np.allclose(other.predict_proba(X), by_hand.predict_proba(X), atol=1e-6)
    p3, pc = by_protocol.predict(m)
    assert p3.shape == (1500, 3) and pc is None and np.allclose(p3.sum(1), 1)
    assert by_protocol.decode_output


def _tiny(ball, period=None, n=None):
    """Two players per team on a line; home at x=-10,-12, away at x=10,12; ball given per frame."""
    ball = np.asarray(ball, dtype=float)
    T = len(ball)
    px = np.tile(np.array([-10, -12, 10, 12], dtype=np.float32), (T, 1))
    py = np.zeros((T, 4), dtype=np.float32)
    per = np.ones(T, dtype=np.int8) if period is None else np.asarray(period, dtype=np.int8)
    return Match(name="t", px=px, py=py, team=np.array([0, 0, 1, 1], dtype=np.int8),
                 ball_x=ball, ball_y=np.zeros(T), period=per, label=np.zeros(T, dtype=np.int8),
                 length=105.0, width=68.0, home_attacks_right={1: True, 2: False})


def _b50(m):
    return np.argmax(Baseline50().predict(m)[0], axis=1)


def test_baseline50_nearest_player_team():
    out = _b50(_tiny([-9.0] * 200))
    assert (out == 0).all()
    out = _b50(_tiny([11.0] * 200))
    assert (out == 1).all()


def test_baseline50_dead_where_ball_unseen():
    out = _b50(_tiny([np.nan] * 300))
    assert (out == 2).all()


def test_baseline50_vote_does_not_cross_periods():
    ball = [-9.0] * 200 + [11.0] * 200
    per = [1] * 200 + [2] * 200
    out = _b50(_tiny(ball, per))
    assert (out[:200] == 0).all() and (out[200:] == 1).all()
    # within one period the vote smooths a short blip
    ball = [-9.0] * 100 + [11.0] * 5 + [-9.0] * 100
    assert (_b50(_tiny(ball)) == 0).all()


def test_lomo_gt_is_clean_match(synthetic_match, monkeypatch):
    import seqmodel.evaluate as ev
    ms = _three(synthetic_match)
    seen = []
    real = ev.pm.evaluate

    def spy(gt, pred, *a, **k):
        seen.append(np.array(gt))
        return real(gt, pred, *a, **k)

    monkeypatch.setattr(ev.pm, "evaluate", spy)
    monkeypatch.setattr(ev, "degrade",
                        lambda m, seed, cfg: dataclasses.replace(m, label=np.full_like(m.label, 2)))
    lomo(ms, Baseline50, seed=1, cfg=CFG["degrade"], use_change=False)
    assert len(seen) == 3
    for gt, m in zip(seen, ms):
        assert np.array_equal(gt, controlled_labels(m.label))


class _NaNPredictor:
    decode_output = False

    def fit(self, train):
        return self

    def predict(self, m):
        p = np.full((len(m.label), 3), 1 / 3)
        p[5, 0] = np.nan
        return p, None


def test_nan_probabilities_raise(synthetic_match):
    with pytest.raises(ValueError, match="non-finite"):
        lomo(_three(synthetic_match, 800), _NaNPredictor, seed=1, cfg=CFG["degrade"],
             use_change=False)
    m = synthetic_match(0, 800)
    fm = FrameModel().fit([m])
    fm.W = fm.W * np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fm.predict(m)


def test_frame_model_symmetric(synthetic_match):
    m = synthetic_match(0, 1500)
    X = frame_features(m)
    y = controlled_labels(m.label)
    fm = FrameModel().fit(X[::3], y[::3])
    Xs = X.copy()
    Xs[:, ODD] *= -1
    p, ps = fm.predict_proba(X), fm.predict_proba(Xs)
    assert np.allclose(p, ps[:, [1, 0, 2]], atol=1e-12)

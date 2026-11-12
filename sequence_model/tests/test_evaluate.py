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
    # asymmetric on purpose: 30 frames of H then 200 of A; a vote crossing the boundary flips period 1
    ball = [-9.0] * 30 + [11.0] * 200
    per = [1] * 30 + [2] * 200
    out = _b50(_tiny(ball, per))
    assert (out[:30] == 0).all() and (out[30:] == 1).all()
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


class _InfPChange:
    decode_output = True

    def fit(self, train):
        return self

    def predict(self, m):
        pc = np.zeros(len(m.label))
        pc[3] = np.inf
        return np.full((len(m.label), 3), 1 / 3), pc


def test_nonfinite_p_change_raises(synthetic_match):
    with pytest.raises(ValueError, match="non-finite"):
        lomo(_three(synthetic_match, 800), _InfPChange, seed=1, cfg=CFG["degrade"], use_change=True)


class _Counting:
    decode_output = True
    fits = 0

    def fit(self, train):
        type(self).fits += 1
        return self

    def predict(self, m):
        p = np.full((len(m.label), 3), 0.2)
        p[:, 0] = 0.6
        return p, np.zeros(len(m.label))


def test_lomo_multi_fits_once_and_reports(synthetic_match):
    from seqmodel.evaluate import lomo_multi, select_decoder, plain_decoder
    ms = _three(synthetic_match, 800)
    seen = []
    _Counting.fits = 0
    res = lomo_multi(ms, _Counting, seed=1, cfg=CFG["degrade"],
                     decoders={"a": plain_decoder(False),
                               "b": select_decoder([0, 5], [0.5, 1.0])},
                     progress=lambda name, sec: seen.append(name))
    assert _Counting.fits == 3                       # one fit per fold, shared by both decodings
    assert set(res) == {"a", "b"} and all(len(d) == 3 for d in res.values())
    assert {"beta", "w"} <= set(res["b"].columns)
    assert res["b"].beta.isin([0, 5]).all() and res["b"].w.isin([0.5, 1.0]).all()
    assert seen == [m.name for m in ms]


def test_lomo_unchanged_signature(synthetic_match):
    df = lomo(_three(synthetic_match, 800), Baseline50, seed=1, cfg=CFG["degrade"], use_change=False)
    assert list(df.columns) == ["match", "spellF1", "macroF1", "n_pred_poss", "n_true_poss",
                                "transF1_25"]


class _Recorder:
    decode_output = True
    calls = []

    def fit(self, train):
        self.train_ids = {id(t) for t in train}
        return self

    def predict(self, m):
        type(self).calls.append((id(self), id(m)))
        p = np.full((len(m.label), 3), 0.2)
        p[:, 0] = 0.6
        return p, np.zeros(len(m.label))


def test_select_decoder_does_not_touch_other_data(synthetic_match):
    from seqmodel.evaluate import lomo_multi, select_decoder
    ms = _three(synthetic_match, 800)
    _Recorder.calls = []
    preds = []

    def make():
        preds.append(_Recorder())
        return preds[-1]

    spy = select_decoder([0, 5], [1.0])

    def dec(c):
        c.test_id = id(c.test)
        n0 = len(_Recorder.calls)
        out = spy(c)
        inner = [m for p, m in _Recorder.calls[n0:] if p == id(c.pred)]
        assert set(inner) == {id(t) for t in c.train} and len(inner) == len(c.train)
        assert id(c.test) not in inner
        return out

    lomo_multi(ms, make, seed=1, cfg=CFG["degrade"], decoders={"b": dec})
    for p in preds:
        heldout = [m for q, m in _Recorder.calls if q == id(p)]
        assert len(heldout) == 1 + 2                   # held-out once + 2 training matches in selection
        assert heldout[0] not in p.train_ids           # first call is the held-out match


def test_select_decoder_nonfinite_inner_and_nan_scores(synthetic_match):
    import pytest
    from seqmodel.evaluate import lomo_multi, select_decoder
    ms = _three(synthetic_match, 800)

    class Bad(_Recorder):
        n = 0

        def predict(self, m):
            p3, pc = super().predict(m)
            type(self).n += 1
            if type(self).n > 1:                       # first call is the held-out match
                p3[0, 0] = np.nan
            return p3, pc
    with pytest.raises(ValueError, match="non-finite"):
        lomo_multi(ms, Bad, seed=1, cfg=CFG["degrade"], decoders={"b": select_decoder([0], [1.0])})

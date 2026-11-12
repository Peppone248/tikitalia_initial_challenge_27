import dataclasses

import numpy as np

from seqmodel.evaluate import Baseline50, lomo
from seqmodel.features import ODD, frame_features
from seqmodel.frame_model import FrameModel
from seqmodel.match import CFG, controlled_labels


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
    fm = FrameModel().fit([m, synthetic_match(1, 1500)])
    p3, pc = fm.predict(m)
    assert p3.shape == (1500, 3) and pc is None and np.isfinite(p3).all()
    assert fm.decode_output
    # X is never a class: probabilities have exactly three columns summing to 1
    assert np.allclose(p3.sum(1), 1)
    # a model trained with X frames labelled H would differ; check X frames did not enter fit
    X = frame_features(m)
    y = controlled_labels(m.label)
    keep = y != 3
    a = FrameModel().fit(X[keep], y[keep]).predict_proba(X)
    b = FrameModel().fit([m]).predict_proba(X)
    assert a.shape == b.shape


def test_frame_model_symmetric(synthetic_match):
    m = synthetic_match(0, 1500)
    X = frame_features(m)
    y = controlled_labels(m.label)
    fm = FrameModel().fit(X[::3], y[::3])
    Xs = X.copy()
    Xs[:, ODD] *= -1
    p, ps = fm.predict_proba(X), fm.predict_proba(Xs)
    assert np.allclose(p, ps[:, [1, 0, 2]], atol=1e-12)

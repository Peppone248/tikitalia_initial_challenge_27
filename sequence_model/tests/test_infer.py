import dataclasses
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqmodel.evaluate import lomo  # noqa: E402
from seqmodel.features import FEATURES, ODD, frame_features  # noqa: E402
from seqmodel.infer import NumpyTCN, NumpyTCNPredictor  # noqa: E402
from seqmodel.match import CFG  # noqa: E402
from seqmodel.tcn import TCN, TCNPredictor, export_weights, symmetric_forward  # noqa: E402

F = len(FEATURES)
SMALL = dict(CFG["tcn"], channels=8, levels=3, window=300, windows_per_epoch=40, max_epochs=2,
             patience=3)


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(0)
    net = TCN(F, channels=8, levels=3, kernel=5).eval()
    rng = np.random.default_rng(0)
    net.mu = rng.normal(size=F)
    net.mu[ODD] = 0.0
    net.sd = rng.uniform(0.5, 2.0, size=F)
    path = str(tmp_path / "w.npz")
    export_weights(net, path)
    return net, path


def test_matches_torch(exported):
    net, path = exported
    X = np.random.default_rng(1).normal(size=(250, F)) * 2.0
    p3, pc = NumpyTCN.load(path).predict(X, ODD, round_outputs=False)
    x = torch.from_numpy(((X - net.mu) / net.sd).astype(np.float32).T[None].copy())
    q3, qc = symmetric_forward(net, x, ODD)
    assert np.abs(p3 - q3[0].numpy().T).max() < 1e-4
    assert np.abs(pc - qc[0, 0].numpy()).max() < 1e-4


def test_deterministic(exported):
    _, path = exported
    X = np.random.default_rng(2).normal(size=(300, F))
    h = [hashlib.sha256(b"".join(a.tobytes() for a in NumpyTCN.load(path).predict(X, ODD))).hexdigest()
         for _ in range(2)]
    assert h[0] == h[1]


def test_rounding_stable(exported):
    _, path = exported
    net = NumpyTCN.load(path)
    X = np.random.default_rng(3).normal(size=(300, F))
    noise = np.random.default_rng(4).uniform(-1e-9, 1e-9, size=X.shape)
    a, b = net.predict(X, ODD), net.predict(X + noise, ODD)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[0], np.round(a[0], 3)) and np.array_equal(a[1], np.round(a[1], 3))


def test_predict_match_per_period(exported, synthetic_match):
    net, path = exported
    m = synthetic_match(0, 800)
    p3, pc = NumpyTCN.load(path).predict_match(m)
    assert p3.shape == (800, 3) and pc.shape == (800,)
    X = frame_features(m)
    cut = int(np.flatnonzero(np.diff(m.period))[0]) + 1
    n = NumpyTCN.load(path)
    a3, ac = n.predict(X[:cut], ODD)
    b3, bc = n.predict(X[cut:], ODD)
    assert np.array_equal(p3, np.vstack([a3, b3])) and np.array_equal(pc, np.concatenate([ac, bc]))
    # agrees with the torch predictor up to output rounding
    tp = TCNPredictor(0, SMALL)
    tp.model = net
    t3, tc = tp.predict(m)
    assert np.abs(p3 - t3).max() < 2e-3 and np.abs(pc - tc).max() < 2e-3


def test_numpy_predictor_lomo(synthetic_match):
    ms = [dataclasses.replace(synthetic_match(s, 1500), name=f"m{s}") for s in range(3)]
    assert NumpyTCNPredictor.decode_output is True
    df = lomo(ms, lambda: NumpyTCNPredictor(0, SMALL), seed=1, cfg=CFG["degrade"], use_change=True)
    assert len(df) == 3 and df.spellF1.between(0, 1).all()

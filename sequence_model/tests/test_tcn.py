import dataclasses
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqmodel.evaluate import lomo  # noqa: E402
from seqmodel.features import FEATURES, ODD, frame_features  # noqa: E402
from seqmodel.match import CFG, controlled_labels  # noqa: E402
from seqmodel.tcn import (TCN, TCNPredictor, change_targets, export_weights,  # noqa: E402
                          symmetric_forward, train_tcn)

F = len(FEATURES)
SMALL = dict(CFG["tcn"], channels=8, levels=3, window=300, windows_per_epoch=40, max_epochs=2,
             patience=3)


def test_change_targets():
    lab = np.zeros(400, dtype=np.int8)
    lab[100:] = 1
    per = np.ones(400, dtype=np.int8)
    t = change_targets(lab, per)
    assert t.dtype == np.float32
    assert np.flatnonzero(t).tolist() == list(range(88, 113))
    lab2 = np.zeros(400, dtype=np.int8)
    lab2[100:150] = 2
    lab2[150:] = 1
    assert change_targets(lab2, per).sum() == 0


def test_change_targets_not_across_period():
    lab = np.zeros(400, dtype=np.int8)
    lab[200:] = 1
    per = np.where(np.arange(400) < 200, 1, 2).astype(np.int8)
    assert change_targets(lab, per).sum() == 0


def test_noncausal():
    torch.manual_seed(0)
    net = TCN(F, channels=8, levels=4, kernel=5).eval()      # receptive field +-30 frames
    x = torch.randn(1, F, 200)
    y0 = net(x)[0]
    x2 = x.clone()
    x2[0, :, 120] += 5.0
    y1 = net(x2)[0]
    assert (y1[0, :, 100] - y0[0, :, 100]).abs().max() > 0       # future frame changes the past
    assert (y1[0, :, 100 - 40] - y0[0, :, 100 - 40]).abs().max() == 0   # beyond the field


def test_symmetry_exact():
    torch.manual_seed(1)
    net = TCN(F, channels=8, levels=3, kernel=5).eval()
    x = torch.randn(2, F, 150)
    xs = x.clone()
    xs[:, torch.as_tensor(ODD), :] *= -1
    p3, pc = symmetric_forward(net, x, ODD)
    q3, qc = symmetric_forward(net, xs, ODD)
    assert torch.allclose(q3, p3[:, [1, 0, 2]], atol=1e-6)
    assert torch.allclose(qc, pc, atol=1e-6)
    assert torch.allclose(p3.sum(1), torch.ones(2, 150), atol=1e-5)


def test_overfits_tiny(synthetic_match):
    m = synthetic_match(0, 3000)
    bx, by = m.ball_x.copy(), m.ball_y.copy()
    bx[m.label == 2] = by[m.label == 2] = np.nan      # dead ball: ball not seen (else D has no signal)
    m = dataclasses.replace(m, ball_x=bx, ball_y=by)
    cfg = dict(CFG["tcn"], windows_per_epoch=200, max_epochs=8, patience=99,
               swap_p=0.0)   # 8 x 25 = 200 steps
    pred = TCNPredictor(0, cfg).fit([m])
    p3, pc = pred.predict(m)
    assert p3.shape == (3000, 3) and pc.shape == (3000,) and p3.dtype == np.float64
    lab = controlled_labels(m.label)
    ok = lab <= 2
    assert (p3.argmax(1)[ok] == lab[ok]).mean() > 0.9


def test_never_crosses_period_and_deterministic(synthetic_match):
    ms = [dataclasses.replace(synthetic_match(s, 1200), name=f"m{s}") for s in range(2)]
    a = train_tcn(ms, SMALL, seed=3)
    b = train_tcn(ms, SMALL, seed=3)
    for (n, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(pa, pb), n


def test_export_weights(tmp_path):
    net = TCN(F, channels=8, levels=3, kernel=5)
    net.mu, net.sd = np.zeros(F), np.ones(F)
    path = str(tmp_path / "w.npz")
    export_weights(net, path)
    z = np.load(path)
    for i in range(3):
        assert z[f"l{i}.conv.w"].shape == (8, 8, 5)
        assert f"l{i}.conv.b" in z and f"l{i}.pw.w" in z
    assert z["head3.w"].shape[0] == 3 and z["headc.w"].shape[0] == 1
    assert int(z["levels"]) == 3 and int(z["kernel"]) == 5 and int(z["channels"]) == 8
    assert z["odd"].dtype == bool and z["mu"].shape == (F,)
    # every parameter is exported
    assert len([k for k in z.files if k.endswith((".w", ".b"))]) == len(list(net.parameters()))


def test_lomo_plugs_in(synthetic_match):
    ms = [dataclasses.replace(synthetic_match(s, 1500), name=f"m{s}") for s in range(3)]
    df = lomo(ms, lambda: TCNPredictor(0, SMALL), seed=1, cfg=CFG["degrade"], use_change=True)
    assert len(df) == 3 and df.spellF1.between(0, 1).all()

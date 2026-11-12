import dataclasses
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqmodel.features import FEATURES, ODD  # noqa: E402
from seqmodel.match import CFG, match_from_atd  # noqa: E402
from seqmodel.tcn import TCN, export_weights  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = len(FEATURES)


def _load_script():
    spec = importlib.util.spec_from_file_location("predict", os.path.join(ROOT, "scripts", "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    net = TCN(F, channels=8, levels=3, kernel=5).eval()
    net.mu = np.zeros(F)
    net.sd = np.ones(F)
    path = str(tmp_path / "w.npz")
    export_weights(net, path)
    return path


def _save_idsse(m, path):
    poss = np.where(m.label == 0, 1, np.where(m.label == 1, 2, 0))
    np.savez(path, ball_possession=poss, ball_alive=(m.label != 2), pitch_x=m.length,
             pitch_y=m.width, px=m.px, py=m.py, player_team=m.team, ball_x=m.ball_x,
             ball_y=m.ball_y, period=m.period)


def _check(out, T):
    z = np.load(out / "probs.npz")
    for k in ("p_h", "p_a", "p_d", "p_change"):
        assert z[k].dtype == np.float32 and z[k].shape == (T,)
    assert np.allclose(z["p_h"] + z["p_a"] + z["p_d"], 1, atol=5e-3)
    lines = (out / "labels.csv").read_text().splitlines()
    assert lines[0] == "frame,label" and len(lines) == T + 1
    assert [int(l.split(",")[0]) for l in lines[1:]] == list(range(T))
    return [l.split(",")[1] for l in lines[1:]]


def test_predict_writes_files(synthetic_match, weights, tmp_path):
    m = synthetic_match(0, 1200)
    src = tmp_path / "m.npz"
    _save_idsse(m, str(src))
    out = tmp_path / "out"
    _load_script().run(weights, str(src), str(out))
    labs = _check(out, 1200)
    assert set(labs) <= {"H", "A", "D", "X"} and "X" not in labs


def test_predict_atd_outside_halves_is_x(synthetic_match, weights, tmp_path, monkeypatch):
    m = synthetic_match(1, 1500)
    period = m.period.copy()
    period[:200] = 0
    period[1300:] = 4
    period[700:800] = 3
    m = dataclasses.replace(m, period=period, home_attacks_right={0: True, 1: True, 2: False, 3: True, 4: True})
    mod = _load_script()
    monkeypatch.setattr(mod, "load_atd", lambda xml, txt: m)
    xml = tmp_path / "g_atd.xml"
    xml.write_text("")
    out = tmp_path / "out"
    mod.run(weights, str(xml), str(out), beta=0.0, w=1.0)
    labs = np.array(_check(out, 1500))
    assert (labs[(period == 0) | (period == 3) | (period == 4)] == "X").all()
    assert set(labs[(period == 1) | (period == 2)]) <= {"H", "A", "D"}


def test_predict_deterministic(synthetic_match, weights, tmp_path):
    src = tmp_path / "m.npz"
    _save_idsse(synthetic_match(0, 900), str(src))
    mod = _load_script()
    for d in ("a", "b"):
        mod.run(weights, str(src), str(tmp_path / d))
    assert (tmp_path / "a" / "labels.csv").read_bytes() == (tmp_path / "b" / "labels.csv").read_bytes()


def _fake_dataset(n, blocks_n=None):
    class P:
        pass
    ps = [P(), P(), P()]
    home, away = SimpleNamespace(players=ps[:1]), SimpleNamespace(players=ps[1:])
    recs = []
    for i in range(n):
        pd = {ps[0]: SimpleNamespace(coordinates=SimpleNamespace(x=0.5 + 0.001 * i, y=0.25)),
              ps[1]: SimpleNamespace(coordinates=None),
              ps[2]: SimpleNamespace(coordinates=SimpleNamespace(x=0.0, y=1.0))}
        ball = SimpleNamespace(x=1.0, y=0.5) if i % 2 == 0 else None
        recs.append(SimpleNamespace(frame_id=i + 1, players_data=pd, ball_coordinates=ball))
    recs.reverse()      # order must not matter
    return SimpleNamespace(records=recs, metadata=SimpleNamespace(teams=[home, away]))


def test_match_from_atd_conversion():
    ds = _fake_dataset(12)
    m = match_from_atd(ds, (100.0, 50.0), [(1, 2, 4), (2, 7, 9)], min_mean_tracked=1)
    assert m.graded_periods == (1, 2)
    assert m.px.shape == (12, 3) and m.team.tolist() == [0, 1, 1]
    assert np.allclose(m.px[3, 0], 0.003 * 100) and np.allclose(m.py[:, 0], -12.5)
    assert np.isnan(m.px[:, 1]).all() and np.allclose(m.px[:, 2], -50) and np.allclose(m.py[:, 2], 25)
    assert np.allclose(m.ball_x[::2], 50) and np.isnan(m.ball_x[1::2]).all()
    assert (m.label == CFG["labels"]["X"]).all() and m.length == 100.0 and m.width == 50.0
    assert m.period.tolist() == [0, 0, 1, 1, 1, 3, 3, 2, 2, 2, 4, 4]
    assert set(m.home_attacks_right) == {0, 1, 2, 3, 4}


def test_match_from_atd_unmatched_players_raises():
    ds = _fake_dataset(12)

    class Q:
        pass
    for r in ds.records:                      # record keys no longer match the metadata players
        r.players_data = {Q(): v for v in r.players_data.values()}
    with pytest.raises(ValueError, match="implausible"):
        match_from_atd(ds, (100.0, 50.0), [(1, 2, 4), (2, 7, 9)], min_mean_tracked=1)


def test_match_from_atd_default_threshold_raises_on_sparse_tracking():
    with pytest.raises(ValueError, match="implausible"):
        match_from_atd(_fake_dataset(12), (100.0, 50.0), [(1, 2, 4), (2, 7, 9)])


def test_predict_uses_match_graded_periods(synthetic_match, weights, tmp_path, monkeypatch):
    m = dataclasses.replace(synthetic_match(2, 900), graded_periods=(2,))
    mod = _load_script()
    monkeypatch.setattr(mod, "load_idsse", lambda p: m)
    src = tmp_path / "m.npz"
    src.write_bytes(b"")
    mod.run(weights, str(src), str(tmp_path / "o"))
    labs = np.array(_check(tmp_path / "o", 900))
    assert (labs[m.period == 1] == "X").all() and "X" not in labs[m.period == 2]

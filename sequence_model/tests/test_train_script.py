import importlib.util
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from seqmodel.infer import NumpyTCN  # noqa: E402
from seqmodel.match import CFG  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_cfg():
    return dict(CFG["tcn"], channels=8, levels=2, window=300, batch=2, windows_per_epoch=4,
                max_epochs=1, patience=1, threads=1)


def test_train_writes_weights_and_sidecar(synthetic_match, tmp_path):
    ms = [synthetic_match(0, 1200), synthetic_match(1, 1200)]
    out = str(tmp_path / "w" / "t.npz")
    side = _script("train").train(ms, 7, out, cfg_tcn=_tiny_cfg())
    assert os.path.exists(out)
    NumpyTCN.load(out)
    with open(str(tmp_path / "w" / "t_decode.json")) as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(json.dumps(side))
    for k in ("trans", "prior", "beta", "w", "seed", "matches", "git_commit", "config_hash", "config", "date"):
        assert k in on_disk
    assert set(on_disk["trans"]) == {"HA", "HD", "AH", "AD", "DH", "DA"}
    assert on_disk["prior"] is None and on_disk["beta"] == 0 and on_disk["w"] == 1.0
    assert on_disk["seed"] == 7 and len(on_disk["matches"]) == 2


def test_predict_uses_sidecar(tmp_path, monkeypatch):
    mod = _script("predict")
    w = tmp_path / "w.npz"
    (tmp_path / "w_decode.json").write_text(json.dumps(
        {"trans": {"HA": 0.1}, "prior": [0.2, 0.3, 0.5], "beta": 3, "w": 0.7}))
    calls = []
    monkeypatch.setattr(mod, "run", lambda *a: calls.append(a))
    monkeypatch.setattr("sys.argv", ["p", "--weights", str(w), "--match", "m.npz", "--out", "o"])
    mod.main()
    assert calls[-1][3:] == (3, 0.7, {"HA": 0.1}, [0.2, 0.3, 0.5])
    monkeypatch.setattr("sys.argv", ["p", "--weights", str(w), "--match", "m.npz", "--out", "o",
                                     "--beta", "1", "--w", "2"])
    mod.main()
    assert calls[-1][3:5] == (1.0, 2.0) and calls[-1][5] == {"HA": 0.1}


def test_predict_warns_without_sidecar(tmp_path, monkeypatch, capsys):
    mod = _script("predict")
    calls = []
    monkeypatch.setattr(mod, "run", lambda *a: calls.append(a))
    monkeypatch.setattr("sys.argv", ["p", "--weights", str(tmp_path / "w.npz"), "--match", "m.npz",
                                     "--out", "o"])
    mod.main()
    assert "WARNING" in capsys.readouterr().err
    assert calls[-1][3:] == (0.0, 1.0, None, None)

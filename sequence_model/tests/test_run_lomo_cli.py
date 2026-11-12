import dataclasses
import importlib.util
import os
import sys

from seqmodel.match import CFG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = dict(CFG["tcn"], channels=8, levels=3, window=300, windows_per_epoch=40, max_epochs=2,
             patience=3)


def test_cli_end_to_end(synthetic_match, monkeypatch, capsys, tmp_path):
    spec = importlib.util.spec_from_file_location("run_lomo", os.path.join(ROOT, "scripts", "run_lomo.py"))
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    ms = [dataclasses.replace(synthetic_match(s, 1500), name=f"m{s}") for s in range(3)]
    data = tmp_path / "d"
    data.mkdir()
    for i in range(3):
        (data / f"{i}.npz").write_bytes(b"")
    it = iter(ms)
    monkeypatch.setattr(rl, "load_idsse", lambda p: next(it))
    monkeypatch.setattr(rl, "ROOT", str(tmp_path))
    monkeypatch.setitem(rl.GROUPS, "tcn", lambda seed: rl.NumpyTCNPredictor(seed, SMALL))
    monkeypatch.setattr(rl, "BETAS", [0, 5])
    monkeypatch.setattr(sys, "argv", ["run_lomo", "--models", "frame", "tcn", "tcn_change",
                                      "--seeds", "1", "2", "--data", str(data)])
    rl.main()
    cap = capsys.readouterr()
    assert cap.err.count("held-out") == 12                      # 2 seeds x 3 folds x (frame, tcn group)
    # 3 matches < 7 paired matches per seed: the criterion cannot pass
    assert cap.out.rstrip().splitlines()[-1] == "criterion: NON SODDISFATTO"
    assert "[train_tcn] epoch" in cap.err and "[fold m0] fit:" in cap.err
    assert os.path.exists(tmp_path / "runs" / "lomo_frame-tcn-tcn_change_1-2.md")


def test_default_seeds_and_w_only_baselines():
    spec = importlib.util.spec_from_file_location("run_lomo", os.path.join(ROOT, "scripts", "run_lomo.py"))
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    import inspect
    assert "default=[2027, 2028]" in inspect.getsource(rl.main)
    # frame and tcn tune w only; tcn_change tunes beta too
    assert inspect.getclosurevars(rl.MODELS["frame"][1]()).nonlocals["betas"] == [0]
    assert inspect.getclosurevars(rl.MODELS["tcn"][1]()).nonlocals["betas"] == [0]
    assert inspect.getclosurevars(rl.MODELS["tcn_change"][1]()).nonlocals["betas"] == rl.BETAS

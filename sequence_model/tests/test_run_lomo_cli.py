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
                                      "--seeds", "1", "--data", str(data)])
    rl.main()
    cap = capsys.readouterr()
    assert cap.err.count("held-out") == 6                       # 3 folds x (frame, tcn group)
    assert cap.out.rstrip().splitlines()[-1] in ("criterion: SODDISFATTO", "criterion: NON SODDISFATTO")
    assert os.path.exists(tmp_path / "runs" / "lomo_frame-tcn-tcn_change_1.md")

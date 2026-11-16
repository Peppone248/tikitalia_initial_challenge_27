"""Train the shipped TCN on all IDSSE matches and write the weights plus the decoding sidecar.
Usage (repo root): sequence_model/.venv/Scripts/python.exe sequence_model/scripts/train.py \
--data sequence_model/data/idsse --seed 2027 --out sequence_model/weights/tcn_idsse.npz
Writes OUT.npz (tcn.export_weights) and OUT_decode.json (trans, prior, beta, w and provenance):
the decoding configuration that was measured (plain TCN, beta 0, w 1, uniform prior, transitions
counted on the controlled labels of the training matches)."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from seqmodel.decode import count_transitions  # noqa: E402
from seqmodel.degrade import degrade  # noqa: E402
from seqmodel.match import CFG, controlled_labels, load_idsse  # noqa: E402
from seqmodel.tcn import TCNPredictor, export_weights  # noqa: E402


def _commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except OSError:
        return None


def train(matches, seed, out, cfg_tcn=None, cfg_degrade=None):
    """matches: list[Match] (clean). The degradation is the evaluation one: degrade(m, seed + j)."""
    cfg_tcn = CFG["tcn"] if cfg_tcn is None else cfg_tcn
    cfg_degrade = CFG["degrade"] if cfg_degrade is None else cfg_degrade
    t0 = time.time()
    deg = [degrade(m, seed + j, cfg_degrade) for j, m in enumerate(matches)]
    print(f"[train] degraded {len(deg)} matches: {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    pred = TCNPredictor(seed, cfg_tcn).fit(deg)
    print(f"[train] fit: {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    export_weights(pred.model, out)
    labs = [controlled_labels(t.label) for t in deg]
    trans = count_transitions(labs, [t.period for t in deg])
    blocks = {"tcn": cfg_tcn, "degrade": cfg_degrade}
    side = dict(trans={k: float(v) for k, v in trans.items()}, prior=None, beta=0, w=1.0, seed=seed,
                matches=[m.name for m in matches], git_commit=_commit(),
                config_hash=hashlib.sha256(json.dumps(blocks, sort_keys=True).encode()).hexdigest(),
                config=blocks, date=datetime.date.today().isoformat())
    side_path = os.path.splitext(out)[0] + "_decode.json"
    with open(side_path, "w") as f:
        json.dump(side, f, indent=1)
    print(f"[train] wrote {out} and {side_path}", file=sys.stderr, flush=True)
    return side


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "idsse"))
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz matches in {a.data}")
    train([load_idsse(p) for p in paths], a.seed, a.out)


if __name__ == "__main__":
    main()

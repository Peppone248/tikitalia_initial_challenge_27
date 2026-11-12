"""Leave-one-match-out comparison of predictors on the IDSSE matches.
Usage: python sequence_model/scripts/run_lomo.py --models baseline50 frame --seeds 2027"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from seqmodel.evaluate import Baseline50, lomo  # noqa: E402
from seqmodel.frame_model import FrameModel  # noqa: E402
from seqmodel.match import CFG, load_idsse  # noqa: E402

# name -> (predictor factory, use_change); add new models here
MODELS = {
    "baseline50": (Baseline50, False),
    "frame": (FrameModel, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2027])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "idsse"))
    a = ap.parse_args()
    matches = [load_idsse(p) for p in sorted(glob.glob(os.path.join(a.data, "*.npz")))]
    out = []
    for name in a.models:
        make, use_change = MODELS[name]
        for seed in a.seeds:
            t0 = time.time()
            df = lomo(matches, make, seed, CFG["degrade"], use_change)
            print(f"{name} seed {seed}: {time.time() - t0:.0f}s", file=sys.stderr)
            out.append(df.assign(model=name, seed=seed))
    df = pd.concat(out)
    cols = ["spellF1", "macroF1", "n_pred_poss", "n_true_poss", "transF1_25"]
    lines = []
    for (name, seed), g in df.groupby(["model", "seed"], sort=False):
        lines += [f"### {name}, seed {seed}", "", "| match | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 1)]
        for _, r in g.iterrows():
            lines.append(f"| {r['match']} | " + " | ".join(
                f"{r[c]:.3f}" if c in ("spellF1", "macroF1", "transF1_25") else f"{int(r[c])}"
                for c in cols) + " |")
        lines.append("| **mean** | " + " | ".join(f"{g[c].mean():.3f}" for c in cols) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    fn = f"lomo_{'-'.join(a.models)}_{'-'.join(map(str, a.seeds))}.md"
    with open(os.path.join(ROOT, "runs", fn), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

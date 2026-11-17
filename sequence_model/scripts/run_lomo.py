"""Leave-one-match-out comparison of predictors on the IDSSE matches.
Usage: python sequence_model/scripts/run_lomo.py --models baseline50 frame --seeds 2027 2028"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from seqmodel.evaluate import (Baseline50, lomo_multi, plain_decoder,  # noqa: E402
                               select_decoder)
from seqmodel.frame_model import FrameModel  # noqa: E402
from seqmodel.infer import NumpyTCNPredictor  # noqa: E402
from seqmodel.match import CFG, load_idsse  # noqa: E402
from seqmodel.report import criterion, render  # noqa: E402,F401

BETAS, WS = [0, 5, 20, 50], [0.5, 1.0]

# model name -> (group, decoder factory). Models of one group share a single fit per fold and seed.
# The factory for tcn takes the seed because the predictor is seeded.
MODELS = {
    "baseline50": ("baseline50", lambda: plain_decoder(False)),
    # frame and tcn select w only (beta fixed 0), tcn_change also beta: its sole extra freedom is the
    # change branch, so the criterion cannot credit a gain that comes from tuning w
    "frame": ("frame", lambda: select_decoder([0], WS)),
    "tcn": ("tcn", lambda: select_decoder([0], WS)),
    "tcn_change": ("tcn", lambda: select_decoder(BETAS, WS)),
}
GROUPS = {
    "baseline50": lambda seed: Baseline50(),
    "frame": lambda seed: FrameModel(),
    "tcn": lambda seed: NumpyTCNPredictor(seed),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "idsse"))
    a = ap.parse_args()
    matches = [load_idsse(p) for p in sorted(glob.glob(os.path.join(a.data, "*.npz")))]
    out = []
    for group in dict.fromkeys(MODELS[n][0] for n in a.models):
        names = [n for n in a.models if MODELS[n][0] == group]
        for seed in a.seeds:
            def progress(match, sec, group=group, seed=seed):
                print(f"[{group} seed {seed}] held-out {match}: {sec:.0f}s", file=sys.stderr,
                      flush=True)
            res = lomo_multi(matches, lambda: GROUPS[group](seed), seed, CFG["degrade"],
                             {n: MODELS[n][1]() for n in names}, progress)
            out += [d.assign(model=n, seed=seed) for n, d in res.items()]
    df = pd.concat(out)
    text = render(df)
    print(text)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    fn = f"lomo_{'-'.join(a.models)}_{'-'.join(map(str, a.seeds))}.md"
    with open(os.path.join(ROOT, "runs", fn), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

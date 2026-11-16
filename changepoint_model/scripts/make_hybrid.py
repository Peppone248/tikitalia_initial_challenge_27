"""v4: the hybrid model. P(home) from a bidirectional GRU trained on the external matches, P(dead)
and the three boundary heads from the boosted trees of an existing model (v2), decoded with that
model's settings (gamma 8, w 0.25) and a fixed emission prior (H, A, D) = (0.3, 0.3, 0.4): exactly
the configuration measured by eval_hybrid.py (+0.016 spellF1, 6/0 held-out matches).

    python changepoint_model/scripts/make_hybrid.py --trees changepoint_model/models/cpm_v2_all18.pkl \
        --metrica ext/metrica/data --idsse data/idsse --pff data/pff \
        --out changepoint_model/models/cpm_v4_hybrid.pkl

Needs torch to train; the saved file predicts with numpy only.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cpm.loaders import load_metrica_csv, load_metrica_epts, load_idsse  # noqa: E402
from cpm import pipeline  # noqa: E402
from cpm.seq import train_seq  # noqa: E402

PRIOR = np.array([0.3, 0.3, 0.4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--metrica")
    ap.add_argument("--idsse")
    ap.add_argument("--pff")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="changepoint_model/models/cpm_v4_hybrid.pkl")
    args = ap.parse_args()
    t0 = time.time()
    trees = pipeline.load_models(args.trees)
    matches = []
    if args.metrica:
        cache_dir = os.path.join(os.path.dirname(args.out), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        matches += [load_metrica_csv(args.metrica, 1), load_metrica_csv(args.metrica, 2),
                    load_metrica_epts(args.metrica, 3, cache=os.path.join(cache_dir, "metrica_g3.npz"))]
    for folder in (args.idsse, args.pff):
        if folder:
            matches += [load_idsse(p) for p in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    print("training matches:", [m.name for m in matches], flush=True)
    preps = [pipeline.prepare(m, "degraded", args.seed + i) for i, m in enumerate(matches)]
    del matches
    sd = train_seq(preps, {"epochs": args.epochs, "seed": args.seed, "k": trees["cfg"]["k"]},
                   log=lambda *a: print(*a, flush=True))
    cfg = dict(trees["cfg"], heads="hybrid")
    models = {"cfg": cfg, "trees": trees, "seq": sd, "prior": PRIOR,
              "even_names": trees["even_names"], "odd_names": trees["odd_names"],
              "training": {"trees": os.path.basename(args.trees),
                           "trees_training": trees.get("training"),
                           "seq_matches": [P.match.name for P in preps], "seq_epochs": args.epochs,
                           "seed": args.seed, "gamma": cfg["gamma"], "w": cfg["w"],
                           "prior": PRIOR.tolist()}}
    pipeline.save_models(models, args.out)
    with open(args.out + ".json", "w") as fh:
        json.dump(models["training"], fh, indent=1, default=str)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print("saved {} (sha256 {}) in {:.0f} s".format(args.out, sha[:16], time.time() - t0))


if __name__ == "__main__":
    main()

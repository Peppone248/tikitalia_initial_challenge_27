"""Train the final models on external matches (never on the provided match).

    python changepoint_model/scripts/train.py --metrica ext/metrica/data --idsse data/idsse \
        --scenario degraded --seed 2027 --out changepoint_model/models/cpm_v1.pkl

The decoder settings (gamma, w) are chosen on the training matches themselves (in-sample, the
same grid as eval_lomo.py) and stored in the model file, so prediction needs no label of any
kind. The file lists the matches, the scenario and the seed it was trained with.
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
from eval_lomo import choose_decoder, GAMMAS, WS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica")
    ap.add_argument("--idsse")
    ap.add_argument("--pff")
    ap.add_argument("--scenario", nargs="+", default=["degraded"])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--backend", default="hgb")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--out", default="changepoint_model/models/cpm_v1.pkl")
    ap.add_argument("--exclude", nargs="*", default=[], help="match names to leave out")
    ap.add_argument("--heads", default="frame", choices=["frame", "seq"])
    ap.add_argument("--seq-epochs", type=int, default=30)
    ap.add_argument("--seq-hidden", type=int, default=64)
    ap.add_argument("--tune-matches", type=int, default=0)
    ap.add_argument("--gammas", nargs="*", type=float, default=list(GAMMAS))
    ap.add_argument("--ws", nargs="*", type=float, default=list(WS))
    ap.add_argument("--subsample", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    matches = []
    if args.metrica:
        cache_dir = os.path.join(os.path.dirname(args.out), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        matches += [load_metrica_csv(args.metrica, 1), load_metrica_csv(args.metrica, 2),
                    load_metrica_epts(args.metrica, 3, cache=os.path.join(cache_dir, "metrica_g3.npz"))]
    for folder in (args.idsse, args.pff):
        if folder:
            matches += [load_idsse(p) for p in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    matches = [m for m in matches if m.name not in args.exclude]
    print("training matches:", [m.name for m in matches], flush=True)
    preps = []
    for sc in args.scenario:
        preps += [pipeline.prepare(m, sc, args.seed + i) for i, m in enumerate(matches)]
    cfg = {"backend": args.backend, "k": args.k, "stack": not args.no_stack, "heads": args.heads,
           "subsample": args.subsample,
           "seq": {"epochs": args.seq_epochs, "hidden": args.seq_hidden}}
    models = pipeline.fit(preps, cfg, verbose=True, log=lambda *a: print(*a, flush=True))
    tune = preps[:args.tune_matches] if args.tune_matches else preps
    (gamma, w), v = choose_decoder(models, tune, tuple(args.gammas), tuple(args.ws), "boundary")
    models["cfg"]["gamma"] = gamma
    models["cfg"]["w"] = w
    models["training"] = {"matches": [m.name for m in matches], "scenarios": args.scenario,
                          "seed": args.seed, "in_sample_spellF1": v, "gamma": gamma, "w": w,
                          "even_names": preps[0].even_names, "odd_names": preps[0].odd_names}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pipeline.save_models(models, args.out)
    with open(args.out + ".json", "w") as fh:
        json.dump(models["training"], fh, indent=1)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print("saved {} (sha256 {}) gamma {} w {} in-sample spellF1 {:.4f}; {:.0f} s".format(
        args.out, sha[:16], gamma, w, v, time.time() - t0))


if __name__ == "__main__":
    main()

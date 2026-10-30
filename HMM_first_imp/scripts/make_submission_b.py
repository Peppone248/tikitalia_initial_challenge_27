"""Raw data -> CSV with approach B (boosted baseline). One command:

    python HMM_first_imp/scripts/make_submission_b.py               # -> outputs/run_b/tikitalia.csv
    python HMM_first_imp/scripts/make_submission_b.py --set broadcast_dead.enabled=true --out-dir outputs/run_b_bc

Steps: (1) approach C v3 (config_c3) for the dead-ball mask, regenerated unless --reuse-c3 and
outputs/run_c3/tikitalia.csv exists; (2) D-SLOW on top; (3) optional broadcast pieces (aligned by
hmm_v0.broadcast.align); (4) nearest player + vote + spread + minimum possession (hmm_v0.baseline_b).
Settings: config_b/baseline_b.yaml. Reads no ground truth.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hmm_v0 import paths  # noqa: E402
from hmm_v0 import baseline_b as BB  # noqa: E402
from tikitalia.preprocessing import team_of_each_track, runs  # noqa: E402
from games import asset_paths, half_video_blocks  # noqa: E402


def _set(cfg, key, value):
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def read_labels(csv_path):
    with open(csv_path) as h:
        next(h)
        return np.array([line.rstrip("\n").split(",")[1] for line in h], dtype="<U1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(paths.HMM_DIR, "config_b", "baseline_b.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out-dir", default="outputs/run_b")
    ap.add_argument("--reuse-c3", action="store_true")
    args = ap.parse_args()
    with open(args.config) as h:
        cfg = yaml.safe_load(h)
    for item in args.set:
        k, v = item.split("=", 1)
        _set(cfg, k, yaml.safe_load(v))

    c3_csv = paths.resolve("outputs/run_c3/tikitalia.csv")
    if not (args.reuse_c3 and os.path.isfile(c3_csv)):
        from hmm_v0.run import run
        run(paths.resolve(cfg["c3_config_dir"]), {}, paths.resolve("outputs/run_c3"), quiet=True)
    c3 = read_labels(c3_csv)

    cache = np.load(paths.resolve("../outputs/cache/atd_arrays.npz"))
    bx, by, px, py = cache["bx"], cache["by"], cache["px"], cache["py"]
    team, _ = team_of_each_track(asset_paths()["atd_meta"])
    n = bx.size
    half = np.zeros(n, np.int8)
    for hh, a, b in half_video_blocks("BAR-ATH"):
        half[a:b + 1] = hh
    graded = half > 0
    dead = (c3 == "D") & graded
    info = {"c3_csv_sha256": hashlib.sha256(open(c3_csv, "rb").read()).hexdigest(),
            "D_c3": float(dead[graded].mean())}

    if cfg["slow_dead"]["enabled"]:
        sd = BB.slow_dead(bx, px, py, half, cfg["slow_dead"])
        info["D_slow_added"] = float((sd & ~dead)[graded].mean())
        dead |= sd

    if cfg["broadcast_dead"]["enabled"] or cfg.get("broadcast_ball"):
        from hmm_v0 import broadcast as BC
        from tikitalia.geometry import Pitch
        from tikitalia.config import load_geometry
        B = BC.load_broadcast(Pitch(load_geometry(paths.TEAM_CONFIG_DIR)))
        al = BC.align(B, {"bx": bx, "by": by, "px": px, "py": py}, team)
        tac = BC.to_tactical(B, al, n)
        info["broadcast_segments"] = al["segments"]
        if cfg["broadcast_dead"]["enabled"]:
            ms = BB.mean_player_speed(px, py)
            bspeed = np.hypot(np.gradient(bx), np.gradient(by)) * 25
            bc = BC.bc_dead(tac, half, ms, bspeed, cfg["broadcast_dead"])
            info["D_bc_added"] = float((bc & ~dead)[graded].mean())
            dead |= bc
        if cfg.get("broadcast_ball"):
            bx = np.where(np.isnan(bx), tac["ball_x"], bx)
            by = np.where(np.isnan(by), tac["ball_y"], by)

    lab = BB.boosted_baseline(bx, by, px, py, team, graded, dead, window=cfg["window"],
                              min_poss_s=cfg["min_poss_s"], unseen=cfg["unseen"],
                              spread_smooth=cfg["spread_smooth"])
    out_dir = paths.resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "tikitalia.csv")
    with open(csv, "w", newline="\n") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, x) for i, x in enumerate(lab))
    g = lab[graded]
    info.update({
        "share": {k: round(float((g == k).mean()), 4) for k in "HAD"},
        "possessions": int(sum(len(runs(lab == k)) for k in "HA")),
        "dead_spells": len(runs(lab == "D")),
        "csv_sha256": hashlib.sha256(open(csv, "rb").read()).hexdigest(),
        "config": cfg,
    })
    with open(os.path.join(out_dir, "run_summary.json"), "w") as h:
        json.dump(info, h, indent=2, default=float)
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=2, default=float))
    val = os.path.join(paths.REPO_ROOT, "evaluation", "validate_submission.py")
    if os.path.isfile(val):
        r = subprocess.run([sys.executable, val, csv], capture_output=True, text=True)
        print(r.stdout[-800:] + r.stderr[-400:])
    print("CSV: " + csv)


if __name__ == "__main__":
    main()

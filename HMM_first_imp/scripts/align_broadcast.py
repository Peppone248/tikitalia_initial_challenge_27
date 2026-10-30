"""Align the broadcast recording to the tactical timeline and write the check report.

    python HMM_first_imp/scripts/align_broadcast.py      # -> outputs/broadcast/alignment.json + .npz

The .npz (tactical-timeline arrays: covered / live / cut / ball) goes under outputs/ (gitignored:
frame-level derived data must not be committed, RULES.md). No frame number is typed anywhere:
the offsets and the switch are found by hmm_v0.broadcast.align.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0 import broadcast as BC  # noqa: E402
from tikitalia.preprocessing import team_of_each_track  # noqa: E402
from tikitalia.geometry import Pitch  # noqa: E402
from tikitalia.config import load_geometry  # noqa: E402
from games import asset_paths, half_video_blocks  # noqa: E402


def main():
    t0 = time.time()
    pitch = Pitch(load_geometry(paths.TEAM_CONFIG_DIR))
    B = BC.load_broadcast(pitch)
    c = np.load(paths.resolve("../outputs/cache/atd_arrays.npz"))
    T = {"bx": c["bx"], "by": c["by"], "px": c["px"], "py": c["py"]}
    team_t, _ = team_of_each_track(asset_paths()["atd_meta"])
    al = BC.align(B, T, team_t)
    nT = T["bx"].size
    tac = BC.to_tactical(B, al, nT)
    half = np.zeros(nT, np.int8)
    for h, a, b in half_video_blocks("BAR-ATH"):
        half[a:b + 1] = h
    g = half > 0
    out_dir = paths.resolve("outputs/broadcast")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "broadcast_on_tactical.npz"), **tac,
                        tframe=al["tframe"], cost=al["cost"], live_b=al["live"])
    # checks
    tb = np.isnan(T["bx"])
    rep = {
        "segments_b_lo_hi_offset_cost": al["segments"],
        "coarse_windows": len(al["coarse"]),
        "coarse_agree_with_modes": float(np.mean([min(abs(o - m) for m in al["modes"]) <= 6
                                                  for _b, o in al["coarse"]])),
        "graded_covered": float(tac["covered"][g].mean()),
        "graded_live": float(tac["live"][g].mean()),
        "graded_cut": float(tac["cut"][g].mean()),
        "live_player_cost_median_m": float(np.nanmedian(al["cost"][al["live"]])),
        "live_ball_diff_median_m": float(np.nanmedian(np.hypot(
            tac["ball_x"] - T["bx"], tac["ball_y"] - T["by"])[g])),
        "tactical_ball_missing_graded": float(tb[g].mean()),
        "filled_by_broadcast_ball_graded": float((tb & ~np.isnan(tac["ball_x"]))[g].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "alignment.json"), "w") as h:
        json.dump(rep, h, indent=2, default=float)
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()

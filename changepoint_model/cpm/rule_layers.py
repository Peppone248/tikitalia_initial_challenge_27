"""The precise dead-ball layers of the team's rules (HMM_first_imp/hmm_v0/dead_b.py, baseline_b.py),
computed on a cpm Match, to be united with the learned dead ball.

Which layers, and why these (REPORT_PFF, dead-ball diagnostics, measured on the 8 PFF games before
this module existed): the false D of the rules come for 92 % from their c_v3 base; the layers on
top are precise on a real tracker -- D-SLOW 0.97, wall 0.98, placed ball 0.96, island 0.97,
slow-from-ball-loss 0.89. REPORT_PFF's own next step was "the HMM's D united with the precise
layers". c_v3, the restart kick R1 (needs Smart Tagging) and the whistle R7a (needs audio) are not
used here: they cannot be measured on the external matches.

    slow     D-SLOW: mean player speed < 1.1 m/s for >= 3 s with the ball mostly unseen
    back     R4: a D-SLOW span starts where the ball was last seen (<= 6 s before)
    placed   ball seen still >= 2 s on the pitch, not in a keeper's hands, to the kick
    wall     >= 3 standing players of one team within 1.5 m, from the stoppage to the kick
    island   R6: a play stretch <= 10 s between two dead spans, ball never seen, players slow

The rule code is imported from the team's HMM_first_imp (and src/tikitalia); its constants are
read from HMM_first_imp/config_b/baseline_b.yaml, unchanged. Paths: $MTAH_HMM_DIR or
<repo>/HMM_first_imp, $MTAH_SRC_DIR or <repo>/src.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LAYERS = ("slow", "back", "placed", "wall")


def _import_rules():
    hmm_dir = os.environ.get("MTAH_HMM_DIR", os.path.join(REPO, "HMM_first_imp"))
    src_dir = os.environ.get("MTAH_SRC_DIR", os.path.join(REPO, "src"))
    for p in (src_dir, hmm_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import yaml
    from hmm_v0 import dead_b as DB
    from hmm_v0 import baseline_b as BB
    with open(os.path.join(hmm_dir, "config_b", "baseline_b.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return DB, BB, cfg


def rule_layers(m):
    """dict name -> (T,) bool for the layers above (island excluded: it needs a dead mask)."""
    import warnings
    DB, BB, cfg = _import_rules()
    L, W = m.pitch
    bx = (m.ball[:, 0] + L / 2.0).astype(np.float64)
    by = (m.ball[:, 1] + W / 2.0).astype(np.float64)
    px = (m.px + L / 2.0).astype(np.float32)
    py = (m.py + W / 2.0).astype(np.float32)
    half = np.where(np.isin(m.period, (1, 2)), m.period, 0).astype(np.int8)
    team = m.team.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slow = BB.slow_dead(bx, px, py, half, cfg["slow_dead"])
        back = DB.slow_from_ball_loss(slow, bx, half, cfg["slow_from_ball_loss"])
        gx, gy, gteam, _ = DB.goalkeeper_team_fix(px, py, team, half, L, W, cfg["goalkeeper_fix"])
        placed = DB.placed_ball(bx, by, gx, gy, gteam, half, L, W, cfg["placed_dead"])
        ms = BB.mean_player_speed(gx, gy)
        F = DB.formations(gx, gy, gteam, half, L, W, cfg["formations"], bx, by)
        wall = DB.formation_dead(F["wall"], bx, by, ms, half, cfg["formation_dead"])
    return {"slow": slow, "back": back, "placed": placed, "wall": wall, "_mspeed": ms,
            "_bx": bx, "_half": half, "_cfg": cfg}


def islands(dead, layers):
    DB, _, _ = _import_rules()
    return DB.dead_islands(np.asarray(dead, bool), layers["_bx"], layers["_half"], layers["_mspeed"],
                           layers["_cfg"]["dead_islands"])


def union(layers, names=LAYERS):
    u = np.zeros_like(layers["slow"], dtype=bool)
    for n in names:
        u |= layers[n]
    return u


def whistle_stop(labels, m, blasts):
    """R7a of the team's rules (dead_b.whistle_dead, REPORT_B 18), unchanged constants, R7b off:
    a whistle blast while the ball is alive, followed within 3 s by a real slowdown of the players,
    starts a stoppage at the blast (up to the next stoppage within 15 s, else the next kick from rest
    within 30 s). `blasts`: [(first, last) video frame]. Returns (labels, info, added mask)."""
    import warnings
    from .match import D, X
    DB, _, cfg = _import_rules()
    L, W = m.pitch
    row_of = {int(f): i for i, f in enumerate(m.frame_index)}
    rows = [(row_of[int(s)], row_of.get(int(e), row_of[int(s)])) for s, e in blasts if int(s) in row_of]
    bx = (m.ball[:, 0] + L / 2.0).astype(np.float64)
    by = (m.ball[:, 1] + W / 2.0).astype(np.float64)
    px = (m.px + L / 2.0).astype(np.float32)
    py = (m.py + W / 2.0).astype(np.float32)
    half = np.where(np.isin(m.period, (1, 2)), m.period, 0).astype(np.int8)
    wc = dict(cfg["whistle"], restart=False)
    dead = np.asarray(labels == D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, layers, info = DB.whistle_dead(dead, rows, bx, by, px, py, half, L, W, wc, cfg["restart_kick"])
    add = layers["whistle_stop"] & (labels != X)
    out = labels.copy()
    out[add] = D
    info = dict(info, frames_set_D=int(add.sum()), blasts=len(rows))
    return out, info, add

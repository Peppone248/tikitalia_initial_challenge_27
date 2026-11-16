r"""Convert the PFF FC / Gradient Sports FIFA World Cup 2022 open data to the same compact .npz as
validation/fetch_idsse.py, so every IDSSE evaluation script can run on it unchanged.

Why this dataset (REPORT_B 17.4, 19): IDSSE is clean optical tracking that we degrade with a SIMULATION
of the BAR-ATH tracker. PFF is broadcast tracking: every player and ball carries a `visibility` flag,
VISIBLE or ESTIMATED (off camera, extrapolated). Keeping only VISIBLE gives REAL tracker gaps, not
simulated ones. Labels come from PFF's own game events (people at PFF, not us, not the rules).

The data is not downloaded by this script: register on the PFF FC / Gradient page and unzip the
download anywhere (default: data/pff/raw/). Then, from the repository root:

    python HMM_first_imp/validation/fetch_pff.py --inspect               # structure of ONE game, no output
    python HMM_first_imp/validation/fetch_pff.py --n 8                   # first 8 games -> data/pff/*.npz
    python HMM_first_imp/validation/fetch_pff.py --games 3812 10517      # chosen games
    python HMM_first_imp/validation/fetch_pff.py --list                  # game ids found

It also writes data/pff/validation_pff.yaml (validation_<folder>.yaml for another --out): validation.yaml with data_dir / matches / out_dir pointed
at the converted games and variants [clean]. NOTE: on PFF the variant 'clean' means "the tracking as
PFF gives it" (VISIBLE only by default), which is already broadcast-degraded: do NOT run 'degraded' on
it (that would degrade twice).

    python HMM_first_imp/scripts/run_validation.py --config data/pff/validation_pff.yaml --st-sim \
        --versions baseline50 c_v3 --variants clean --out-dir outputs/validation_pff
    # (--config is relative to the current folder, the repository root.) Scripts that call
    # validation.load_config() without --config (eval_supervised.py, eval_hmm3.py, ...): set
    $env:MTAH_VALIDATION_CONFIG = "data\pff\validation_pff.yaml"

What goes into <game>.npz (fetch_idsse.py fields, resampled from 29.97 to 25 fps by nearest timestamp):
    period          1 / 2 (extra time as 3 / 4: not graded, like IDSSE)
    frame_n         PFF video frame number of the source frame
    ball_x/y/z      metres, origin at the centre spot (validation.load_idsse shifts it), NaN = not VISIBLE
                    (raw positions: at the kick-offs they are 1-1.6 m from the centre spot, the smoothed ones 3 m)
    ball_alive      the label: D where False
    ball_possession 1 = home, 2 = away, 0 = unknown -> the label: H / A
    px, py          (frames, players) metres, NaN where the player is not on the pitch (or not VISIBLE with
                    --estimated none). Default --estimated players: the broadcast sees ~12 of 22 players, BAR-ATH's
                    tactical camera almost all, so the extrapolated players are kept and only the ball is VISIBLE-only
    player_team     (players,) 0 = home, 1 = away
    pitch_x/pitch_y pitch size in metres (metadata)
  frames with no PFF frame within 0.1 s (missing video) get no positions and label X (not graded)
  restart_type    on the true dead frames: the setpieceType of the restart that ends the spell (T C G F P D K,
                  O = open play, ? = no event file, - = never restarts), '' elsewhere (scripts/eval_pff_dead.py)
  extra fields (ignored by the scripts, for checks): ball_estimated (the ball was there but ESTIMATED),
  player_jersey, source

Labels, from the game events PFF attaches to the tracking frames (kloppy reads them the same way):
    OUT, END                         -> dead from the event's start frame
    FIRST..FOURTHKICKOFF             -> alive, team = the kicking team
    OTB (on-the-ball touch)          -> team = the touching team (home_ball) from the touch's start frame;
                                        after a dead spell, alive only from its END frame (the kick, Law 9 /
                                        16: placing the ball is not play; --restart-at start to change)
    VID, or videoMissing in the event file -> X (not graded) over the event's span
  + the per-game event file (events/<game>.json, joined on gameEventId, else on type + start time):
    an OTB whose setpieceType is a restart (not 'O') while the ball is alive = play was stopped with no
    OUT (foul, offside, goal, injury): dead from the END of the previous touch to the kick of the restart.
    Without the event file those stoppages stay alive (inspect says so).
"""

import argparse
import bz2
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HMM_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(HMM_DIR)
FPS_OUT = 25.0
MAX_GAP_S = 0.1          # a 25 fps frame with no PFF frame within this many seconds is a hole (X, nothing seen)
KICKOFFS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")
SETPIECE_KEYS = ("setpieceType", "setpiece_type", "set_piece_type", "setPieceType")


# -- files -----------------------------------------------------------------------------------------
def scan(root):
    """{game_id: {"tracking": path, "meta": path, "roster": path}} from any folder layout."""
    games = collections.defaultdict(dict)
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if f.endswith(".jsonl.bz2") or f.endswith(".jsonl"):
                gid = f.split(".")[0]
                if gid.isdigit():
                    games[gid]["tracking"] = p
            elif f.endswith(".json") and f.split(".")[0].isdigit():
                gid = f.split(".")[0]
                kind = classify_json(p)
                if kind:
                    games[gid][kind] = p
    return dict(games)


def classify_json(path):
    """'meta' (list with homeTeam / fps), 'roster' (shirtNumber rows), 'events' (gameEvents rows) or None."""
    try:
        with open(path, encoding="utf-8") as h:
            head = h.read(4096)
    except OSError:
        return None
    if any(k in head for k in ('"gameEvents"', '"possessionEvents"', '"game_event"')):
        return "events"                              # event data (one file per game), not metadata
    if '"homeTeam"' in head and '"fps"' in head:
        return "meta"
    if '"shirtNumber"' in head and '"positionGroupType"' in head:
        return "roster"
    return None


def open_lines(path):
    return bz2.open(path, "rt", encoding="utf-8") if path.endswith(".bz2") else open(path, encoding="utf-8")


def read_meta(path):
    with open(path, encoding="utf-8") as h:
        m = json.load(h)
    m = m[0] if isinstance(m, list) else m
    pitch = (m.get("stadium") or {}).get("pitches") or [{}]
    return {"home_id": str(m["homeTeam"]["id"]), "away_id": str(m["awayTeam"]["id"]),
            "home": m["homeTeam"].get("name"), "away": m["awayTeam"].get("name"),
            "fps": float(m.get("fps") or 29.97),
            "length": float(pitch[0].get("length") or 105.0), "width": float(pitch[0].get("width") or 68.0)}


# -- tracking --------------------------------------------------------------------------------------
def _ball_of(row, source):
    if source == "smoothed":
        b = row.get("ballsSmoothed")
        return b if isinstance(b, dict) and b.get("x") is not None and b.get("y") is not None else None
    for b in row.get("balls") or []:
        if b.get("x") is not None and b.get("y") is not None:
            return b
    return None


def read_tracking(path, source, estimated, limit=None):
    """Stream the jsonl. Returns per-source-frame arrays, sparse player observations and game events."""
    fr_n, per, el = [], [], []
    bx, by, bz, best = [], [], [], []
    obs_f, obs_c, obs_x, obs_y = [], [], [], []
    cols = {}                                    # (team 0/1, jersey) -> column
    events = {}
    vis_count = collections.Counter()
    key_side = {"raw": ("homePlayers", "awayPlayers"), "smoothed": ("homePlayersSmoothed", "awayPlayersSmoothed")}
    with open_lines(path) as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            ge = row.get("game_event")
            if ge and ge.get("game_event_type"):
                k = (ge.get("game_event_type"), ge.get("start_frame"), ge.get("end_frame"))
                if k not in events:
                    ge = dict(ge)
                    ge["_id"] = row.get("game_event_id")
                    events[k] = ge
            p = row.get("period")
            if p not in (1, 2, 3, 4):
                continue
            i = len(fr_n)
            fr_n.append(int(row["frameNum"]))
            per.append(int(p))
            el.append(float(row.get("periodElapsedTime") or 0.0))
            b = _ball_of(row, source)
            vis = (b or {}).get("visibility")
            vis_count["ball_" + str(vis)] += 1
            if b is not None and (vis == "VISIBLE" or estimated in ("ball", "both")):
                bx.append(float(b["x"]))
                by.append(float(b["y"]))
                bz.append(float(b.get("z") or 0.0))
            else:
                bx.append(np.nan)
                by.append(np.nan)
                bz.append(np.nan)
            best.append(b is not None and vis != "VISIBLE")
            for team, key in enumerate(key_side[source]):
                for pl in row.get(key) or []:
                    v = pl.get("visibility")
                    vis_count["player_" + str(v)] += 1
                    if pl.get("x") is None or pl.get("y") is None or pl.get("jerseyNum") is None:
                        continue
                    if v != "VISIBLE" and estimated not in ("players", "both"):
                        continue
                    c = cols.setdefault((team, int(pl["jerseyNum"])), len(cols))
                    obs_f.append(i)
                    obs_c.append(c)
                    obs_x.append(float(pl["x"]))
                    obs_y.append(float(pl["y"]))
            if limit and len(fr_n) >= limit:
                break
    order = np.argsort(np.asarray(fr_n), kind="stable")
    if not np.all(order == np.arange(order.size)):
        raise ValueError("frames not in frameNum order in " + path)
    return {"frame_n": np.asarray(fr_n, np.int64), "period": np.asarray(per, np.int8),
            "elapsed": np.asarray(el, float), "bx": np.asarray(bx), "by": np.asarray(by), "bz": np.asarray(bz),
            "ball_est": np.asarray(best, bool), "obs": (np.asarray(obs_f, np.int64), np.asarray(obs_c, np.int64),
                                                        np.asarray(obs_x, np.float32), np.asarray(obs_y, np.float32)),
            "cols": cols, "events": list(events.values()), "vis": vis_count}


# -- event file ------------------------------------------------------------------------------------
def read_event_file(path):
    """{gameEventId: info} and {(type, start time 0.01 s): info} from the per-game event file:
    info = setpiece (gameEvents.setpieceType), foul (any fouls field set), video_missing, type."""
    with open(path, encoding="utf-8") as h:
        rows = json.load(h)
    by_id, by_time = {}, {}
    for r in rows:
        g = r.get("gameEvents") or {}
        fouls = r.get("fouls") or {}
        info = {"setpiece": g.get("setpieceType"), "type": g.get("gameEventType"),
                "foul": any(v is not None for v in fouls.values()), "video_missing": bool(g.get("videoMissing"))}
        if r.get("gameEventId") is not None:
            cur = by_id.setdefault(int(r["gameEventId"]), dict(info))
            cur["foul"] = cur["foul"] or info["foul"]          # several possession events per game event
        if r.get("startTime") is not None:
            by_time.setdefault((info["type"], round(float(r["startTime"]), 2)), info)
    return by_id, by_time


def attach_event_info(events, by_id, by_time):
    """Copy setpiece / foul / video_missing onto the tracking's game events. Returns the match count."""
    hit = 0
    for e in events:
        info = None
        if e.get("_id") is not None:
            info = by_id.get(int(e["_id"]))
        if info is None and e.get("start_time") is not None:
            info = by_time.get((e.get("game_event_type"), round(float(e["start_time"]), 2)))
        if info is not None:
            e["_setpiece"], e["_foul"], e["_video_missing"] = info["setpiece"], info["foul"], info["video_missing"]
            hit += 1
    return hit


def is_setpiece(code):
    """PFF setpieceType: open play is 'O' (or empty); anything else is a restart."""
    return code is not None and str(code).strip().upper() not in ("", "O", "NONE", "N")


# -- labels ----------------------------------------------------------------------------------------
def labels_from_events(frame_n, events, restart_at="end"):
    """(alive bool, possession int8 1/2/0, ungraded bool) per source frame, from the PFF game events."""
    n = frame_n.size
    idx = lambda f: int(np.clip(np.searchsorted(frame_n, int(f)), 0, n - 1))   # noqa: E731
    evs = sorted((e for e in events if e.get("start_frame") is not None),
                 key=lambda e: (int(e["start_frame"]), int(e.get("end_frame") or e["start_frame"])))
    points = []                                  # (source index, order, alive or None, team or None)
    restart_code = {}                            # source index where play restarts -> restart type
    alive = False
    last_end = None                              # end frame of the last touch / kick-off (the last play seen)
    stats = collections.Counter()
    for k, e in enumerate(evs):
        t = e.get("game_event_type")
        s = int(e["start_frame"])
        en = int(e.get("end_frame") or s)
        team = None if e.get("home_ball") is None else (1 if e["home_ball"] else 2)
        if t in ("OUT", "END"):
            points.append((idx(s), k, False, None))
            alive = False
        elif t in KICKOFFS:
            points.append((idx(s), k, True, team))
            restart_code.setdefault(idx(s), "K")
            alive = True
            last_end = en
        elif t == "OTB":
            sp = is_setpiece(e.get("_setpiece"))
            if sp:
                stats["setpiece_" + str(e.get("_setpiece"))] += 1
            if alive and sp and last_end is not None:
                # a restart with no OUT before it: foul, offside, goal, injury... The referee stopped play
                # after the last touch: dead from the end of that touch to the kick of the restart.
                stats["stoppage_without_out"] += 1
                stats["stoppage_with_foul"] += int(any(x.get("_foul") for x in evs[max(0, k - 3):k + 1]))
                points.append((idx(min(last_end, s)), k - 0.5, False, None))
                alive = False
            if alive:
                points.append((idx(s), k, None, team))
            else:
                j = idx(en if restart_at == "end" else s)
                points.append((j, k, True, team))
                code = str(e.get("_setpiece")).strip().upper() if sp else ("?" if e.get("_setpiece") is None else "O")
                restart_code.setdefault(j, code[:1])
                alive = True
            last_end = en
    points.sort(key=lambda x: (x[0], x[1]))
    a = np.zeros(n, bool)
    poss = np.zeros(n, np.int8)
    cur_a, cur_t, last = False, 0, 0
    for i, _k, pa, pt in points:
        a[last:i], poss[last:i] = cur_a, cur_t
        if pa is not None:
            cur_a = pa
        if pt is not None:
            cur_t = pt
        last = i
    a[last:], poss[last:] = cur_a, cur_t
    ungraded = np.zeros(n, bool)
    for e in evs:
        if e.get("game_event_type") == "VID" or e.get("_video_missing"):
            s, en = idx(e["start_frame"]), idx(e.get("end_frame") or e["start_frame"])
            ungraded[s:en + 1] = True
    # restart type of every dead spell, on its frames: the setpieceType of the touch that ends it
    # (T throw-in, C corner, G goal kick, F free kick, P penalty, D drop ball, K kick-off, O open play,
    # ? no event file), '' on alive frames
    restart = np.full(n, "", dtype="<U1")
    dead = ~a
    starts = np.flatnonzero(dead & np.r_[True, ~dead[:-1]])
    ends = np.flatnonzero(dead & np.r_[~dead[1:], True]) + 1
    for s0, e0 in zip(starts, ends):
        restart[s0:e0] = restart_code.get(int(e0), "-")         # '-' = the spell never restarts (end of half)
    labels_from_events.stats = stats
    labels_from_events.restart = restart
    return a, poss, ungraded


# -- resampling ------------------------------------------------------------------------------------
def resample_index(period, elapsed):
    """Source index for every 1/25 s of each period (nearest timestamp), and the time gap to it (s)."""
    out_idx, out_per, out_gap = [], [], []
    for p in (1, 2, 3, 4):
        src = np.flatnonzero(period == p)
        if not src.size:
            continue
        t = elapsed[src]
        tt = np.arange(0.0, t.max() + 1e-9, 1.0 / FPS_OUT)
        j = np.clip(np.searchsorted(t, tt), 1, t.size - 1)
        j = np.where(np.abs(t[j - 1] - tt) <= np.abs(t[j] - tt), j - 1, j)
        out_idx.append(src[j])
        out_per.append(np.full(tt.size, p, np.int8))
        out_gap.append(np.abs(t[j] - tt))
    return np.concatenate(out_idx), np.concatenate(out_per), np.concatenate(out_gap)


def convert(files, source="raw", estimated="players", restart_at="end", limit=None):
    """estimated: which ESTIMATED (off-camera, extrapolated) positions to keep: none / players / ball / both."""
    meta = read_meta(files["meta"])
    tr = read_tracking(files["tracking"], source, estimated, limit)
    tr["event_file_matched"] = None
    if files.get("events"):
        by_id, by_time = read_event_file(files["events"])
        tr["event_file_matched"] = (attach_event_info(tr["events"], by_id, by_time), len(tr["events"]))
    alive_s, poss_s, ung_s = labels_from_events(tr["frame_n"], tr["events"], restart_at)
    restart_s = labels_from_events.restart
    tr["label_stats"] = dict(labels_from_events.stats)
    poss_s = np.where(ung_s, 0, poss_s).astype(np.int8)
    alive_s = alive_s | ung_s                           # alive + possession 0 -> X in validation.labels_of
    sel, period, gap = resample_index(tr["period"], tr["elapsed"])
    hole = gap > MAX_GAP_S                              # no source frame near this time: nothing seen, not graded
    of, oc, ox, oy = tr["obs"]
    ns, nc = tr["frame_n"].size, len(tr["cols"])
    PX = np.full((ns, nc), np.nan, np.float32)
    PY = PX.copy()
    PX[of, oc], PY[of, oc] = ox, oy
    order = sorted(tr["cols"], key=lambda k: (k[0], k[1]))
    cidx = np.array([tr["cols"][k] for k in order], np.int64)
    nan_if = lambda a: np.where(hole if a.ndim == 1 else hole[:, None], np.nan, a)   # noqa: E731
    return meta, tr, {
        "period": period, "frame_n": tr["frame_n"][sel],
        "ball_x": nan_if(tr["bx"][sel]), "ball_y": nan_if(tr["by"][sel]), "ball_z": nan_if(tr["bz"][sel]),
        "ball_alive": alive_s[sel] | hole, "ball_possession": np.where(hole, 0, poss_s[sel]).astype(np.int8),
        "px": nan_if(PX[sel][:, cidx]).astype(np.float32), "py": nan_if(PY[sel][:, cidx]).astype(np.float32),
        "player_team": np.array([k[0] for k in order], np.int8),
        "player_jersey": np.array([k[1] for k in order], np.int16),
        "ball_estimated": tr["ball_est"][sel],
        "restart_type": np.where(hole | ung_s[sel], "", restart_s[sel])}


# -- inspect ---------------------------------------------------------------------------------------
def keys_tree(obj, depth=0, max_depth=3):
    """Names and types only, never values."""
    pad = "  " * depth
    if isinstance(obj, dict):
        lines = []
        for k, v in list(obj.items())[:60]:
            lines.append("{}{}: {}".format(pad, k, type(v).__name__))
            if depth < max_depth and isinstance(v, (dict, list)):
                lines += keys_tree(v, depth + 1, max_depth)
        return lines
    if isinstance(obj, list) and obj:
        return ["{}[{} items] of {}".format(pad, len(obj), type(obj[0]).__name__)] + keys_tree(obj[0], depth + 1, max_depth)
    return []


def inspect(games, gid, root, limit):
    f = games[gid]
    print("game", gid, {k: os.path.relpath(v, root) for k, v in f.items()})
    meta = read_meta(f["meta"])
    print("meta:", {k: meta[k] for k in ("fps", "length", "width")})
    for source, est in (("raw", "players"), ("raw", "none")):
        t0 = time.time()
        meta, tr, out = convert(f, source=source, estimated=est, limit=limit)
        g = np.isin(out["period"], (1, 2))
        types = collections.Counter(e.get("game_event_type") for e in tr["events"])
        print("\n== source: {}, ESTIMATED kept: {} ({:.0f} s)".format(source, est, time.time() - t0))
        print("visibility counts:", dict(tr["vis"]))
        print("frames 25 fps graded: {}   ball VISIBLE: {:.1%}   players VISIBLE per frame (median): {:.0f}".format(
            int(g.sum()), float(np.mean(~np.isnan(out["ball_x"][g]))),
            float(np.median((~np.isnan(out["px"][g])).sum(1)))))
        lab = np.where(~out["ball_alive"], "D", np.where(out["ball_possession"] == 1, "H",
                                                         np.where(out["ball_possession"] == 2, "A", "X")))
        c = collections.Counter(lab[g])
        print("labels on graded frames: D {:.1%}  H {:.1%}  A {:.1%}  X {:.1%}  (IDSSE DFL: D about 38-40%)".format(
            *(c[k] / max(1, g.sum()) for k in "DHAX")))
        print("dead spells:", int(np.sum(np.diff(np.r_[1, out["ball_alive"][g].astype(int)]) == -1)),
              "  game event types:", dict(types))
        ko = [e for e in tr["events"] if e.get("game_event_type") in KICKOFFS]
        dist = []
        for e in ko:
            i = np.searchsorted(tr["frame_n"], int(e["start_frame"]))
            if i < tr["frame_n"].size and not np.isnan(tr["bx"][i]):
                dist.append(float(np.hypot(tr["bx"][i], tr["by"][i])))
        print("ball distance from the centre spot at the kick-offs (m, should be ~0):", [round(x, 1) for x in dist])
    print("\nevent file:", os.path.relpath(f["events"], root) if f.get("events") else "NOT FOUND (fouls not dead)")
    if tr.get("event_file_matched"):
        print("tracking game events matched in the event file: {} of {}".format(*tr["event_file_matched"]))
    print("label building:", tr.get("label_stats"))
    print("\nstructure and counts only.")


# -- main ------------------------------------------------------------------------------------------
def write_config(out_dir, games):
    import yaml
    with open(os.path.join(HMM_DIR, "config", "validation.yaml"), encoding="utf-8") as h:
        vc = yaml.safe_load(h)
    vc.update({"data_dir": os.path.abspath(out_dir), "matches": list(games), "variants": ["clean"],
               "out_dir": "outputs/validation_" + os.path.basename(os.path.normpath(out_dir))})
    path = os.path.join(out_dir, "validation_{}.yaml".format(os.path.basename(os.path.normpath(out_dir))))
    with open(path, "w", encoding="utf-8") as h:
        h.write("# generated by validation/fetch_pff.py from config/validation.yaml: PFF World Cup 2022.\n"
                "# 'clean' here = the PFF broadcast tracking as given (already degraded): never run 'degraded'.\n")
        yaml.safe_dump(vc, h, sort_keys=False, allow_unicode=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.join(REPO, "data", "pff", "raw"), help="the unzipped PFF download")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "pff"))
    ap.add_argument("--games", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=8, help="without --games: the first N game ids (sorted)")
    ap.add_argument("--source", choices=("raw", "smoothed"), default="raw")
    ap.add_argument("--estimated", choices=("players", "none", "ball", "both"), default="players",
                    help="ESTIMATED positions kept. players (default): like BAR-ATH's tactical tracking, players almost "
                         "always there, the ball lost often; none: the broadcast worst case (use another --out)")
    ap.add_argument("--restart-at", choices=("end", "start"), default="end")
    ap.add_argument("--force", action="store_true", help="convert again games already in --out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N tracking frames (tests)")
    args = ap.parse_args()
    games = {g: f for g, f in scan(args.root).items() if {"tracking", "meta", "roster"} <= set(f) or
             {"tracking", "meta"} <= set(f)}
    if not games:
        print("no PFF game found under", args.root, "(need <id>.jsonl.bz2 + metadata <id>.json)")
        return 1
    ids = sorted(games, key=int)
    if args.list:
        print(len(ids), "games:", " ".join(ids))
        return 0
    chosen = args.games or ids[:args.n]
    if args.inspect:
        inspect(games, chosen[0], args.root, args.limit)
        return 0
    os.makedirs(args.out, exist_ok=True)
    done = []
    for gid in chosen:
        target = os.path.join(args.out, "{}.npz".format(gid))
        if os.path.isfile(target) and not args.force:
            print(gid, "already converted")
            done.append(gid)
            continue
        t0 = time.time()
        meta, _tr, out = convert(games[gid], args.source, args.estimated, args.restart_at, args.limit)
        np.savez_compressed(target, pitch_x=meta["length"], pitch_y=meta["width"], match_id=gid,
                            source="pff_{}_est-{}".format(args.source, args.estimated), **out)
        g = np.isin(out["period"], (1, 2))
        print("{} {} v {}: {} frames, {} players, ball seen {:.0%}, D {:.0%}, stoppages without OUT {}, "
              "event file {}, {:.0f} s".format(
                  gid, meta["home"], meta["away"], int(g.sum()), out["px"].shape[1],
                  float(np.mean(~np.isnan(out["ball_x"][g]))), float(np.mean(~out["ball_alive"][g])),
                  _tr["label_stats"].get("stoppage_without_out", 0),
                  "{}/{} matched".format(*_tr["event_file_matched"]) if _tr["event_file_matched"] else "MISSING",
                  time.time() - t0), flush=True)
        done.append(gid)
    print("config:", write_config(args.out, done))
    print("done: pff ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

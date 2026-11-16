"""FIFA / EPTS tracking export (Metrica's format): a metadata XML and a raw text file, one line
per frame. This is the format of the provided ATD and of Metrica's Sample Game 3, so the same
parser serves both. Standard library + numpy only (no kloppy).

The metadata gives, per block of frames, the exact layout of a line (DataFormatSpecification):
nested registers with separators, each leaf a player or ball channel. The parser follows that
specification, so any channel order and any extra channel (speed, z) is handled without a
special case.

Coordinates in the file are normalised to [0, 1] of the pitch; they become metres centred on
the centre spot. The first team of the metadata is the home team (the convention of the starter
code and of kloppy for the provided match).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..match import Match, X


def _strip(tag):
    return tag.split("}")[-1]


def _iter(el, name):
    for child in el.iter():
        if _strip(child.tag) == name:
            yield child


def _text(el, name, default=None):
    for child in _iter(el, name):
        return child.text
    return default


def read_metadata(path):
    root = ET.parse(path).getroot()
    fps = float(_text(root, "FrameRate", "25"))
    pitch = (105.0, 68.0)
    for fs in _iter(root, "FieldSize"):
        w, h = _text(fs, "Width"), _text(fs, "Height")
        if w and h:
            pitch = (float(w), float(h))
    teams = []
    for t in _iter(root, "Team"):
        tid = t.get("id")
        if tid is None:
            continue
        attack = None
        for pp in _iter(t, "ProviderParameter"):
            if _text(pp, "Name") == "attack_direction_first_half":
                attack = _text(pp, "Value")
        teams.append({"id": tid, "name": _text(t, "Name", tid), "attack_first_half": attack})
    players = []
    for p in _iter(root, "Player"):
        pid, tid = p.get("id"), p.get("teamId")
        if pid is None:
            continue
        pos = None
        for pp in _iter(p, "ProviderParameter"):
            if _text(pp, "Name") == "position_type":
                pos = _text(pp, "Value")
        players.append({"id": pid, "team_id": tid, "position": pos,
                        "shirt": _text(p, "ShirtNumber")})
    channels = {}
    for pc in _iter(root, "PlayerChannel"):
        channels[pc.get("id")] = (pc.get("playerId"), pc.get("channelId"))
    periods = {}
    for pp in _iter(root, "ProviderParameter"):
        n, v = _text(pp, "Name"), _text(pp, "Value")
        if n and v and n.endswith(("_start", "_end")) and "half" in n:
            try:
                periods[n] = int(float(v))
            except ValueError:
                pass
    specs = []
    for spec in _iter(root, "DataFormatSpecification"):
        specs.append({"start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
                      "sep": spec.get("separator", ":"), "node": spec})
    return {"fps": fps, "pitch": pitch, "teams": teams, "players": players,
            "channels": channels, "periods": periods, "specs": specs}


def _layout(node):
    """Flatten a register node into a nested structure of (separator, children) and leaves."""
    kids = []
    for child in list(node):
        tag = _strip(child.tag)
        if tag == "SplitRegister":
            kids.append(("split", child.get("separator"), _layout(child)))
        elif tag == "PlayerChannelRef":
            kids.append(("pchan", child.get("playerChannelId")))
        elif tag == "BallChannelRef":
            kids.append(("ball", child.get("channelId")))
        elif tag == "StringRegister":
            kids.append(("string", child.get("name")))
        elif tag == "SplitRegister" or tag.endswith("Register"):
            kids.append(("other", None))
    return kids


def _compile(spec, channels, player_col):
    """Return a function line -> (frame_id, player_values dict col->(x,y), ball (x,y))."""
    layout = _layout(spec["node"])
    top_sep = spec["sep"]

    def parse_kids(kids, sep, text, out):
        parts = text.split(sep) if sep else [text]
        # a SplitRegister with children [A, B, C] matched one by one to the parts
        for i, kid in enumerate(kids):
            part = parts[i] if i < len(parts) else ""
            kind = kid[0]
            if kind == "split":
                parse_kids(kid[2], kid[1], part, out)
            elif kind == "string":
                out["_" + kid[1]] = part
            elif kind == "pchan":
                pid, ch = channels.get(kid[1], (None, kid[1]))
                out[("p", pid, ch)] = part
            elif kind == "ball":
                out[("b", kid[1])] = part

    def parse_line(line):
        out = {}
        parse_kids(layout, top_sep, line.rstrip("\n\r"), out)
        return out

    return parse_line


def load_epts(meta_path, raw_path, name=None, period_of_frame=None, cache=None,
              max_players=None):
    """Read an EPTS pair into a Match.

    period_of_frame(frame_id) -> period (0 = outside the match) may be given to impose the halves
    (the provided match: games.half_video_blocks). Otherwise the metadata half boundaries are
    used. `cache`: an .npz path; written after the first parse, read afterwards.
    """
    meta = read_metadata(meta_path)
    name = name or os.path.splitext(os.path.basename(raw_path))[0]
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        arrays = {k: z[k] for k in z.files}
    else:
        arrays = _parse_raw(meta, raw_path)
        if cache:
            np.savez_compressed(cache, **arrays)
    frame_id = arrays["frame_id"]
    T = len(frame_id)
    if period_of_frame is None:
        pr = meta["periods"]
        period = np.zeros(T, dtype=np.int16)
        for k, (a, b) in enumerate([("first_half_start", "first_half_end"),
                                    ("second_half_start", "second_half_end"),
                                    ("first_extra_half_start", "first_extra_half_end"),
                                    ("second_extra_half_start", "second_extra_half_end")], 1):
            if a in pr and b in pr:
                period[(frame_id >= pr[a]) & (frame_id <= pr[b])] = k
    else:
        period = np.array([period_of_frame(int(f)) for f in frame_id], dtype=np.int16)
    L, W = meta["pitch"]
    px = (arrays["px"] - 0.5) * L
    py = (0.5 - arrays["py"]) * W
    ball = np.stack([(arrays["bx"] - 0.5) * L, (0.5 - arrays["by"]) * W], axis=1)
    team = arrays["team"]
    attack = None
    if meta["teams"] and meta["teams"][0].get("attack_first_half"):
        d = meta["teams"][0]["attack_first_half"]
        s = 1 if d == "left_to_right" else -1
        attack = {1: s, 2: -s, 3: s, 4: -s}
    extra = {"player_ids": arrays["player_ids"], "positions": arrays["positions"],
             "frame_id": frame_id, "meta_attack": attack}
    m = Match(name, period, ball, px, py, team, meta["pitch"], labels=None,
              frame_index=frame_id - 1, fps=meta["fps"], extra=extra)
    if attack is not None:
        for p, s in attack.items():
            if p in m.attack and m.attack[p] != s:
                print("warning: {} period {}: inferred attack direction {} disagrees with the "
                      "metadata {}; the metadata is used".format(name, p, m.attack[p], s))
                m.attack[p] = s
    return m


def _parse_raw(meta, raw_path):
    players = meta["players"]
    team_ids = [t["id"] for t in meta["teams"]]
    pid_col = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([team_ids.index(p["team_id"]) if p["team_id"] in team_ids else -1
                     for p in players], dtype=np.int8)
    P = len(players)
    specs = sorted(meta["specs"], key=lambda s: s["start"])
    parsers = [(s["start"], s["end"], _compile(s, meta["channels"], pid_col)) for s in specs]
    frames, PX, PY, BX, BY = [], [], [], [], []
    k = 0
    with open(raw_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fid = int(line[:line.index(":")])
            while k < len(parsers) - 1 and fid > parsers[k][1]:
                k += 1
            parse = parsers[k][2]
            d = parse(line)
            row_x = np.full(P, np.nan, dtype=np.float32)
            row_y = np.full(P, np.nan, dtype=np.float32)
            bx = by = np.nan
            for key, val in d.items():
                if not isinstance(key, tuple):
                    continue
                if key[0] == "p":
                    _, pid, ch = key
                    if pid in pid_col and ch in ("x", "y"):
                        try:
                            v = float(val)
                        except ValueError:
                            v = np.nan
                        if ch == "x":
                            row_x[pid_col[pid]] = v
                        else:
                            row_y[pid_col[pid]] = v
                elif key[0] == "b":
                    ch = key[1]
                    try:
                        v = float(val)
                    except ValueError:
                        v = np.nan
                    if ch.endswith("x"):
                        bx = v
                    elif ch.endswith("y"):
                        by = v
            frames.append(fid)
            PX.append(row_x)
            PY.append(row_y)
            BX.append(bx)
            BY.append(by)
    return {"frame_id": np.array(frames, dtype=np.int64), "px": np.array(PX), "py": np.array(PY),
            "bx": np.array(BX, dtype=np.float64), "by": np.array(BY, dtype=np.float64),
            "team": team, "player_ids": np.array([p["id"] for p in players]),
            "positions": np.array([p["position"] or "" for p in players])}

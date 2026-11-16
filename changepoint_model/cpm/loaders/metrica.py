"""Metrica Sports sample data (github.com/metrica-sports/sample-data).

Games 1 and 2: two CSV tracking files (home / away) + an events CSV. Game 3: EPTS (metadata XML +
raw text) + an events JSON. Coordinates normalised to [0, 1] of a 105 x 68 pitch. Labels are
built from the events by `cpm.labels.labels_from_events`, the same mechanical rule for every game.
"""

import csv
import json
import os

import numpy as np

from ..match import Match
from ..labels import labels_from_events
from .epts import load_epts

PITCH = (105.0, 68.0)


def _read_tracking_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header_team, header_num, header = rows[0], rows[1], rows[2]
    data = rows[3:]
    # columns: Period, Frame, Time, then pairs (x, y) per player, then Ball x, y
    names = []
    for i in range(3, len(header), 2):
        names.append(header[i])
    arr = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r] for r in data],
                   dtype=np.float64)
    period = arr[:, 0].astype(int)
    frame = arr[:, 1].astype(int)
    xy = arr[:, 3:]
    n_pairs = xy.shape[1] // 2
    ball = xy[:, 2 * (n_pairs - 1):2 * n_pairs]
    players = xy[:, :2 * (n_pairs - 1)]
    px = players[:, 0::2]
    py = players[:, 1::2]
    return period, frame, px, py, ball, names[:-1]


def load_metrica_csv(folder, game):
    """folder/Sample_Game_<g>/..., game = 1 or 2."""
    stem = os.path.join(folder, "Sample_Game_{}".format(game), "Sample_Game_{}_".format(game))
    ph, fh_, pxh, pyh, ballh, nh = _read_tracking_csv(stem + "RawTrackingData_Home_Team.csv")
    pa, fa, pxa, pya, balla, na = _read_tracking_csv(stem + "RawTrackingData_Away_Team.csv")
    assert np.array_equal(fh_, fa), "home and away tracking files differ in frames"
    L, W = PITCH
    px = np.concatenate([pxh, pxa], axis=1)
    py = np.concatenate([pyh, pya], axis=1)
    team = np.array([0] * pxh.shape[1] + [1] * pxa.shape[1], dtype=np.int8)
    ball = np.where(np.isnan(ballh), balla, ballh)
    px = (px - 0.5) * L
    py = (0.5 - py) * W
    ball = np.stack([(ball[:, 0] - 0.5) * L, (0.5 - ball[:, 1]) * W], axis=1)
    events = _read_events_csv(stem + "RawEventsData.csv")
    period = ph.astype(np.int16)
    m = Match("metrica_g{}".format(game), period, ball, px, py, team, PITCH,
              frame_index=fh_ - 1)
    m.labels = labels_from_events(m, events)
    return m


def _read_events_csv(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({"team": 0 if r["Team"].strip().lower() == "home" else 1,
                        "type": r["Type"].strip().upper(),
                        "subtype": (r["Subtype"] or "").strip().upper(),
                        "period": int(r["Period"]),
                        "start": int(r["Start Frame"]), "end": int(r["End Frame"])})
    return out


def _read_events_json(path, team_ids):
    with open(path) as fh:
        data = json.load(fh)["data"]
    out = []
    for e in data:
        tid = (e.get("team") or {}).get("id")
        if tid not in team_ids:
            continue
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        else:
            sub = ""
        out.append({"team": team_ids.index(tid), "type": e["type"]["name"].upper(),
                    "subtype": (sub or "").upper(), "period": int(e["period"]),
                    "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"])})
    return out


def load_metrica_epts(folder, game=3, cache=None):
    stem = os.path.join(folder, "Sample_Game_{}".format(game), "Sample_Game_{}_".format(game))
    from .epts import read_metadata
    meta = read_metadata(stem + "metadata.xml")
    m = load_epts(stem + "metadata.xml", stem + "tracking.txt", name="metrica_g{}".format(game),
                  cache=cache)
    team_ids = [t["id"] for t in meta["teams"]]
    events = _read_events_json(stem + "events.json", team_ids)
    m.labels = labels_from_events(m, events)
    return m

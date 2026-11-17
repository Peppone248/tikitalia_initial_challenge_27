"""Download the IDSSE open dataset (DFL / Sportec, Bundesliga 2022/23, CC-BY 4.0) and convert each
match to a compact .npz for the validation of HMM_first_imp. Run it on a machine with internet:

    python sequence_model/scripts/fetch_idsse.py                 # all 7 matches -> sequence_model/data/idsse/
    python sequence_model/scripts/fetch_idsse.py --matches J03WPY --keep-raw

Needs only the Python standard library + numpy (no kloppy): the positions XML (hundreds of MB per
match) is parsed as a stream, so memory stays low. The raw XML is deleted after conversion unless
--keep-raw. Output goes under data/ (gitignored): the data is CC-BY but must not end up in git.

Source: Bassek, Rein, Weber, Memmert, "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y
Mirror: https://huggingface.co/datasets/pysport/idsse-data

What goes into <match>.npz (per frame of the positions file, both halves back to back):
    period          1 / 2 (extra time sections, if any, as 3 / 4)
    frame_n         DFL frame number N
    ball_x/y/z      metres, DFL coordinates (origin at the centre spot)
    ball_alive      BallStatus == 1          -> the label: D where False
    ball_possession BallPossession 1 = home, 2 = away, 0 = missing -> the label: H / A
    px, py          (frames, players) metres, NaN where the player is not on the pitch
    player_team     (players,) 0 = home, 1 = away
    pitch_x/pitch_y pitch size in metres (match information), 0 if not found
"""

import argparse
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_ROOT = os.path.dirname(HERE)  # sequence_model/
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
COMPETITION = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
FILES = {
    "meta": "DFL_02_01_matchinformation_{c}_DFL-MAT-{m}.xml",
    "tracking": "DFL_04_03_positions_raw_observed_{c}_DFL-MAT-{m}.xml",
}
SECTION = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def download(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print("  already there:", os.path.basename(path))
        return path
    tmp = path + ".part"
    for attempt in range(3):
        try:
            print("  downloading", os.path.basename(path), flush=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (50 << 20) < (1 << 20):
                        print("    {:.0f} MB ({:.0f} s)".format(total / 2 ** 20, time.time() - t0),
                              flush=True)
            os.replace(tmp, path)
            return path
        except Exception as error:  # network hiccup: retry
            print("    failed ({}), retry {}/3".format(error, attempt + 1))
            time.sleep(5)
    raise RuntimeError("could not download " + url)


def read_meta(path):
    """{team_id: 0 home / 1 away}, pitch (x, y)."""
    root = ET.parse(path).getroot()
    teams, pitch = {}, (0.0, 0.0)
    for el in root.iter():
        role = (el.get("Role") or "").lower()
        if el.tag == "Team" and el.get("TeamId") and role in ("home", "guest", "away"):
            teams[el.get("TeamId")] = 0 if role == "home" else 1
        if el.get("PitchX") and el.get("PitchY"):
            pitch = (float(el.get("PitchX")), float(el.get("PitchY")))
    if len(teams) != 2:
        raise ValueError("match information: expected 2 teams with Role home/guest, got {}".format(teams))
    return teams, pitch


def read_positions(path, teams):
    """Stream the positions XML. Returns per-object dicts of frame arrays."""
    ball = {}                       # period -> lists
    players = {}                    # (person_id) -> {"team": t, "frames": {period: [N, X, Y]}}
    cur = None
    n_sets = 0
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            period = SECTION.get(el.get("GameSection"))
            team_id, person = el.get("TeamId"), el.get("PersonId")
            if period is None:
                cur = None
            elif team_id == "BALL":
                cur = ("ball", period, ball.setdefault(period, {"N": [], "X": [], "Y": [], "Z": [],
                                                              "P": [], "S": []}))
            elif team_id in teams and person:
                p = players.setdefault(person, {"team": teams[team_id], "frames": {}})
                cur = ("player", period, p["frames"].setdefault(period, {"N": [], "X": [], "Y": []}))
            else:
                cur = None                                  # referees and other objects
            n_sets += 1
        elif event == "end" and el.tag == "Frame":
            if cur is not None:
                kind, period, store = cur
                store["N"].append(int(el.get("N")))
                store["X"].append(float(el.get("X", "nan")))
                store["Y"].append(float(el.get("Y", "nan")))
                if kind == "ball":
                    store["Z"].append(float(el.get("Z", "nan")))
                    store["P"].append(int(el.get("BallPossession", "0") or 0))
                    store["S"].append(int(el.get("BallStatus", "0") or 0))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            el.clear()
    print("  framesets: {}, players: {}, periods: {}".format(n_sets, len(players), sorted(ball)))
    return ball, players


def to_arrays(ball, players):
    periods = sorted(ball)
    starts, frame_n, period_of = {}, [], []
    offset = 0
    for per in periods:
        n = np.array(ball[per]["N"])
        lo, hi = int(n.min()), int(n.max())
        starts[per] = (lo, offset)
        frame_n.append(np.arange(lo, hi + 1))
        period_of.append(np.full(hi - lo + 1, per, dtype=np.int8))
        offset += hi - lo + 1
    T = offset
    out = {"period": np.concatenate(period_of), "frame_n": np.concatenate(frame_n)}
    bx = np.full(T, np.nan)
    by, bz = bx.copy(), bx.copy()
    alive = np.zeros(T, dtype=bool)
    poss = np.zeros(T, dtype=np.int8)
    for per in periods:
        lo, off = starts[per]
        b = ball[per]
        idx = np.array(b["N"]) - lo + off
        bx[idx], by[idx], bz[idx] = b["X"], b["Y"], b["Z"]
        alive[idx] = np.array(b["S"]) == 1
        poss[idx] = b["P"]
    ids = sorted(players, key=lambda k: (players[k]["team"], k))
    px = np.full((T, len(ids)), np.nan, dtype=np.float32)
    py = px.copy()
    for j, pid in enumerate(ids):
        for per, fr in players[pid]["frames"].items():
            if per not in starts:
                continue
            lo, off = starts[per]
            idx = np.array(fr["N"]) - lo + off
            ok = (idx >= off) & (idx < T)
            px[idx[ok], j] = np.array(fr["X"])[ok]
            py[idx[ok], j] = np.array(fr["Y"])[ok]
    out.update(ball_x=bx, ball_y=by, ball_z=bz, ball_alive=alive, ball_possession=poss, px=px,
               py=py, player_team=np.array([players[p]["team"] for p in ids], dtype=np.int8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="+", default=sorted(COMPETITION))
    ap.add_argument("--out", default=os.path.join(SEQ_ROOT, "data", "idsse"))
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for m in args.matches:
        target = os.path.join(args.out, "{}.npz".format(m))
        if os.path.isfile(target):
            print(m, "already converted:", target)
            continue
        print(m)
        c = COMPETITION[m]
        paths = {k: download(BASE_URL.format(v.format(c=c, m=m)),
                             os.path.join(raw_dir, v.format(c=c, m=m))) for k, v in FILES.items()}
        teams, pitch = read_meta(paths["meta"])
        t0 = time.time()
        ball, players = read_positions(paths["tracking"], teams)
        arrays = to_arrays(ball, players)
        np.savez_compressed(target, pitch_x=pitch[0], pitch_y=pitch[1], match_id=m, **arrays)
        alive = arrays["ball_alive"]
        print("  -> {} frames, {} players, alive {:.1%}, parsed in {:.0f} s: {}".format(
            alive.size, arrays["px"].shape[1], alive.mean(), time.time() - t0, target))
        if not args.keep_raw:
            os.remove(paths["tracking"])
    print("done. Tell Claude 'idsse ready'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

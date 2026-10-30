"""From the raw files to a Match: arrays in metres, cleaned ball, derived speeds, Smart Tagging.

Only numpy and the standard library, like starter/baseline.py, so it runs in the scorer's
environment. Parsing the raw ATD takes well under a minute; the result can be cached to an .npz
under outputs/ (gitignored -- frame-level derived data must never be committed, RULES.md).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1
FIELD_SEP, GROUP_SEP = ":", ";"


class Match(object):
    """Everything the rules read, frame-indexed by VIDEO frame (0 .. n-1).

    ball_x, ball_y      metres, NaN where not visible (after cleaning)
    ball_vx, ball_vy    m/s from the smoothed track, NaN at the edges of visible stretches
    ball_speed          m/s
    ball_still_for      seconds the ball has been visible and slower than V_BALL_STILL (0 if not)
    px, py              (n, n_slots) player positions in metres, NaN where missing
    team_of_slot        (n_slots,) HOME/AWAY
    half                (n,) 1 / 2 on graded frames, 0 elsewhere
    smart_tagging       list of dicts: code, start_frame, end_frame, team (0/1/None), tags
    """

    def __init__(self, fps, ball_x, ball_y, px, py, team_of_slot, half, smart_tagging=None,
                 team_names=("home", "away")):
        self.fps = float(fps)
        self.ball_x = np.asarray(ball_x, dtype=float)
        self.ball_y = np.asarray(ball_y, dtype=float)
        self.px = np.asarray(px, dtype=float)
        self.py = np.asarray(py, dtype=float)
        self.team_of_slot = np.asarray(team_of_slot, dtype=np.int8)
        self.half = np.asarray(half, dtype=np.int8)
        self.smart_tagging = smart_tagging or []
        self.team_names = tuple(team_names)
        self.n = len(self.ball_x)
        self.ball_vx = self.ball_vy = self.ball_speed = None
        self.ball_still_for = None
        self.diagnostics = {}

    @property
    def ball_visible(self):
        return ~np.isnan(self.ball_x)

    @property
    def graded(self):
        return self.half > 0


# -- raw files -------------------------------------------------------------------------------------
def team_of_each_track(meta_path):
    """HOME/AWAY per track slot; first team in the metadata is home (as in starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = tuple(team.findtext("Name") for team in root.iter("Team"))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8), names


def parse_raw_atd(raw_path, n_frames, n_slots, pitch):
    """(ball_x, ball_y, px, py) in metres. Raw line: "<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,s>"."""
    length, width = pitch.length, pitch.width
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                       # frame_id counts from 1, video from 0
            values = np.array(parts[1].replace(GROUP_SEP, ",").split(","), dtype=float)
            if values.size == 3 * n_slots:
                values = values.reshape(n_slots, 3)
                px[i] = values[:, 0] * length
                py[i] = values[:, 1] * width
            ball = parts[2].split(",")
            bx[i] = float(ball[0]) * length
            by[i] = float(ball[1]) * width
    return bx, by, px, py


def load_smart_tagging(path, fps, team_names):
    """Smart Tagging intervals in video frames, Team mapped to HOME/AWAY (None for N/A)."""
    lookup = {name.strip().lower(): side for side, name in enumerate(team_names)}
    out = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team = (tags.get("Team") or "").strip().lower()
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(instance.findtext("start")) * fps)),
            "end_frame": int(round(float(instance.findtext("end")) * fps)),
            "team": lookup.get(team),
            "type": (tags.get("Type") or "").strip().lower(),
            "tags": tags,
        })
    return out


def half_array(n_frames, blocks):
    """1 / 2 on the graded blocks [(half, first, last)], 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in blocks:
        half[first:last + 1] = h
    return half


# -- ball ------------------------------------------------------------------------------------------
def runs(mask):
    """[(start, end_exclusive)] of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def remove_jumps(bx, by, fps, v_jump):
    """NaN out isolated points whose speed from BOTH visible neighbours exceeds v_jump."""
    idx = np.flatnonzero(~np.isnan(bx))
    if idx.size < 3:
        return bx, by, 0
    step = np.hypot(np.diff(bx[idx]), np.diff(by[idx])) / (np.diff(idx) / fps)
    v_prev = np.concatenate(([0.0], step))
    v_next = np.concatenate((step, [0.0]))
    spike = (v_prev > v_jump) & (v_next > v_jump)
    bx, by = bx.copy(), by.copy()
    bx[idx[spike]] = np.nan
    by[idx[spike]] = np.nan
    return bx, by, int(spike.sum())


def smooth_and_differentiate(bx, by, fps, window):
    """Centred moving average on each visible stretch, then velocity by central differences.

    Returns (sx, sy, vx, vy, residuals): residuals are raw - smoothed on stretches long enough
    for the average, used to estimate the position jitter.
    """
    n = len(bx)
    sx, sy = np.full(n, np.nan), np.full(n, np.nan)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    residuals = []
    half_w = int(window) // 2
    kernel = np.ones(2 * half_w + 1) / (2 * half_w + 1)
    for start, end in runs(~np.isnan(bx)):
        seg_x, seg_y = bx[start:end], by[start:end]
        if end - start >= len(kernel):
            # 'same' with edge padding keeps the length; the edges are less smoothed, which is fine
            pad_x = np.pad(seg_x, half_w, mode="edge")
            pad_y = np.pad(seg_y, half_w, mode="edge")
            sx[start:end] = np.convolve(pad_x, kernel, mode="valid")
            sy[start:end] = np.convolve(pad_y, kernel, mode="valid")
            inner = slice(half_w, end - start - half_w)
            residuals.append(seg_x[inner] - sx[start:end][inner])
            residuals.append(seg_y[inner] - sy[start:end][inner])
        else:
            sx[start:end], sy[start:end] = seg_x, seg_y
        if end - start >= 3:
            vx[start:end] = np.gradient(sx[start:end]) * fps
            vy[start:end] = np.gradient(sy[start:end]) * fps
    residuals = np.concatenate(residuals) if residuals else np.array([])
    return sx, sy, vx, vy, residuals


def robust_sigma(values):
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def seconds_still(speed, visible, fps, v_still):
    """Seconds the ball has been visible and slower than v_still, as a running count."""
    still = visible & (np.nan_to_num(speed, nan=np.inf) < v_still)
    out = np.zeros(len(speed))
    for start, end in runs(still):
        out[start:end] = np.arange(1, end - start + 1) / fps
    return out


def prepare_ball(match, params):
    """Clean the ball, derive velocity, estimate SIGMA_POS. Fills match in place."""
    bx, by, n_jumps = remove_jumps(match.ball_x, match.ball_y, match.fps, params.V_JUMP)
    _, _, vx, vy, residuals = smooth_and_differentiate(bx, by, match.fps, params.SMOOTH_WIN)
    match.ball_x, match.ball_y = bx, by                # rules use raw (cleaned) positions
    match.ball_vx, match.ball_vy = vx, vy
    match.ball_speed = np.hypot(vx, vy)
    match.ball_still_for = seconds_still(match.ball_speed, match.ball_visible, match.fps,
                                         params.V_BALL_STILL)
    sigma = robust_sigma(residuals)
    sigma = max(sigma, params.SIGMA_POS_FLOOR) if np.isfinite(sigma) else params.SIGMA_POS_FLOOR
    params.set_estimated("SIGMA_POS", sigma)
    visible_graded = match.ball_visible & match.graded
    still_share = float(np.mean(match.ball_still_for[visible_graded] > 0)) \
        if visible_graded.any() else float("nan")
    match.diagnostics.update({"ball_jumps_removed": n_jumps, "sigma_pos_m": sigma,
                              "still_share": still_share})
    return match


# -- entry point -----------------------------------------------------------------------------------
def load_match(pitch, params, game="BAR-ATH", data_dir=None, cache_path=None):
    """Read the tactical ATD + Smart Tagging of a game into a prepared Match."""
    from games import GAMES, half_video_blocks, require, source_frames

    meta_path, raw_path, st_path = require(["atd_meta", "atd_raw", "smart_tagging"], game,
                                           data_dir)
    fps = GAMES[game]["fps"]
    n_frames = source_frames("tactical", game)
    team_of_slot, names = team_of_each_track(meta_path)

    if cache_path and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        bx, by, px, py = cached["bx"], cached["by"], cached["px"], cached["py"]
    else:
        bx, by, px, py = parse_raw_atd(raw_path, n_frames, len(team_of_slot), pitch)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, bx=bx, by=by, px=px, py=py)

    match = Match(fps, bx, by, px, py, team_of_slot,
                  half_array(n_frames, half_video_blocks(game)),
                  load_smart_tagging(st_path, fps, names), names)
    return prepare_ball(match, params)

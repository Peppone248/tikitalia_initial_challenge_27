"""IDSSE matches from the .npz written by HMM_first_imp/validation/fetch_idsse.py (or the copy in
sequence_model/scripts/fetch_idsse.py): DFL coordinates are already metres with the origin at the
centre spot. Labels: D where BallStatus is not 'alive', else H / A from BallPossession."""

import os

import numpy as np

from ..match import Match, H, A, D, X


def load_idsse(path, name=None):
    z = np.load(path)
    period = z["period"].astype(np.int16)
    ball = np.stack([z["ball_x"], z["ball_y"]], axis=1).astype(np.float64)
    px, py = z["px"], z["py"]
    team = z["player_team"].astype(np.int8)
    pitch = (float(z["pitch_x"]) or 105.0, float(z["pitch_y"]) or 68.0)
    alive = z["ball_alive"].astype(bool)
    poss = z["ball_possession"].astype(int)
    labels = np.full(len(period), X, dtype=np.int8)
    labels[~alive] = D
    labels[alive & (poss == 1)] = H
    labels[alive & (poss == 2)] = A
    labels[period <= 0] = X
    frame_index = z["frame_n"] if "frame_n" in z else None
    extra = {"ball_z": z["ball_z"]} if "ball_z" in z else {}
    extra["source"] = str(z["source"]) if "source" in z.files else "idsse"
    return Match(name or os.path.splitext(os.path.basename(path))[0], period, ball, px, py, team,
                 pitch, labels=labels, frame_index=frame_index, extra=extra)

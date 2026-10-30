"""Pitch geometry, vectorised. Coordinates in metres: x in [0, length], y in [0, width].

Nothing here depends on which team attacks where, except the methods taking `attack_dir`.
attack_dir is computed from the data (events.attack_directions), never hard-coded.
"""

import numpy as np

from .config import load_geometry

HOME, AWAY = 0, 1


class Pitch(object):

    def __init__(self, geometry):
        self.length = float(geometry["pitch"]["length"])
        self.width = float(geometry["pitch"]["width"])
        self.goal_half_width = float(geometry["goal"]["width"]) / 2.0
        self.box_depth = float(geometry["penalty_box"]["depth"])
        self.box_half_width = float(geometry["penalty_box"]["width"]) / 2.0
        self.centre_radius = float(geometry["centre_circle"]["radius"])

    @classmethod
    def from_config(cls, config_dir=None):
        return cls(load_geometry(config_dir))

    @property
    def centre(self):
        return self.length / 2.0, self.width / 2.0

    # -- boundaries ------------------------------------------------------------------------------
    def dist_outside(self, x, y):
        """Signed distance from the pitch boundary: > 0 outside (how far), < 0 inside (how deep).

        Outside, this is the distance beyond the furthest-crossed line (Chebyshev-like, which is
        what 'how far over the line' means for a ball near a corner).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.maximum.reduce([-x, x - self.length, -y, y - self.width])

    def crossed_line(self, x, y):
        """'goal' if the ball is beyond a goal line, 'touch' if only beyond a touch line, '' inside.

        Beyond both (corner area): the one crossed by more.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        over_goal = np.maximum(-x, x - self.length)
        over_touch = np.maximum(-y, y - self.width)
        out = np.full(x.shape, "", dtype="<U5")
        out[(over_goal > 0) & (over_goal >= over_touch)] = "goal"
        out[(over_touch > 0) & (over_touch > over_goal)] = "touch"
        return out

    def goal_end(self, x):
        """0.0 or length: the goal line nearest to x."""
        x = np.asarray(x, dtype=float)
        return np.where(x < self.length / 2.0, 0.0, self.length)

    def dist_beyond_goal_line(self, x):
        """How far beyond its nearest goal line (> 0 behind the line, < 0 in front of it)."""
        x = np.asarray(x, dtype=float)
        return np.where(x < self.length / 2.0, -x, x - self.length)

    def mouth_margin(self, y, tol=0.0):
        """How far inside the posts (+ tol) the ball is laterally: > 0 between the posts."""
        y = np.asarray(y, dtype=float)
        return self.goal_half_width + tol - np.abs(y - self.width / 2.0)

    def dist_to_centre(self, x, y):
        cx, cy = self.centre
        return np.hypot(np.asarray(x, dtype=float) - cx, np.asarray(y, dtype=float) - cy)

    # -- team-relative ---------------------------------------------------------------------------
    @staticmethod
    def attacker_of_end(end_x, half, attack_dir, length):
        """Team attacking the goal at end_x (0 or length) in that half, or -1 if unknown.

        attack_dir[(team, half)] = +1 means that team attacks the goal at x = length.
        """
        sign = 1 if end_x > length / 2.0 else -1
        for team in (HOME, AWAY):
            if attack_dir.get((team, half)) == sign:
                return team
        return -1

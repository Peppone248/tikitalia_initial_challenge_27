import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seqmodel.match import Match  # noqa: E402


def make_synthetic_match(seed=0, n_frames=3000):
    rng = np.random.default_rng(seed)
    T, P = n_frames, 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    period = np.where(np.arange(T) < T // 2, 1, 2).astype(np.int8)
    # period 1: home attacks right, so the home goalkeeper is at negative x, the away one at positive x
    base_x = np.concatenate([np.linspace(-45, 20, 11), np.linspace(45, -20, 11)])
    base_y = np.tile(np.linspace(-25, 25, 11), 2)
    sign = np.where(period == 1, 1.0, -1.0)[:, None]
    px = (sign * base_x[None, :] + rng.normal(0, 1.0, (T, P))).astype(np.float32)
    py = (base_y[None, :] + rng.normal(0, 1.0, (T, P))).astype(np.float32)
    label = np.zeros(T, dtype=np.int8)
    t, who = 0, 0
    while t < T:
        n = int(rng.integers(200, 401))
        label[t:t + n] = who
        t, who = t + n, 1 - who
    for start in range(500, T - 100, 700):      # dead-ball stretches
        label[start:start + 100] = 2
    ball_x, ball_y = np.zeros(T), np.zeros(T)
    for i in range(T):
        pool = np.where(team == (label[i] if label[i] < 2 else 0))[0]
        j = pool[(i // 50) % len(pool)]
        ball_x[i] = px[i, j] + rng.normal(0, 0.3)
        ball_y[i] = py[i, j] + rng.normal(0, 0.3)
    return Match(name="synthetic", px=px, py=py, team=team, ball_x=ball_x, ball_y=ball_y,
                 period=period, label=label, length=105.0, width=68.0,
                 home_attacks_right={1: True, 2: False})


@pytest.fixture
def synthetic_match():
    return make_synthetic_match

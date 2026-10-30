"""Strength of an activation from its margins (docs/REGOLE_PSEUDO_LABEL.md 4.1).

    z_i = margin_i / sigma_i          margin > 0 means the condition holds
    z   = min_i z_i                   conditions in AND: the weakest one decides
    s   = clip(z / Z_SAT, 0, 1)       0 on the threshold, 1 at Z_SAT sigmas or more
    duration factor: s *= min(1, observed / (2 * required))
"""

import numpy as np


def z_score(margin, sigma):
    return np.asarray(margin, dtype=float) / float(sigma)


def combine_and(*zs):
    return np.minimum.reduce([np.asarray(z, dtype=float) for z in zs])


def to_strength(z, z_sat):
    return np.clip(np.asarray(z, dtype=float) / float(z_sat), 0.0, 1.0)


def duration_factor(observed_s, required_s):
    if required_s <= 0:
        return 1.0
    return float(min(1.0, observed_s / (2.0 * required_s)))

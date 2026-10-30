"""Noise ("rumore", NOMENCLATURA 2) of every feature a rule condition reads, and the strength.

    z_i = m_i / sigma_i          margin over the threshold in units of noise
    s_i = clip(z_i / Z_SAT, 0, 1) * d_i
    s   = min_i s_i              (conditions in AND)
    w   = q * s                  (the vote weight, NOMENCLATURA 4)

How sigma_i is obtained, without labels, on the FEATURE ITSELF (not composed from the position
noise, which on BAR-ATH sits on the 0.10 m floor):

    residual = x - moving_average(x, SMOOTH_WIN)        on every finite stretch of x
    mode std      sigma = std(residual)                 previous definition (NOMENCLATURA 2)
    mode mad      sigma = 1.4826 * MAD(residual)        robust: ignores jumps / nearest-player
                                                        switches, which are not "noise" but
                                                        events; equals std for Gaussian jitter
    mode mad_tol  sigma = max(1.4826 * MAD, tol_i)      + the a-priori semantic tolerance of the
                                                        condition (STATO_LAVORI 2.1 proposal)

`floor` (per feature) only guards against sigma = 0 on a frozen signal. The choice of the mode
is made by scripts/eval_sigma_synthetic.py on synthetic data (report in docs of HMM_first_imp).
"""

import numpy as np

from tikitalia.preprocessing import runs

MAD_TO_SIGMA = 1.4826
MODES = ("std", "mad", "mad_tol")


def residual(x, window):
    """x - centred moving average, on every finite stretch at least `window` long (inner part)."""
    x = np.asarray(x, dtype=float)
    half = int(window) // 2
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    out = []
    for start, end in runs(np.isfinite(x)):
        if end - start < len(kernel) + 2 * half:
            continue
        seg = x[start:end]
        smooth = np.convolve(np.pad(seg, half, mode="edge"), kernel, mode="valid")
        out.append((seg - smooth)[half:len(seg) - half])
    return np.concatenate(out) if out else np.array([])


def mad_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def std_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else np.nan


def estimate(x, window):
    """{'mad': ..., 'std': ..., 'n': ...} of the residual of x."""
    r = residual(x, window)
    return {"mad": mad_sigma(r), "std": std_sigma(r), "n": int(r.size)}


def pick_sigma(est, mode, tol, floor):
    if mode not in MODES:
        raise ValueError("noise mode must be one of {}".format(MODES))
    if mode == "std":
        sigma = est["std"]
    else:
        sigma = est["mad"]
    if not np.isfinite(sigma):
        sigma = floor
    sigma = max(sigma, floor)
    if mode == "mad_tol":
        sigma = max(sigma, tol)
    return float(sigma)


class NoiseModel(object):
    """sigma per named feature. `sources` maps a name to a (n,) or (n, k) array (columns pooled)."""

    def __init__(self, sources, spec, mode, window):
        self.mode = mode
        self.table = {}
        for name, entry in spec.items():
            if name not in sources:
                raise KeyError("noise: no source array for feature {!r}".format(name))
            arr = np.asarray(sources[name], dtype=float)
            cols = [arr] if arr.ndim == 1 else [arr[:, j] for j in range(arr.shape[1])]
            res = np.concatenate([residual(c, window) for c in cols])
            est = {"mad": mad_sigma(res), "std": std_sigma(res), "n": int(res.size)}
            self.table[name] = dict(est, tol=float(entry["tol"]), floor=float(entry["floor"]),
                                    unit=entry.get("unit", ""),
                                    sigma=pick_sigma(est, mode, float(entry["tol"]),
                                                     float(entry["floor"])))

    def sigma(self, name):
        return self.table[name]["sigma"]

    def z(self, margin, name):
        return np.asarray(margin, dtype=float) / self.sigma(name)

    def summary(self):
        return {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in e.items()}
                for k, e in self.table.items()}


def strength(zs, z_sat, duration_factor=1.0):
    """s = clip(min_i z_i / Z_SAT, 0, 1) * d. zs: list of arrays (AND)."""
    z = np.minimum.reduce([np.asarray(v, dtype=float) for v in zs])
    return np.clip(np.nan_to_num(z, nan=0.0) / float(z_sat), 0.0, 1.0) * duration_factor


def sources_from(match, feats, pitch):
    """The arrays the rules compare with thresholds, keyed as in hmm.yaml noise.features."""
    return {
        "ball_pos": np.column_stack([match.ball_x, match.ball_y]),
        "dist_outside": pitch.dist_outside(match.ball_x, match.ball_y),
        "ball_speed": match.ball_speed,
        "nearest_dist": feats.nearest_dist,
        "rel_speed": feats.rel_speed,
        "pressure_dist": feats.pressure_dist,
        "align_cos": feats.align_cos,
        "slowdown": feats.slowdown[:, :2],
    }

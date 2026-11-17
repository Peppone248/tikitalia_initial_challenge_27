"""The referee's whistle: blasts from the match audio, then Law 5 (play stops at the whistle).

    read_audio(path)                     mono float32 16 kHz from any file ffmpeg reads
    detect_blasts(pcm, n_frames)         [(first, last)] video frames of whistle blasts
    whistle_stop(labels, match, blasts)  labels with the stoppages that start at a blast

Detector: signal processing only, no training and nothing measured on the match. A referee's
whistle is two narrow, flat tones sounding together (two-chamber whistle). The band follows the
published description of the whistle in broadcast audio (Raventos et al., SpringerPlus 2015,
arXiv 1411.6496: 3.5-4.5 kHz, tonal, 2-3 spectral peaks); the other constants are a priori.

Rule: a blast while the ball is alive starts a stoppage at the blast only if the players really
slow down within 3 s (a whistle from the crowd, or advantage played, does not stop the players);
the stoppage runs to the model's next stoppage if one starts within 15 s, else to the next kick
from rest within 30 s (a free kick is usually taken within that time), else nothing changes.
Blasts during a stoppage are ignored: they restart play, they do not stop it. The tracking alone
notices a stoppage only when the players have stopped and the ball is placed; the whistle dates
its start.
"""

import subprocess
import warnings

import numpy as np

from .match import D, X

FPS = 25

# -- detector ----------------------------------------------------------------------------------------
SR = 16000
HOP = SR // FPS               # one video frame of audio
NFFT = 1024
BIN_HZ = SR / NFFT
BAND = (2500.0, 5000.0)       # where the first tone is searched
GAP_HZ = (300.0, 600.0)       # where the second tone is searched, above the first
NEIGH = (2, 10)               # neighbourhood of a peak, bins on each side (local background)
PROFILE = {
    "snr_db": 12.0,           # both tones this far above their neighbourhood
    "pitch": [[3500.0, 4200.0]],  # accepted first tone
    "gap": [300.0, 600.0],    # second tone this far above the first
    "steady_hz": 2 * BIN_HZ,  # a whistle is flat: frame-to-frame change of the pitch at most 2 bins
    "min_frames": 3,          # >= 0.12 s
    "merge_frames": 8,        # blasts closer than 0.32 s are one
}

# -- rule --------------------------------------------------------------------------------------------
SLOW_S = 3.0                  # the slowdown must come within this after the blast:
BEFORE_S = 1.0                # lowest 1 s mean player speed <= SLOW_RATIO x the speed of BEFORE_S
SLOW_RATIO = 0.7              # before the blast, or <= SLOW_MS (walking pace)
SLOW_MS = 1.1
MERGE_S = 15.0                # stoppage up to the model's next stoppage within this,
MAX_S = 30.0                  # else up to the next kick from rest within this
KICK_SPEED = 8.0              # kick from rest: ball seen leaving faster than this (over MEASURE_S),
MEASURE_S = 0.2
STILL_SPEED = 2.0             # after being seen slower than this, or unseen, within REST_S before
REST_S = 1.0
MARGIN_M = 2.0                # at most this far outside the lines


def read_audio(path):
    """mono float32 at 16 kHz, decoded by ffmpeg."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(out, np.float32)


def spectrogram(pcm, n_frames):
    """(n_frames, NFFT / 2 + 1) power; a Hann window centred on each video frame."""
    pcm = np.asarray(pcm, np.float32)
    pad = np.concatenate([np.zeros(NFFT, np.float32), pcm, np.zeros(2 * NFFT, np.float32)])
    win = np.hanning(NFFT).astype(np.float32)
    S = np.empty((n_frames, NFFT // 2 + 1), np.float32)
    for s in range(0, n_frames, 5000):
        e = min(n_frames, s + 5000)
        idx = (np.arange(s, e) * HOP + HOP // 2 - NFFT // 2 + NFFT)[:, None] + np.arange(NFFT)[None, :]
        idx = np.minimum(idx, pad.size - 1)
        S[s:e] = (np.abs(np.fft.rfft(pad[idx] * win, axis=1)) ** 2).astype(np.float32)
    return S


def _local_snr(S, band):
    """(n, k) dB of each bin of the band against the mean of its neighbourhood; first bin index."""
    L = 10.0 * np.log10(S.astype(np.float64) + 1e-20)
    n = L.shape[0]
    c = np.cumsum(np.c_[np.zeros(n), L], axis=1)
    k0, k1 = int(band[0] / BIN_HZ), int(band[1] / BIN_HZ)
    a, b = NEIGH
    snr = np.empty((n, k1 - k0), np.float32)
    for j, k in enumerate(range(k0, k1)):
        side = 0.5 * ((c[:, k - a] - c[:, k - b]) / (b - a) + (c[:, k + b + 1] - c[:, k + a + 1]) / (b - a))
        snr[:, j] = L[:, k] - side
    return snr, k0


def two_tone(S):
    """per frame: score (dB, the weaker of the two tones), first and second tone (Hz)."""
    snr, k0 = _local_snr(S, (BAND[0], BAND[1] + GAP_HZ[1]))
    n = snr.shape[0]
    g0, g1 = int(GAP_HZ[0] / BIN_HZ), int(GAP_HZ[1] / BIN_HZ)
    best = np.full(n, -99.0)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    for j in range(0, int(BAND[1] / BIN_HZ) - k0):
        hi = snr[:, j + g0:j + g1 + 1]
        sc = np.minimum(snr[:, j], hi.max(1))
        up = sc > best
        best[up] = sc[up]
        f1[up] = (j + k0) * BIN_HZ
        f2[up] = (j + g0 + hi.argmax(1)[up] + k0) * BIN_HZ
    return best.astype(np.float32), f1.astype(np.float32), f2.astype(np.float32)


def runs(mask):
    """[(start, end_exclusive)] of consecutive True values."""
    idx = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, bool).astype(np.int8), 0]))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def detect_blasts(pcm, n_frames, profile=PROFILE):
    """[(first, last)] video frames of whistle blasts."""
    score, f, f2 = two_tone(spectrogram(pcm, n_frames))
    g = f2 - f
    ok = (score > profile["snr_db"]) & (g >= profile["gap"][0]) & (g <= profile["gap"][1])
    inside = np.zeros(f.size, bool)
    for lo, hi in profile["pitch"]:
        inside |= (f >= lo) & (f <= hi)
    ok &= inside
    steady = np.r_[True, np.abs(np.diff(f)) <= profile["steady_hz"]]
    ok &= steady | ~np.r_[False, ok[:-1]]
    blasts = []
    for s, e in runs(ok):
        if e - s >= profile["min_frames"]:
            if blasts and s - blasts[-1][1] <= profile["merge_frames"]:
                blasts[-1] = (blasts[-1][0], int(e - 1))
            else:
                blasts.append((int(s), int(e - 1)))
    return blasts


def restart_kicks(bx, by, length, width):
    """(n,) bool: kicks from rest, dated to the start of the kick."""
    m = max(1, int(round(MEASURE_S * FPS)))
    rest = max(1, int(round(REST_S * FPS)))
    fwd = np.full(bx.size, np.nan)
    if bx.size > m:
        fwd[:-m] = np.hypot(bx[m:] - bx[:-m], by[m:] - by[:-m]) / m * FPS
    seen = ~np.isnan(bx)
    near = seen & (bx > -MARGIN_M) & (bx < length + MARGIN_M) & (by > -MARGIN_M) & (by < width + MARGIN_M)
    fast = near & (np.nan_to_num(fwd, nan=0.0) > KICK_SPEED)
    slow = seen & (np.nan_to_num(fwd, nan=np.inf) < STILL_SPEED)
    out = np.zeros(bx.size, bool)
    for s, _ in runs(fast):
        lo = max(0, s - rest)
        r = np.flatnonzero(slow[lo:s])
        u = np.flatnonzero(~seen[lo:s])
        if r.size and (not u.size or r[-1] > u[-1]):
            out[lo + int(r[-1]) + 1] = True
        elif u.size:
            out[lo + int(u[-1]) + 1] = True
    return out


def whistle_stop(labels, m, blasts):
    """labels with D from each accepted blast; also (info, mask of the frames set to D).
    `blasts`: [(first, last)] video frames; `m`: the cpm Match the labels were computed on."""
    L, W = m.pitch
    row_of = {int(f): i for i, f in enumerate(m.frame_index)}
    blasts = [(row_of[int(s)], row_of.get(int(e), row_of[int(s)])) for s, e in blasts if int(s) in row_of]
    bx = (m.ball[:, 0] + L / 2.0).astype(np.float64)        # corner origin
    by = (m.ball[:, 1] + W / 2.0).astype(np.float64)
    px = (m.px + L / 2.0).astype(np.float32)
    py = (m.py + W / 2.0).astype(np.float32)
    half = np.where(np.isin(m.period, (1, 2)), m.period, 0).astype(np.int8)
    dead = np.asarray(labels == D)
    n = dead.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nan_to_num(np.nanmean(np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * FPS, 1))
    speed_1s = np.convolve(speed, np.ones(FPS) / FPS, "same")
    kicks = np.flatnonzero(restart_kicks(bx, by, L, W))
    slow_k, before_k = int(SLOW_S * FPS), int(BEFORE_S * FPS)
    merge_k, max_k = int(MERGE_S * FPS), int(MAX_S * FPS)
    stop = np.zeros(n, bool)
    info = {"stop": 0, "stop_no_slowdown": 0, "stop_no_end": 0}
    for s, e in blasts:
        if s >= n or half[s] == 0 or dead[s]:
            continue
        v0 = speed[max(0, s - before_k):s].mean() if s > 0 else speed[0]
        vmin = speed_1s[s:min(n, s + slow_k)].min()
        if not (vmin <= SLOW_RATIO * v0 or vmin <= SLOW_MS):
            info["stop_no_slowdown"] += 1
            continue
        nxt = np.flatnonzero(dead[s:min(n, s + merge_k)] & (half[s:min(n, s + merge_k)] == half[s]))
        if nxt.size:
            end = s + int(nxt[0])
        else:
            k = kicks[(kicks > e) & (kicks <= s + max_k)]
            k = k[half[k] == half[s]] if k.size else k
            if not k.size:
                info["stop_no_end"] += 1
                continue
            end = int(k[0])
        stop[s:end] = True
        info["stop"] += 1
    add = stop & ~dead & (labels != X)
    out = labels.copy()
    out[add] = D
    info.update(frames_set_D=int(add.sum()), blasts=len(blasts))
    return out, info, add

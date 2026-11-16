"""The referee's whistle from the match audio (REPORT_B 18). Signal processing only: no training, no
labels, no frame of the match typed anywhere.

    spectrogram(pcm, n_frames)      power spectrum per VIDEO frame (16 kHz audio, 640 samples = 1 frame)
    dual_tone(S)                    per frame: the two-tone score, its lower and upper frequency
    calibrate(score, f1, f2)        the whistle's own pitch: the mode of the strong two-tone frames
    detect(score, f1, f2, cal)      whistle blasts: (start_frame, end_frame) list
    features(S), detect_profile()   the same, with a profile learned on OTHER matches: nothing measured on the match
                                    being labelled (scripts/fit_whistle_profile.py, REPORT_B 30)

A referee's whistle on the BAR-ATH feed is two flat lines sounding together (two-chamber whistle):
a narrow peak f1 and a second one ~1.12 f1. Commentary and crowd make tonal lines too, but they
glide and come in harmonic stacks; the test is therefore: two narrow peaks at once, both well above
their neighbourhood, at the whistle's pitch (calibrated on the match itself from the distribution of
two-tone frames, origin "distribution" like the other self-measured constants), steady for >= 3
frames. The constants below are a priori.
"""

import numpy as np

SR = 16000
HOP = 640                      # 16000 / 25: one video frame
NFFT = 1024
BIN_HZ = SR / NFFT
BAND = (3300.0, 4700.0)        # where two-chamber referee whistles sound (fundamental)
BAND_WIDE = (2500.0, 5000.0)   # where the profile learned on other matches may look (scripts/fit_whistle_profile.py)
GAP_HZ = (300.0, 600.0)        # search range of the second tone above the first
NEIGH = (2, 10)                # neighbourhood of a peak, in bins on each side (local background)
CAL_SCORE = 15.0               # dB: strong two-tone frames used to calibrate the pitch
SCORE_DB = 12.0                # dB: both tones at least this far above their neighbourhood
PITCH_TOL_HZ = 150.0           # f1 within this of the calibrated pitch
GAP_TOL_HZ = 70.0              # f2 - f1 within this of the calibrated gap
MIN_FRAMES = 3                 # >= 0.12 s
MERGE_FRAMES = 8               # blasts closer than 0.32 s are one


def spectrogram(pcm, n_frames):
    """(n_frames, NFFT / 2 + 1) float32 power; window centred on each video frame."""
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


def local_snr(S, band):
    """(n, k) dB: each bin of band against the mean of its neighbourhood (NEIGH bins on each side); and
    the index of the band's first bin."""
    L = 10.0 * np.log10(S.astype(np.float64) + 1e-20)
    n = L.shape[0]
    c = np.cumsum(np.c_[np.zeros(n), L], axis=1)
    k0, k1 = int(band[0] / BIN_HZ), int(band[1] / BIN_HZ)
    a, b = NEIGH
    snr = np.empty((n, k1 - k0), np.float32)
    for j, k in enumerate(range(k0, k1)):
        side = 0.5 * ((c[:, k - a + 0] - c[:, k - b]) / (b - a) + (c[:, k + b + 1] - c[:, k + a + 1]) / (b - a))
        snr[:, j] = L[:, k] - side
    return snr, k0


def single_tone(S, band=None):
    """(score dB, f Hz) per frame: the strongest narrow peak of the band (one-chamber whistles, or a two-chamber one
    whose second tone is masked)."""
    band = band or BAND_WIDE
    snr, k0 = local_snr(S, band)
    j = snr.argmax(1)
    return snr[np.arange(snr.shape[0]), j].astype(np.float64), (j + k0) * BIN_HZ


def dual_tone(S, band=None):
    """(score dB, f1 Hz, f2 Hz) per frame: score = the weaker of the two tones' local SNR. band: where f1 is
    searched (default BAND, the setting of REPORT_B 18)."""
    if band is not None:
        return _dual_tone_band(S, band)
    L = 10.0 * np.log10(S.astype(np.float64) + 1e-20)
    n = L.shape[0]
    c = np.cumsum(np.c_[np.zeros(n), L], axis=1)
    k0, k1 = int(BAND[0] / BIN_HZ), int((BAND[1] + GAP_HZ[1]) / BIN_HZ)
    a, b = NEIGH
    snr = np.empty((n, k1 - k0))
    for j, k in enumerate(range(k0, k1)):
        side = 0.5 * ((c[:, k - a + 0] - c[:, k - b]) / (b - a) + (c[:, k + b + 1] - c[:, k + a + 1]) / (b - a))
        snr[:, j] = L[:, k] - side
    g0, g1 = int(GAP_HZ[0] / BIN_HZ), int(GAP_HZ[1] / BIN_HZ)
    best = np.full(n, -99.0)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    last = int(BAND[1] / BIN_HZ) - k0
    for j in range(0, last):
        hi = snr[:, j + g0:j + g1 + 1]
        bm = hi.max(1)
        sc = np.minimum(snr[:, j], bm)
        up = sc > best
        best[up] = sc[up]
        f1[up] = (j + k0) * BIN_HZ
        f2[up] = (j + g0 + hi.argmax(1)[up] + k0) * BIN_HZ
    return best, f1, f2


def _dual_tone_band(S, band):
    snr, k0 = local_snr(S, (band[0], band[1] + GAP_HZ[1]))
    n = snr.shape[0]
    g0, g1 = int(GAP_HZ[0] / BIN_HZ), int(GAP_HZ[1] / BIN_HZ)
    best = np.full(n, -99.0)
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    for j in range(0, int(band[1] / BIN_HZ) - k0):
        hi = snr[:, j + g0:j + g1 + 1]
        bm = hi.max(1)
        sc = np.minimum(snr[:, j], bm)
        up = sc > best
        best[up] = sc[up]
        f1[up] = (j + k0) * BIN_HZ
        f2[up] = (j + g0 + hi.argmax(1)[up] + k0) * BIN_HZ
    return best, f1, f2


def features(S):
    """everything the profile detector needs, per video frame, over the wide band: the two-tone and the one-tone
    scores and their frequencies. Cheap to keep on disk for external matches."""
    s2, f1, f2 = dual_tone(S, BAND_WIDE)
    s1, fs = single_tone(S, BAND_WIDE)
    return {"s2": s2.astype(np.float32), "f1": f1.astype(np.float32), "f2": f2.astype(np.float32),
            "s1": s1.astype(np.float32), "fs": fs.astype(np.float32)}


def _runs(ok):
    idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def detect_profile(feat, profile):
    """[(start, end_inclusive)] blasts with a FIXED profile learned on other matches (scripts/fit_whistle_profile.py):
    no calibration on the match being labelled.

    profile: mode "dual" | "single"; snr_db; pitch: [[lo, hi], ...] accepted f1 (or f) intervals; gap: [lo, hi]
    (dual); steady_hz: largest frame-to-frame change of the frequency; min_frames; merge_frames."""
    if profile["mode"] == "dual":
        score, f = feat["s2"], feat["f1"]
        ok = (score > profile["snr_db"])
        g = feat["f2"] - feat["f1"]
        ok &= (g >= profile["gap"][0]) & (g <= profile["gap"][1])
    else:
        score, f = feat["s1"], feat["fs"]
        ok = score > profile["snr_db"]
    inside = np.zeros(f.size, bool)
    for lo, hi in profile["pitch"]:
        inside |= (f >= lo) & (f <= hi)
    ok &= inside
    # inside a run the frequency must stay put (a whistle is flat; voices and chants glide)
    steady = np.r_[True, np.abs(np.diff(f)) <= profile["steady_hz"]]
    prev = np.r_[False, ok[:-1]]
    ok &= steady | ~prev
    ev = []
    for s, e in _runs(ok):
        if e - s >= profile["min_frames"]:
            if ev and s - ev[-1][1] <= profile["merge_frames"]:
                ev[-1] = (ev[-1][0], int(e - 1))
            else:
                ev.append((int(s), int(e - 1)))
    return ev


def calibrate(score, f1, f2):
    """{"pitch": Hz, "gap": Hz}: modes of f1 and f2 - f1 over the strong two-tone frames."""
    strong = score > CAL_SCORE
    h, e = np.histogram(f1[strong], bins=np.arange(BAND[0], BAND[1] + BIN_HZ, 2 * BIN_HZ))
    pitch = float(e[h.argmax()] + BIN_HZ)
    near = strong & (np.abs(f1 - pitch) <= PITCH_TOL_HZ)
    gaps = (f2 - f1)[near]
    h, e = np.histogram(gaps, bins=np.arange(GAP_HZ[0], GAP_HZ[1] + BIN_HZ, 2 * BIN_HZ))
    gap = float(e[h.argmax()] + BIN_HZ) if gaps.size else float(np.mean(GAP_HZ))
    return {"pitch": pitch, "gap": gap, "strong_frames": int(strong.sum()), "near_frames": int(near.sum())}


def detect(score, f1, f2, cal):
    """[(start, end_inclusive)] whistle blasts."""
    ok = (score > SCORE_DB) & (np.abs(f1 - cal["pitch"]) <= PITCH_TOL_HZ) \
        & (np.abs((f2 - f1) - cal["gap"]) <= GAP_TOL_HZ)
    ev = []
    idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    for s, e in zip(idx[::2], idx[1::2]):
        if e - s >= MIN_FRAMES:
            if ev and s - ev[-1][1] <= MERGE_FRAMES:
                ev[-1] = (ev[-1][0], int(e - 1))
            else:
                ev.append((int(s), int(e - 1)))
    return ev


def read_audio(path):
    """mono float32 16 kHz from any file ffmpeg reads (the match video itself, or an extracted .flac)."""
    import subprocess
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(out, np.float32)

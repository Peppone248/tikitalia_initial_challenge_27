"""Referee whistle by band-pass short-time energy and duration (Kathirvel, Manikandan, Soman, "Automated Referee Whistle
Sound Detection for Extraction of Highlights from Sports Video", IJCA 12(11), 2011). A second detector, next to the
two-tone profile of hmm_v0/whistle.py; every constant is the paper's:

    16 kHz mono, non-overlapping blocks of 6 s, each with its mean removed and divided by its largest |sample|;
    band-pass FIR 3750-4100 Hz (the paper: least-squares linear phase; here a Hamming windowed-sinc of 255 taps, the
    number of taps is not given in the paper); short-time energy = squared output summed over W = 0.3 s;
    amplitude threshold 10 on that energy; a block is a whistle block if more than 0.3 s of samples are above it
    (a referee whistle lasts 0.5-3.5 s: short tonal spikes are rejected by duration, not by pitch).

The output is the same as hmm_v0.whistle.detect_profile: [(start_frame, end_frame)] in video frames (640 samples), so
it goes through dead_b.whistle_dead (R7a) unchanged. Profile file: config_b/whistle_profile_ste.json ("type": "ste").
"""
import numpy as np

SR = 16000
HOP = 640


def bandpass_taps(lo, hi, numtaps, sr=SR):
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = (2 * hi / sr) * np.sinc(2 * hi / sr * n) - (2 * lo / sr) * np.sinc(2 * lo / sr * n)
    return (h * np.hamming(numtaps)).astype(np.float64)


def _filter(x, taps):
    n = x.size + taps.size - 1
    m = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(taps, m), m)[:n]
    d = (taps.size - 1) // 2                      # linear phase: centre the output on the input
    return y[d:d + x.size]


def _runs(ok):
    idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def detect(pcm, prof):
    """[(start_frame, end_frame)] whistle blasts of a 16 kHz mono signal with an "ste" profile."""
    pcm = np.asarray(pcm, np.float64)
    taps = bandpass_taps(prof["band"][0], prof["band"][1], int(prof.get("numtaps", 255)))
    B = int(round(prof.get("block_s", 6.0) * SR))
    W = int(round(prof.get("window_s", 0.3) * SR))
    amp = float(prof.get("amp_thr", 10.0))
    tmin = int(round(prof.get("time_thr_s", 0.3) * SR))
    keep = np.zeros(pcm.size, bool)
    for b0 in range(0, pcm.size, B):
        x = pcm[b0:b0 + B]
        if x.size < W:
            break
        x = x - x.mean()
        peak = np.abs(x).max()
        if peak <= 0:
            continue
        f = _filter(x / peak, taps)
        c = np.cumsum(np.r_[0.0, f * f])
        s = np.empty(x.size)
        s[:W] = c[1:W + 1]                          # first samples: the partial window
        s[W:] = c[W + 1:] - c[1:x.size - W + 1]
        above = s > amp
        if above.sum() > tmin:
            # the energy at n covers [n - W + 1, n]: the sound starts about W before the first sample above
            for a, e in _runs(above):
                keep[b0 + max(0, a - W + 1):b0 + e] = True
    ev = []
    merge = int(prof.get("merge_frames", 8))
    for a, e in _runs(keep):
        s, t = a // HOP, (e - 1) // HOP
        if ev and s - ev[-1][1] <= merge:
            ev[-1] = (ev[-1][0], t)
        else:
            ev.append((int(s), int(t)))
    return ev

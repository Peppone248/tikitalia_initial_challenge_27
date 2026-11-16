"""The referee's whistles of BAR-ATH from the audio of the tactical video (REPORT_B 18).

    python HMM_first_imp/scripts/detect_whistles.py [--audio data/tactical/BAR-ATH_tactical_audio.flac]

--audio may be the tactical .mp4 itself (ffmpeg reads its audio track) or the .flac extracted with
    ffmpeg -i data/tactical/BAR-ATH_tactical.mp4 -vn -ac 1 -ar 16000 -c:a flac data/tactical/BAR-ATH_tactical_audio.flac
Writes outputs/whistles/BAR-ATH_whistles.json (ignored by git: derived data of the match, RULES.md):
the calibrated whistle pitch and the blasts as video frames. Reads no ground truth.
With --profile the detector uses a profile learned on other matches (scripts/fit_whistle_profile.py) and measures nothing
on this audio:
    python HMM_first_imp/scripts/detect_whistles.py --profile config_b/whistle_profile.json --out outputs/whistles/BAR-ATH_whistles_ext.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hmm_v0 import paths  # noqa: E402
from hmm_v0 import whistle as Wh  # noqa: E402
from hmm_v0 import whistle_ste as STE  # noqa: E402

N_FRAMES = 199319


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", default="../data/tactical/BAR-ATH_tactical_audio.flac")
    ap.add_argument("--out", default="outputs/whistles/BAR-ATH_whistles.json")
    ap.add_argument("--profile", default=None, help="config_b/whistle_profile.json: a profile learned on other matches; "
                    "nothing is then measured on this audio (REPORT_B 30)")
    args = ap.parse_args()
    pcm = Wh.read_audio(paths.resolve(args.audio))
    S = Wh.spectrogram(pcm, N_FRAMES)
    if args.profile:
        with open(paths.resolve(args.profile)) as h:
            prof = json.load(h)
        if prof["profile"].get("type") == "ste":   # band energy (Kathirvel et al. 2011), config_b/whistle_profile_ste.json
            ev = [e for e in STE.detect(pcm, prof["profile"]) if e[0] < N_FRAMES]
        else:
            ev = Wh.detect_profile(Wh.features(S), prof["profile"])
        out = paths.resolve(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as h:
            json.dump({"calibration": {"profile": args.profile, "chosen": prof.get("chosen", prof["profile"])}, "blasts": ev,
                       "audio_samples": int(pcm.size)}, h, indent=1)
        print("profile {} ({}): {} blasts -> {}".format(prof.get("chosen", prof["profile"]), prof.get("source", "{} external halves".format(prof.get("halves"))), len(ev), out))
        return
    score, f1, f2 = Wh.dual_tone(S)
    cal = Wh.calibrate(score, f1, f2)
    ev = Wh.detect(score, f1, f2, cal)
    out = paths.resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as h:
        json.dump({"calibration": cal, "blasts": ev, "audio_samples": int(pcm.size)}, h, indent=1)
    print("whistle pitch {:.0f} Hz + {:.0f} Hz, {} blasts -> {}".format(cal["pitch"], cal["gap"], len(ev), out))


if __name__ == "__main__":
    main()

"""Does the pipeline give the same file on this machine as on the reference machine? (RULES.md 3:
"Running your repository against the provided data must reproduce your submission. Not
approximately -- the same file.")

    python changepoint_model/scripts/check_determinism.py                      # a teammate
    python changepoint_model/scripts/check_determinism.py --write-reference    # the reference machine

Four steps, each run twice in a fresh process (same machine) and compared with
changepoint_model/determinism_reference.json (other machine). A difference tells where it starts:

    1 metrica_g3    public Metrica sample game 3 (no competition data): the model alone
    2 v41           the provided match, model + kick-off rule (no audio involved)
    3 whistles      the referee's whistle blasts from the match audio (ffmpeg decoding)
    4 v42           the provided match, model + kick-off rule + whistle: the candidate submission

Needs: the data bundle in data/ (or MTAH_DATA_DIR); ffmpeg on the PATH for step 3; for step 1 the
Metrica sample data (git clone --depth 1 https://github.com/metrica-sports/sample-data ext/metrica),
otherwise step 1 is skipped. The audio is extracted once, with the command documented in
HMM_first_imp/scripts/detect_whistles.py, to data/tactical/BAR-ATH_tactical_audio.flac.

Writes outputs/determinism/report_<machine>.json: SHA-256 values and software versions only, no data.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PY = sys.executable
PREDICT = os.path.join(HERE, "predict_atd.py")
MODEL = os.path.join(PKG, "models", "cpm_v4_hybrid.pkl")
DETECT = os.path.join(REPO, "HMM_first_imp", "scripts", "detect_whistles.py")
PROFILE = "config_b/whistle_profile_literature.json"            # relative to HMM_first_imp
WHISTLES = os.path.join(REPO, "HMM_first_imp", "outputs", "whistles", "BAR-ATH_whistles_lit_check.json")
REFERENCE = os.path.join(PKG, "determinism_reference.json")
OUT = os.path.join(REPO, "outputs", "determinism")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log):
    t = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n" + p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError("failed ({}): see {}".format(p.returncode, log))
    return time.time() - t


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
         "processor": platform.processor()}
    for mod in ("numpy", "sklearn", "numba", "yaml", "scipy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        v["blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # noqa: BLE001
        v["blas"] = None
    try:
        v["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        v["ffmpeg"] = None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica", default=os.path.join(REPO, "ext", "metrica", "data"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--skip-barath", action="store_true", help="only the public Metrica step")
    ap.add_argument("--write-reference", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    results, times = {}, {}

    def step(name, make_cmd, out_file):
        hashes = []
        for i in range(args.runs):
            out_i = out_file.format(i)
            times.setdefault(name, []).append(round(run(make_cmd(out_i), os.path.join(OUT, "{}_{}.log".format(name, i))), 1))
            hashes.append(sha(out_i))
        results[name] = {"sha256": hashes[0], "same_on_this_machine": len(set(hashes)) == 1}
        print("{:<11} {}  same in {} runs: {}  ({} s)".format(name, hashes[0][:16], args.runs,
                                                           len(set(hashes)) == 1, times[name]), flush=True)

    meta = os.path.join(args.metrica, "Sample_Game_3", "Sample_Game_3_metadata.xml")
    raw = os.path.join(args.metrica, "Sample_Game_3", "Sample_Game_3_tracking.txt")
    if os.path.isfile(meta) and os.path.isfile(raw):
        step("metrica_g3", lambda o: [PY, PREDICT, "--models", MODEL, "--meta", meta, "--raw", raw, "--out", o],
             os.path.join(OUT, "metrica_g3_{}.csv"))
    else:
        print("metrica_g3  skipped: Metrica sample data not found in {}".format(args.metrica))

    if not args.skip_barath:
        step("v41", lambda o: [PY, PREDICT, "--models", MODEL, "--out", o], os.path.join(OUT, "v41_{}.csv"))
        sys.path.insert(0, REPO)
        from games import asset_paths
        flac = os.path.join(os.path.dirname(asset_paths()["video"]), "BAR-ATH_tactical_audio.flac")
        if not os.path.isfile(flac):
            print("extracting the audio once -> {}".format(flac), flush=True)
            run(["ffmpeg", "-y", "-i", asset_paths()["video"], "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", flac],
                os.path.join(OUT, "ffmpeg.log"))
        step("whistles", lambda o: [PY, DETECT, "--audio", flac, "--profile", PROFILE, "--out", o],
             WHISTLES.replace(".json", "_{}.json"))
        step("v42", lambda o: [PY, PREDICT, "--models", MODEL, "--whistle", WHISTLES.replace(".json", "_0.json"),
                               "--out", o], os.path.join(OUT, "v42_{}.csv"))

    env = versions()
    report = {"machine": socket.gethostname(), "env": env, "results": results, "seconds": times,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(OUT, "report_{}.json".format(socket.gethostname()))
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    if args.write_reference:
        ref = {"machine": report["machine"], "env": env, "time": report["time"],
               "sha256": {k: v["sha256"] for k, v in results.items()}}
        if os.path.isfile(REFERENCE):
            with open(REFERENCE) as fh:
                old = json.load(fh)
            for k, v in old.get("sha256", {}).items():
                if k in ref["sha256"] and ref["sha256"][k] != v:
                    print("  note: {} differs from the previous reference ({}): {} -> {}".format(
                        k, old.get("machine"), v[:16], ref["sha256"][k][:16]))
                ref["sha256"].setdefault(k, v)
        with open(REFERENCE, "w") as fh:
            json.dump(ref, fh, indent=1)
        print("reference written: {}".format(REFERENCE))
    elif os.path.isfile(REFERENCE):
        with open(REFERENCE) as fh:
            ref = json.load(fh)
        print("\nagainst the reference ({}, {}):".format(ref.get("machine"), ref.get("env", {}).get("platform")))
        ok_all = True
        for k, v in results.items():
            r = ref.get("sha256", {}).get(k)
            ok = r == v["sha256"] and v["same_on_this_machine"]
            ok_all &= ok
            print("  {:<11} {}".format(k, "SAME FILE" if ok else ("DIFFERENT" if r else "no reference")))
        print("\nRESULT: {}".format("deterministic across machines" if ok_all else
                                    "differences found: send {} to the team".format(path)))
    print("report: {}".format(path))


if __name__ == "__main__":
    main()

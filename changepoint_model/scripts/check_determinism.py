"""Does this machine write the same files as the reference machine? (RULES.md 3: "the same file")

    python changepoint_model/scripts/check_determinism.py                    # any machine
    python changepoint_model/scripts/check_determinism.py --write-reference  # the reference machine

Three runs, each twice, compared with changepoint_model/determinism_reference.json:

    metrica_g3   public Metrica sample game 3: the model alone, no competition data
                 (git clone --depth 1 https://github.com/metrica-sports/sample-data ext/metrica)
    v41          the provided match, model + kick-off rule, no audio
    v42          the provided match, model + kick-off + whistle rule: the submission

A difference in v42 only points at the audio decoding (ffmpeg version). Writes
outputs/determinism/report_<machine>.json: hashes and software versions only.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
PREDICT = os.path.join(PKG, "predict.py")
REFERENCE = os.path.join(PKG, "determinism_reference.json")
OUT = os.path.join(REPO, "outputs", "determinism")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run(args, out, log):
    p = subprocess.run([sys.executable, PREDICT, "--out", out] + args, cwd=REPO, capture_output=True, text=True)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit("failed: see {}".format(log))
    return sha(out)


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "numba"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = None
    try:
        v["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except OSError:
        v["ffmpeg"] = None
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrica", default=os.path.join(REPO, "ext", "metrica", "data", "Sample_Game_3"))
    ap.add_argument("--write-reference", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    steps = {}
    meta = os.path.join(args.metrica, "Sample_Game_3_metadata.xml")
    raw = os.path.join(args.metrica, "Sample_Game_3_tracking.txt")
    if os.path.isfile(meta) and os.path.isfile(raw):
        steps["metrica_g3"] = ["--meta", meta, "--raw", raw]
    else:
        print("metrica_g3 skipped: no Metrica sample data in {}".format(args.metrica))
    steps["v41"] = ["--no-whistle"]
    steps["v42"] = []
    results = {}
    for name, extra in steps.items():
        h = [run(extra, os.path.join(OUT, "{}_{}.csv".format(name, i)), os.path.join(OUT, "{}_{}.log".format(name, i)))
             for i in range(2)]
        results[name] = h[0] if h[0] == h[1] else None
        print("{:<11} {}".format(name, h[0][:16] if results[name] else "DIFFERENT between two runs"), flush=True)

    report = {"machine": socket.gethostname(), "env": versions(), "sha256": results}
    with open(os.path.join(OUT, "report_{}.json".format(socket.gethostname())), "w") as fh:
        json.dump(report, fh, indent=1)
    if args.write_reference:
        with open(REFERENCE, "w") as fh:
            json.dump(report, fh, indent=1)
        print("reference written: {}".format(REFERENCE))
        return
    with open(REFERENCE) as fh:
        ref = json.load(fh)["sha256"]
    same = {k: v is not None and ref.get(k) == v for k, v in results.items()}
    for k, ok in same.items():
        print("  {:<11} {}".format(k, "SAME FILE" if ok else "DIFFERENT"))
    print("RESULT:", "deterministic across machines" if all(same.values()) else "differences found")


if __name__ == "__main__":
    main()

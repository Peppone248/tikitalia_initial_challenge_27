"""Raw data -> CSV for the Initial Challenge, with the HMM engine (v0). One command:

    python HMM_first_imp/scripts/make_submission.py

Options
    --config-dir DIR      folder with YAML files that replace config/<same name>.yaml
    --set KEY=VALUE       override one value, e.g. --set hmm.nn.epochs=3
                          --set hmm.noise.mode=std   --set pipeline.team_name=myteam
    --out-dir DIR         where to write (default: pipeline.yaml output_dir)

Writes <out-dir>/<team_name>.csv, run_summary.json, run_config_snapshot.json, episodes/*.csv
and runs evaluation/validate_submission.py on the CSV. Reads no ground truth.
"""

import argparse
import json
import os
import sys

# Reproducibility (RULES 3): one BLAS thread, set BEFORE numpy is imported. A multi-threaded
# matrix product may sum in a different order from run to run / machine to machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0.run import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    res = run(args.config_dir, overrides, args.out_dir)
    s = res["summary"]
    print("\n== validation ==\n" + res["validation"])
    print("\n== output (no ground truth involved) ==")
    print(json.dumps(s["output"], indent=2))
    print("\nCSV: " + res["csv"])
    return 0 if s["validation_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

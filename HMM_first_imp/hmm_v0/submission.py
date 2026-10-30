"""CSV writer + the official validator (SUBMISSION.md): header `frame,label`, one row per video
frame 0..n-1, labels H/A/D/X."""

import os
import subprocess
import sys

import numpy as np

from . import paths


def write_csv(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    labels = np.asarray(labels)
    bad = set(np.unique(labels)) - {"H", "A", "D", "X"}
    if bad:
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(bad)))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, l) for i, l in enumerate(labels)))
    return path


def validate(path):
    """Runs evaluation/validate_submission.py on the file. Returns (ok, output)."""
    script = os.path.join(paths.REPO_ROOT, "evaluation", "validate_submission.py")
    proc = subprocess.run([sys.executable, script, path], capture_output=True, text=True,
                          cwd=paths.REPO_ROOT)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()

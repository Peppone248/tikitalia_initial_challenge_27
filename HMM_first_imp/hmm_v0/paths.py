"""Locations. HMM_first_imp sits inside the team repository: ../src holds `tikitalia`, the repo
root holds games.py and evaluation/. Importing this module makes both importable."""

import os
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
HMM_DIR = os.path.dirname(PKG_DIR)
REPO_ROOT = os.path.dirname(HMM_DIR)
CONFIG_DIR = os.path.join(HMM_DIR, "config")
TEAM_CONFIG_DIR = os.path.join(REPO_ROOT, "src", "tikitalia", "config")

for _p in (os.path.join(REPO_ROOT, "src"), REPO_ROOT, HMM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def resolve(path):
    """Relative paths in the YAML files are relative to HMM_first_imp/."""
    if path is None:
        return None
    path = os.path.expandvars(os.path.expanduser(str(path)))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HMM_DIR, path))

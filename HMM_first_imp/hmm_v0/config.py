"""Configuration: the team YAML files + the ones of HMM_first_imp.

    ../src/tikitalia/config/params.yaml    team thresholds (D-OUT, D-GOAL, kick-offs, features)
    config/params_hmm.yaml                 thresholds added here (same schema: value, unit,
                                           origin, why) -- merged on top of the team file
    config/rules_hmm.yaml                  rule registry used by this pipeline
    config/hmm.yaml                        noise model, NN, HMM topology and transitions
    config/pipeline.yaml                   game, data paths, cache, outputs, seed

Any number used by the pipeline is in one of these files. Override any of them with
`--config-dir`: a directory holding files with the same names replaces the defaults.
"""

import copy
import os

import yaml

from . import paths
from tikitalia.config import Params, REQUIRED_PARAM_KEYS, ORIGINS  # noqa: E402

FILES = ("params_hmm.yaml", "rules_hmm.yaml", "hmm.yaml", "pipeline.yaml")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _pick(name, config_dir):
    if config_dir and os.path.isfile(os.path.join(config_dir, name)):
        return os.path.join(config_dir, name)
    return os.path.join(paths.CONFIG_DIR, name)


def validate_params(entries):
    problems = []
    for name, entry in entries.items():
        missing = [k for k in REQUIRED_PARAM_KEYS if k not in entry]
        if missing:
            problems.append("{}: missing {}".format(name, missing))
        elif entry["origin"] not in ORIGINS:
            problems.append("{}: origin must be one of {}".format(name, ORIGINS))
        elif entry["origin"] == "distribution" and "estimator" not in entry:
            problems.append("{}: origin 'distribution' needs an 'estimator'".format(name))
    if problems:
        raise ValueError("params are not compliant:\n  " + "\n  ".join(problems))
    return entries


class Config(object):
    """All configuration of one run. `params` is a tikitalia Params over the merged tables."""

    def __init__(self, config_dir=None, overrides=None):
        team = _read(os.path.join(paths.TEAM_CONFIG_DIR, "params.yaml"))
        mine = _read(_pick("params_hmm.yaml", config_dir))
        clash = sorted(set(team) & set(mine))
        merged = copy.deepcopy(team)
        merged.update(copy.deepcopy(mine))          # params_hmm.yaml wins (declared below)
        self.overridden_team_params = clash
        self.param_entries = validate_params(merged)
        self.params = Params(self.param_entries)
        self.rules = _read(_pick("rules_hmm.yaml", config_dir))
        self.hmm = _read(_pick("hmm.yaml", config_dir))
        self.pipeline = _read(_pick("pipeline.yaml", config_dir))
        seg = _pick("segment.yaml", config_dir)
        self.segment = _read(seg) if os.path.isfile(seg) else {}       # approach C only
        for dotted, value in (overrides or {}).items():
            self.set(dotted, value)

    def set(self, dotted, value):
        """'hmm.nn.epochs=3' style override (section.key.subkey)."""
        section, *keys = dotted.split(".")
        node = getattr(self, section)
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value

    def snapshot(self):
        """Everything that decides the output, for the run log."""
        return {"params": {k: self.param_entries[k]["value"] for k in self.param_entries},
                "rules": self.rules, "hmm": self.hmm, "pipeline": self.pipeline,
                "segment": self.segment}

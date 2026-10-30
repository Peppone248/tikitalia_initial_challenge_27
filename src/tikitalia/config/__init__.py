"""Load the YAML configuration: pitch geometry, rule parameters, rule registry."""

import os

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML is needed for the rule configuration:\n    pip install pyyaml")

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PARAM_KEYS = ("value", "unit", "origin", "why")
ORIGINS = ("a_priori", "distribution")


def _load(name, config_dir=None):
    with open(os.path.join(config_dir or CONFIG_DIR, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_geometry(config_dir=None):
    return _load("geometry.yaml", config_dir)


def load_rules(config_dir=None):
    return _load("rules.yaml", config_dir)


def load_params(config_dir=None):
    """{name: entry} with every entry validated. Fails loudly on an undocumented threshold."""
    params = _load("params.yaml", config_dir)
    problems = []
    for name, entry in params.items():
        missing = [key for key in REQUIRED_PARAM_KEYS if key not in entry]
        if missing:
            problems.append("{}: missing {}".format(name, missing))
        elif entry["origin"] not in ORIGINS:
            problems.append("{}: origin must be one of {}".format(name, ORIGINS))
        elif entry["origin"] == "distribution" and "estimator" not in entry:
            problems.append("{}: origin 'distribution' needs an 'estimator'".format(name))
        elif not str(entry["why"]).strip():
            problems.append("{}: empty 'why'".format(name))
    if problems:
        raise ValueError("params.yaml is not compliant:\n  " + "\n  ".join(problems))
    return params


class Params(object):
    """Attribute access to parameter values: p.OUT_MARGIN.

    Values with origin 'distribution' start as None and are filled by the preprocessing
    (set_estimated). Reading one that was never estimated is an error, not a silent None.
    """

    def __init__(self, entries):
        self._entries = entries
        self._values = {name: entry["value"] for name, entry in entries.items()}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._values:
            raise AttributeError("no parameter {!r} in params.yaml".format(name))
        value = self._values[name]
        if value is None:
            raise ValueError("parameter {} (origin: distribution) was never estimated".format(name))
        return value

    def set_estimated(self, name, value):
        if self._entries[name]["origin"] != "distribution":
            raise ValueError("{} is a priori: it cannot be overwritten at run time".format(name))
        self._values[name] = float(value)

    def subset(self, names):
        return {name: self._values.get(name) for name in names}

    def entry(self, name):
        return self._entries[name]

    def names(self):
        return list(self._entries)

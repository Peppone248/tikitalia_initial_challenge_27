"""Pseudo-labelling rules. See docs/REGOLE_PSEUDO_LABEL.md for the definitions."""

from .base import Rule, RuleOutput, ABSTAIN  # noqa: F401
from .registry import register, build_rules  # noqa: F401

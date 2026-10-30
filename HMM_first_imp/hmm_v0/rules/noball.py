"""X rule: the frame is unobservable (ball not tracked). Emits the label 'X' = the HMM states
XH / XA, "possession continues but nobody can see it" (hmm.py)."""

import numpy as np

from tikitalia.preprocessing import runs
from .base import HmmRule
from .registry import register


@register
class NoBall(HmmRule):
    """X-NOBALL -- ANCHOR (an observed fact): ball not seen for at least T_X_MIN. Span: the whole gap, from its first
    frame. Strength ramps from 0.5 to 1 over the first 2 * T_X_MIN (a short gap is weak)."""
    emits = ("X",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = self.new_output(match)
        need = max(1, int(round(p.T_X_MIN * fps)))
        for start, end in runs(~match.ball_visible):
            if end - start < need:
                continue
            ramp = np.clip(0.5 + 0.5 * np.arange(end - start) / (2.0 * need), 0.5, 1.0)
            out.emit(start, end, "X", ramp, dur_s=round((end - start) / fps, 1))
        return out

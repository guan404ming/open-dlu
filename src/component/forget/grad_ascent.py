"""Gradient ascent forget: unbounded ascent on per-token CE.

Used by GA (forget only) and GD (= GA + CE retain). No cap, no reference,
so the forget pressure never saturates.
"""

from src.component.forget.base import ForgetLoss


class GradAscent(ForgetLoss):
    needs_ref = False

    def __call__(self, *, ce_model, ce_ref=None, **_):
        return -ce_model.mean()

"""Capped GradDiff: hard-clamped gradient ascent on per-token CE."""

import torch

from src.component.forget.base import ForgetLoss


class CappedForget(ForgetLoss):
    r"""Implements

        L = - mean_i clamp(CE_theta_i, max=cap)

    Hard cap zeros the gradient once a token's CE reaches ``cap``,
    preventing the unbounded representation drift of plain GradDiff.
    """

    needs_ref = False

    def __init__(self, cap: float = 5.0):
        self.cap = cap

    def __call__(self, *, ce_model, ce_ref=None, **kwargs):
        return -torch.clamp(ce_model, max=self.cap).mean()

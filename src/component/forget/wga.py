"""WGA forget: weighted gradient ascent.

Weights each token by the model's current confidence on the gold token,
``w_i = stop_grad(exp(-gamma * CE_theta_i))``, so already-forgotten tokens
stop contributing and ascent concentrates on still-memorized tokens:

    L = - mean_i  w_i * CE_theta_i
"""

import torch

from src.component.forget.base import ForgetLoss


class WgaForget(ForgetLoss):
    needs_ref = False

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, *, ce_model, ce_ref=None, **_):
        w = torch.exp(-self.gamma * ce_model).detach()
        return -(w * ce_model).mean()

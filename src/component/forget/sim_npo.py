"""SimNPO forget: reference-free NPO (Fan et al. 2024).

Replaces NPO's frozen reference with a fixed margin ``gamma`` on the
length-normalized CE, so no reference forward pass is needed:

    L = - (2/beta) * log sigma( beta * mean_i CE_theta_i - gamma )
"""

import torch.nn.functional as F

from src.component.forget.base import ForgetLoss


class SimNpoForget(ForgetLoss):
    needs_ref = False

    def __init__(self, beta: float = 0.2, gamma: float = 0.0):
        self.beta = beta
        self.gamma = gamma

    def __call__(self, *, ce_model, ce_ref=None, **_):
        arg = self.beta * ce_model.mean() - self.gamma
        return -(2.0 / self.beta) * F.logsigmoid(arg)

"""NPO forget: Negative Preference Optimization (Zhang et al. 2024).

Bounded gradient ascent against a frozen reference:

    L = - (2/beta) * mean_i log sigma( beta (CE_theta_i - CE_ref_i) )

As a token is forgotten its CE rises above the reference, the sigmoid
saturates, and the gradient vanishes, bounding the forget pressure.
"""

import torch.nn.functional as F

from src.component.forget.base import ForgetLoss


class NpoForget(ForgetLoss):
    needs_ref = True

    def __init__(self, beta: float = 0.2):
        self.beta = beta

    def __call__(self, *, ce_model, ce_ref, **_):
        assert ce_ref is not None, "NpoForget requires ce_ref"
        return -(2.0 / self.beta) * F.logsigmoid(self.beta * (ce_model - ce_ref)).mean()

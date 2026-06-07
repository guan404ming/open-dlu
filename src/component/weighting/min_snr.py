"""Min-SNR-γ weighting (Hang et al, ICCV 2023): bounded SNR.

For absorbing/mask diffusion, ``SNR(t) = (1-t) / t``. ``w(t) = min(SNR, γ)``
clamps the low-t spike that 1/t weighting suffers, while keeping the
SNR-style low-t emphasis. Bridges Uniform (γ→0) and 1/t (γ→∞).
"""
from src.component.weighting.base import TrajectoryWeighting

class MinSnr(TrajectoryWeighting):
    def __init__(self, gamma: float = 5.0, eps: float = 1e-3):
        self.gamma = gamma
        self.eps = eps

    def __call__(self, t: float) -> float:
        snr = (1.0 - t) / max(t, self.eps)
        return min(snr, self.gamma)

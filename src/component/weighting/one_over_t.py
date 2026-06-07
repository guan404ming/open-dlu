"""1/t weighting from the mask-diffusion ELBO."""

from src.component.weighting.base import TrajectoryWeighting


class OneOverT(TrajectoryWeighting):
    def __call__(self, t: float) -> float:
        return 1.0 / t

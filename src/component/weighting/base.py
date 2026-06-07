"""Trajectory weighting interface (ELBO-style coefficient on per-step losses)."""

from abc import ABC, abstractmethod


class TrajectoryWeighting(ABC):
    @abstractmethod
    def __call__(self, t: float) -> float: ...

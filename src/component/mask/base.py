"""Mask sampler interface."""
from abc import ABC, abstractmethod
import torch

class MaskSampler(ABC):
    """Maps (input_ids, t) → boolean mask of the same shape.

    Samplers advertise needs via class attributes so the pipeline knows
    when to forward through the frozen reference.
    """

    needs_frozen: bool = False

    @abstractmethod
    def __call__(
        self,
        input_ids: torch.Tensor,
        t: float,
        *,
        frozen: torch.nn.Module | None = None,
        mask_id: int = 0,
    ) -> torch.Tensor:
        """Return a boolean mask of the same shape as input_ids."""

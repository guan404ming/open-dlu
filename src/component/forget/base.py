"""ForgetLoss abstract interface."""
from abc import ABC, abstractmethod
import torch

class ForgetLoss(ABC):
    """Maps per-token CE on the forget batch to a scalar forget loss.

    Subclasses receive per-masked-token CE from both the trainable model
    (``ce_model``) and the frozen reference model (``ce_ref``), and any
    extra signals via ``**kwargs``.
    """

    needs_ref: bool = False

    @abstractmethod
    def __call__(
        self,
        *,
        ce_model: torch.Tensor,
        ce_ref: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        ...

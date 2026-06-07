"""RetainLoss abstract interface."""

from abc import ABC, abstractmethod
import torch


class RetainLoss(ABC):
    """Maps the retain batch to a scalar retain loss.

    Components advertise their requirements via class attributes so the
    pipeline knows which extra forward passes / captures are needed.
    """

    needs_frozen_logits: bool = False
    needs_hidden_states: bool = (
        False  # set True to receive model/frozen layer activations
    )
    is_null: bool = False  # if True, the pipeline skips this term's forward pass

    @abstractmethod
    def __call__(
        self,
        *,
        model_logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        frozen_logits: torch.Tensor | None = None,
        model_hidden: dict | None = None,
        frozen_hidden: dict | None = None,
    ) -> torch.Tensor: ...

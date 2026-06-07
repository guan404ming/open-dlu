"""Adapter interface: chooses which parameters of the model are trainable."""
from abc import ABC, abstractmethod
import torch

class Adapter(ABC):
    @abstractmethod
    def setup(self, model: torch.nn.Module) -> None:
        """Configure requires_grad on the trainable model in place."""

    @abstractmethod
    def state_dict(self, model: torch.nn.Module) -> dict:
        """Return the subset of state_dict the adapter actually changes."""

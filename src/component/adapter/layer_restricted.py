"""Layer-restricted adapter: freeze everything outside a chosen set of blocks."""
import torch

from src.component.adapter.base import Adapter

class LayerRestricted(Adapter):
    def __init__(self, layers=(5, 6, 7), block_key: str = None):
        self.layers = tuple(layers)
        self.block_key = block_key

    def _in_scope(self, name: str) -> bool:
        return any(f"{self.block_key}{i}." in name for i in self.layers)

    def _detect_block_key(self, model) -> str:
        """Auto-detect transformer block prefix (e.g. ``blocks.`` LLaDA, ``layers.`` Dream)."""
        names = [n for n, _ in model.named_modules()]
        for cand in ("blocks.", "layers.", "h."):
            if any(cand in n for n in names):
                return cand
        raise ValueError("Could not detect transformer block prefix in model")

    def setup(self, model):
        if self.block_key is None:
            self.block_key = self._detect_block_key(model)
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if self._in_scope(name):
                p.requires_grad = True

    def state_dict(self, model):
        return {k: v.detach().cpu() for k, v in model.state_dict().items()
                if self._in_scope(k)}

"""Independent per-token Bernoulli(t) mask."""
import torch

from src.component.mask.base import MaskSampler

class BernoulliMask(MaskSampler):
    def __call__(self, input_ids, t, **_):
        return torch.rand(input_ids.shape, device=input_ids.device) < t

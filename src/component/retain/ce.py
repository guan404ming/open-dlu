"""CE retain: cross-entropy on masked retain positions (no frozen reference)."""

import torch
import torch.nn.functional as F

from src.component.retain.base import RetainLoss


class CERetain(RetainLoss):
    needs_frozen_logits = False

    def __call__(self, *, model_logits, target, mask, **_):
        if mask.sum().item() == 0:
            return model_logits.sum() * 0.0
        return F.cross_entropy(model_logits[mask].float(), target[mask])

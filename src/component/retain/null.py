"""Null retain: no retain regularizer (forget-only objective)."""

from src.component.retain.base import RetainLoss


class NullRetain(RetainLoss):
    needs_frozen_logits = False

    def __call__(self, *, model_logits, target, mask, **_):
        return model_logits.sum() * 0.0

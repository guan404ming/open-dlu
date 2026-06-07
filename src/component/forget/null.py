"""Null forget: no forget objective (used by finetune = learn the corpus only)."""

from src.component.forget.base import ForgetLoss


class NullForget(ForgetLoss):
    is_null = True  # pipeline skips the forget pass entirely

    def __call__(self, *, ce_model, ce_ref=None, **_):
        return ce_model.sum() * 0.0

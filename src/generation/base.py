"""Sampler interface: turns a prompt into a generated answer string.

Generation is model-family-specific (LLaDA and Dream denoise differently), so
each model config points at its own sampler via `model.generator`. Metrics that
need generation receive the instantiated sampler and stay model-agnostic.
"""

from abc import ABC, abstractmethod


class Sampler(ABC):
    # Markers at which a decoded completion is cut (chat/EOS/turn boundaries).
    _STOPS = ("<|endoftext|>", "<|im_end|>", "\nQ:", "\n\n")

    @classmethod
    def _truncate(cls, text: str) -> str:
        """Cut a decoded completion at the first stop marker and strip it."""
        for stop in cls._STOPS:
            k = text.find(stop)
            if k >= 0:
                text = text[:k]
        return text.strip()

    @abstractmethod
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        mask_id: int,
        device: str,
        max_new: int = 64,
        chat: bool = True,
    ) -> str:
        """Return the model's answer to `prompt` as a decoded string.

        ``chat=True`` wraps the prompt in the model's chat template (QA tasks);
        ``chat=False`` feeds the prompt verbatim (raw text completion, e.g. MUSE
        verbatim memorization).
        """

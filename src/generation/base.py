"""Sampler interface: turns a prompt into a generated answer string.

Generation is model-family-specific (LLaDA and Dream denoise differently), so
each model config points at its own sampler via `model.generator`. Metrics that
need generation receive the instantiated sampler and stay model-agnostic.
"""

from abc import ABC, abstractmethod


class Sampler(ABC):
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

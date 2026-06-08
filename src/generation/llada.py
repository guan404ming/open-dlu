"""LLaDA sampler: official block-wise low-confidence remasking (generate.py)."""

import torch

from src.generation.base import Sampler


def _num_transfer(mask_index, steps):
    n = mask_index.sum(dim=1, keepdim=True)
    out = (
        torch.zeros(n.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + n // steps
    )
    for i in range(n.size(0)):
        out[i, : (n % steps)[i]] += 1
    return out


class LladaSampler(Sampler):
    def __init__(self, steps: int = 64, block_length: int = 32, max_new: int = 64):
        self.steps = steps
        self.block_length = block_length
        self.max_new = max_new

    def _format(self, tok, q: str, chat: bool = True) -> str:
        if not chat:
            return q
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"Q: {q}\nA:"

    @torch.no_grad()
    def generate(
        self, model, tokenizer, prompt, mask_id, device, max_new=None, chat=True
    ):
        gen_length = max_new or self.max_new
        pids = torch.tensor(
            tokenizer(
                self._format(tokenizer, prompt, chat), add_special_tokens=False
            ).input_ids,
            dtype=torch.long,
            device=device,
        )[None, :]
        plen = pids.shape[1]
        x = torch.full((1, plen + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :plen] = pids
        n_blocks = gen_length // self.block_length
        spb = self.steps // n_blocks
        for b in range(n_blocks):
            lo, hi = plen + b * self.block_length, plen + (b + 1) * self.block_length
            n_xfer = _num_transfer((x[:, lo:hi] == mask_id), spb)
            for i in range(spb):
                mi = x == mask_id
                logits = model(x).logits
                x0 = logits.argmax(-1)
                p = logits.softmax(-1).gather(-1, x0[..., None]).squeeze(-1)
                p[:, hi:] = -float("inf")
                x0 = torch.where(mi, x0, x)
                conf = torch.where(mi, p, torch.full_like(p, -float("inf")))
                for j in range(conf.shape[0]):
                    _, sel = torch.topk(conf[j], k=int(n_xfer[j, i]))
                    x[j, sel] = x0[j, sel]
        text = tokenizer.decode(x[0, plen:].tolist(), skip_special_tokens=True)
        for stop in ["<|endoftext|>", "<|im_end|>", "\nQ:", "\n\n"]:
            k = text.find(stop)
            if k >= 0:
                text = text[:k]
        return text.strip()

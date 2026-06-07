"""Dream sampler: Dream's official ``diffusion_generate`` (entropy remasking)."""
import torch

from src.generation.base import Sampler


class DreamSampler(Sampler):
    def __init__(self, steps: int = 64, max_new: int = 64,
                 alg: str = "entropy", temperature: float = 0.0):
        self.steps = steps
        self.max_new = max_new
        self.alg = alg
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, model, tokenizer, prompt, mask_id, device, max_new=None, chat=True):
        gen_length = max_new or self.max_new
        if chat:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt", add_generation_prompt=True,
            ).to(device)
        else:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.diffusion_generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=gen_length,
            steps=self.steps,
            temperature=self.temperature,
            alg=self.alg,
            alg_temp=0.0,
            output_history=False,
            return_dict_in_generate=True,
        )
        text = tokenizer.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        for stop in ["<|endoftext|>", "<|im_end|>", "\nQ:", "\n\n"]:
            k = text.find(stop)
            if k >= 0:
                text = text[:k]
        return text.strip()

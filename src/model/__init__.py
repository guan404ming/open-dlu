"""Model loading. The id + mask_id live in configs/model/*.yaml."""

import torch
from transformers import AutoModel, AutoTokenizer


def load_model(model_id: str, device: str = "cuda:0", eval_mode: bool = False):
    """Return (model, tokenizer) for an MDLM, bf16 + trust_remote_code."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if eval_mode:
        model.eval()
    return model, tok

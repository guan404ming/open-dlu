"""Model loading. The id + mask_id live in configs/model/*.yaml."""

import json
import os

import torch
from transformers import AutoModel, AutoTokenizer


def _tokenizer(model_id: str):
    """Load the tokenizer, falling back to the base repo when a re-saved
    checkpoint has had its custom tokenizer code stripped (the save canonicalises
    auto_map to the remote repo and drops the local *.py)."""
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (OSError, EnvironmentError):
        cfg = os.path.join(model_id, "config.json")
        amap = json.load(open(cfg)).get("auto_map", {}) if os.path.exists(cfg) else {}
        base = next((v.split("--")[0] for v in amap.values() if "--" in v), None)
        if not base:
            raise
        return AutoTokenizer.from_pretrained(base, trust_remote_code=True)


def load_model(model_id: str, device: str = "cuda:0", eval_mode: bool = False):
    """Return (model, tokenizer) for an MDLM, bf16 + trust_remote_code."""
    tok = _tokenizer(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if eval_mode:
        model.eval()
    return model, tok

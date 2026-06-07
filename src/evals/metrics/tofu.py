"""TOFU (Maini 2024) generation metric: rougeL recall on forget / retain QA.

Model-agnostic: generation goes through the model's `generator` (Sampler).
"""
import json

from huggingface_hub import hf_hub_download

from src.evals.metrics.base import unlearning_metric
from src.evals.metrics._text import rouge_l_recall


def _qa(split):
    path = hf_hub_download("locuslab/TOFU", f"{split}.json", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _gen_rouge(generator, model, tok, qa, mask_id, device, cap):
    rls = []
    for x in qa[:cap]:
        pred = generator.generate(model, tok, x["question"], mask_id, device)
        rls.append(rouge_l_recall(pred, x["answer"]))
    return sum(rls) / max(len(rls), 1)


@unlearning_metric(name="tofu")
def tofu(model, tokenizer, mask_id, device, generator=None,
         forget_split="forget10", retain_split="retain_perturbed",
         max_eval=100, **kw):
    assert generator is not None, "tofu metric needs a Sampler (set model.generator)"
    return {
        "tofu_forget_rougeL": _gen_rouge(
            generator, model, tokenizer, _qa(forget_split), mask_id, device, max_eval),
        "tofu_retain_rougeL": _gen_rouge(
            generator, model, tokenizer, _qa(retain_split), mask_id, device, max_eval),
    }

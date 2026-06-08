"""RWKU (Cao 2024) generation metric: rougeL recall on forget / neighbor probes.

Model-agnostic: generation is delegated to the model's `generator` (a Sampler),
so this metric works for any MDLM that provides one.
"""

from datasets import load_dataset

from src.evals.metrics.base import unlearning_metric
from src.evals.metrics._text import rouge_l_recall


def _probe(generator, model, tok, configs, names, mask_id, device, cap):
    rls = []
    for cfg in configs:
        rows = [
            r
            for r in load_dataset("jinzhuoran/RWKU", cfg, split="test")
            if r["subject"] in names
        ][:cap]
        for r in rows:
            pred = generator.generate(model, tok, r["query"], mask_id, device)
            rls.append(rouge_l_recall(pred, r["answer"]))
    return sum(rls) / max(len(rls), 1)


@unlearning_metric(name="rwku")
def rwku(
    model,
    tokenizer,
    mask_id,
    device,
    generator=None,
    n_entities=10,
    max_per_split=100,
    **kw,
):
    assert generator is not None, "rwku metric needs a Sampler (set model.generator)"
    ft = load_dataset("jinzhuoran/RWKU", "forget_target", split="train").select(
        range(n_entities)
    )
    names = {r["target"] for r in ft}
    return {
        "rwku_forget_rougeL": _probe(
            generator,
            model,
            tokenizer,
            ("forget_level1", "forget_level2"),
            names,
            mask_id,
            device,
            max_per_split,
        ),
        "rwku_neighbor_rougeL": _probe(
            generator,
            model,
            tokenizer,
            ("neighbor_level1", "neighbor_level2"),
            names,
            mask_id,
            device,
            max_per_split,
        ),
    }

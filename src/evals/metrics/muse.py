"""MUSE (Shi 2024) generation metrics, aligned to OpenUnlearning's defaults.

Three rougeL-f1 scores: verbatim memorization (raw text continuation) and
knowledge memorization QA on the forget and retain splits (few-shot, raw
completion). Model-agnostic: generation goes through the model's `generator`.

OU additionally reports privleak / extraction_strength, which need loss/MIA
probes rather than generation; left out until those probes exist here.
"""

from datasets import load_dataset

from src.evals.metrics.base import unlearning_metric
from src.evals.metrics._text import rouge_l_f1


def _verbmem(generator, model, tok, path, mask_id, device, cap):
    rows = list(load_dataset(path, "verbmem", split="forget"))[:cap]
    rls = [
        rouge_l_f1(
            generator.generate(
                model, tok, r["prompt"], mask_id, device, max_new=128, chat=False
            ),
            r["gt"],
        )
        for r in rows
    ]
    return sum(rls) / max(len(rls), 1)


def _knowmem(generator, model, tok, path, qa_split, icl_split, mask_id, device, cap):
    icl = list(load_dataset(path, "knowmem", split=icl_split))[:5]
    shots = "".join(
        f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in icl
    )
    rows = list(load_dataset(path, "knowmem", split=qa_split))[:cap]
    rls = []
    for r in rows:
        pred = generator.generate(
            model,
            tok,
            f"{shots}Question: {r['question']}\nAnswer:",
            mask_id,
            device,
            max_new=32,
            chat=False,
        )
        rls.append(rouge_l_f1(pred, r["answer"]))
    return sum(rls) / max(len(rls), 1)


@unlearning_metric(name="muse")
def muse(
    model,
    tokenizer,
    mask_id,
    device,
    generator=None,
    data_split="News",
    max_eval=100,
    **kw,
):
    assert generator is not None, "muse metric needs a Sampler (set model.generator)"
    path = f"muse-bench/MUSE-{data_split}"
    return {
        "forget_verbmem_ROUGE": _verbmem(
            generator, model, tokenizer, path, mask_id, device, max_eval
        ),
        "forget_knowmem_ROUGE": _knowmem(
            generator,
            model,
            tokenizer,
            path,
            "forget_qa",
            "forget_qa_icl",
            mask_id,
            device,
            max_eval,
        ),
        "retain_knowmem_ROUGE": _knowmem(
            generator,
            model,
            tokenizer,
            path,
            "retain_qa",
            "retain_qa_icl",
            mask_id,
            device,
            max_eval,
        ),
    }

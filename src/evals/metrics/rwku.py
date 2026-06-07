"""RWKU (Cao 2024) generation metric: rougeL recall on forget / neighbor probes.

Generation-based, unlike the MCQ metrics: it answers each probe query with the
model and scores rougeL recall against the gold answer. Uses LLaDA's official
block-wise low-confidence remasking sampler (LLaDA-specific).
"""
import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.evals.metrics.base import unlearning_metric

_SCORER = None


def _rouge_l_recall(pred: str, gt: str) -> float:
    global _SCORER
    if not pred or not gt:
        return 0.0
    if _SCORER is None:
        from rouge_score import rouge_scorer

        _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _SCORER.score(gt, pred)["rougeL"].recall


def _format_prompt(tok, question: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"Q: {question}\nA:"


def _num_transfer(mask_index, steps):
    n = mask_index.sum(dim=1, keepdim=True)
    out = torch.zeros(n.size(0), steps, device=mask_index.device, dtype=torch.int64) + n // steps
    for i in range(n.size(0)):
        out[i, : (n % steps)[i]] += 1
    return out


@torch.no_grad()
def _generate(model, prompt_ids, mask_id, gen_length=64, steps=64, block_length=32):
    """Greedy block-wise low-confidence remasking (LLaDA generate.py)."""
    x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
                   mask_id, dtype=torch.long, device=prompt_ids.device)
    x[:, : prompt_ids.shape[1]] = prompt_ids
    n_blocks = gen_length // block_length
    spb = steps // n_blocks
    for b in range(n_blocks):
        lo = prompt_ids.shape[1] + b * block_length
        hi = lo + block_length
        n_xfer = _num_transfer((x[:, lo:hi] == mask_id), spb)
        for i in range(spb):
            mask_index = x == mask_id
            logits = model(x).logits
            x0 = logits.argmax(dim=-1)
            p = logits.softmax(dim=-1).gather(-1, x0[..., None]).squeeze(-1)
            p[:, hi:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, p, torch.full_like(p, -float("inf")))
            for j in range(conf.shape[0]):
                _, sel = torch.topk(conf[j], k=int(n_xfer[j, i]))
                x[j, sel] = x0[j, sel]
    return x


@torch.no_grad()
def _answer(model, tok, query, mask_id, device, max_new=64):
    pids = torch.tensor(tok(_format_prompt(tok, query), add_special_tokens=False).input_ids,
                        dtype=torch.long, device=device)[None, :]
    out = _generate(model, pids, mask_id, gen_length=max_new)
    text = tok.decode(out[0, pids.shape[1]:].tolist(), skip_special_tokens=True)
    for stop in ["<|endoftext|>", "<|im_end|>", "\nQ:", "\n\n"]:
        i = text.find(stop)
        if i >= 0:
            text = text[:i]
    return text.strip()


def _probe(model, tok, configs, names, mask_id, device, cap):
    rls = []
    for cfg in configs:
        rows = [r for r in load_dataset("jinzhuoran/RWKU", cfg, split="test")
                if r["subject"] in names][:cap]
        for r in rows:
            rls.append(_rouge_l_recall(_answer(model, tok, r["query"], mask_id, device), r["answer"]))
    return sum(rls) / max(len(rls), 1)


@unlearning_metric(name="rwku")
def rwku(model, tokenizer, mask_id, device, n_entities=10, max_per_split=100, **kw):
    ft = load_dataset("jinzhuoran/RWKU", "forget_target", split="train").select(range(n_entities))
    names = {r["target"] for r in ft}
    return {
        "rwku_forget_rougeL": _probe(
            model, tokenizer, ("forget_level1", "forget_level2"),
            names, mask_id, device, max_per_split),
        "rwku_neighbor_rougeL": _probe(
            model, tokenizer, ("neighbor_level1", "neighbor_level2"),
            names, mask_id, device, max_per_split),
    }

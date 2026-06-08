"""TOFU (Maini 2024) metrics, following MDU's protocol (Lee 2026, arXiv 2605.18253).

Two metrics on each of four splits (forget / retain / real_authors / world_facts):
  rL : rougeL recall between a sampled answer and the ground truth (via generator).
  p  : per-token answer probability exp(-L_rec), where L_rec is a Monte-Carlo
       reconstruction NLL over randomly masked answer positions (paper Eq.7). It
       is model-family-agnostic, needing only a forward pass and the mask id.

Matches MDU's eval_tofu_llada.py: rougeL *recall* and the Eq.(14) estimator.
"""

import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.data import encode_qa
from src.evals.metrics.base import unlearning_metric
from src.evals.metrics._text import rouge_l_recall

# Output-key prefix -> TOFU split file. real_authors / world_facts carry MCQ
# options too, but we score the `answer` field exactly as MDU does.
SPLITS = {
    "forget": "forget10.json",
    "retain": "retain_perturbed.json",
    "real_authors": "real_authors.json",
    "world_facts": "world_facts.json",
}


def _load(split_file):
    path = hf_hub_download("locuslab/TOFU", split_file, repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build(tok, question, answer, device):
    prompt, ans = encode_qa(tok, question, answer)
    ids = torch.tensor(prompt + ans, dtype=torch.long, device=device)[None, :]
    ans_idx = torch.arange(len(prompt), len(prompt) + len(ans), device=device)
    return ids, ans_idx


@torch.no_grad()
def _answer_prob(model, ids, ans_idx, mask_id, n_mc, mc_batch):
    """p = exp(-L_rec): MC reconstruction NLL over masked answer subsets (Eq.7)."""
    L = ans_idx.numel()
    if L == 0:
        return 0.0
    seq = ids.expand(n_mc, -1).clone()
    flags = torch.zeros(n_mc, ids.shape[1], dtype=torch.bool, device=ids.device)
    norms = []
    for s in range(n_mc):
        ln = int(torch.randint(1, L + 1, (1,)))
        sel = ans_idx[torch.randperm(L, device=ids.device)[:ln]]
        seq[s, sel] = mask_id
        flags[s, sel] = True
        norms.append(ln)
    tgt = ids.expand(n_mc, -1)
    total = 0.0
    for i in range(0, n_mc, mc_batch):
        j = min(i + mc_batch, n_mc)
        nll = F.cross_entropy(
            model(seq[i:j]).logits.transpose(1, 2), tgt[i:j], reduction="none"
        )
        for k in range(j - i):
            total += nll[k][flags[i + k]].sum().item() / norms[i + k]
    return float(torch.tensor(-total / n_mc).exp())


def _eval_split(model, tok, generator, rows, mask_id, device, n_mc, mc_batch):
    rl, pr = [], []
    for r in rows:
        pred = generator.generate(
            model, tok, r["question"], mask_id, device, max_new=128
        )
        rl.append(rouge_l_recall(pred, r["answer"]))
        ids, ans_idx = _build(tok, r["question"], r["answer"], device)
        pr.append(_answer_prob(model, ids, ans_idx, mask_id, n_mc, mc_batch))
    n = max(len(rows), 1)
    return sum(rl) / n, sum(pr) / n


@unlearning_metric(name="tofu")
def tofu(
    model,
    tokenizer,
    mask_id,
    device,
    generator=None,
    max_eval=100,
    n_mc=128,
    mc_batch=16,
    **kw,
):
    assert generator is not None, "tofu metric needs a Sampler (set model.generator)"
    torch.manual_seed(42)
    out = {}
    for prefix, fname in SPLITS.items():
        rl, pr = _eval_split(
            model,
            tokenizer,
            generator,
            _load(fname)[:max_eval],
            mask_id,
            device,
            n_mc,
            mc_batch,
        )
        out[f"tofu_{prefix}_rougeL"] = rl
        out[f"tofu_{prefix}_prob"] = pr
    return out

"""Corpora: load a dataset into training items.

A corpus is any object with ``load(tokenizer, n_chunks, seq_len) -> items``,
where each item is either a bare token list (the whole sequence is trainable,
e.g. text corpora) or an ``(ids, loss_mask)`` pair (only the masked-in span is
trainable, e.g. answer-supervised QA). ``HFTextCorpus`` covers any HF dataset
with a text column (wmdp-bio, wikitext, ...); ``TofuCorpus`` is answer-masked QA.
Add a new one by writing a class and a ``configs/data/*.yaml`` that points at it.
"""

import random

import torch
from datasets import load_dataset


def chunkify(texts, tokenizer, n_chunks, seq_len, min_len=50):
    ids = []
    for t in texts:
        if not t or len(t) < min_len:
            continue
        ids += tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) >= n_chunks * seq_len:
            break
    n = min(n_chunks, len(ids) // seq_len)
    return [ids[i * seq_len : (i + 1) * seq_len] for i in range(n)]


def encode_qa(tokenizer, question, answer):
    """Tokenize a chat-template QA pair into ``(prompt_ids, answer_ids)``.

    The prompt is wrapped in the model's chat template with a generation prompt;
    the answer is appended raw. Shared by ``TofuCorpus`` (answer-masked SFT) and
    the TOFU metric (answer-probability reconstruction).
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ans = tokenizer(answer, add_special_tokens=False).input_ids
    return prompt, ans


def batch_sampler(items, batch_size, pad_id=0):
    """Sample a padded batch as ``(ids, loss_mask)``.

    Items are either bare token lists or ``(ids, loss_mask)`` pairs. The loss
    mask flows to the pipeline, which never masks a non-trainable position; an
    equal-length, fully-trainable corpus behaves exactly as a plain ``[B, L]``
    batch (loss mask all-True, no padding).
    """
    norm = [x if isinstance(x, tuple) else (x, [1] * len(x)) for x in items]

    def sample():
        batch = random.sample(norm, batch_size)
        L = max(len(a) for a, _ in batch)
        ids = torch.full((batch_size, L), pad_id, dtype=torch.long)
        lm = torch.zeros((batch_size, L), dtype=torch.bool)
        for i, (a, m) in enumerate(batch):
            ids[i, : len(a)] = torch.tensor(a, dtype=torch.long)
            lm[i, : len(m)] = torch.tensor(m, dtype=torch.bool)
        return ids, lm

    return sample


class HFTextCorpus:
    """Any HuggingFace dataset exposing a text column."""

    def __init__(self, path, split="train", name=None, text_key="text", min_len=50):
        self.path = path
        self.split = split
        self.name = name
        self.text_key = text_key
        self.min_len = min_len

    def load(self, tokenizer, n_chunks, seq_len):
        ds = load_dataset(self.path, self.name, split=self.split)
        return chunkify(
            (x.get(self.text_key, "") for x in ds),
            tokenizer,
            n_chunks,
            seq_len,
            self.min_len,
        )


class MmluCorpus:
    """MMLU MCQ formatted as QA text, for use as a retain anchor. Uses the
    validation split so it stays disjoint from the test split the mmlu metric
    scores; the prompt format matches the evaluator."""

    def __init__(self, split: str = "validation"):
        self.split = split

    def load(self, tokenizer, n_chunks, seq_len):
        from src.evals.metrics.mcq import MMLU_SUBJECTS

        texts = []
        for subj in MMLU_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subj, split=self.split)
            except Exception:
                continue
            for x in ds:
                opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", x["choices"]))
                texts.append(f"{x['question']}\n{opts}\nAnswer: {'ABCD'[x['answer']]}")
        return chunkify(texts, tokenizer, n_chunks, seq_len)


class RwkuCorpus:
    """RWKU (Cao 2024) forget entities: intro bios + cloze/QA probes, chunked."""

    def __init__(self, n_entities: int = 10, repeat: int = 300):
        self.n_entities = n_entities
        self.repeat = repeat

    def load(self, tokenizer, n_chunks, seq_len):
        ft = load_dataset("jinzhuoran/RWKU", "forget_target", split="train").select(
            range(self.n_entities)
        )
        names = {r["target"] for r in ft}
        texts = [r["intro"] for r in ft]
        for cfg in ("forget_level1", "forget_level2"):
            ds = load_dataset("jinzhuoran/RWKU", cfg, split="test")
            texts += [
                f"{r['query']} {r['answer']}" for r in ds if r["subject"] in names
            ]
        return chunkify(texts * self.repeat, tokenizer, n_chunks, seq_len)


class TofuCorpus:
    """TOFU (Maini 2024) fictional-author QA, answer-masked.

    Each item is ``(ids, loss_mask)`` where ``ids`` is the chat prompt followed
    by the answer and only the answer span is trainable, matching the MDLM SFT /
    unlearning regime (the prompt is kept as clean context).
    """

    def __init__(self, split: str = "forget10"):
        self.split = split

    def load(self, tokenizer, n_chunks, seq_len):
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "locuslab/TOFU", f"{self.split}.json", repo_type="dataset"
        )
        with open(path) as f:
            qa = [json.loads(line) for line in f if line.strip()]
        items = []
        eos = tokenizer.eos_token_id
        for x in qa[:n_chunks]:
            prompt, ans = encode_qa(tokenizer, x["question"], x["answer"])
            if eos is not None:
                ans = ans + [eos]  # train termination so generation stops cleanly
            ids = (prompt + ans)[:seq_len]
            loss_mask = ([0] * len(prompt) + [1] * len(ans))[:seq_len]
            if any(loss_mask):
                items.append((ids, loss_mask))
        return items

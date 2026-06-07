"""Corpora: load text -> tokenize -> fixed-length chunks.

A corpus is any object with ``load(tokenizer, n_chunks, seq_len) -> list[list[int]]``.
Add a new one by writing a class and a ``configs/data/*.yaml`` that points at it;
nothing else changes. ``HFTextCorpus`` covers any HF dataset with a text column
(wmdp-bio, wikitext, ...); QA corpora like TOFU get their own class later.
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


def batch_sampler(chunks, batch_size):
    return lambda: torch.tensor(random.sample(chunks, batch_size), dtype=torch.long)


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

"""Build components from a composed config and train. Shared by train.py / run.py."""

import time

from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer.config import TrainConfig
from src.data import batch_sampler
from src.model import load_model
from src.trainer.pipeline import UnlearnPipeline


def _needs_frozen(forget, retain, mask) -> bool:
    return (
        getattr(forget, "needs_ref", False)
        or getattr(forget, "needs_uncond_self", False)
        or getattr(retain, "needs_frozen_logits", False)
        or getattr(retain, "needs_hidden_states", False)
        or getattr(mask, "needs_frozen", False)
    )


def build_and_train(cfg, device: str = "cuda:0") -> dict:
    t = cfg.trainer
    forget = instantiate(t.forget)
    retain = instantiate(t.retain)
    weighting = instantiate(t.weighting)
    adapter = instantiate(t.adapter)
    mask = instantiate(t.mask)
    train_cfg = TrainConfig(**OmegaConf.to_container(t.args, resolve=True))

    print(f"[load] {cfg.model.model_id}")
    model, tok = load_model(cfg.model.model_id, device)
    frozen = (
        load_model(cfg.model.model_id, device, eval_mode=True)[0]
        if _needs_frozen(forget, retain, mask)
        else None
    )

    fc = instantiate(cfg.data.forget).load(tok, cfg.data.n_chunks, train_cfg.seq_len)
    rc = instantiate(cfg.data.retain).load(tok, cfg.data.n_chunks, train_cfg.seq_len)
    print(f"[data] forget={len(fc)} retain={len(rc)} chunks")

    pipe = UnlearnPipeline(
        model, frozen, forget, retain, weighting, adapter, mask, train_cfg
    )
    t0 = time.time()
    pipe.train(
        batch_sampler(fc, train_cfg.batch_forget),
        batch_sampler(rc, train_cfg.batch_retain),
        log_every=1,
    )
    train_min = (time.time() - t0) / 60

    scores = {}
    if cfg.get("eval"):
        from src.evals import run_evaluators

        model.eval()
        generator = instantiate(cfg.model.generator) if cfg.model.get("generator") else None
        scores = run_evaluators(
            cfg.eval,
            model=model,
            tokenizer=tok,
            mask_id=train_cfg.mask_id,
            device=device,
            generator=generator,
        )
        print(f"[eval] {scores}")

    return {
        "train_min": train_min,
        "steps": train_cfg.steps,
        "model": cfg.model.name,
        "scores": scores,
    }

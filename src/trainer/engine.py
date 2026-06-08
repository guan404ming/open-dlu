"""Build components from a composed config and train. Shared by train.py / run.py."""

import math
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer.config import TrainConfig
from src.data import batch_sampler
from src.model import load_model
from src.trainer.pipeline import UnlearnPipeline


def _canonicalize_auto_map(out_dir: str) -> None:
    """Point a re-saved remote-code checkpoint's auto_map back at one consistent
    source. ``save_pretrained`` copies only part of the custom code locally and
    rewrites some auto_map entries to local while others stay remote; the two
    then resolve to different config classes and loading fails. Force every
    entry to the original remote repo (and drop the partial local copies), which
    is the form that reloads cleanly."""
    import glob
    import json
    import os

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    amap = cfg.get("auto_map")
    repo = next((v.split("--")[0] for v in (amap or {}).values() if "--" in v), None)
    if not repo:
        return  # no remote ref to anchor on; leave a fully-local save as-is
    cfg["auto_map"] = {k: f"{repo}--{v.split('--')[-1]}" for k, v in amap.items()}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    for py in glob.glob(os.path.join(out_dir, "*.py")):
        os.remove(py)


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

    # A null term needs no corpus (finetune drops forget, GA drops retain).
    nc, sl = cfg.data.n_chunks, train_cfg.seq_len
    fc = [] if forget.is_null else instantiate(cfg.data.forget).load(tok, nc, sl)
    rc = [] if retain.is_null else instantiate(cfg.data.retain).load(tok, nc, sl)
    print(f"[data] forget={len(fc)} retain={len(rc)} items")

    if train_cfg.epochs:  # convert epochs over the active corpus into optimizer steps
        n = len(fc) or len(rc)
        bs = train_cfg.batch_forget if fc else train_cfg.batch_retain
        eff = max(bs * train_cfg.grad_accum, 1)
        train_cfg.steps = max(1, train_cfg.epochs * math.ceil(n / eff))
        print(
            f"[epochs] {train_cfg.epochs} x ceil({n}/{eff}) -> {train_cfg.steps} steps"
        )

    pipe = UnlearnPipeline(
        model, frozen, forget, retain, weighting, adapter, mask, train_cfg
    )
    t0 = time.time()
    pipe.train(
        batch_sampler(fc, train_cfg.batch_forget, train_cfg.mask_id),
        batch_sampler(rc, train_cfg.batch_retain, train_cfg.mask_id),
        log_every=1,
    )
    train_min = (time.time() - t0) / 60

    scores = {}
    if cfg.get("eval"):
        from src.evals import run_evaluators

        model.eval()
        generator = (
            instantiate(cfg.model.generator) if cfg.model.get("generator") else None
        )
        scores = run_evaluators(
            cfg.eval,
            model=model,
            tokenizer=tok,
            mask_id=train_cfg.mask_id,
            device=device,
            generator=generator,
        )
        print(f"[eval] {scores}")

    # Persist the trained model as a HuggingFace dir so it can be reused as a
    # `model.model_id` (e.g. a finetuned TOFU / MUSE target to then unlearn).
    if cfg.get("output_dir"):
        import os

        pipe.acc.unwrap_model(pipe.model).save_pretrained(cfg.output_dir)
        tok.save_pretrained(cfg.output_dir)
        _canonicalize_auto_map(cfg.output_dir)
        # Pin the exact recipe next to the weights so a checkpoint chain stays
        # reproducible even when configs are edited between runs. `data` records
        # the corpus actually trained on (forget/retain item counts) since a
        # capped n_chunks can differ from what loaded.
        snap = OmegaConf.create(
            {"config": cfg, "resolved": {"steps": train_cfg.steps,
                                         "forget_items": len(fc),
                                         "retain_items": len(rc)}}
        )
        OmegaConf.save(snap, os.path.join(cfg.output_dir, "run_config.yaml"))
        print(f"[save] -> {cfg.output_dir}  (steps={train_cfg.steps} "
              f"forget={len(fc)} retain={len(rc)})")

    return {
        "train_min": train_min,
        "steps": train_cfg.steps,
        "model": cfg.model.name,
        "scores": scores,
    }

"""Composable MDLM unlearning loop.

Method-agnostic: the loop never names a method. Each forget/retain component
declares the extra signals it needs (a frozen reference, an unconditional pass,
hidden states) via flags, and the loop supplies exactly those.
"""
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from src.trainer.config import TrainConfig


class UnlearnPipeline:
    def __init__(self, model, frozen, forget_loss, retain_loss, weighting,
                 adapter, mask_sampler, config: TrainConfig, device="cuda:0"):
        self.model = model
        self.frozen = frozen
        self.forget_loss = forget_loss
        self.retain_loss = retain_loss
        self.weighting = weighting
        self.adapter = adapter
        self.mask_sampler = mask_sampler
        self.cfg = config
        self.device = device

        adapter.setup(model)
        if frozen is not None:
            for p in frozen.parameters():
                p.requires_grad = False
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if config.use_8bit_optim:
            import bitsandbytes as bnb
            self.optim = bnb.optim.AdamW8bit(trainable, lr=config.lr)
        else:
            self.optim = torch.optim.AdamW(trainable, lr=config.lr)
        print(f"[pipeline] trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params")

    # --- helpers ---
    def _lr_scale(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return (step + 1) / c.warmup_steps
        if c.cosine_decay:
            prog = (step - c.warmup_steps) / max(1, c.steps - c.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * prog))
        return 1.0

    def _sample_t(self) -> float:
        c = self.cfg
        return c.t_min + (c.t_max - c.t_min) * torch.rand(()).item()

    def _mask(self, ids, t):
        kw = {"mask_id": self.cfg.mask_id}
        if self.mask_sampler.needs_frozen:
            kw["frozen"] = self.frozen
        if getattr(self.mask_sampler, "needs_model", False):
            kw["model"] = self.model
        return self.mask_sampler(ids, t, **kw)

    def _ce(self, logits, target, mask):
        return F.cross_entropy(logits[mask].float(), target[mask], reduction="none")

    def _hooks(self, model, layers, out: dict) -> list:
        prefix = getattr(self.adapter, "block_key", None) or "blocks."
        hooks = []
        for name, mod in model.named_modules():
            for i in layers:
                if name.endswith(f"{prefix}{i}"):
                    hooks.append(mod.register_forward_hook(
                        lambda _m, _i, o, k=i:
                        out.__setitem__(k, o[0] if isinstance(o, tuple) else o)))
        return hooks

    # --- loss terms ---
    def _forget_term(self, ids, t, w):
        c, fl = self.cfg, self.forget_loss
        mask = self._mask(ids, t)
        x = torch.where(mask, c.mask_id, ids)
        hid, hooks = {}, []
        if getattr(fl, "needs_hidden", False):
            hooks = self._hooks(self.model, getattr(fl, "hidden_layers", (7,)), hid)
        logits = self.model(x).logits
        for h in hooks:
            h.remove()
        if not mask.any():
            return logits.sum() * 0.0, 0.0

        ce_model = self._ce(logits, ids, mask)
        kw = {"ce_model": ce_model, "ce_ref": None, "t": t}
        if fl.needs_ref:
            with torch.no_grad():
                kw["ce_ref"] = self._ce(self.frozen(x).logits, ids, mask).detach()
        if getattr(fl, "needs_uncond_self", False):
            with torch.no_grad():
                u = torch.full_like(ids, c.mask_id)
                kw["ce_uncond"] = self._ce(self.model(u).logits, ids, mask).detach()
        if getattr(fl, "needs_hidden", False):
            kw["model_hidden"] = hid
        return w * fl(**kw), ce_model.mean().item()

    def _retain_term(self, ids, t, w):
        c, rl = self.cfg, self.retain_loss
        mask = self._mask(ids, t)
        x = torch.where(mask, c.mask_id, ids)
        mh, fh, hooks = {}, {}, []
        if rl.needs_hidden_states:
            layers = getattr(rl, "hidden_layers", (6,))
            hooks = self._hooks(self.model, layers, mh) + self._hooks(self.frozen, layers, fh)
        logits = self.model(x).logits
        ref = None
        if rl.needs_frozen_logits or rl.needs_hidden_states:
            with torch.no_grad():
                out = self.frozen(x)
            ref = out.logits if rl.needs_frozen_logits else None
        for h in hooks:
            h.remove()
        loss = rl(model_logits=logits, target=ids, mask=mask, frozen_logits=ref,
                  model_hidden=mh or None, frozen_hidden=fh or None)
        return c.alpha_retain * w * loss

    def _step(self, fids, rids, step):
        c = self.cfg
        for g in self.optim.param_groups:
            g["lr"] = c.lr * self._lr_scale(step)
        t_f, t_r = self._sample_t(), self._sample_t()
        forget, ce = self._forget_term(fids, t_f, self.weighting(t_f))
        retain = self._retain_term(rids, t_r, self.weighting(t_r))
        loss = forget + retain
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], c.grad_clip)
        self.optim.step()
        return loss.item(), ce

    def train(self, get_forget, get_retain, log_every: int = 100):
        c = self.cfg
        self.model.train()
        t0 = time.time()
        for step in range(c.steps):
            loss, ce = self._step(get_forget().to(self.device),
                                  get_retain().to(self.device), step)
            if (step + 1) % log_every == 0:
                print(f"[step {step + 1}/{c.steps}] ce={ce:.2f} loss={loss:.3f}")
        print(f"[train] done in {(time.time() - t0) / 60:.1f} min")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.adapter.state_dict(self.model), path)
        print(f"[save] -> {path}")

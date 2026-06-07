"""Composable MDLM unlearning loop, on top of HuggingFace Accelerate.

Accelerate owns the mechanical parts (device placement, gradient accumulation,
gradient clipping, multi-GPU); this class owns the *diffusion* parts (sampling a
mask ratio, masking, and the per-token loss).

It is method-agnostic: the loop never names a method. Each forget/retain
component declares the extra signals it needs (a frozen reference, an
unconditional pass, hidden states) via flags, and the loop supplies exactly
those. The objective is always  loss = gamma * forget + alpha * retain.
"""

import os
import random
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import get_scheduler

from src.trainer.config import TrainConfig


class UnlearnPipeline:
    def __init__(
        self,
        model,
        frozen,
        forget_loss,
        retain_loss,
        weighting,
        adapter,
        mask_sampler,
        config: TrainConfig,
    ):
        self.forget_loss = forget_loss
        self.retain_loss = retain_loss
        self.weighting = weighting
        self.adapter = adapter
        self.mask_sampler = mask_sampler
        self.cfg = config

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        # Adapter picks the trainable parameters (must run before the optimizer).
        adapter.setup(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if config.use_8bit_optim:
            import bitsandbytes as bnb

            optim = bnb.optim.AdamW8bit(trainable, lr=config.lr)
        else:
            optim = torch.optim.AdamW(trainable, lr=config.lr)
        sched = get_scheduler(
            config.scheduler,
            optim,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.steps,
        )

        self.acc = Accelerator(gradient_accumulation_steps=config.grad_accum)
        self.model, self.optim, self.sched = self.acc.prepare(model, optim, sched)
        self.frozen = (
            self.acc.prepare_model(frozen, evaluation_mode=True)
            if frozen is not None
            else None
        )
        print(
            f"[pipeline] trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params"
        )

    # --- diffusion helpers ---
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
        """Per-masked-token cross-entropy (fp32 for stability)."""
        return F.cross_entropy(logits[mask].float(), target[mask], reduction="none")

    def _hooks(self, model, layers, out: dict) -> list:
        """Capture block outputs at `layers` into `out` (for activation-space methods)."""
        prefix = getattr(self.adapter, "block_key", None) or "blocks."
        hooks = []
        for name, mod in model.named_modules():
            for i in layers:
                if name.endswith(f"{prefix}{i}"):
                    hooks.append(
                        mod.register_forward_hook(
                            lambda _m, _i, o, k=i: out.__setitem__(
                                k, o[0] if isinstance(o, tuple) else o
                            )
                        )
                    )
        return hooks

    # --- loss terms (component-driven; the loop never special-cases a method) ---
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
            hooks = self._hooks(self.model, layers, mh) + self._hooks(
                self.frozen, layers, fh
            )
        logits = self.model(x).logits
        ref = None
        if rl.needs_frozen_logits or rl.needs_hidden_states:
            with torch.no_grad():
                out = self.frozen(x)
            ref = out.logits if rl.needs_frozen_logits else None
        for h in hooks:
            h.remove()
        loss = rl(
            model_logits=logits,
            target=ids,
            mask=mask,
            frozen_logits=ref,
            model_hidden=mh or None,
            frozen_hidden=fh or None,
        )
        return c.alpha_retain * w * loss

    def _loss(self, fids, rids):
        """Objective over whichever terms are active. A null component (e.g.
        finetune has no forget, GA has no retain) contributes nothing and its
        forward pass is skipped, so `fids` / `rids` may be None."""
        c = self.cfg
        forget, ce = 0.0, 0.0
        if fids is not None:
            t_f = self._sample_t()
            forget, ce = self._forget_term(fids, t_f, self.weighting(t_f))
        retain = 0.0
        if rids is not None:
            t_r = self._sample_t()
            retain = self._retain_term(rids, t_r, self.weighting(t_r))
        loss = c.gamma_forget * forget + retain
        return loss, (ce if fids is not None else loss.item())

    def train(self, get_forget, get_retain, log_every: int = 100):
        c = self.cfg
        skip_f = getattr(self.forget_loss, "is_null", False)
        skip_r = getattr(self.retain_loss, "is_null", False)
        self.model.train()
        t0, done = time.time(), 0
        while done < c.steps:
            with self.acc.accumulate(self.model):
                fids = None if skip_f else get_forget().to(self.acc.device)
                rids = None if skip_r else get_retain().to(self.acc.device)
                loss, ce = self._loss(fids, rids)
                self.acc.backward(loss)
                if self.acc.sync_gradients:
                    self.acc.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad()
            if self.acc.sync_gradients:  # one real optimizer step
                done += 1
                if done % log_every == 0:
                    print(f"[step {done}/{c.steps}] ce={ce:.2f} loss={loss.item():.3f}")
        print(f"[train] done in {(time.time() - t0) / 60:.1f} min")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.adapter.state_dict(self.acc.unwrap_model(self.model)), path)
        print(f"[save] -> {path}")

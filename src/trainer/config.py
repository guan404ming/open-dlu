"""Training hyperparameters."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    steps: int = 500  # optimizer steps
    warmup_steps: int = 150
    lr: float = 5e-5
    scheduler: str = (
        "cosine"  # get_scheduler name: cosine / constant_with_warmup / linear
    )
    grad_accum: int = (
        1  # gradient accumulation; effective batch = batch_forget * grad_accum
    )
    gamma_forget: float = 1.0  # forget weight in  loss = gamma*forget + alpha*retain
    alpha_retain: float = 1.0
    batch_forget: int = 2
    batch_retain: int = 2
    seq_len: int = 512
    grad_clip: float = 1.0
    t_min: float = 0.1
    t_max: float = 0.9
    seed: int = 42
    mask_id: int = 126336
    use_8bit_optim: bool = False  # 8-bit AdamW (bitsandbytes) for full-model FT


DEFAULT = TrainConfig()

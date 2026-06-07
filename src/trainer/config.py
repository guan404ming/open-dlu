"""Training hyperparameters."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    steps: int = 500
    warmup_steps: int = 150
    lr: float = 5e-5
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
    cosine_decay: bool = False  # cosine LR decay after warmup


DEFAULT = TrainConfig()

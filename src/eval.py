"""Standalone evaluation entrypoint (mirrors src/train.py).

python -m src.eval                       # eval base llada on wmdp
python -m src.eval model=dream
"""

import hydra
from hydra.utils import instantiate

from src.evals import run_evaluators
from src.model import load_model


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    device = "cuda:0"
    model, tok = load_model(cfg.model.model_id, device, eval_mode=True)
    generator = instantiate(cfg.model.generator) if cfg.model.get("generator") else None
    scores = run_evaluators(
        cfg.eval, model=model, tokenizer=tok, mask_id=cfg.model.mask_id,
        device=device, generator=generator,
    )
    print(scores)


if __name__ == "__main__":
    main()

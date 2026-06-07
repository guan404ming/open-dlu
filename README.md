# Open-DLU

Weight-space **unlearning for diffusion language models** (LLaDA, Dream).
Config-driven and method-agnostic: the training loop never names a method, it
just composes five replaceable components.

| Component | Role |
|-----------|------|
| **forget** | per-token forget loss (cap, NPO, SimNPO, WGA, GA) |
| **retain** | preserve unrelated knowledge (CE, null) |
| **weighting** | trajectory weight `w(t)` (Min-SNR, 1/t) |
| **mask** | which positions to mask (Bernoulli) |
| **adapter** | which params train (layer-restricted, full model) |

## Install

```bash
pip install -e ".[train]"
```

## Run

```bash
# Modal (GPU)
modal run src/run.py                              # cap on LLaDA / WMDP-bio
modal run src/run.py --overrides "trainer=npo"   # swap method
modal run src/run.py --experiment unlearn/wmdp   # open-unlearning-aligned recipe

# local
python -m src.train                              # train
python -m src.eval                               # evaluate
python -m src.train -m trainer.args.seed=42,7,1337   # sweep
```

Override any field on the CLI: `trainer.forget.cap=2`, `trainer.args.lr=2e-5`,
`model=dream`.

## Configs

```
configs/
  unlearn.yaml  eval.yaml          # entry points
  experiment/   *.yaml             # full presets (model+data+trainer+eval+args)
  trainer/      base.yaml          # shared training args
                <method>.yaml      # defaults:[base] + components (+ overrides)
  model/  data/  eval/  *.yaml
```

A `trainer` inherits `base` and adds its five components; experiments set
dataset-level args (steps, lr). Add a method = one class in
`src/component/<axis>/` + one `trainer/<name>.yaml`.

## Acknowledgements

Layout follows [open-unlearning](https://github.com/locuslab/open-unlearning),
adapted from autoregressive to masked-diffusion LMs.

## License

MIT

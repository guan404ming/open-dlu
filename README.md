# Open-DLU

**Open framework for weight-space unlearning in diffusion language models** (masked diffusion LMs such as LLaDA and Dream).

Most unlearning tooling targets autoregressive LLMs. Open-DLU brings the same
config-driven, plug-in workflow to MDLMs, whose masked-denoising training and
parallel generation open extra axes (mask sampling, trajectory weighting,
layer locality) that have no AR analogue.

The library is deliberately **method-agnostic**: the training loop never names a
method. Everything is assembled from five replaceable components, instantiated
from Hydra configs.

| Component | Role | Examples |
|-----------|------|----------|
| **forget** | per-token forget loss | cap, NPO, WGA, GA |
| **retain** | preserve unrelated knowledge | CE, KL, hidden-match |
| **weighting** | trajectory / noise-level `w(t)` | Min-SNR, 1/t, uniform |
| **mask** | which positions to mask each step | Bernoulli, saliency, reliance |
| **adapter** | which parameters are trainable | full model, layer-restricted |

A component advertises what it needs (a frozen reference, hidden states, an
unconditional pass, ...) via class flags; the pipeline reads the flags and
provides exactly those signals. Adding a method never touches the core.

## Install

```bash
pip install -e ".[train]"
```

## Quickstart

Local (CPU/GPU, single process):

```bash
python -m src.train                                           # simplest cap recipe
python -m src.train trainer.forget.cap=2 trainer.adapter.layers=[10,11,12]
python -m src.train -m train.seed=42,7,1337                   # Hydra multirun sweep
```

On Modal (GPU):

```bash
modal run src/run.py --steps 1                    # 1-step smoke test
modal run src/run.py --steps 500 --model dream
modal run src/run.py --overrides "trainer.forget.cap=2 trainer.adapter.layers=[10,11,12]"
```

## Configuration

Hydra config groups under `configs/`:

```
configs/
  unlearn.yaml          # main entry: composes the groups below
  model/   *.yaml       # HF id + mask_id  (llada, dream)
  data/    *.yaml       # forget / retain corpora
  trainer/ *.yaml       # a method = the five components, composed in one file
  train/   default.yaml # TrainConfig hyperparameters
```

A `trainer` config inlines the five components (`forget`, `retain`, `weighting`,
`adapter`, `mask`). Swap a method with `trainer=<name>`; tweak one knob with
`trainer.forget.cap=2`.

## Extend

Adding a method = one class + one config, no core changes. See
[docs/components.md](docs/components.md) and [docs/contributing.md](docs/contributing.md).

## Acknowledgements

Structure follows [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning),
adapted from autoregressive to masked-diffusion LMs.

## License

MIT (see [LICENSE](LICENSE)).

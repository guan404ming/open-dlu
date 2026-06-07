"""Modal runner: compose a recipe from configs/ and train on a GPU.

modal run src/run.py                       # 1-step smoke test (cap, llada)
modal run src/run.py --steps 500 --model dream
modal run src/run.py --overrides "trainer.forget.cap=2 trainer.adapter.layers=[10,11,12]"
"""

import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import src` resolves when run as a script

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install uv",
        "uv venv /opt/venv",
        ". /opt/venv/bin/activate && uv pip install "
        "torch==2.4.0 transformers==4.46.3 accelerate==0.34.2 "
        "datasets==2.21.0 huggingface_hub==0.25.0 numpy==1.26.4 "
        "sentencepiece==0.2.0 hydra-core==1.3.2 setuptools bitsandbytes==0.43.1 "
        "rouge-score==0.1.2",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_python_source("src")
    .add_local_dir(str(ROOT / "configs"), "/root/configs")
)

app = modal.App("open-dlu", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="H100:1",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=8,
)
def train_remote(overrides: list):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.trainer.engine import build_and_train

    with initialize_config_dir(version_base=None, config_dir="/root/configs"):
        cfg = compose(config_name="unlearn", overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    return build_and_train(cfg)


@app.local_entrypoint()
def main(
    experiment: str = "",
    model: str = "",
    steps: int = 0,
    seed: int = -1,
    trainers: str = "",
    overrides: str = "",
):
    # Only force a key when given, so an experiment preset can own steps/seed.
    base = []
    if experiment:
        base.append(f"experiment={experiment}")
    if model:
        base.append(f"model={model}")
    if steps:
        base.append(f"trainer.args.steps={steps}")
    if seed >= 0:
        base.append(f"trainer.args.seed={seed}")
    if overrides:
        base += overrides.split()

    if not trainers:                       # single run
        print(f"[overrides] {base}")
        print(train_remote.remote(base))
        return

    names = [t.strip() for t in trainers.split(",") if t.strip()]
    jobs = [(base + [f"trainer={n}"],) for n in names]
    results = list(train_remote.starmap(jobs))   # parallel, one container each

    print(f"\n{'trainer':9s} {'bio↓':>7} {'mmlu':>7} {'bio_adj':>8} {'non_bio':>8}")
    for n, r in zip(names, results):
        s = (r or {}).get("scores", {})
        print(f"{n:9s} {s.get('wmdp_bio', 0):>7.3f} {s.get('mmlu_full', 0):>7.3f} "
              f"{s.get('mmlu_bio_adj', 0):>8.3f} {s.get('mmlu_non_bio', 0):>8.3f}")

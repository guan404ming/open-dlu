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
    timeout=60 * 60 * 8,  # SFT rounds can be many epochs; billed on actual use
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=8,
)
def train_remote(overrides: list):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.trainer.engine import build_and_train

    volume.reload()  # see the latest committed checkpoints (for --model-id resume)
    with initialize_config_dir(version_base=None, config_dir="/root/configs"):
        cfg = compose(config_name="unlearn", overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    result = build_and_train(cfg)
    if cfg.get("output_dir"):
        import os

        os.sync()  # flush page cache to the FUSE mount before snapshotting
        volume.commit()  # then persist the (now fully-written) large shards
    return result


@app.local_entrypoint()
def main(
    experiment: str = "",
    model: str = "",
    steps: int = 0,
    seed: int = -1,
    trainers: str = "",
    overrides: str = "",
    output_dir: str = "",
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
    if output_dir:
        base.append(f"+output_dir={output_dir}")  # save trained model as a HF dir
    if overrides:
        base += overrides.split()

    if not trainers:  # single run
        print(f"[overrides] {base}")
        print(train_remote.remote(base))
        return

    names = [t.strip() for t in trainers.split(",") if t.strip()]
    jobs = [(base + [f"trainer={n}"],) for n in names]
    results = list(train_remote.starmap(jobs))  # parallel, one container each

    print(f"\n{'trainer':9s} {'bio↓':>7} {'mmlu':>7} {'bio_adj':>8} {'non_bio':>8}")
    for n, r in zip(names, results):
        s = (r or {}).get("scores", {})
        print(
            f"{n:9s} {s.get('wmdp_bio', 0):>7.3f} {s.get('mmlu_full', 0):>7.3f} "
            f"{s.get('mmlu_bio_adj', 0):>8.3f} {s.get('mmlu_non_bio', 0):>8.3f}"
        )


@app.local_entrypoint()
def finetune_loop(
    experiment: str = "finetune/tofu",
    rounds: int = 5,
    epochs_per_round: int = 20,
    output_root: str = "/root/.cache/huggingface/open-dlu/tofu_sft",
    eval_max: int = 50,
    n_mc: int = 32,
    model_id: str = "",
    start_ep: int = 0,
):
    """Checkpointed SFT toward the MDU Base-SFT target. Each round resumes from
    the previous checkpoint, trains `epochs_per_round` epochs, saves a checkpoint
    on the volume, and evals. Resume a stopped run by passing the last checkpoint
    as --model-id and its epoch count as --start-ep. Stop when the metrics reach
    the paper targets."""
    prev, rows = model_id, []
    for r in range(rounds):
        total_ep = start_ep + (r + 1) * epochs_per_round
        ckpt = f"{output_root}/ckpt_{total_ep}ep"
        ov = [
            f"experiment={experiment}",
            f"trainer.args.epochs={epochs_per_round}",
            f"+output_dir={ckpt}",
            f"eval.tofu.metrics.tofu.max_eval={eval_max}",
            f"eval.tofu.metrics.tofu.n_mc={n_mc}",
        ]
        if prev:
            ov.append(f"model.model_id={prev}")
        print(
            f"\n===== round {r + 1}/{rounds}: resume {prev or 'base'} -> {ckpt} ====="
        )
        rows.append((total_ep, (train_remote.remote(ov) or {}).get("scores", {})))
        prev = ckpt

    print("\ntarget  fgt 0.884/0.380  ret 0.870/0.330  RA 0.611/0.041  WF 0.835/0.143")
    print(
        f"{'epoch':>6}{'fgt_p':>8}{'fgt_rL':>8}{'ret_p':>8}{'ret_rL':>8}"
        f"{'RA_p':>8}{'RA_rL':>8}{'WF_p':>8}{'WF_rL':>8}"
    )
    for ep, s in rows:
        g = lambda k: s.get(k, 0)  # noqa: E731
        print(
            f"{ep:>6}{g('tofu_forget_prob'):>8.3f}{g('tofu_forget_rougeL'):>8.3f}"
            f"{g('tofu_retain_prob'):>8.3f}{g('tofu_retain_rougeL'):>8.3f}"
            f"{g('tofu_real_authors_prob'):>8.3f}{g('tofu_real_authors_rougeL'):>8.3f}"
            f"{g('tofu_world_facts_prob'):>8.3f}{g('tofu_world_facts_rougeL'):>8.3f}"
        )


@app.local_entrypoint()
def cap_sweep(grid: str = "1.0:500,2.0:500,3.0:500", layers: str = "", lr: float = 0.0):
    """Sweep the cap recipe on the default config (cap / wmdp_bio / wmdp) to find
    the Pareto-best operating point. `grid` is comma-separated `cap:steps` points;
    optionally override the localized `layers` (e.g. "4,5,6,7,8") and `lr`."""
    pts = [p.split(":") for p in grid.split(",")]
    jobs = []
    for cap, steps in pts:
        ov = [f"trainer.forget.cap={cap}", f"trainer.args.steps={steps}"]
        if layers:
            ov.append(f"trainer.adapter.layers=[{layers}]")
        if lr:
            ov.append(f"trainer.args.lr={lr}")
        jobs.append((ov,))
    results = list(train_remote.starmap(jobs))

    print(f"\n{'cap':>5}{'steps':>7}{'bio↓':>8}{'mmlu':>8}{'bio_adj':>9}{'non_bio':>9}")
    for (cap, steps), r in zip(pts, results):
        s = (r or {}).get("scores", {})
        print(
            f"{cap:>5}{steps:>7}{s.get('wmdp_bio', 0):>8.3f}{s.get('mmlu_full', 0):>8.3f}"
            f"{s.get('mmlu_bio_adj', 0):>9.3f}{s.get('mmlu_non_bio', 0):>9.3f}"
        )

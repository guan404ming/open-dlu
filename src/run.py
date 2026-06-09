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

# torch 2.7/cu128 covers both Hopper (H100/H200, sm_90) and Blackwell (B200,
# sm_100) from one image. torch comes from the cu128 index, the rest from PyPI
# (two installs so the index URL doesn't leak to the non-torch wheels). No
# bitsandbytes: it's imported only when trainer.args.use_8bit_optim is set, and
# full-model SFT now runs on B200's 192 GB where 8-bit optim is unnecessary.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install uv",
        "uv venv /opt/venv",
        ". /opt/venv/bin/activate && uv pip install torch==2.7.0 "
        "--index-url https://download.pytorch.org/whl/cu128",
        ". /opt/venv/bin/activate && uv pip install "
        "transformers==4.46.3 accelerate==0.34.2 "
        "datasets==2.21.0 huggingface_hub==0.25.0 numpy==1.26.4 "
        "sentencepiece==0.2.0 hydra-core==1.3.2 setuptools rouge-score==0.1.2",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_python_source("src")
    .add_local_dir(str(ROOT / "configs"), "/root/configs")
)

app = modal.App("open-dlu", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _train(overrides: list):
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


# Same body on two GPU tiers so we can pick speed/cost per run. Each tier needs
# a distinctly-named wrapper: Modal registers a function under its __name__, so
# decorating the one `_train` twice collides and only the last survives.
_fn = dict(
    timeout=60 * 60 * 8,  # SFT rounds can be many epochs; billed on actual use
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=8,
)


def _train_h100(overrides: list):
    return _train(overrides)


def _train_b200(overrides: list):
    return _train(overrides)


train_remote = app.function(gpu="H100:1", **_fn)(_train_h100)
train_remote_b200 = app.function(gpu="B200:1", **_fn)(_train_b200)


def _gen_probe(overrides: list):
    """Dump raw generations on a handful of TOFU forget + real_authors questions
    to inspect the failure mode (garbage / wrong content / scoring mismatch)."""
    import json
    import os

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    volume.reload()
    with initialize_config_dir(version_base=None, config_dir="/root/configs"):
        cfg = compose(config_name="unlearn", overrides=overrides)
    ckpt = cfg.model.model_id
    # SFT save strips the custom tokenizer code, so read the tokenizer from the
    # base repo named in the checkpoint's auto_map; weights load from the ckpt.
    amap = json.load(open(os.path.join(ckpt, "config.json"))).get("auto_map", {})
    base = next((v.split("--")[0] for v in amap.values() if "--" in v), ckpt)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .to("cuda:0")
        .eval()
    )
    gen = instantiate(cfg.model.generator)
    out = []
    for split in ("forget10.json", "real_authors.json"):
        path = hf_hub_download("locuslab/TOFU", split, repo_type="dataset")
        rows = [json.loads(x) for x in open(path) if x.strip()][:4]
        for r in rows:
            pred = gen.generate(
                model, tok, r["question"], cfg.model.mask_id, "cuda:0", max_new=128
            )
            out.append({"split": split, "q": r["question"], "gt": r["answer"], "pred": pred})
    return out


probe_remote = app.function(gpu="B200:1", **_fn)(_gen_probe)


@app.function(
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def push_hf(local_dir: str, repo_name: str, private: bool = False):
    """Upload a saved checkpoint folder to the token owner's HF account."""
    import os

    from huggingface_hub import HfApi

    volume.reload()
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    api = HfApi(token=token)  # token from the mounted huggingface secret
    repo_id = f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id)
    return repo_id


def _fn_for(gpu: str):
    """Pick the GPU-tier function ("b200" -> Blackwell, else the H100 default)."""
    return train_remote_b200 if gpu.lower().startswith("b200") else train_remote


# Shared WMDP results table (bio forget + MMLU utility), printed by every sweep.
_WMDP_KEYS = ("wmdp_bio", "mmlu_full", "mmlu_bio_adj", "mmlu_non_bio")
_WMDP_TITLES = ("bio↓", "mmlu", "bio_adj", "non_bio")
_WMDP_W = (8, 8, 9, 9)


def _wmdp_titles() -> str:
    return "".join(f"{t:>{w}}" for t, w in zip(_WMDP_TITLES, _WMDP_W))


def _wmdp_cells(r) -> str:
    """Format the four WMDP score columns from a (possibly None) run result."""
    s = (r or {}).get("scores", {})
    return "".join(f"{s.get(k, 0):>{w}.3f}" for k, w in zip(_WMDP_KEYS, _WMDP_W))


@app.local_entrypoint()
def spawn_train(
    experiment: str = "", overrides: str = "", output_dir: str = "", gpu: str = ""
):
    """Fire-and-forget a single training run server-side and exit immediately.
    Robust to local disconnect/shutdown (unlike a blocking `.remote()` stream).
    Launch with `modal run --detach` so the app outlives the client; retrieve via
    `modal volume ls` (the saved checkpoint) and `modal app logs` (eval scores)."""
    base = []
    if experiment:
        base.append(f"experiment={experiment}")
    if output_dir:
        base.append(f"+output_dir={output_dir}")
    if overrides:
        base += overrides.split()
    call = _fn_for(gpu).spawn(base)
    print(f"[spawned] {call.object_id}  gpu={gpu or 'H100'}  overrides={base}")


@app.local_entrypoint()
def main(
    experiment: str = "",
    model: str = "",
    steps: int = 0,
    seed: int = -1,
    trainers: str = "",
    overrides: str = "",
    output_dir: str = "",
    gpu: str = "",
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

    fn = _fn_for(gpu)
    if not trainers:  # single run
        print(f"[overrides] {base}  gpu={gpu or 'H100'}")
        print(fn.remote(base))
        return

    names = [t.strip() for t in trainers.split(",") if t.strip()]
    jobs = [(base + [f"trainer={n}"],) for n in names]
    results = list(fn.starmap(jobs))  # parallel, one container each

    print(f"\n{'trainer':9s}" + _wmdp_titles())
    for n, r in zip(names, results):
        print(f"{n:9s}" + _wmdp_cells(r))


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
    model: str = "",
    gpu: str = "",
    extra: str = "",
):
    """Checkpointed SFT toward the MDU Base-SFT target. Each round resumes from
    the previous checkpoint, trains `epochs_per_round` epochs, saves a checkpoint
    on the volume, and evals. Resume a stopped run by passing the last checkpoint
    as --model-id and its epoch count as --start-ep. `extra` injects per-backbone
    overrides (e.g. "model=dream trainer.args.lr=1e-5"). Stop when the metrics
    reach the paper targets."""
    fn, prev, rows = _fn_for(gpu), model_id, []
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
        if model:
            ov.append(f"model={model}")
        if extra:
            ov += extra.split()
        if prev:
            ov.append(f"model.model_id={prev}")
        print(
            f"\n===== round {r + 1}/{rounds}: resume {prev or 'base'} -> {ckpt} ====="
        )
        rows.append((total_ep, (fn.remote(ov) or {}).get("scores", {})))
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
def cap_sweep(grid: str = "1.0:500,2.0:500,3.0:500", layers: str = ""):
    """Sweep the cap recipe on the default config (cap / wmdp_bio / wmdp) to find
    the Pareto-best operating point. `grid` is comma-separated `cap:steps[:lr]`
    points (lr optional); optionally override the localized `layers` ("4,5,6,7,8")."""
    pts = [p.split(":") for p in grid.split(",")]
    jobs = []
    for p in pts:
        ov = [f"trainer.forget.cap={p[0]}", f"trainer.args.steps={p[1]}"]
        if len(p) > 2:
            ov.append(f"trainer.args.lr={p[2]}")
        if layers:
            ov.append(f"trainer.adapter.layers=[{layers}]")
        jobs.append((ov,))
    results = list(train_remote.starmap(jobs))

    print(f"\n{'cap':>5}{'steps':>7}{'lr':>9}" + _wmdp_titles())
    for p, r in zip(pts, results):
        lr = p[2] if len(p) > 2 else "base"
        print(f"{p[0]:>5}{p[1]:>7}{lr:>9}" + _wmdp_cells(r))


@app.local_entrypoint()
def variant_sweep(specs: str):
    """Run labelled method variants in parallel on the default config (wmdp_bio,
    domain retain). `specs` is semicolon-separated `label|override override ...`,
    e.g. "npo_b.05|trainer=npo trainer.forget.beta=0.05;wga_g2|trainer=wga
    trainer.forget.gamma=2.0". Lets each method tune its own hyperparameters."""
    items = [s.split("|", 1) for s in specs.split(";") if s.strip()]
    jobs = [(ov.split(),) for _, ov in items]
    results = list(train_remote.starmap(jobs))

    print(f"\n{'variant':16s}" + _wmdp_titles())
    for (label, _), r in zip(items, results):
        print(f"{label:16s}" + _wmdp_cells(r))


# TOFU 4-split table (rougeL recall + answer prob, in the MDU column order).
_TOFU_SPLITS = ("forget", "retain", "real_authors", "world_facts")
_TOFU_TITLES = ("fgt_rL", "fgt_p", "ret_rL", "ret_p", "RA_rL", "RA_p", "WF_rL", "WF_p")


def _tofu_titles() -> str:
    return "".join(f"{t:>8}" for t in _TOFU_TITLES)


def _tofu_cells(r) -> str:
    s = (r or {}).get("scores", {})
    return "".join(
        f"{s.get(f'tofu_{sp}_rougeL', 0):>8.3f}{s.get(f'tofu_{sp}_prob', 0):>8.3f}"
        for sp in _TOFU_SPLITS
    )


@app.local_entrypoint()
def tofu_sweep(
    trainers: str = "ga,gd,npo,sim_npo,wga,cap",
    model_id: str = "",
    gpu: str = "",
    overrides: str = "",
):
    """Unlearn the TOFU SFT target with each method in parallel and print the
    4-split table (rougeL recall + answer prob). `model_id` is the checkpoint to
    forget from (the SFT target, e.g. .../ckpt_150ep)."""
    base = ["experiment=unlearn/tofu"]
    if model_id:
        base.append(f"model.model_id={model_id}")
    if overrides:
        base += overrides.split()
    names = [t.strip() for t in trainers.split(",") if t.strip()]
    jobs = [(base + [f"trainer={n}"],) for n in names]
    results = list(_fn_for(gpu).starmap(jobs))

    print(f"\n{'method':9s}" + _tofu_titles())
    for n, r in zip(names, results):
        print(f"{n:9s}" + _tofu_cells(r))


@app.local_entrypoint()
def tofu_variant_sweep(specs: str, model_id: str = "", gpu: str = ""):
    """Unlearn the TOFU target with labelled method variants in parallel, to tune
    one method's hyperparameters. `specs` is semicolon-separated `label|override
    ...`, e.g. "sn_gf1|trainer=sim_npo trainer.args.gamma_forget=1.0"."""
    items = [s.split("|", 1) for s in specs.split(";") if s.strip()]
    base = ["experiment=unlearn/tofu"]
    if model_id:
        base.append(f"model.model_id={model_id}")
    jobs = [(base + ov.split(),) for _, ov in items]
    results = list(_fn_for(gpu).starmap(jobs))

    print(f"\n{'variant':16s}" + _tofu_titles())
    for (label, _), r in zip(items, results):
        print(f"{label:16s}" + _tofu_cells(r))


@app.local_entrypoint()
def upload_hf(local_dir: str, repo_name: str, private: bool = False):
    """Push a checkpoint on the volume to HF, e.g.
    modal run src/run.py::upload_hf \
      --local-dir /root/.cache/huggingface/open-dlu/tofu_sft/ckpt_150ep \
      --repo-name llada-8b-tofu-sft"""
    print(f"[uploaded] https://huggingface.co/{push_hf.remote(local_dir, repo_name, private)}")


@app.local_entrypoint()
def gen_probe(model_id: str, model: str = "dream"):
    """Print raw generations for a few TOFU forget + real_authors questions, to
    see how a (possibly broken) SFT checkpoint actually answers."""
    ov = [f"model={model}", f"model.model_id={model_id}", "+model.generator.steps=256"]
    for r in probe_remote.remote(ov) or []:
        print(f"\n[{r['split']:18s}] Q: {r['q']}")
        print(f"  GT  : {r['gt'][:220]}")
        print(f"  PRED: {r['pred'][:220]}")

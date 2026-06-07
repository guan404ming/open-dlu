"""Local Hydra entrypoint.

python -m src.train                       # simplest cap recipe
python -m src.train trainer.forget.cap=2 trainer.adapter.layers=[10,11,12]
python -m src.train -m trainer.args.seed=42,7,1337   # multirun sweep
"""

import hydra

from src.trainer.engine import build_and_train


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn")
def main(cfg):
    print(build_and_train(cfg))


if __name__ == "__main__":
    main()

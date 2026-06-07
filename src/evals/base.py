"""Evaluator: runs a named set of metrics over a model and merges their scores."""

import json
import os

from omegaconf import OmegaConf

from src.evals.metrics import get_metric


class Evaluator:
    def __init__(self, name, eval_cfg):
        self.name = name
        self.metrics, self.metric_kwargs = {}, {}
        for key, mc in eval_cfg.metrics.items():
            mc = OmegaConf.to_container(mc, resolve=True)
            self.metrics[key] = get_metric(mc.pop("handler"))
            self.metric_kwargs[key] = mc

    def evaluate(self, output_dir: str = None, **shared) -> dict:
        scores = {}
        for key, metric in self.metrics.items():
            scores.update(metric(**self.metric_kwargs[key], **shared))
        if output_dir:
            self.save_logs(scores, output_dir)
        return scores

    def save_logs(self, scores: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(scores, f, indent=2, sort_keys=True)
        print(f"[eval] {self.name} -> {path}")

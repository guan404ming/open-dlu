"""Evaluator registry. `get_evaluators(cfg.eval)` -> {name: Evaluator}."""
from src.evals.base import Evaluator

_EVALUATORS = {"Evaluator": Evaluator}


def get_evaluators(eval_cfg) -> dict:
    return {name: _EVALUATORS[c.handler](name, c) for name, c in eval_cfg.items()}


def run_evaluators(eval_cfg, **shared) -> dict:
    """Run every evaluator and merge their scores. Shared by train + eval entrypoints."""
    scores = {}
    for ev in get_evaluators(eval_cfg).values():
        scores.update(ev.evaluate(**shared))
    return scores

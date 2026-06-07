"""Metric primitive: a named function wrapped by @unlearning_metric, auto-registered.

    @unlearning_metric(name="wmdp_bio")
    def wmdp_bio(model, tokenizer, mask_id, device, **kw): ...

The Evaluator looks metrics up by name via ``get_metric``.
"""

from typing import Any, Callable

_METRICS: dict = {}


class UnlearningMetric:
    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self.fn = fn

    def __call__(self, model, **kwargs):
        return self.fn(model=model, **kwargs)


class unlearning_metric:  # noqa: N801  (decorator, lowercase by convention)
    def __init__(self, name: str = None):
        self.name = name

    def __call__(self, fn: Callable[..., Any]) -> UnlearningMetric:
        metric = UnlearningMetric(self.name or fn.__name__, fn)
        _METRICS[metric.name] = metric
        return metric


def get_metric(name: str) -> UnlearningMetric:
    if name not in _METRICS:
        raise KeyError(f"unknown metric {name}; have {sorted(_METRICS)}")
    return _METRICS[name]

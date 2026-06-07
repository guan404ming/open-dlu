"""Metric registry. Importing a metric module registers its metrics by name."""

from src.evals.metrics.base import UnlearningMetric, get_metric, unlearning_metric
from src.evals.metrics import mcq  # noqa: F401  (registers wmdp_bio, mmlu)

__all__ = ["UnlearningMetric", "unlearning_metric", "get_metric"]

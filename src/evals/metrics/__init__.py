"""Metric registry. Importing a metric module registers its metrics by name."""

from src.evals.metrics.base import UnlearningMetric, get_metric, unlearning_metric
from src.evals.metrics import mcq  # noqa: F401  (registers wmdp_bio, wmdp_cyber, mmlu)
from src.evals.metrics import rwku  # noqa: F401  (registers rwku)
from src.evals.metrics import tofu  # noqa: F401  (registers tofu)
from src.evals.metrics import muse  # noqa: F401  (registers muse)

__all__ = ["UnlearningMetric", "unlearning_metric", "get_metric"]

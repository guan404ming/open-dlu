"""Shared text-overlap scoring for generation metrics."""
_SCORER = None


def _score(pred: str, gt: str):
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer

        _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _SCORER.score(gt, pred)["rougeL"]


def rouge_l_recall(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    return _score(pred, gt).recall


def rouge_l_f1(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    return _score(pred, gt).fmeasure

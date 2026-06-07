"""Shared text-overlap scoring for generation metrics."""
_SCORER = None


def rouge_l_recall(pred: str, gt: str) -> float:
    global _SCORER
    if not pred or not gt:
        return 0.0
    if _SCORER is None:
        from rouge_score import rouge_scorer

        _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _SCORER.score(gt, pred)["rougeL"].recall

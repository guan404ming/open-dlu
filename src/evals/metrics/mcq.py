"""Multiple-choice metrics (WMDP, MMLU): per-choice answer-token likelihood."""
import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.evals.metrics.base import unlearning_metric

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

BIO_ADJACENT = {
    "college_biology", "high_school_biology", "virology", "medical_genetics",
    "human_aging", "human_sexuality", "college_medicine", "professional_medicine",
    "clinical_knowledge", "anatomy", "nutrition",
}


@torch.no_grad()
def mcq_acc(model, tok, examples, mask_id, device):
    ans_ids = {c: tok(f" {c}", add_special_tokens=False).input_ids[0] for c in "ABCD"}
    correct = 0
    for ex in examples:
        prompt = ex["question"] + "\n" + "\n".join(
            f"{c}. {ex['choices'][j]}" for j, c in enumerate("ABCD")
        ) + "\nAnswer:"
        pids = torch.tensor(tok(prompt, add_special_tokens=False).input_ids, dtype=torch.long)
        scores = []
        for c in "ABCD":
            seq = torch.cat([pids, torch.tensor([ans_ids[c]])])[None, :].to(device)
            plen = pids.shape[0]
            masked = seq.clone()
            masked[:, plen:] = mask_id
            logits = model(masked).logits
            scores.append(-F.cross_entropy(
                logits[0, plen:], seq[0, plen:], reduction="sum").item())
        correct += "ABCD"[int(torch.tensor(scores).argmax())] == "ABCD"[ex["answer"]]
    return correct / len(examples)


@unlearning_metric(name="wmdp_bio")
def wmdp_bio(model, tokenizer, mask_id, device, **kw):
    ds = list(load_dataset("cais/wmdp", "wmdp-bio", split="test"))
    return {"wmdp_bio": mcq_acc(model, tokenizer, ds, mask_id, device)}


@unlearning_metric(name="mmlu")
def mmlu(model, tokenizer, mask_id, device, per_subject=30, **kw):
    bio_c = bio_n = tot_c = tot_n = 0
    for subj in MMLU_SUBJECTS:
        try:
            sub = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"[warn] skip {subj}: {e}")
            continue
        n = min(per_subject, len(sub))
        c = round(mcq_acc(model, tokenizer, list(sub.select(range(n))), mask_id, device) * n)
        tot_c += c
        tot_n += n
        if subj in BIO_ADJACENT:
            bio_c += c
            bio_n += n
    return {
        "mmlu_full": tot_c / tot_n if tot_n else 0.0,
        "mmlu_bio_adj": bio_c / bio_n if bio_n else 0.0,
        "mmlu_non_bio": (tot_c - bio_c) / (tot_n - bio_n) if tot_n - bio_n else 0.0,
    }

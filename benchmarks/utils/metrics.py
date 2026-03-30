import re
import string
from collections import Counter
from typing import List, Dict

def normalize_text(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = 1.0 * num_common / len(pred_tokens)
    recall = 1.0 * num_common / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact_match(prediction: str, truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(truth))

# mean reciprocal rank
# <=> how quickly a user finds the first relevant result in a list
def compute_mrr(relevant_docs: List[List[str]], retrieved_docs: List[List[str]]) -> float:
    mrr = 0.0
    for rel, ret in zip(relevant_docs, retrieved_docs):
        for rank, doc_id in enumerate(ret, 1):
            if doc_id in rel:
                mrr += 1.0 / rank
                break
    return mrr / max(1, len(relevant_docs))

def compute_recall_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int = 5) -> float:
    recall = 0.0
    for rel, ret in zip(relevant_docs, retrieved_docs):
        ret_k = set(ret[:k])
        hits = sum(1 for doc_id in rel if doc_id in ret_k)
        recall += hits / max(1, len(rel))
    return recall / max(1, len(relevant_docs))
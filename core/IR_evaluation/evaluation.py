"""
Evaluation Module for IR Project
Person 3 (J Sai Varun)
--------------------------------
Implements standard IR metrics:
- Precision@K
- Recall@K
- nDCG@K
- MRR
Handles both int and string doc_id formats automatically.
"""

import math
import json
from typing import List, Dict, Tuple


# -------------------------------------------------------
# 🔧 Utility helpers
# -------------------------------------------------------

def _normalize_ids(ranked_list, relevance_judgments):
    """Ensure consistent string comparison for doc_ids."""
    ranked_list = [str(x) for x in ranked_list]
    relevance_judgments = {str(k): int(v) for k, v in relevance_judgments.items()}
    return ranked_list, relevance_judgments


# -------------------------------------------------------
# 📈 Metric Functions
# -------------------------------------------------------

def precision_at_k(ranked_list: List[str], relevance_judgments: Dict[str, int], k: int) -> float:
    ranked_list, relevance_judgments = _normalize_ids(ranked_list, relevance_judgments)
    if k <= 0 or len(ranked_list) == 0:
        return 0.0
    top_k = ranked_list[:k]
    relevant_count = sum(1 for doc_id in top_k if relevance_judgments.get(doc_id, 0) > 0)
    return relevant_count / k


def recall_at_k(ranked_list: List[str], relevance_judgments: Dict[str, int], k: int) -> float:
    ranked_list, relevance_judgments = _normalize_ids(ranked_list, relevance_judgments)
    total_relevant = sum(1 for score in relevance_judgments.values() if score > 0)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_list[:k]
    retrieved_relevant = sum(1 for doc_id in top_k if relevance_judgments.get(doc_id, 0) > 0)
    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(ranked_list: List[str], relevance_judgments: Dict[str, int]) -> float:
    ranked_list, relevance_judgments = _normalize_ids(ranked_list, relevance_judgments)
    for rank, doc_id in enumerate(ranked_list, start=1):
        if relevance_judgments.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_list: List[str], relevance_judgments: Dict[str, int], k: int) -> float:
    ranked_list, relevance_judgments = _normalize_ids(ranked_list, relevance_judgments)
    actual_dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k], start=1):
        rel = relevance_judgments.get(doc_id, 0)
        actual_dcg += (2**rel - 1) / math.log2(i + 1)

    ideal_gains = sorted(
        [v for v in relevance_judgments.values() if v > 0],
        reverse=True
    )[:k]
    ideal_dcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# -------------------------------------------------------
# 🧮 Evaluation Functions
# -------------------------------------------------------

def evaluate_query(ranked_list: List[str], relevance_judgments: Dict[str, int], k_values: List[int]) -> Dict[str, float]:
    """Compute all metrics for a single query."""
    results = {}
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(ranked_list, relevance_judgments, k)
        results[f'recall@{k}'] = recall_at_k(ranked_list, relevance_judgments, k)
        results[f'ndcg@{k}'] = ndcg_at_k(ranked_list, relevance_judgments, k)
    results['mrr'] = mean_reciprocal_rank(ranked_list, relevance_judgments)
    return results


def evaluate_all_queries(model_results: Dict[str, List[str]], qrels_file: str,
                         k_values: List[int] = [5, 10, 20]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Evaluate all queries and return per-query and averaged metrics."""
    with open(qrels_file, 'r', encoding='utf-8') as f:
        qrels = json.load(f)

    per_query_results = {}
    aggregated = {f'precision@{k}': 0.0 for k in k_values}
    aggregated.update({f'recall@{k}': 0.0 for k in k_values})
    aggregated.update({f'ndcg@{k}': 0.0 for k in k_values})
    aggregated['mrr'] = 0.0

    total_queries = 0
    for q in qrels:
        qid = q['query_id']
        rel_judgments = q['relevance_judgments']
        if qid not in model_results:
            continue
        ranked = model_results[qid]
        per_query_results[qid] = evaluate_query(ranked, rel_judgments, k_values)
        total_queries += 1
        for m, v in per_query_results[qid].items():
            aggregated[m] += v

    if total_queries > 0:
        for m in aggregated:
            aggregated[m] /= total_queries

    return per_query_results, aggregated


# -------------------------------------------------------
# ⚔️ Model Comparison
# -------------------------------------------------------

def compare_models(model1_results, model2_results, qrels_file,
                   model1_name='TF-IDF', model2_name='BM25', k_values=[5, 10, 20]):
    """Compare two models and compute improvements."""
    _, stats1 = evaluate_all_queries(model1_results, qrels_file, k_values)
    _, stats2 = evaluate_all_queries(model2_results, qrels_file, k_values)

    metrics = {}
    for k in k_values:
        for metric_base in ['precision', 'recall', 'ndcg']:
            m = f'{metric_base}@{k}'
            val1, val2 = stats1[m], stats2[m]
            improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0.0
            metrics[m] = {
                f'{model1_name}_mean': val1,
                f'{model2_name}_mean': val2,
                'improvement_%': improvement,
                'winner': model1_name if val1 >= val2 else model2_name
            }
    # Add MRR
    val1, val2 = stats1['mrr'], stats2['mrr']
    improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0.0
    metrics['mrr'] = {
        f'{model1_name}_mean': val1,
        f'{model2_name}_mean': val2,
        'improvement_%': improvement,
        'winner': model1_name if val1 >= val2 else model2_name
    }

    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'metrics': metrics
    }

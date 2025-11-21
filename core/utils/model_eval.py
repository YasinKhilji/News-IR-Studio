import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.IR_core.search_pipeline import SearchPipeline
from core.IR_evaluation.evaluation import evaluate_query

EVAL_QUERY_PATH = Path("data/Evaluate_models.json")
EVAL_TOP_K = 20
EVAL_CANDIDATE_POOL = 300
MODEL_EVAL_CACHE_PATH = Path("core/results/evaluation_results/model_eval_snapshot.json")


def load_eval_queries(path: str) -> List[Dict]:
    eval_file = Path(path)
    if not eval_file.exists():
        return []
    with open(eval_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_models(
    pipeline: SearchPipeline,
    models: Sequence[str],
    eval_queries: Sequence[Dict],
    top_k: int,
    candidate_pool: int,
) -> Tuple[List[Dict[str, float]], Optional[str]]:
    if not eval_queries:
        return [], None

    summary: List[Dict[str, float]] = []
    best_model: Optional[str] = None
    best_score = -1.0

    for model_name in models:
        totals: Dict[str, float] = {"ndcg@10": 0.0, "precision@10": 0.0, "recall@10": 0.0, "mrr": 0.0}
        count = 0
        for payload in eval_queries:
            _, ranked_ids = pipeline.search_ranked(
                payload["query_text"],
                model_name,
                top_k=top_k,
                embedding_candidate_pool=candidate_pool,
            )
            if not ranked_ids:
                continue
            metrics = evaluate_query(ranked_ids, payload["relevance_judgments"], [5, 10, 20])
            totals["ndcg@10"] += metrics.get("ndcg@10", 0.0)
            totals["precision@10"] += metrics.get("precision@10", 0.0)
            totals["recall@10"] += metrics.get("recall@10", 0.0)
            totals["mrr"] += metrics.get("mrr", 0.0)
            count += 1
        if count:
            for key in totals:
                totals[key] /= count
        row = {"model": model_name, **totals}
        summary.append(row)
        if totals["ndcg@10"] > best_score:
            best_score = totals["ndcg@10"]
            best_model = model_name
    return summary, best_model


def load_persisted_eval_cache() -> Optional[Dict[str, Any]]:
    if not MODEL_EVAL_CACHE_PATH.exists():
        return None
    try:
        with open(MODEL_EVAL_CACHE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if not payload.get("summary"):
        return None
    return payload


def save_persisted_eval_cache(payload: Dict[str, Any]) -> None:
    MODEL_EVAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_EVAL_CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)



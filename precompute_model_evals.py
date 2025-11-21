import argparse
from datetime import datetime
from pathlib import Path

from core.IR_core.search_pipeline import SearchPipeline
from core.utils.model_eval import (
    EVAL_CANDIDATE_POOL,
    EVAL_QUERY_PATH,
    EVAL_TOP_K,
    MODEL_EVAL_CACHE_PATH,
    evaluate_models,
    load_eval_queries,
    save_persisted_eval_cache,
)


def discover_index_dirs(base_dir: str):
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}

    discovered = {}
    candidates = [base_path] if (base_path / "inverted_index.pkl").exists() else []
    candidates += [p for p in base_path.iterdir() if p.is_dir()]

    for path in candidates:
        if (path / "inverted_index.pkl").exists():
            label = path.name
            if label in discovered:
                label = path.resolve().as_posix()
            discovered[label] = str(path.resolve())
    return discovered


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute model evaluation snapshot for News IR Studio.")
    parser.add_argument("--index-base", default="core/results", help="Base directory to look for built indexes.")
    parser.add_argument("--dataset", default=None, help="Dataset label to evaluate (defaults to the first discovered).")
    parser.add_argument("--qrels", default="data/queries_relevance_auto.json", help="Path to qrels JSON.")
    parser.add_argument("--models-dir", default="core/results/models", help="Directory containing trained models.")
    parser.add_argument("--eval-queries", default=str(EVAL_QUERY_PATH), help="JSON file with labeled queries.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = discover_index_dirs(args.index_base)
    if not datasets:
        raise RuntimeError("No index directories found. Build an index first.")

    dataset_label = args.dataset or next(iter(datasets))
    if dataset_label not in datasets:
        raise RuntimeError(f"Dataset '{dataset_label}' not found in {datasets.keys()}.")

    pipeline = SearchPipeline(
        index_dir=datasets[dataset_label],
        qrels_path=args.qrels,
        models_dir=args.models_dir,
        preload_text=False,
    )

    eval_queries = load_eval_queries(args.eval_queries)
    summary, best_model = evaluate_models(
        pipeline,
        pipeline.rank_models(),
        eval_queries,
        top_k=EVAL_TOP_K,
        candidate_pool=EVAL_CANDIDATE_POOL,
    )

    payload = {
        "summary": summary,
        "best_model": best_model,
        "computed_at": datetime.utcnow().isoformat(),
    }
    save_persisted_eval_cache(payload)

    print(f"Snapshot written to {MODEL_EVAL_CACHE_PATH.resolve()}")
    if best_model:
        print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()



"""
IR Project - Person 3 (J Sai Varun)
LTR Training Script: Train and evaluate Learning-to-Rank model
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core.IR_core.config import Config
from core.IR_core.io_utils import load_pickle
from core.IR_core.ranking import BM25Ranker, TfIdfRanker
from core.IR_evaluation.evaluation import compare_models
from core.IR_evaluation.itr import FeatureExtractor, LTRRanker, LTRTrainer


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_retrieval_results(results_file: Path):
    """Load retrieval results from JSON"""
    return load_json(results_file)


def run_ltr_retrieval(queries_file: Path, ltr_ranker: LTRRanker, top_k: int = 100):
    """Run LTR retrieval for all queries"""
    queries = load_json(queries_file)
    
    results = {}
    for query in queries:
        query_id = query['query_id']
        query_text = query['query_text']
        
        print(f"  Processing {query_id}: {query_text}")
        
        ranked_docs = ltr_ranker.rank(query_text, top_k=top_k)
        results[query_id] = [doc_id for doc_id, _ in ranked_docs]
    
    return results


def load_query_config(index_dir: Path) -> Config:
    cfg_path = index_dir / "config_used.json"
    if cfg_path.exists():
        return Config.from_json(str(cfg_path))
    print("WARNING: config_used.json not found. Falling back to default Config.")
    return Config(output_dir=str(index_dir))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LTR model")
    parser.add_argument('--indexdir', required=True, help='Path to built index directory')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file')
    parser.add_argument('--tfidf_results', required=True, 
                       help='Path to TF-IDF results JSON (from run_experiments.py)')
    parser.add_argument('--bm25_results', required=True,
                       help='Path to BM25 results JSON (from run_experiments.py)')
    parser.add_argument('--output', default='./core/results/models',
                       help='Output directory for trained model')
    parser.add_argument('--eval_output', default='./core/results/evaluation_results/ltr',
                       help='Output directory for LTR evaluation results')
    parser.add_argument('--top_k', type=int, default=100, help='Top documents to consider during ranking')
    
    args = parser.parse_args()
    index_dir = Path(args.indexdir)
    queries_path = Path(args.queries)
    tfidf_path = Path(args.tfidf_results)
    bm25_path = Path(args.bm25_results)
    
    # Create output directories
    model_dir = Path(args.output)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    eval_dir = Path(args.eval_output)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LEARNING-TO-RANK TRAINING & EVALUATION")
    print("="*80)
    
    # Step 1: Load index and statistics
    print("\n[1/7] Loading index and statistics...")
    inverted_index = load_pickle(index_dir / 'inverted_index.pkl')
    doc_stats = load_json(index_dir / 'doc_stats.json')
    term_stats = load_json(index_dir / 'term_stats.json')
    doc_store_path = index_dir / 'doc_store.jsonl'
    query_config = load_query_config(index_dir)
    print(f"  ✓ Loaded index with {len(inverted_index)} terms")
    
    # Step 2: Initialize base rankers
    print("\n[2/7] Initializing base rankers...")
    tfidf_ranker = TfIdfRanker(index_dir=str(index_dir))
    bm25_ranker = BM25Ranker(index_dir=str(index_dir), k1=query_config.bm25_k1, b=query_config.bm25_b)
    print("  ✓ TF-IDF and BM25 rankers initialized")
    
    # Step 3: Initialize feature extractor
    print("\n[3/7] Initializing feature extractor...")
    feature_extractor = FeatureExtractor(
        inverted_index, doc_stats, term_stats, doc_store_path
    )
    print("  ✓ Feature extractor ready (10 features per query-doc pair)")
    
    # Step 4: Load previous results
    print("\n[4/7] Loading TF-IDF and BM25 results...")
    tfidf_results = load_retrieval_results(tfidf_path)
    bm25_results = load_retrieval_results(bm25_path)
    print(f"  ✓ Loaded results for {len(tfidf_results)} queries")
    
    # Step 5: Prepare training data and train
    print("\n[5/7] Preparing training data and training LTR model...")
    print("  This may take a few minutes...")
    
    ltr_trainer = LTRTrainer(feature_extractor, query_config)
    
    X, y, groups = ltr_trainer.prepare_training_data(
        str(queries_path),
        tfidf_results,
        bm25_results,
        tfidf_ranker,
        bm25_ranker
    )
    
    if len(X) == 0:
        raise RuntimeError("No training samples were generated. Check your input data.")
    
    print(f"\n  Training data prepared:")
    print(f"    - Total samples: {len(X)}")
    print(f"    - Features per sample: {X.shape[1]}")
    print(f"    - Relevance labels distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100 if len(y) else 0
        print(f"      Label {int(label)}: {count} samples ({pct:.1f}%)")
    
    # Train the model
    ltr_trainer.train(X, y, groups, test_size=0.2, random_state=42)
    
    # Save model
    model_path = model_dir / 'ltr_model.json'
    scaler_path = model_dir / 'ltr_scaler.pkl'
    ltr_trainer.save_model(str(model_path), str(scaler_path))
    
    # Step 6: Run LTR retrieval
    print("\n[6/7] Running LTR retrieval for all queries...")
    ltr_ranker = LTRRanker(ltr_trainer, tfidf_ranker, bm25_ranker, feature_extractor)
    ltr_results = run_ltr_retrieval(queries_path, ltr_ranker, top_k=args.top_k)
    
    # Save LTR results
    ltr_results_file = eval_dir / 'ltr_results.json'
    with open(ltr_results_file, 'w', encoding='utf-8') as f:
        json.dump(ltr_results, f, indent=2)
    print(f"  ✓ LTR results saved to {ltr_results_file}")
    
    # Step 7: Evaluate and compare
    print("\n[7/7] Evaluating LTR model and comparing with baselines...")
    
    # Compare LTR vs BM25
    comparison_ltr_bm25 = compare_models(
        bm25_results,
        ltr_results,
        str(queries_path),
        model1_name="BM25",
        model2_name="LTR"
    )
    
    # Compare LTR vs TF-IDF
    comparison_ltr_tfidf = compare_models(
        tfidf_results,
        ltr_results,
        str(queries_path),
        model1_name="TF-IDF",
        model2_name="LTR"
    )
    
    # Save comparisons
    with open(eval_dir / 'ltr_vs_bm25_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_ltr_bm25, f, indent=2)
    
    with open(eval_dir / 'ltr_vs_tfidf_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_ltr_tfidf, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 LTR vs BM25 Comparison:")
    print(f"{'Metric':<20} {'BM25':<15} {'LTR':<15} {'Improvement':<15}")
    print("-" * 70)
    
    key_metrics = ['ndcg@10', 'ndcg@5', 'precision@10', 'mrr']
    for metric in key_metrics:
        if metric in comparison_ltr_bm25['metrics']:
            m = comparison_ltr_bm25['metrics'][metric]
            bm25_val = m['BM25_mean']
            ltr_val = m['LTR_mean']
            improvement = m['improvement_%']
            print(f"{metric:<20} {bm25_val:<15.4f} {ltr_val:<15.4f} {improvement:>+14.2f}%")
    
    print("\n📊 LTR vs TF-IDF Comparison:")
    print(f"{'Metric':<20} {'TF-IDF':<15} {'LTR':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in key_metrics:
        if metric in comparison_ltr_tfidf['metrics']:
            m = comparison_ltr_tfidf['metrics'][metric]
            tfidf_val = m['TF-IDF_mean']
            ltr_val = m['LTR_mean']
            improvement = m['improvement_%']
            print(f"{metric:<20} {tfidf_val:<15.4f} {ltr_val:<15.4f} {improvement:>+14.2f}%")
    
    # Success criteria
    print("\n" + "="*80)
    ndcg_improvement = comparison_ltr_bm25['metrics']['ndcg@10']['improvement_%']
    
    if ndcg_improvement > 5:
        print("✅ SUCCESS! LTR model shows significant improvement!")
        print(f"   nDCG@10 improved by {ndcg_improvement:.2f}% over BM25 baseline")
    elif ndcg_improvement > 0:
        print("✓ LTR model shows positive improvement")
        print(f"   nDCG@10 improved by {ndcg_improvement:.2f}% over BM25 baseline")
    else:
        print("⚠ LTR model needs tuning - showing negative improvement")
        print("   Consider:")
        print("   - Adding more training queries with judgments")
        print("   - Adjusting XGBoost hyperparameters")
        print("   - Engineering better features")
    
    print("\n" + "="*80)
    print("\n✓ All files saved to:")
    print(f"  Model: {model_dir}/")
    print(f"  Results: {eval_dir}/")
    print("\nFiles generated:")
    print("  • ltr_model.json - Trained XGBoost model")
    print("  • ltr_scaler.pkl - Feature scaler")
    print("  • feature_names.json - Feature name mapping")
    print("  • ltr_results.json - LTR ranked results")
    print("  • ltr_vs_bm25_comparison.json - Detailed comparison")
    print("  • ltr_vs_tfidf_comparison.json - Detailed comparison")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
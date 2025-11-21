#!/usr/bin/env python3

"""
IR Project - Person 3 (J Sai Varun)
Experiment Runner: Compare TF-IDF and BM25 models
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.IR_core.config import Config
from core.IR_core.io_utils import load_pickle
from core.IR_core.preprocessing import preprocess_text
from core.IR_core.ranking import BM25Ranker, TfIdfRanker
from core.IR_evaluation.evaluation import compare_models


def load_queries(queries_file: str):
    """Load queries from JSON file"""
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    return queries


def load_query_config(index_dir: Path) -> Config:
    """Load the preprocessing config used to build the index."""
    cfg_path = Path(index_dir) / "config_used.json"
    if cfg_path.exists():
        return Config.from_json(str(cfg_path))
    print("WARNING: config_used.json not found. Falling back to default Config.")
    return Config(output_dir=str(index_dir))


def preprocess_query(text: str, config: Config):
    """Apply the same preprocessing used during indexing."""
    return preprocess_text(
        text=text,
        lowercase=config.lowercase,
        remove_punct=config.remove_punct,
        remove_digits=config.remove_digits,
        remove_stopwords=config.remove_stopwords,
        use_stemming=config.use_stemming and not config.use_lemmatize,
        use_lemmatize=config.use_lemmatize,
        min_token_len=config.min_token_len,
        max_token_len=config.max_token_len,
    )


def run_retrieval_for_all_queries(queries, ranker, config: Config, top_k=100):
    """Run retrieval for all queries using given ranker"""
    results = {}
    for query in queries:
        query_id = query['query_id']
        query_text = query['query_text']
        print(f"Processing {query_id}: {query_text}")
        tokens = preprocess_query(query_text, config)
        if not tokens:
            results[query_id] = []
            continue

        scores = ranker.rank(tokens)
        if not scores:
            results[query_id] = []
            continue

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ranked_ids = [doc_id for doc_id, _ in ranked_docs]
        results[query_id] = ranked_ids
    return results


def generate_comparison_table(comparison_results, output_file):
    """Generate markdown/CSV comparison table"""
    metrics_data = []
    model1_name = comparison_results['model1_name']
    model2_name = comparison_results['model2_name']

    for metric, values in comparison_results['metrics'].items():
        metrics_data.append({
            'Metric': metric,
            model1_name: f"{values[f'{model1_name}_mean']:.4f}",
            model2_name: f"{values[f'{model2_name}_mean']:.4f}",
            'Improvement %': f"{values['improvement_%']:.2f}%",
            'Winner': values['winner']
        })

    df = pd.DataFrame(metrics_data)

    # Save as CSV
    csv_file = output_file.replace('.txt', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Comparison table saved to {csv_file}")

    # Save as text
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL COMPARISON: {model1_name} vs {model2_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")

    print(f"Comparison report saved to {output_file}")
    return df


def plot_comparison(comparison_results, output_dir):
    """Generate visualization plots comparing models"""
    metrics, model1_values, model2_values = [], [], []
    model1 = comparison_results['model1_name']
    model2 = comparison_results['model2_name']

    for metric, values in comparison_results['metrics'].items():
        if '@' in metric:
            metrics.append(metric)
            model1_values.append(values[f"{model1}_mean"])
            model2_values.append(values[f"{model2}_mean"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], model1_values, width, label=model1, alpha=0.8)
    ax.bar([i + width/2 for i in x], model2_values, width, label=model2, alpha=0.8)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison: {model1} vs {model2}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_file = Path(output_dir) / 'model_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")
    plt.close()

    improvements = [comparison_results['metrics'][m]['improvement_%'] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.barh(metrics, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('Improvement %', fontsize=12)
    ax.set_title(f'{model2} Improvement over {model1}', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    improvement_file = Path(output_dir) / 'improvement_chart.png'
    plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
    print(f"Improvement chart saved to {improvement_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run IR evaluation experiments")
    parser.add_argument('--indexdir', required=True, help='Path to built index directory')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file')
    parser.add_argument('--output', default='./core/results/evaluation_results', help='Output directory')
    parser.add_argument('--top_k', type=int, default=100, help='Number of top docs to retrieve')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("IR EVALUATION EXPERIMENT RUNNER")
    print("="*80)

    print("\n[1/6] Loading index and statistics...")
    inverted_index = load_pickle(Path(args.indexdir) / 'inverted_index.pkl')
    with open(Path(args.indexdir) / 'doc_stats.json', 'r') as f:
        doc_stats = json.load(f)
    with open(Path(args.indexdir) / 'term_stats.json', 'r') as f:
        term_stats = json.load(f)
    query_config = load_query_config(Path(args.indexdir))

    print(f"  ✓ Loaded index with {len(inverted_index)} terms")
    print(f"  ✓ Document stats: N={doc_stats['N']}, avgdl={doc_stats['avgdl']:.2f}")
    print(f"  ✓ Term stats available for {len(term_stats['df'])} terms")

    print("\n[2/6] Loading queries...")
    queries = load_queries(args.queries)
    print(f"  ✓ Loaded {len(queries)} queries")

    print("\n[3/6] Initializing ranking models...")
    tfidf_ranker = TfIdfRanker(index_dir=args.indexdir)
    bm25_ranker = BM25Ranker(index_dir=args.indexdir, k1=1.5, b=0.75)
    print("  ✓ TF-IDF ranker initialized")
    print("  ✓ BM25 ranker initialized")

    print(f"\n[4/6] Running retrieval for {len(queries)} queries...")
    print("\n  TF-IDF Retrieval:")
    tfidf_results = run_retrieval_for_all_queries(queries, tfidf_ranker, query_config, args.top_k)

    print("\n  BM25 Retrieval:")
    bm25_results = run_retrieval_for_all_queries(queries, bm25_ranker, query_config, args.top_k)

    print("\n[5/6] Evaluating models...")
    comparison = compare_models(tfidf_results, bm25_results, args.queries, model1_name="TF-IDF", model2_name="BM25")

    print("\n[6/6] Generating reports and visualizations...")
    with open(output_dir / 'tfidf_results.json', 'w') as f: json.dump(tfidf_results, f, indent=2)
    with open(output_dir / 'bm25_results.json', 'w') as f: json.dump(bm25_results, f, indent=2)
    with open(output_dir / 'comparison.json', 'w') as f: json.dump(comparison, f, indent=2)

    generate_comparison_table(comparison, str(output_dir / 'comparison_report.txt'))
    plot_comparison(comparison, output_dir)

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("\nKey Metrics (Mean Scores):")
    print(f"{'Metric':<20} {'TF-IDF':<15} {'BM25':<15} {'Winner':<10}")
    print("-" * 65)
    for metric in ['ndcg@10', 'precision@10', 'recall@10', 'mrr']:
        if metric in comparison['metrics']:
            m = comparison['metrics'][metric]
            print(f"{metric:<20} {m['TF-IDF_mean']:<15.4f} {m['BM25_mean']:<15.4f} {m['winner']:<10}")
    print("\n✓ Results saved to:", output_dir)
    print("="*80)


if __name__ == "__main__":
    main()

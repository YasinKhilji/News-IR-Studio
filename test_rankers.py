import heapq
import json
import os
from typing import Dict

from core.IR_core.config import Config
from core.IR_core.preprocessing import preprocess_text
from core.IR_core.ranking import BM25Ranker, TfIdfRanker


def load_doc_store(path: str) -> Dict[int, Dict[str, str]]:
    """Loads the lightweight doc_store.jsonl for printing results."""
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            store[int(doc["doc_id"])] = doc
    return store

def print_results(results, doc_store):
    """Helper function to print a ranked list nicely."""
    if not results:
        print("No results found.")
        return

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_store.get(doc_id, {})
        title = doc.get("title", f"Unknown Title (doc_id: {doc_id})")
        date = doc.get("date", "Unknown Date")
        
        print(f"\n{rank}. (Score: {score:.4f})")
        print(f"   Title: {title}")
        print(f"   Date:  {date}")
        print(f"   (doc_id: {doc_id})")

def load_query_config(index_dir: str) -> Config:
    cfg_path = os.path.join(index_dir, "config_used.json")
    if os.path.exists(cfg_path):
        return Config.from_json(cfg_path)
    print("WARNING: config_used.json not found. Using default config for queries.")
    return Config(output_dir=index_dir)


def preprocess_query(text: str, cfg: Config):
    return preprocess_text(
        text=text,
        lowercase=cfg.lowercase,
        remove_punct=cfg.remove_punct,
        remove_digits=cfg.remove_digits,
        remove_stopwords=cfg.remove_stopwords,
        use_stemming=cfg.use_stemming and not cfg.use_lemmatize,
        use_lemmatize=cfg.use_lemmatize,
        min_token_len=cfg.min_token_len,
        max_token_len=cfg.max_token_len,
    )


def main():
    INDEX_DIR = "./core/results/built_index"
    QUERY = "covid boosters and new vaccine" # <-- You can change this query
    TOP_K = 5
    query_cfg = load_query_config(INDEX_DIR)

    # 1. Load the document metadata (only needs to be done once)
    doc_store_path = os.path.join(INDEX_DIR, "doc_store.jsonl")
    print(f"Loading doc store from {doc_store_path}...")
    doc_store = load_doc_store(doc_store_path)

    # 2. Preprocess the query (only needs to be done once)
    print(f"Original query: '{QUERY}'")
    query_terms = preprocess_query(QUERY, query_cfg)
    print(f"Processed query terms: {query_terms}")

    # --- 3. Test BM25 Ranker ---
    print("\n" + "="*30)
    print("  TESTING BM25 RANKER")
    print("="*30)
    
    bm25_ranker = BM25Ranker(index_dir=INDEX_DIR)
    bm25_scores = bm25_ranker.rank(query_terms)
    top_k_bm25 = heapq.nlargest(TOP_K, bm25_scores.items(), key=lambda item: item[1])
    
    print_results(top_k_bm25, doc_store)

    # --- 4. Test TF-IDF Ranker ---
    print("\n" + "="*30)
    print("  TESTING TF-IDF RANKER")
    print("="*30)
    
    # This init will take a moment to pre-compute norms
    tfidf_ranker = TfIdfRanker(index_dir=INDEX_DIR)
    tfidf_scores = tfidf_ranker.rank(query_terms)
    top_k_tfidf = heapq.nlargest(TOP_K, tfidf_scores.items(), key=lambda item: item[1])
    
    print_results(top_k_tfidf, doc_store)


if __name__ == "__main__":
    main()
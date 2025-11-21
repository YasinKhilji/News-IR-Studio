import json
import math
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set
from collections import Counter  # <-- This is needed for TfIdfRanker

import numpy as np

from core.IR_core.document_store import DocumentStore
from core.IR_core.io_utils import load_pickle

# --- Your Official BM25 Ranker ---

class BM25Ranker:
    """
    Your official BM25 ranking model.
    """
    def __init__(
        self,
        index_dir: str,
        k1: float = 1.2,
        b: float = 0.75,
        index_data: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        term_stats_data: Optional[Dict[str, Any]] = None,
        doc_stats_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Loads all the necessary index files and stats.
        """
        print(f"Loading BM25 Ranker from directory: {index_dir}")
        
        # 1. Load the main index
        if index_data is None:
            index_path = os.path.join(index_dir, "inverted_index.pkl")
            self.index: Dict[str, Dict[int, Dict[str, Any]]] = load_pickle(index_path)
        else:
            self.index = index_data

        # 2. Load term statistics
        if term_stats_data is None:
            term_stats_path = os.path.join(index_dir, "term_stats.json")
            with open(term_stats_path, "r", encoding="utf-8") as f:
                term_stats = json.load(f)
        else:
            term_stats = term_stats_data
        self.idf: Dict[str, float] = term_stats["idf"]

        # 3. Load document statistics
        if doc_stats_data is None:
            doc_stats_path = os.path.join(index_dir, "doc_stats.json")
            with open(doc_stats_path, "r", encoding="utf-8") as f:
                doc_stats = json.load(f)
        else:
            doc_stats = doc_stats_data

        self.doc_len: Dict[int, int] = {int(k): v for k, v in doc_stats["doc_len"].items()}
        self.N: int = doc_stats["N"]
        self.avgdl: float = doc_stats["avgdl"]
        
        # 4. Store BM25 parameters
        self.k1 = k1
        self.b = b
        
        print("BM25 Ranker loaded successfully.")

    def rank(self, query_terms: List[str]) -> Dict[int, float]:
        """
        Takes a list of preprocessed query terms and returns a dict 
        of {doc_id: score}
        """
        scores: Dict[int, float] = {}
        
        for t in query_terms:
            postings = self.index.get(t)
            if not postings:
                continue
            
            idf = self.idf.get(t, 0.0)
            if idf == 0.0:
                continue
                
            for doc_id, payload in postings.items():
                tf = payload["tf"]
                dl = self.doc_len.get(doc_id, 0)
                
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                increment = idf * (tf * (self.k1 + 1)) / (denom + 1e-9)
                
                scores[doc_id] = scores.get(doc_id, 0.0) + increment
                
        return scores

# --- Your Official TF-IDF Ranker ---

class TfIdfRanker:
    """
    Your official TF-IDF (with Cosine Similarity) ranking model.
    Uses log-frequency weighting.
    """
    def __init__(
        self,
        index_dir: str,
        index_data: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        term_stats_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Loads all the necessary index files and pre-computes
        document vector norms.
        """
        print(f"Loading TfIdfRanker from directory: {index_dir}")
        
        # 1. Load the main index
        if index_data is None:
            index_path = os.path.join(index_dir, "inverted_index.pkl")
            self.index: Dict[str, Dict[int, Dict[str, Any]]] = load_pickle(index_path)
        else:
            self.index = index_data

        # 2. Load term statistics
        if term_stats_data is None:
            term_stats_path = os.path.join(index_dir, "term_stats.json")
            with open(term_stats_path, "r", encoding="utf-8") as f:
                term_stats = json.load(f)
        else:
            term_stats = term_stats_data
        self.idf: Dict[str, float] = term_stats["idf"]
        
        # 3. Pre-compute document norms (vector magnitudes)
        print("Pre-computing document norms (this may take a moment)...")
        self.doc_norms = self._precompute_doc_norms()
        print(f"Computed norms for {len(self.doc_norms)} documents.")

    def _get_log_tf(self, tf: int) -> float:
        """Calculates log-frequency weight for TF."""
        return (1 + math.log10(tf)) if tf > 0 else 0.0

    def _precompute_doc_norms(self) -> Dict[int, float]:
        """
        Calculates the norm |D| for every document D in the collection.
        |D| = sqrt( sum( (tf_idf_weight)^2 ) )
        """
        doc_norms_squared: Dict[int, float] = {}
        
        for term, idf_score in self.idf.items():
            postings = self.index.get(term)
            if not postings:
                continue
            
            for doc_id, payload in postings.items():
                tf = payload["tf"]
                weight = self._get_log_tf(tf) * idf_score
                doc_norms_squared[doc_id] = doc_norms_squared.get(doc_id, 0.0) + (weight ** 2)
        
        doc_norms = {doc_id: math.sqrt(norm_sq) for doc_id, norm_sq in doc_norms_squared.items()}
        return doc_norms

    def rank(self, query_terms: List[str]) -> Dict[int, float]:
        """
        Takes a list of preprocessed query terms and returns a dict 
        of {doc_id: cosine_similarity_score}
        """
        scores: Dict[int, float] = {}
        if not query_terms:
            return scores

        # --- 1. Calculate Query Vector and Query Norm ---
        query_vector: Dict[str, float] = {}
        query_norm_squared = 0.0
        
        query_tf_counts = Counter(query_terms)
        
        for term, tf in query_tf_counts.items():
            if term in self.idf:
                idf = self.idf[term]
                weight = self._get_log_tf(tf) * idf
                query_vector[term] = weight
                query_norm_squared += weight ** 2
        
        query_norm = math.sqrt(query_norm_squared)
        if query_norm == 0:
            return scores

        # --- 2. Calculate Dot Products ---
        dot_products: Dict[int, float] = {}
        
        for term, query_weight in query_vector.items():
            postings = self.index.get(term)
            if not postings:
                continue
            
            idf = self.idf[term]
            
            for doc_id, payload in postings.items():
                doc_tf = payload["tf"]
                doc_weight = self._get_log_tf(doc_tf) * idf
                dot_products[doc_id] = dot_products.get(doc_id, 0.0) + (query_weight * doc_weight)

        # --- 3. Calculate Final Cosine Similarity ---
        for doc_id, dot_prod in dot_products.items():
            doc_norm = self.doc_norms.get(doc_id)
            if not doc_norm or doc_norm == 0:
                continue
            
            scores[doc_id] = dot_prod / (query_norm * doc_norm)
            
        return scores


class LanguageModelRanker:
    """
    Query-likelihood language-model ranker with Dirichlet smoothing.
    """

    def __init__(
        self,
        index_dir: str,
        mu: float = 2000.0,
        index_data: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
        doc_stats_data: Optional[Dict[str, Any]] = None,
    ):
        self.mu = mu

        if index_data is None:
            index_path = os.path.join(index_dir, "inverted_index.pkl")
            self.index: Dict[str, Dict[int, Dict[str, Any]]] = load_pickle(index_path)
        else:
            self.index = index_data

        if doc_stats_data is None:
            doc_stats_path = os.path.join(index_dir, "doc_stats.json")
            with open(doc_stats_path, "r", encoding="utf-8") as handle:
                doc_stats = json.load(handle)
        else:
            doc_stats = doc_stats_data

        self.doc_len: Dict[int, int] = {int(k): v for k, v in doc_stats["doc_len"].items()}
        self.collection_len: float = float(sum(self.doc_len.values()))
        self.term_cf: Dict[str, int] = self._compute_collection_frequencies()

    def _compute_collection_frequencies(self) -> Dict[str, int]:
        cf: Dict[str, int] = {}
        for term, postings in self.index.items():
            cf[term] = sum(payload["tf"] for payload in postings.values())
        return cf

    def rank(
        self,
        query_terms: Sequence[str],
        candidate_docs: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        if not query_terms:
            return {}

        candidate_filter: Optional[Set[int]] = set(candidate_docs) if candidate_docs else None
        scores: Dict[int, float] = {}

        for term in query_terms:
            postings = self.index.get(term)
            if not postings or self.collection_len == 0:
                continue

            cf = self.term_cf.get(term, 0)
            p_collection = cf / self.collection_len if self.collection_len > 0 else 0.0

            for doc_id, payload in postings.items():
                if candidate_filter and doc_id not in candidate_filter:
                    continue
                dl = self.doc_len.get(doc_id, 0)
                denom = dl + self.mu
                if denom <= 0:
                    continue
                prob = (payload["tf"] + self.mu * p_collection) / denom
                if prob <= 0:
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + math.log(prob)

        return scores


class TemporalBM25Ranker(BM25Ranker):
    """
    Adds a recency-aware boost on top of standard BM25.
    """

    def __init__(
        self,
        index_dir: str,
        document_store: DocumentStore,
        k1: float = 1.2,
        b: float = 0.75,
        half_life_days: float = 30.0,
        alpha: float = 0.35,
        reference_date: Optional[datetime] = None,
        **kwargs: Any,
    ):
        super().__init__(index_dir=index_dir, k1=k1, b=b, **kwargs)
        self.document_store = document_store
        self.alpha = alpha
        self.reference_date = reference_date or document_store.max_date or datetime.utcnow()
        self.decay_lambda = math.log(2) / max(half_life_days, 1.0)

    def rank(self, query_terms: List[str]) -> Dict[int, float]:
        base_scores = super().rank(query_terms)
        if not base_scores:
            return base_scores

        boosted: Dict[int, float] = {}
        for doc_id, score in base_scores.items():
            doc_dt = self.document_store.get_datetime(doc_id)
            if not doc_dt:
                boosted[doc_id] = score
                continue
            days_old = max((self.reference_date - doc_dt).days, 0)
            recency = math.exp(-self.decay_lambda * days_old)
            boosted_score = (1 - self.alpha) * score + self.alpha * recency
            boosted[doc_id] = boosted_score
        return boosted


class EmbeddingRanker:
    """
    Lightweight semantic ranker that reorders BM25 candidates using MiniLM embeddings.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        bm25_ranker: BM25Ranker,
        preprocess_query_fn: Callable[[str], Sequence[str]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EmbeddingRanker. "
                "Install it via `pip install sentence-transformers`."
            ) from exc

        self.embedder = SentenceTransformer(model_name, device=device)
        self.doc_store = document_store
        self.bm25_ranker = bm25_ranker
        self.preprocess_query_fn = preprocess_query_fn

    def rank(
        self,
        query_text: str,
        top_k: int = 20,
        candidate_pool: int = 200,
    ) -> Dict[int, float]:
        query_text = (query_text or "").strip()
        if not query_text:
            return {}

        query_embedding = self._normalize_vector(
            self.embedder.encode([query_text], convert_to_numpy=True)[0]
        )

        query_terms = list(self.preprocess_query_fn(query_text))
        base_scores = self.bm25_ranker.rank(query_terms)
        if not base_scores:
            return {}

        sorted_candidates = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [doc_id for doc_id, _ in sorted_candidates[:candidate_pool]]

        corpus_texts = [
            self.doc_store.get_text(doc_id)
            or (self.doc_store.get(doc_id, include_text=False) or {}).get("title", "")
            or ""
            for doc_id in candidate_ids
        ]

        doc_embeddings = self.embedder.encode(
            corpus_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False
        )
        doc_embeddings = self._normalize_matrix(doc_embeddings)
        scores = doc_embeddings @ query_embedding

        paired = sorted(
            zip(candidate_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]

        return {doc_id: float(score) for doc_id, score in paired}

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _normalize_matrix(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
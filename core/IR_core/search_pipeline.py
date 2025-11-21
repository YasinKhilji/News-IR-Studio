from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.IR_core.boolean_search import BooleanQueryEngine
from core.IR_core.config import Config
from core.IR_core.document_store import DocumentStore
from core.IR_core.io_utils import load_pickle
from core.IR_core.preprocessing import preprocess_text
from core.IR_core.ranking import (
    BM25Ranker,
    EmbeddingRanker,
    LanguageModelRanker,
    TemporalBM25Ranker,
    TfIdfRanker,
)
from core.IR_evaluation.evaluation import evaluate_query
from core.IR_evaluation.itr import FeatureExtractor, LTRRanker, LTRTrainer


@dataclass
class SearchResult:
    rank: int
    doc_id: int
    score: float
    title: Optional[str]
    date: Optional[str]
    category: Optional[str]
    link: Optional[str]
    snippet: Optional[str]


class SearchPipeline:
    """
    High-level orchestrator for ranked retrieval, Boolean search, evaluation, and
    metadata-aware filtering. Designed to back the Gradio UI and programmatic
    experiments.
    """

    def __init__(
        self,
        index_dir: str,
        qrels_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        preload_text: bool = False,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.models_dir = Path(models_dir or (self.index_dir.parent / "models"))
        self.qrels_path = Path(qrels_path) if qrels_path else None

        cfg_path = self.index_dir / "config_used.json"
        self.config = Config.from_json(str(cfg_path)) if cfg_path.exists() else Config(output_dir=str(index_dir))

        self.doc_store = DocumentStore(index_dir, self.config, preload_text=preload_text)
        self._load_index_artifacts()

        self._query_processor = self._build_query_processor()
        self.boolean_engine = BooleanQueryEngine(
            self.inverted_index,
            self._query_processor,
            self.doc_len.keys(),
        )

        self.tfidf_ranker = TfIdfRanker(
            index_dir=str(self.index_dir),
            index_data=self.inverted_index,
            term_stats_data=self.term_stats_raw,
        )
        self.bm25_ranker = BM25Ranker(
            index_dir=str(self.index_dir),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            index_data=self.inverted_index,
            term_stats_data=self.term_stats_raw,
            doc_stats_data=self.doc_stats_raw,
        )
        self.language_model_ranker = LanguageModelRanker(
            index_dir=str(self.index_dir),
            index_data=self.inverted_index,
            doc_stats_data=self.doc_stats_raw,
        )
        self.temporal_ranker = TemporalBM25Ranker(
            index_dir=str(self.index_dir),
            document_store=self.doc_store,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            index_data=self.inverted_index,
            term_stats_data=self.term_stats_raw,
            doc_stats_data=self.doc_stats_raw,
        )
        self._embedding_ranker: Optional[EmbeddingRanker] = None

        self.qrels_map = self._load_qrels()
        self.eval_cache: Dict[str, Dict[str, float]] = {}

        self.feature_extractor: Optional[FeatureExtractor] = None
        self.ltr_trainer: Optional[LTRTrainer] = None
        self.ltr_ranker: Optional[LTRRanker] = None
        self._maybe_bootstrap_ltr()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_index_artifacts(self) -> None:
        index_path = self.index_dir / "inverted_index.pkl"
        term_stats_path = self.index_dir / "term_stats.json"
        doc_stats_path = self.index_dir / "doc_stats.json"

        self.inverted_index = load_pickle(str(index_path))
        with open(term_stats_path, "r", encoding="utf-8") as handle:
            self.term_stats_raw = json.load(handle)
        with open(doc_stats_path, "r", encoding="utf-8") as handle:
            self.doc_stats_raw = json.load(handle)

        self.doc_len: Dict[int, int] = {int(k): v for k, v in self.doc_stats_raw["doc_len"].items()}

    def _build_query_processor(self):
        def processor(text: str) -> List[str]:
            return preprocess_text(
                text=text,
                lowercase=self.config.lowercase,
                remove_punct=self.config.remove_punct,
                remove_digits=self.config.remove_digits,
                remove_stopwords=self.config.remove_stopwords,
                use_stemming=self.config.use_stemming and not self.config.use_lemmatize,
                use_lemmatize=self.config.use_lemmatize,
                min_token_len=self.config.min_token_len,
                max_token_len=self.config.max_token_len,
            )

        return processor

    def _load_qrels(self) -> Dict[str, Dict[str, Any]]:
        if not self.qrels_path or not self.qrels_path.exists():
            return {}
        with open(self.qrels_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {entry["query_id"]: entry for entry in data}

    def _maybe_bootstrap_ltr(self) -> None:
        model_path = self.models_dir / "ltr_model.json"
        scaler_path = self.models_dir / "ltr_scaler.pkl"
        if not (model_path.exists() and scaler_path.exists()):
            return

        feature_extractor = FeatureExtractor(
            self.inverted_index,
            self.doc_stats_raw,
            self.term_stats_raw,
            self.index_dir / "doc_store.jsonl",
        )
        trainer = LTRTrainer(feature_extractor, self.config)
        trainer.load_model(str(model_path), str(scaler_path))

        self.feature_extractor = feature_extractor
        self.ltr_trainer = trainer
        self.ltr_ranker = LTRRanker(
            ltr_trainer=trainer,
            tfidf_ranker=self.tfidf_ranker,
            bm25_ranker=self.bm25_ranker,
            feature_extractor=feature_extractor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_query(self, text: str) -> List[str]:
        return self._query_processor(text)

    def rank_models(self) -> List[str]:
        models = ["BM25", "TF-IDF", "BM25 (Temporal)", "Language Model"]
        if self.ltr_ranker:
            models.append("Learning-to-Rank")
        models.append("Embedding (semantic)")
        return models

    def available_categories(self) -> List[str]:
        cats = {record.category for record in self.doc_store.iter_metadata() if record.category}
        return sorted(cats)

    def list_queries(self) -> List[Tuple[str, str]]:
        return [(qid, data["query_text"]) for qid, data in self.qrels_map.items()]

    def search_ranked(
        self,
        query_text: str,
        model_name: str,
        top_k: int = 10,
        categories: Optional[List[str]] = None,
        date_range: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None,
        embedding_candidate_pool: int = 200,
    ) -> Tuple[List[SearchResult], List[int]]:
        query_terms = self.preprocess_query(query_text)
        if model_name != "Embedding (semantic)" and not query_terms:
            return [], []

        allowed_docs = (
            self.doc_store.filter_doc_ids(categories, date_range) if categories or date_range else None
        )

        scores: Dict[int, float]
        if model_name == "BM25":
            scores = self.bm25_ranker.rank(query_terms)
        elif model_name == "TF-IDF":
            scores = self.tfidf_ranker.rank(query_terms)
        elif model_name == "BM25 (Temporal)":
            scores = self.temporal_ranker.rank(query_terms)
        elif model_name == "Language Model":
            scores = self.language_model_ranker.rank(query_terms, candidate_docs=allowed_docs)
        elif model_name == "Learning-to-Rank":
            if not self.ltr_ranker:
                raise RuntimeError("LTR model not trained yet. Run train_itr.py first.")
            ranked = self.ltr_ranker.rank(query_text, top_k=max(top_k * 2, 50))
            scores = {doc_id: float(score) for doc_id, score in ranked}
        elif model_name == "Embedding (semantic)":
            embedding_ranker = self._get_embedding_ranker()
            scores = embedding_ranker.rank(
                query_text,
                top_k=max(top_k, 10),
                candidate_pool=max(embedding_candidate_pool, top_k * 4),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if allowed_docs is not None and model_name != "Language Model":
            scores = {doc_id: score for doc_id, score in scores.items() if doc_id in allowed_docs}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._enrich_results(ranked), [doc_id for doc_id, _ in ranked]

    def search_boolean(
        self,
        query_text: str,
        limit: int = 100,
        categories: Optional[List[str]] = None,
        date_range: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None,
        rerank_with_bm25: bool = True,
    ) -> List[SearchResult]:
        doc_ids = self.boolean_engine.search(query_text)
        if not doc_ids:
            return []

        allowed = (
            self.doc_store.filter_doc_ids(categories, date_range) if categories or date_range else None
        )
        if allowed is not None:
            doc_ids = [doc_id for doc_id in doc_ids if doc_id in allowed]

        if rerank_with_bm25:
            query_terms = self.preprocess_query(query_text)
            bm25_scores = self.bm25_ranker.rank(query_terms)
            scored = sorted(
                [(doc_id, bm25_scores.get(doc_id, 0.0)) for doc_id in doc_ids],
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            scored = [(doc_id, 0.0) for doc_id in doc_ids]

        return self._enrich_results(scored[:limit])

    def evaluate_model(
        self,
        model_name: str,
        top_k: int = 20,
        k_values: Sequence[int] = (5, 10, 20),
    ) -> Dict[str, float]:
        cache_key = f"{model_name}:{top_k}:{','.join(map(str, k_values))}"
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        if not self.qrels_map:
            raise RuntimeError("No relevance judgments available for evaluation.")

        aggregated: Dict[str, float] = {f"precision@{k}": 0.0 for k in k_values}
        aggregated.update({f"recall@{k}": 0.0 for k in k_values})
        aggregated.update({f"ndcg@{k}": 0.0 for k in k_values})
        aggregated["mrr"] = 0.0

        total = 0
        for qid, payload in self.qrels_map.items():
            results, ranked_ids = self.search_ranked(payload["query_text"], model_name, top_k=top_k)
            if not ranked_ids:
                continue
            metrics = evaluate_query(ranked_ids, payload["relevance_judgments"], list(k_values))
            for key, value in metrics.items():
                aggregated[key] += value
            total += 1

        if total:
            for key in aggregated:
                aggregated[key] /= total

        self.eval_cache[cache_key] = aggregated
        return aggregated

    def evaluate_single_query(
        self,
        ranked_ids: List[int],
        query_id: str,
        k_values: Sequence[int] = (5, 10),
    ) -> Optional[Dict[str, float]]:
        payload = self.qrels_map.get(query_id)
        if not payload:
            return None
        return evaluate_query(ranked_ids, payload["relevance_judgments"], list(k_values))

    def prepare_rag_context(
        self,
        doc_ids: Sequence[int],
        limit: int = 3,
        query_text: Optional[str] = None,
        window_sentences: int = 3,
    ) -> List[Dict[str, Optional[str]]]:
        query_tokens = set(self.preprocess_query(query_text)) if query_text else set()
        contexts: List[Dict[str, Optional[str]]] = []
        for doc_id in doc_ids[:limit]:
            record = self.doc_store.get(doc_id, include_text=True)
            if record:
                snippet = self._select_relevant_snippet(
                    record.get("text") or "",
                    query_tokens,
                    window_sentences=window_sentences,
                )
                record["snippet"] = snippet
                contexts.append(record)
        return contexts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_results(self, ranked_pairs: Iterable[Tuple[int, float]]) -> List[SearchResult]:
        results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(ranked_pairs, start=1):
            record = self.doc_store.get(doc_id, include_text=False) or {}
            snippet = self.doc_store.build_snippet(doc_id)
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    score=float(score),
                    title=record.get("title"),
                    date=record.get("date"),
                    category=record.get("category"),
                    link=record.get("link"),
                    snippet=snippet,
                )
            )
        return results

    def _get_embedding_ranker(self) -> EmbeddingRanker:
        if self._embedding_ranker is None:
            self._embedding_ranker = EmbeddingRanker(
                document_store=self.doc_store,
                bm25_ranker=self.bm25_ranker,
                preprocess_query_fn=self.preprocess_query,
            )
        return self._embedding_ranker

    def _select_relevant_snippet(
        self,
        text: str,
        query_tokens: set,
        window_sentences: int = 3,
    ) -> str:
        if not text:
            return ""
        sentences = _split_sentences(text)
        if not sentences:
            return text[:500]
        if not query_tokens:
            return " ".join(sentences[:window_sentences])[:500]

        best_score = -1
        best_window = sentences[:window_sentences]
        num_sentences = len(sentences)
        for start in range(0, num_sentences):
            window = sentences[start : start + window_sentences]
            if not window:
                continue
            joined = " ".join(window)
            tokens = set(joined.lower().split())
            score = sum(1 for token in query_tokens if token in tokens)
            if score > best_score:
                best_score = score
                best_window = window
        snippet = " ".join(best_window).strip()
        if len(snippet) > 600:
            snippet = snippet[:597].rstrip() + "..."
        return snippet


def _split_sentences(text: str) -> List[str]:
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


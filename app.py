import argparse
import json
from collections import Counter
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import streamlit as st

from core.IR_core.rag import RAGAnswerer
from core.IR_core.search_pipeline import SearchPipeline, SearchResult
from core.utils.model_eval import (
    EVAL_CANDIDATE_POOL,
    EVAL_QUERY_PATH,
    EVAL_TOP_K,
    MODEL_EVAL_CACHE_PATH,
    evaluate_models,
    load_eval_queries,
    load_persisted_eval_cache,
    save_persisted_eval_cache,
)


def serialize_date_value(value: Optional[Union[str, date, datetime]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.isoformat()



def discover_index_dirs(base_dir: str) -> Dict[str, str]:
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}

    discovered: Dict[str, str] = {}
    candidates = [base_path] if (base_path / "inverted_index.pkl").exists() else []
    candidates += [p for p in base_path.iterdir() if p.is_dir()]

    for path in candidates:
        if (path / "inverted_index.pkl").exists():
            label = path.name
            if label in discovered:
                label = path.resolve().as_posix()
            discovered[label] = str(path.resolve())
    return discovered


class PipelineManager:
    """Cache of SearchPipeline instances keyed by dataset label."""

    def __init__(self, datasets: Dict[str, str], qrels_path: Optional[str], models_root: Optional[str]):
        if not datasets:
            raise RuntimeError("No index directories found. Build an index first.")
        self.datasets = datasets
        self.qrels_path = qrels_path
        self.models_root = models_root
        self._pipelines: Dict[str, SearchPipeline] = {}

    def list_datasets(self) -> Dict[str, str]:
        return self.datasets

    def add_dataset(self, label: str, path: str) -> None:
        resolved = Path(path).resolve()
        if not (resolved / "inverted_index.pkl").exists():
            raise FileNotFoundError(f"{resolved} does not look like an index directory.")
        self.datasets[label] = str(resolved)
        self._pipelines.pop(label, None)

    def get(self, label: str) -> SearchPipeline:
        if label not in self.datasets:
            raise KeyError(f"Unknown dataset '{label}'")
        if label not in self._pipelines:
            idx_dir = self.datasets[label]
            models_dir = (
                Path(idx_dir).parent / "models" if self.models_root is None else Path(self.models_root)
            )
            pipeline = SearchPipeline(
                index_dir=idx_dir,
                qrels_path=self.qrels_path,
                models_dir=str(models_dir),
                preload_text=False,
            )
            self._pipelines[label] = pipeline
        return self._pipelines[label]


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def format_query_choices(pipeline: SearchPipeline) -> List[str]:
    return [f"{qid} | {text}" for qid, text in pipeline.list_queries()]


def extract_query_id(selection: Optional[str]) -> Optional[str]:
    if not selection:
        return None
    return selection.split("|", 1)[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the News IR Studio UI (Streamlit).")
    parser.add_argument("--index-base", default="core/results", help="Base directory for index folders.")
    parser.add_argument("--qrels", default="data/queries_relevance_auto.json", help="Relevance judgments JSON path.")
    parser.add_argument("--models-dir", default="core/results/models", help="Directory containing trained LTR models.")
    parser.add_argument(
        "--rag-model",
        default="google/flan-t5-base",
        help="HuggingFace model name for the RAG answer generator.",
    )
    parser.add_argument(
        "--eval-queries",
        default=str(EVAL_QUERY_PATH),
        help="JSON file with 100 labeled queries used for automatic evaluation.",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_resource(show_spinner=False)
def load_manager(index_base: str, qrels: str, models_dir: str) -> PipelineManager:
    datasets = discover_index_dirs(index_base)
    fallback = Path("core/results/built_index")
    if "built_index" not in datasets and fallback.exists():
        datasets["built_index"] = str(fallback.resolve())
    return PipelineManager(datasets, qrels, models_dir)


def ensure_state_defaults(pipeline: SearchPipeline) -> None:
    st.session_state.setdefault("dataset_label", list(st.session_state["datasets"].keys())[0])
    st.session_state.setdefault("backend_categories", [])
    st.session_state.setdefault("backend_start_date", None)
    st.session_state.setdefault("backend_end_date", None)
    st.session_state.setdefault("backend_start_date_enabled", False)
    st.session_state.setdefault("backend_end_date_enabled", False)
    st.session_state.setdefault("backend_top_k", 10)
    st.session_state.setdefault("backend_candidate_pool", 200)
    st.session_state.setdefault("backend_model_override", False)
    st.session_state.setdefault("search_state", {})
    st.session_state.setdefault("ranked_results", [])
    st.session_state.setdefault("ranked_status", "Run a search to see ranked results.")
    st.session_state.setdefault("rag_answer", "")
    st.session_state.setdefault("frontend_results", [])
    st.session_state.setdefault("frontend_rag", "")
    st.session_state.setdefault("frontend_trigger", False)
    st.session_state.setdefault("backend_query", "")
    st.session_state.setdefault("frontend_query", "")
    st.session_state.setdefault("labeled_query_option", None)
    st.session_state.setdefault("live_eval_message", "Run a ranked search to populate Live Evaluation.")

    for key in ("backend_start_date", "backend_end_date"):
        value = st.session_state.get(key)
        if value == "":
            st.session_state[key] = None
        elif isinstance(value, str):
            try:
                st.session_state[key] = datetime.fromisoformat(value).date()
            except ValueError:
                st.session_state[key] = None

    available_models = pipeline.rank_models()
    st.session_state["available_models"] = available_models
    if "backend_model" not in st.session_state:
        st.session_state["backend_model"] = available_models[0]


def ensure_rag(model_name: str) -> RAGAnswerer:
    if "rag_answerer" not in st.session_state:
        st.session_state["rag_answerer"] = RAGAnswerer(model_name=model_name)
    return st.session_state["rag_answerer"]


def load_dataset_overview(pipeline: SearchPipeline) -> Tuple[pd.DataFrame, Dict[str, str]]:
    sample_size = 8
    rows = []
    category_counts: Counter[str] = Counter()
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None

    for idx, record in enumerate(pipeline.doc_store.iter_metadata()):
        if idx < sample_size:
            rows.append(
                {
                    "Doc ID": record.doc_id,
                    "Title": record.title,
                    "Date": record.date,
                    "Category": record.category,
                    "Link": record.link,
                }
            )
        if record.category:
            category_counts.update([record.category])
        dt = parse_date(record.date)
        if dt:
            earliest = dt if earliest is None or dt < earliest else earliest
            latest = dt if latest is None or dt > latest else latest

    df = pd.DataFrame(rows)
    stats = {
        "Total Docs": pipeline.doc_store.total_docs,
        "Categories": len(category_counts),
        "Earliest": earliest.date().isoformat() if earliest else "n/a",
        "Latest": latest.date().isoformat() if latest else "n/a",
    }
    return df, stats


def run_ranked_search(
    pipeline: SearchPipeline,
    query_text: str,
        model_name: str,
        top_k: int,
    categories: Sequence[str],
    start_date: str,
    end_date: str,
        candidate_pool: int,
    labeled_option: Optional[str],
) -> Tuple[List[SearchResult], List[int], str, Dict]:
    if not query_text.strip():
        return [], [], "Enter a query to search.", {}

    date_range = None
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if start_dt or end_dt:
        date_range = (start_dt, end_dt)

    results, ranked_ids = pipeline.search_ranked(
        query_text=query_text,
        model_name=model_name,
        top_k=top_k,
        categories=list(categories) or None,
        date_range=date_range,
        embedding_candidate_pool=candidate_pool,
    )

    qid = extract_query_id(labeled_option)
    state = {
        "doc_ids": ranked_ids,
        "query_text": query_text,
        "model_name": model_name,
        "dataset": st.session_state["dataset_label"],
        "query_id": qid,
    }

    status = f"Retrieved {len(ranked_ids)} documents with {model_name}."
    if qid:
        status += f" Labeled query: {qid}"
    return results, ranked_ids, status, state


def results_to_dataframe(results: List[SearchResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": item.rank,
                "Score": round(item.score, 4),
                "Title": item.title,
                "Date": item.date,
                "Category": item.category,
                "Snippet": item.snippet,
                "Doc ID": item.doc_id,
                "Link": item.link,
            }
            for item in results
        ]
    )


def run_boolean_search(
    pipeline: SearchPipeline,
        query_text: str,
        limit: int,
    categories: Sequence[str],
    start_date: str,
    end_date: str,
        rerank: bool,
) -> Tuple[pd.DataFrame, str]:
    date_range = None
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if start_dt or end_dt:
        date_range = (start_dt, end_dt)

    results = pipeline.search_boolean(
        query_text=query_text,
        limit=limit,
        categories=list(categories) or None,
        date_range=date_range,
        rerank_with_bm25=rerank,
    )
    return results_to_dataframe(results), f"Found {len(results)} matching documents."


def run_rag_answer(pipeline: SearchPipeline, search_state: Dict, rag_model: str) -> str:
    if not search_state or not search_state.get("doc_ids"):
        return "Run a ranked search first."

    doc_ids = search_state["doc_ids"]
    doc_limit = min(3, len(doc_ids))

    if search_state["model_name"] not in {"BM25", "Learning-to-Rank"}:
        _, fallback_ids = pipeline.search_ranked(
            search_state["query_text"],
            "BM25",
            top_k=doc_limit,
        )
        doc_ids = fallback_ids or doc_ids[:doc_limit]

    contexts = pipeline.prepare_rag_context(
        doc_ids,
        limit=doc_limit,
        query_text=search_state["query_text"],
    )
    answerer = ensure_rag(rag_model)
    return answerer.generate(search_state["query_text"], contexts)


def format_model_label(model: str, best_model: Optional[str]) -> str:
    if model == best_model:
        return f"{model} ⭐ best"
    return model


def set_global_styles():
    st.set_page_config(
        page_title="News IR Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>

    /* ---------------------------- ROOT COLORS ---------------------------- */
    :root {
        --lav-light: #f5f7ff;
        --lav-mid: #e8eaff;
        --lav-accent: #b9bfff;
        --lav-purple: #8b95ff;
        --lav-deep: #6470ff;

        --pink-soft: #ffe8f4;
        --pink-mid: #ffcee9;
        --pink-strong: #ff9ed1;

        --text-dark: #2b2d42;
        --text-light: #4b4d63;

        --card-bg: #ffffff;
        --page-bg: #fafbff;
        --border-soft: #e0e3f0;
        --shadow-soft: 0 10px 25px rgba(0,0,0,0.07);
    }

    /* ---------------------------- PAGE BACKGROUND ---------------------------- */
    html, body, .stApp {
        background: var(--page-bg) !important;
        color: var(--text-dark) !important;
    }

    .main .block-container {
        background: linear-gradient(135deg, var(--lav-light), var(--pink-soft));
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        box-shadow: var(--shadow-soft);
        border: none;
        position: relative;
        overflow: hidden;
    }

    .main .block-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--lav-purple), var(--pink-strong));
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 1rem;
    }

    /* ---------------------------- TABS ---------------------------- */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, var(--pink-soft), var(--lav-light));
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 1.7rem;
        border: none !important;
        box-shadow: var(--shadow-soft);
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px;
        margin: 3px;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid transparent;
        color: var(--text-light);
        transition: 0.25s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--lav-purple), var(--pink-strong));
        color: white !important;
        box-shadow: 0 6px 20px rgba(168,139,235,0.35);
    }

    /* ---------------------------- INPUTS & SELECTBOXES ---------------------------- */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stDateInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid var(--border-soft) !important;
        color: var(--text-dark) !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-soft) !important;
    }

    .stSelectbox [data-baseweb="select"] span,
    .stMultiSelect [data-baseweb="select"] span {
        color: var(--text-dark) !important;
    }

    .stSelectbox [data-baseweb="select"] span[data-id="placeholder"],
    .stMultiSelect [data-baseweb="select"] span[data-id="placeholder"] {
        color: var(--text-dark) !important;
    }

    .stSelectbox [data-baseweb="select"] svg,
    .stMultiSelect [data-baseweb="select"] svg {
        color: var(--text-dark) !important;
        fill: var(--text-dark) !important;
    }

    label {
        color: var(--text-dark) !important;
        font-weight: 600;
    }

    /* ---------------------------- SLIDERS ---------------------------- */
    .stSlider > div > div > div > div {
        background: var(--lav-purple) !important;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }

    /* ---------------------------- BUTTONS ---------------------------- */
    .stButton > button {
        background: linear-gradient(45deg, var(--lav-purple), var(--pink-strong));
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0 6px 16px rgba(140, 119, 255, 0.25);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(140, 119, 255, 0.35);
    }

    /* ---------------------------- CARDS (RAG, Results, Stats) ---------------------------- */
    .hero-card, .result-card, .rag-card, .stat-box {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-soft);
        color: var(--text-dark);
    }

    .hero-card h2 {
        color: var(--text-dark);
        margin-bottom: 0.4rem;
    }

    .hero-card p {
        color: var(--text-base);
        margin: 0;
    }

    .hero-card small {
        color: var(--text-muted);
        margin-top: 0.6rem;
        display: block;
    }

    .result-card a {
        color: var(--lav-deep);
        font-weight: 600;
        text-decoration: none;
    }

    .rag-card strong {
        display: block;
        color: var(--text-dark);
        margin-bottom: 6px;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }

    /* ---------------------------- DATAFRAME ---------------------------- */
    .stDataFrame {
        background: white !important;
        border-radius: 14px !important;
        box-shadow: var(--shadow-soft);
        overflow: hidden;
        border: none;
    }

    .stDataFrame table {
        color: var(--text-dark) !important;
    }

    /* ---------------------------- HEADERS ---------------------------- */
    h1, h2, h3, h4, h5 {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }

    /* ---------------------------- ALERTS ---------------------------- */
    .stAlert {
        border-radius: 14px;
        background: var(--lav-light);
        border: none;
        box-shadow: var(--shadow-soft);
        color: var(--text-dark);
    }

    .eval-highlight {
        margin-top: 1rem;
        padding: 1rem 1.25rem;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--pink-soft), var(--lav-mid));
        color: var(--text-dark);
        font-weight: 600;
        box-shadow: var(--shadow-soft);
    }

    .eval-note {
        color: var(--text-light);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    </style>
    """, unsafe_allow_html=True)


def sync_best_model(best_model: Optional[str]):
    st.session_state["best_model"] = best_model


def trigger_frontend():
    st.session_state["frontend_trigger"] = True


def mark_model_override():
    st.session_state["backend_model_override"] = True


def perform_ranked_workflow(pipeline: SearchPipeline, args, source: str = "backend"):
    query_text = (
        st.session_state["backend_query"] if source == "backend" else st.session_state.get("frontend_query", "")
    )
    model_name = st.session_state["backend_model"]
    top_k = st.session_state["backend_top_k"]
    categories = st.session_state.get("backend_categories", [])
    candidate_pool = st.session_state["backend_candidate_pool"]
    start_date = (
        serialize_date_value(st.session_state.get("backend_start_date"))
        if st.session_state.get("backend_start_date_enabled")
        else ""
    )
    end_date = (
        serialize_date_value(st.session_state.get("backend_end_date"))
        if st.session_state.get("backend_end_date_enabled")
        else ""
    )
    results, ranked_ids, status, state = run_ranked_search(
        pipeline,
        query_text=query_text,
        model_name=model_name,
        top_k=top_k,
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        candidate_pool=candidate_pool,
        labeled_option=st.session_state.get("labeled_query_option"),
    )
    st.session_state["ranked_results"] = results
    st.session_state["ranked_status"] = status
    st.session_state["search_state"] = state
    st.session_state["live_eval_message"] = (
        f"Last search: `{query_text}` via **{model_name}**"
        if results
        else "Run a ranked search to populate Live Evaluation."
    )

    if state:
        rag_answer = run_rag_answer(pipeline, state, args.rag_model)
        st.session_state["rag_answer"] = rag_answer
        if source == "frontend":
            st.session_state["frontend_results"] = results
            st.session_state["frontend_rag"] = rag_answer
            st.session_state["backend_query"] = query_text
    return results


def render_search_engine_tab(pipeline: SearchPipeline, args):
    with st.container():
        st.markdown(
            f"""
            <div class="hero-card">
                <h2>Search Engine</h2>
                <p>Ask any news. Hit <strong>ENTER</strong> to search.</p>
                <small>Currently exploring <strong>{st.session_state['dataset_label']}</strong>.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "Search the news knowledge base",
            key="frontend_query",
            placeholder="Search the full news knowledge base...",
            label_visibility="collapsed",
            on_change=trigger_frontend,
        )

        if st.session_state.get("frontend_trigger") and st.session_state["frontend_query"].strip():
            perform_ranked_workflow(pipeline, args, source="frontend")
            st.session_state["frontend_trigger"] = False

        rag_answer = st.session_state.get("frontend_rag")
        if rag_answer:
            st.markdown(
                f"""
                <div class="rag-card">
                    <strong>RAG Insight</strong>
                    <p>{rag_answer}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        results = st.session_state.get("frontend_results", [])
        if results:
            st.subheader("Top results")
            for item in results:
                description = item.snippet or item.title or "Open article"
                url = item.link or "#"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <a href="{url}" target="_blank">{description}</a>
                        <p style="margin:6px 0 0 0;color:#475569;">{item.title or ''}</p>
                        <small style="color:#94a3b8;">Rank #{item.rank} • {item.category or 'Uncategorized'} • {item.date or 'n/a'}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("Type a query and press Enter to see intelligent results.")


def render_ranking_tab(pipeline: SearchPipeline):
    st.subheader("Ranking Workspace")
    col1, col2 = st.columns([2, 1])
    categories = pipeline.available_categories()
    choices = format_query_choices(pipeline)
    best_model = st.session_state.get("best_model")

    if best_model and not st.session_state.get("backend_model_override"):
        st.session_state["backend_model"] = best_model

    with col1:
        st.selectbox(
            "Ranking model",
            options=st.session_state["available_models"],
            key="backend_model",
            format_func=lambda value: format_model_label(value, best_model),
            on_change=mark_model_override,
        )
        st.text_area(
            "Query",
            key="backend_query",
            height=80,
            placeholder="Refine the search query here...",
        )
        st.slider("Top K", min_value=5, max_value=50, key="backend_top_k")

        if st.session_state["backend_model"] == "Embedding (semantic)":
            st.slider(
                "Semantic candidate pool",
                min_value=50,
                max_value=400,
                    step=10,
                key="backend_candidate_pool",
            )

    with col2:
        st.multiselect(
            "Category filter",
            options=categories,
            key="backend_categories",
            placeholder="Choose categories",
        )
        st.toggle(
            "Apply start date filter",
            key="backend_start_date_enabled",
            help="Toggle on to apply the selected start date.",
        )
        st.date_input(
            "Start date",
            key="backend_start_date",
            value=st.session_state.get("backend_start_date") or date.today(),
            format="YYYY-MM-DD",
        )
        st.toggle(
            "Apply end date filter",
            key="backend_end_date_enabled",
            help="Toggle on to apply the selected end date.",
        )
        st.date_input(
            "End date",
            key="backend_end_date",
            value=st.session_state.get("backend_end_date") or date.today(),
            format="YYYY-MM-DD",
        )
        st.selectbox(
            "Labeled query (optional)",
            options=[""] + choices if choices else [""],
            key="labeled_query_option",
        )

    backend_run = st.button("Run search ⚡", use_container_width=True, type="primary")
    if backend_run and st.session_state["backend_query"].strip():
        perform_ranked_workflow(pipeline, st.session_state["args"])

    st.info(st.session_state["ranked_status"])
    if st.session_state.get("ranked_results"):
        st.dataframe(results_to_dataframe(st.session_state["ranked_results"]), use_container_width=True, hide_index=True)

    rag_btn = st.button("Generate RAG answer", use_container_width=True)
    if rag_btn:
        st.session_state["rag_answer"] = run_rag_answer(
            pipeline,
            st.session_state["search_state"],
            st.session_state["args"].rag_model,
        )
    if st.session_state.get("rag_answer"):
        st.markdown(f"**RAG Answer**: {st.session_state['rag_answer']}")


def render_dataset_overview_tab(pipeline: SearchPipeline):
    st.subheader("Dataset overview")
    df, stats = load_dataset_overview(pipeline)
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='stat-box'><h3>{stats['Total Docs']}</h3><p>Documents</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-box'><h3>{stats['Categories']}</h3><p>Categories</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-box'><h3>{stats['Earliest']}</h3><p>Earliest article</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stat-box'><h3>{stats['Latest']}</h3><p>Latest article</p></div>", unsafe_allow_html=True)


def render_boolean_tab(pipeline: SearchPipeline):
    st.subheader("Boolean lab")
    bool_query = st.text_area("Boolean query (AND / OR / NOT)", key="bool_query")
    bool_categories = st.multiselect(
        "Category filter",
        options=pipeline.available_categories(),
        key="bool_categories",
        placeholder="Choose categories",
    )
    bool_limit = st.slider("Max results", min_value=10, max_value=200, value=50, step=10)
    bool_rerank = st.checkbox("Re-rank with BM25", value=True)
    run_bool = st.button("Run Boolean Search", use_container_width=True)
    if run_bool:
        df, status = run_boolean_search(
            pipeline,
            bool_query,
            bool_limit,
            bool_categories,
            serialize_date_value(st.session_state.get("backend_start_date"))
            if st.session_state.get("backend_start_date_enabled")
            else "",
            serialize_date_value(st.session_state.get("backend_end_date"))
            if st.session_state.get("backend_end_date_enabled")
            else "",
            bool_rerank,
        )
        st.success(status)
        st.dataframe(df, use_container_width=True)


def render_live_eval_tab(manager: PipelineManager):
    st.subheader("Live evaluation")
    st.write(st.session_state["live_eval_message"])
    if st.button("Evaluate last search", type="primary"):
        state = st.session_state.get("search_state")
        if not state or not state.get("doc_ids"):
            st.warning("Run a ranked search first.")
            return
        qid = state.get("query_id")
        if not qid:
            st.warning("Select a labeled query in Ranking Search to evaluate.")
            return
        pipeline = manager.get(st.session_state["dataset_label"])
        metrics = pipeline.evaluate_single_query(state["doc_ids"], qid)
        if not metrics:
            st.info("No judgments available for that query.")
            return
        eval_df = pd.DataFrame(
            [{"Metric": key, "Score": round(value, 4)} for key, value in metrics.items()]
        )
        st.dataframe(eval_df, hide_index=True, use_container_width=True)


def ensure_model_evaluation(
    pipeline: SearchPipeline,
    args,
    show_spinner: bool = False,
    force_refresh: bool = False,
):
    if not force_refresh:
        cached = st.session_state.get("model_eval_cache")
        if cached and cached.get("summary"):
            return cached["summary"], cached.get("best_model")

        persisted = load_persisted_eval_cache()
        if persisted:
            st.session_state["model_eval_cache"] = persisted
            return persisted.get("summary", []), persisted.get("best_model")

        return [], None

    eval_queries = load_eval_queries(args.eval_queries)
    context = st.spinner("Evaluating models on labeled queries...") if show_spinner else nullcontext()
    with context:
        summary, best_model = evaluate_models(
            pipeline,
            st.session_state["available_models"],
            eval_queries,
            top_k=EVAL_TOP_K,
            candidate_pool=EVAL_CANDIDATE_POOL,
        )

    payload = {
        "summary": summary,
        "best_model": best_model,
        "computed_at": datetime.utcnow().isoformat(),
    }
    st.session_state["model_eval_cache"] = payload
    save_persisted_eval_cache(payload)
    return summary, best_model


def render_evaluate_models_tab(pipeline: SearchPipeline, args):
    st.subheader("Evaluate models")
    force_refresh = st.button(
        "Recompute evaluation (slow)",
        help="Runs all ranking models across the 100 labeled queries and refreshes the cached snapshot.",
    )
    summary, best_model = ensure_model_evaluation(
        pipeline,
        args,
        show_spinner=True,
        force_refresh=force_refresh,
    )
    sync_best_model(best_model)

    if not summary:
        st.info(
            "No cached snapshot found. Run `python precompute_model_evals.py` (recommended) or click "
            "“Recompute evaluation” to generate the metrics.",
        )
        return

    st.write("Each model is automatically benchmarked on the 100 labeled queries.")
    df = pd.DataFrame(summary)
    st.dataframe(df.round(4), use_container_width=True, hide_index=True)
    st.markdown(
        "<p class='eval-note'>Cached snapshot — reused on startup until you choose to recompute.</p>",
        unsafe_allow_html=True,
    )
    if best_model:
        st.markdown(
            f"<div class='eval-highlight'>Best performing model: <strong>{best_model}</strong> "
            "— applied automatically to the Search Engine (unless overridden).</div>",
            unsafe_allow_html=True,
        )


def render_backend_tab(manager: PipelineManager, pipeline: SearchPipeline, args):
    backend_tabs = st.tabs(["Ranking Search", "Dataset Overview", "Boolean Lab", "Live Evaluation", "Evaluate Models"])
    with backend_tabs[0]:
        render_ranking_tab(pipeline)
    with backend_tabs[1]:
        render_dataset_overview_tab(pipeline)
    with backend_tabs[2]:
        render_boolean_tab(pipeline)
    with backend_tabs[3]:
        render_live_eval_tab(manager)
    with backend_tabs[4]:
        render_evaluate_models_tab(pipeline, args)


def main():
    args = parse_args()
    set_global_styles()
    manager = load_manager(args.index_base, args.qrels, args.models_dir)
    st.session_state["datasets"] = manager.list_datasets()
    st.session_state["args"] = args
    dataset_label = list(manager.list_datasets().keys())[0]
    st.session_state["dataset_label"] = dataset_label
    pipeline = manager.get(dataset_label)
    ensure_state_defaults(pipeline)

    if "model_eval_cache" in st.session_state and st.session_state["model_eval_cache"]:
        sync_best_model(st.session_state["model_eval_cache"].get("best_model"))
    else:
        persisted_eval = load_persisted_eval_cache()
        if persisted_eval:
            st.session_state["model_eval_cache"] = persisted_eval
            sync_best_model(persisted_eval.get("best_model"))
        else:
            sync_best_model(None)

    st.markdown('<div class="page-title">News IR Studio</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Search Engine", "Backend"])
    with tabs[0]:
        render_search_engine_tab(pipeline, args)
    with tabs[1]:
        render_backend_tab(manager, pipeline, args)


if __name__ == "__main__":
    main()

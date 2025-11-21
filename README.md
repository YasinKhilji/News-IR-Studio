# IR Project

End-to-end information retrieval pipeline for the news category dataset. The project now follows a `core/`-centric layout so that indexing, ranking, evaluation, and experiment artifacts live under one umbrella.

## Layout

- `data/`: raw corpus plus query/relevance templates.
- `core/build_index/`: CLI utilities to build the inverted index (`python -m core.build_index.build_index ...`).
- `core/IR_core/`: engine modules for preprocessing, indexing, storage utilities, and rankers (BM25 + TF-IDF).
- `core/IR_evaluation/`: metrics, experiment runners, and learning-to-rank training/inference utilities.
- `core/results/`
  - `built_index/`: default target for index artifacts (inverted index, stats, doc store, config).
  - `evaluation_results/`: default sink for TF-IDF/BM25 experiment outputs.
  - `models/` & `evaluation_results/ltr/`: created on demand for learning-to-rank training.
- `test_rankers.py`: quick sanity harness for both rankers using the latest built index.

## Typical Workflow

1. **Build the index**
   ```bash
   python -m core.build_index.build_index --data data/News_Category_Dataset_v3.json
   ```
   This writes artifacts to `core/results/built_index/` and stores the exact preprocessing config alongside them.

2. **Run baseline experiments**
   ```bash
   python -m core.IR_evaluation.run_experiments \
     --indexdir core/results/built_index \
     --queries data/queries_relevance_auto.json \
     --output core/results/evaluation_results
   ```
   Produces ranked lists, aggregate metrics, and plots for both TF-IDF and BM25.

3. **(Optional) Train the LTR model**
   ```bash
   python -m core.IR_evaluation.train_itr \
     --indexdir core/results/built_index \
     --queries data/queries_relevance_auto.json \
     --tfidf_results core/results/evaluation_results/tfidf_results.json \
     --bm25_results core/results/evaluation_results/bm25_results.json
   ```
   Saves the booster, scaler, and comparison reports under `core/results/models/` and `core/results/evaluation_results/ltr/`.

4. **Ad-hoc testing**
   Update the `QUERY` constant in `test_rankers.py` and run `python test_rankers.py` to inspect the top hits in the console.

## Data Prep for Ready-to-Use Core Functions

If you only need the searchable artifacts (for example, before wiring up a UI), step 1 is the only mandatory action. Running the build command above will populate `core/results/built_index/` with everything the online app needs:

- `inverted_index.pkl` – postings and term statistics.
- `doc_stats.json` / `term_stats.json` – collection summaries used by the rankers.
- `doc_store.jsonl` – lightweight metadata that the UI can display alongside results.
- `config_used.json` – the exact preprocessing switches so future queries are tokenized the same way.

Once those files exist, the core rankers (`BM25Ranker`, `TfIdfRanker`, future LTR models) can be instantiated immediately without any additional preprocessing.

## Optional Evaluation & Why It Exists

`python -m core.IR_evaluation.run_experiments ...` is optional. It simply replays the prepared queries/relevance labels so you can:

- sanity-check that the rankers are wired correctly,
- compare TF-IDF vs BM25 metrics (Precision/Recall/nDCG/MRR),
- generate CSV/text/plot artifacts for reports.

If you don’t need those diagnostics (e.g., when powering the UI), you can skip this step entirely. Nothing in the runtime path depends on the saved evaluation outputs.

## What's New (UI + Advanced Retrieval)

- 🔎 **Boolean query support** via the `BooleanQueryEngine` (`AND/OR/NOT`, parentheses, quoted literals) that runs on the same inverted index as the ranked models.
- 🧠 **Additional rankers**: Dirichlet query-likelihood LM, recency-aware Temporal BM25, lightweight semantic re-ranker (SentenceTransformers), plus the fully wired learning-to-rank XGBoost stack.
- 🗂️ **Dataset manager & temporal filters** so you can toggle between multiple index directories, restrict by category/date, and keep query-time preprocessing consistent with each snapshot.
- 📊 **Real-time evaluation** directly in the UI (per-query metrics, manual relevance overrides, and batch experiments).
- 🤖 **RAG answer generation**: grab the top-3 BM25 or LTR documents, feed them into a FLAN-T5 generator, and summarize answers inline.
- 🖥️ **Gradio multi-tab UI** (“News IR Studio”) to make all of the above usable without extra scripts.

## Gradio UI (News IR Studio)

Spin up the full experience, including RAG, Boolean search, analytics, and dataset management:

```bash
python app.py --index-base core/results --qrels data/queries_relevance_auto.json --listen
```

- `--listen` binds to `0.0.0.0` for remote demos (omit for localhost only).
- Use `--rag-model` to change the HuggingFace seq2seq model (defaults to `google/flan-t5-base`).

### Tabs & Features

| Tab | Highlights |
| --- | --- |
| **Ranked Search** | Choose dataset + ranker (BM25, TF-IDF, Language Model, Temporal BM25, LTR, Embedding), apply category/date filters, view live metrics (ground-truth or manual), and fire the “Generate Answer (RAG)” button. |
| **Boolean Lab** | Run structured queries with `AND/OR/NOT`, parentheses, optional re-ranking via BM25, and the same metadata filters. |
| **Live Evaluation** | Batch-evaluate any ranker against the labeled queries without leaving the UI; outputs Precision/Recall/nDCG/MRR averages. |
| **Datasets** | View/register additional index directories (multi-dataset support). Each dataset keeps its own preprocessing config and artifacts. |

### Boolean Query Syntax

- Operators: `AND`, `OR`, `NOT` (case-insensitive).
- Grouping: parentheses `(` `)`.
- Exact/phrase-like behavior: wrap literals in quotes (`"climate change"`). When exact phrase positions are unavailable, the engine falls back to AND-ing the constituent terms.
- Default operator between bare tokens is `AND`.

### RAG Button (“Generate Answer”)

1. Run a ranked search using BM25 or LTR (other models auto-fallback to BM25 for the RAG step).
2. Click **Generate Answer (RAG)** to summarize the top-3 supporting docs with FLAN-T5 (customizable via `--rag-model`).
3. Responses cite document numbers so you can trace them back in the table.

### Advanced Rankers On Tap

| Model name (UI) | Implementation |
| --------------- | -------------- |
| `BM25` | Classic Robertson/Sparck Jones with configurable `k1/b`. |
| `TF-IDF` | Log-weighted vector space with cosine normalization. |
| `Language Model` | Query-likelihood with Dirichlet smoothing. |
| `BM25 (Temporal)` | BM25 + exponential recency boost (half-life + alpha configurable in code). |
| `Learning-to-Rank` | XGBoost model trained via `core/IR_evaluation/train_itr.py`. |
| `Embedding (semantic)` | SentenceTransformers MiniLM re-ranker over BM25’s candidate pool. |

### Real-Time Evaluation & Manual Judgments

- Select any pre-labeled query (dropdown uses `data/queries_relevance_auto.json`) to see per-query Precision/Recall/nDCG/MRR immediately after a search.
- Supply your own relevant doc IDs (comma-separated) to overlay manual Precision/Recall/MRR in the metrics JSON.
- Use the **Live Evaluation** tab to compute dataset-level averages for any ranker without invoking the CLI experiment runner.

### Dataset & Temporal Support

- The dataset dropdown is backed by a registry (`core/results/**/inverted_index.pkl`). Use the “Datasets” tab to point the UI at additional index folders (e.g., alternate preprocessing snapshots or future corpora).
- Date filters and the temporal ranker rely on the stored metadata; DocumentStore automatically materializes `doc_texts.jsonl` if it’s missing by replaying the original corpus (`config_used.json` keeps the source path).

## `test_rankers.py`

This mini script is just a console smoke test. Point it at `core/results/built_index`, tweak the `QUERY` string, and it will:

1. preprocess the query using the stored config,
2. run both rankers,
3. print the top-k titles/dates from the doc store.

It’s helpful when you want to quickly verify that the newly built index behaves as expected without running the full evaluation harness.

## Requirements

Install dependencies with:

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Ensure the NLTK resources mentioned in `core/IR_core/preprocessing.py` are downloaded before running the pipeline. The UI/RAG features also depend on the heavier packages listed near the bottom of `requirements.txt` (Gradio, transformers, sentence-transformers, torch).


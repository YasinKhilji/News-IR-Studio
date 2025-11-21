from collections import defaultdict
from typing import Dict, Any, List, Tuple
import math, os

from .preprocessing import preprocess_text
from .io_utils import save_pickle, save_json, write_jsonl

def build_inverted_index(docs: Dict[int, Dict[str, Any]],
                         config) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    docs: {doc_id: {"text": "...", "title": "...", "date": "...", "category": "...", "link": "..."}}
    returns: inverted index: term -> {doc_id: {"tf": int, "pos": [int,...]?}}
    """
    index = defaultdict(dict)  # term -> doc_id -> payload
    doc_len = {}  # doc_id -> length (number of tokens)

    for doc_id, d in docs.items():
        tokens = preprocess_text(
            d["text"],
            lowercase=config.lowercase,
            remove_punct=config.remove_punct,
            remove_digits=config.remove_digits,
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming and not config.use_lemmatize,
            use_lemmatize=config.use_lemmatize,
            min_token_len=config.min_token_len,
            max_token_len=config.max_token_len,
        )

        doc_len[doc_id] = len(tokens)

        if not tokens:
            continue

        positions_map = defaultdict(list)
        for pos, tok in enumerate(tokens):
            positions_map[tok].append(pos)

        for tok, pos_list in positions_map.items():
            payload = {"tf": len(pos_list)}
            if config.keep_positions:
                payload["pos"] = pos_list
            index[tok][doc_id] = payload

    # compute stats
    N = len(docs)
    df = {t: len(postings) for t, postings in index.items()}
    # idf for BM25 (Robertson/Sparck Jones idf)
    idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9) for t, df_t in df.items()}  # +1e-9 for numeric safety
    avgdl = sum(doc_len.values()) / N if N else 0.0

    stats = {
        "N": N,
        "avgdl": avgdl,
        "doc_len": doc_len,
    }
    term_stats = {
        "df": df,
        "idf": idf,
    }
    return index, stats, term_stats

def save_artifacts(index, stats, term_stats, docs, outdir: str, config):
    os.makedirs(outdir, exist_ok=True)
    # index
    save_pickle(index, os.path.join(outdir, "inverted_index.pkl"))
    # stats
    save_json(stats, os.path.join(outdir, "doc_stats.json"))
    save_json(term_stats, os.path.join(outdir, "term_stats.json"))
    # doc store (minimal, one json per line)
    rows = ({
        "doc_id": int(doc_id),
        "title": d.get("title"),
        "date": d.get("date"),
        "category": d.get("category"),
        "link": d.get("link"),
    } for doc_id, d in docs.items())
    write_jsonl(rows, os.path.join(outdir, "doc_store.jsonl"))
    config.to_json(os.path.join(outdir, "config_used.json"))

import argparse
import os
from typing import Dict, Any

from core.IR_core.config import Config
from core.IR_core.io_utils import iter_jsonl
from core.IR_core.index import build_inverted_index, save_artifacts

def load_news_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    docs = {}
    for i, row in enumerate(iter_jsonl(path)):
        text = " ".join(filter(None, [row.get("headline",""), row.get("short_description","")]))
        docs[i] = {
            "text": text,
            "title": row.get("headline"),
            "date": row.get("date"),
            "category": row.get("category"),
            "link": row.get("link"),
        }
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to news JSONL (one JSON per line). Supports .gz")
    ap.add_argument("--outdir", default="./core/results/built_index", help="Where to write artifacts")
    ap.add_argument("--positions", action="store_true", help="Store positions in postings (more space)")
    ap.add_argument("--lemmatize", action="store_true", help="Use lemmatization instead of stemming")
    args = ap.parse_args()

    cfg = Config(input_path=args.data, output_dir=args.outdir)
    if args.positions: cfg.keep_positions = True
    if args.lemmatize:
        cfg.use_lemmatize = True
        cfg.use_stemming = False

    docs = load_news_jsonl(cfg.input_path)
    index, stats, term_stats = build_inverted_index(docs, cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_artifacts(index, stats, term_stats, docs, cfg.output_dir, cfg)
    cfg.to_json(os.path.join(cfg.output_dir, "config_used.json"))
    print(f"Built index for {stats['N']} docs. avgdl={stats['avgdl']:.2f}. Out: {cfg.output_dir}")

if __name__ == "__main__":
    main()

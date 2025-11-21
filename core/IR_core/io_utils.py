from typing import Dict, Any, Iterable, Tuple, Generator, Optional
import json, os, gzip, pickle

def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def iter_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def write_jsonl(rows, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

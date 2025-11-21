from dataclasses import dataclass, asdict
from typing import Optional
import json, os

@dataclass
class Config:
    # Preprocessing
    lowercase: bool = True
    remove_punct: bool = True
    remove_digits: bool = False
    remove_stopwords: bool = True
    use_stemming: bool = True      # Porter stemmer
    use_lemmatize: bool = False    # WordNet lemmatizer (heavier). If True, stemming is ignored.
    keep_positions: bool = False   # store positions for phrase/proximity queries
    
    # Tokenization
    min_token_len: int = 2
    max_token_len: int = 40

    # Files/IO
    input_path: Optional[str] = None
    output_dir: str = "./core/results/built_index"

    # BM25 defaults (for smoke tests)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Misc
    random_seed: int = 1337

    def to_json(self, out_path: str):
        directory = os.path.dirname(out_path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import Config
from .io_utils import iter_jsonl, write_jsonl


@dataclass
class DocumentRecord:
    """Container for lightweight document metadata."""

    doc_id: int
    title: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    link: Optional[str] = None
    text: Optional[str] = None


class DocumentStore:
    """
    Utility around the lightweight doc_store plus optional full-text payloads.
    Handles:
      • Loading metadata / titles / categories
      • Materializing doc_texts.jsonl when only the raw dataset exists
      • Lazy access to per-doc text (needed for UI snippets + RAG)
      • Date/category filtering helpers for temporal ranking
    """

    def __init__(
        self,
        index_dir: str,
        config: Optional[Config] = None,
        preload_text: bool = False,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.doc_store_path = self.index_dir / "doc_store.jsonl"
        self.doc_texts_path = self.index_dir / "doc_texts.jsonl"
        self.config = config or self._load_config()

        if not self.doc_store_path.exists():
            raise FileNotFoundError(
                f"Expected doc_store.jsonl at {self.doc_store_path} – build the index first."
            )

        self._metadata: Dict[int, DocumentRecord] = self._load_metadata()
        self._text_bank: Dict[int, str] = {}
        self._dates_cache: Dict[int, Optional[datetime]] = {}
        self._max_date: Optional[datetime] = self._compute_max_date()

        if preload_text:
            self.ensure_text_cache(force_memory_load=True)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Optional[Config]:
        cfg_path = self.index_dir / "config_used.json"
        if cfg_path.exists():
            return Config.from_json(str(cfg_path))
        return None

    def _load_metadata(self) -> Dict[int, DocumentRecord]:
        metadata: Dict[int, DocumentRecord] = {}
        with open(self.doc_store_path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                doc_id = int(payload["doc_id"])
                metadata[doc_id] = DocumentRecord(
                    doc_id=doc_id,
                    title=payload.get("title"),
                    date=payload.get("date"),
                    category=payload.get("category"),
                    link=payload.get("link"),
                )
        return metadata

    def _load_text_snapshot(self) -> Dict[int, str]:
        if not self.doc_texts_path.exists():
            return {}
        bank: Dict[int, str] = {}
        with open(self.doc_texts_path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                bank[int(payload["doc_id"])] = payload.get("text", "")
        return bank

    def ensure_text_cache(self, force_memory_load: bool = False) -> None:
        """
        Make sure doc_texts.jsonl exists (build it if we still have access to
        the raw dataset) and optionally load the whole thing into memory.
        """
        if not self.doc_texts_path.exists():
            self._materialize_doc_texts_from_source()

        if force_memory_load and not self._text_bank:
            self._text_bank = self._load_text_snapshot()

    def _materialize_doc_texts_from_source(self) -> None:
        """Generate doc_texts.jsonl by replaying the original dataset."""
        if not self.config or not self.config.input_path:
            return

        source_path = Path(self.config.input_path)
        if not source_path.exists():
            return

        def iter_rows():
            for idx, row in enumerate(iter_jsonl(str(source_path))):
                headline = (row.get("headline") or "").strip()
                short_desc = (row.get("short_description") or "").strip()
                text = " ".join(filter(None, [headline, short_desc])).strip()
                yield {"doc_id": idx, "text": text}

        write_jsonl(iter_rows(), str(self.doc_texts_path))

    def _compute_max_date(self) -> Optional[datetime]:
        max_dt: Optional[datetime] = None
        for doc_id in self._metadata:
            dt = self.get_datetime(doc_id)
            if dt and (max_dt is None or dt > max_dt):
                max_dt = dt
        return max_dt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_docs(self) -> int:
        return len(self._metadata)

    @property
    def max_date(self) -> Optional[datetime]:
        return self._max_date

    def get(self, doc_id: int, include_text: bool = True) -> Optional[Dict[str, Optional[str]]]:
        record = self._metadata.get(doc_id)
        if not record:
            return None
        data = {
            "doc_id": record.doc_id,
            "title": record.title,
            "date": record.date,
            "category": record.category,
            "link": record.link,
        }
        if include_text:
            data["text"] = self.get_text(doc_id)
        return data

    def get_many(self, doc_ids: Sequence[int], include_text: bool = True) -> List[Dict[str, Optional[str]]]:
        return [self.get(doc_id, include_text=include_text) for doc_id in doc_ids if self.get(doc_id)]

    def get_text(self, doc_id: int) -> Optional[str]:
        if doc_id in self._text_bank:
            return self._text_bank[doc_id]

        if not self._text_bank:
            # Lazy load when first requested.
            self.ensure_text_cache(force_memory_load=True)

        return self._text_bank.get(doc_id)

    def get_datetime(self, doc_id: int) -> Optional[datetime]:
        if doc_id in self._dates_cache:
            return self._dates_cache[doc_id]

        record = self._metadata.get(doc_id)
        if not record or not record.date:
            self._dates_cache[doc_id] = None
            return None

        dt = self._parse_date(record.date)
        self._dates_cache[doc_id] = dt
        return dt

    def filter_doc_ids(
        self,
        categories: Optional[Iterable[str]] = None,
        date_range: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None,
    ) -> Set[int]:
        allowed: Set[int] = set(self._metadata.keys())

        if categories:
            category_set = {c.lower() for c in categories}
            allowed = {
                doc_id
                for doc_id, record in self._metadata.items()
                if record.category and record.category.lower() in category_set
            }

        if date_range:
            start, end = date_range
            result: Set[int] = set()
            for doc_id in allowed:
                dt = self.get_datetime(doc_id)
                if not dt:
                    continue
                if start and dt < start:
                    continue
                if end and dt > end:
                    continue
                result.add(doc_id)
            allowed = result

        return allowed

    def build_snippet(self, doc_id: int, max_chars: int = 280) -> str:
        text = self.get_text(doc_id) or ""
        if not text:
            title = self._metadata.get(doc_id).title if doc_id in self._metadata else ""
            return (title or "")[:max_chars]
        snippet = text.strip().replace("\n", " ")
        if len(snippet) <= max_chars:
            return snippet
        return snippet[: max_chars - 3].rstrip() + "..."

    def iter_metadata(self):
        for record in self._metadata.values():
            yield record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None



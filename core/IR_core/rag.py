from __future__ import annotations

from typing import Dict, List, Optional


class RAGAnswerer:
    """
    Tiny wrapper around a seq2seq LLM (default: FLAN-T5) to generate
    summaries/answers from the top-k retrieved documents.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_context_chars: int = 1200,
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for RAGAnswerer. "
                "Install it via `pip install transformers sentencepiece`."
            ) from exc

        task = "text2text-generation"
        self.generator = pipeline(task, model=model_name, device=device)
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, question: str, documents: List[Dict[str, Optional[str]]]) -> str:
        if not question:
            return "Please provide a question."
        if not documents:
            return "No supporting documents were provided."

        prompt = self._build_prompt(question, documents)
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        return outputs[0]["generated_text"].strip()

    def _build_prompt(self, question: str, documents: List[Dict[str, Optional[str]]]) -> str:
        blocks = []
        for idx, doc in enumerate(documents, start=1):
            title = (doc.get("title") or "").strip()
            snippet = (doc.get("snippet") or doc.get("text") or "").strip()
            snippet = snippet[: self.max_context_chars]
            blocks.append(f"Document {idx} - {title}\n{snippet}")

        context = "\n\n".join(blocks)
        prompt = (
            "You are a news intelligence analyst. Study the brief summaries below and craft a helpful reply "
            "that directly addresses the user's request.\n\n"
            f"{context}\n\n"
            f"Question: {question.strip()}\n"
            "Write 2-3 natural sentences that synthesize the most relevant facts. Reference supporting material "
            "in-line using (Doc #) rather than listing document titles, and avoid generic statements."
        )
        return prompt


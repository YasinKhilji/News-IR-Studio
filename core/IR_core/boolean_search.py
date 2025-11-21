from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Sequence, Set


class BooleanQueryEngine:
    """
    Minimal Boolean retrieval module that works on top of the inverted index.
    Supports:
      • AND / OR / NOT operators (case-insensitive)
      • Parentheses for grouping
      • Default implicit operator (configurable, defaults to AND)
      • Phrase handling via quoted literals (falls back to AND-ing the terms)
    """

    OP_PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}

    def __init__(
        self,
        index: Dict[str, Dict[int, Dict[str, int]]],
        preprocess_fn: Callable[[str], Sequence[str]],
        universe_docs: Iterable[int],
    ) -> None:
        self.index = index
        self.preprocess_fn = preprocess_fn
        self.universe: Set[int] = set(universe_docs)

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(self, query: str, default_operator: str = "AND") -> List[int]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        tokens = self._inject_default_ops(tokens, default_operator.upper())
        rpn = self._to_rpn(tokens)
        result_set = self._evaluate_rpn(rpn)
        return sorted(result_set)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _tokenize(self, query: str) -> List[str]:
        pattern = r'(".*?"|\(|\)|\bAND\b|\bOR\b|\bNOT\b)'
        raw_tokens = re.split(pattern, query, flags=re.IGNORECASE)
        cleaned = [tok.strip() for tok in raw_tokens if tok and tok.strip()]
        normalized: List[str] = []
        for tok in cleaned:
            upper = tok.upper()
            if upper in self.OP_PRECEDENCE or tok in ("(", ")"):
                normalized.append(upper)
            else:
                normalized.append(tok)
        return normalized

    def _inject_default_ops(self, tokens: List[str], default_operator: str) -> List[str]:
        if default_operator not in {"AND", "OR"}:
            raise ValueError("default_operator must be AND or OR")
        output: List[str] = []
        prev = None
        for tok in tokens:
            if self._is_literal(tok) and prev and self._is_literal(prev):
                output.append(default_operator)
            elif tok == "(" and prev and self._is_literal(prev):
                output.append(default_operator)
            elif self._is_literal(tok) and prev == ")":
                output.append(default_operator)
            output.append(tok)
            prev = tok
        return output

    def _to_rpn(self, tokens: List[str]) -> List[str]:
        output: List[str] = []
        stack: List[str] = []

        for tok in tokens:
            if self._is_literal(tok):
                output.append(tok)
            elif tok in self.OP_PRECEDENCE:
                while (
                    stack
                    and stack[-1] in self.OP_PRECEDENCE
                    and self.OP_PRECEDENCE[stack[-1]] >= self.OP_PRECEDENCE[tok]
                ):
                    output.append(stack.pop())
                stack.append(tok)
            elif tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if stack and stack[-1] == "(":
                    stack.pop()

        while stack:
            output.append(stack.pop())
        return output

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_rpn(self, rpn_tokens: List[str]) -> Set[int]:
        stack: List[Set[int]] = []
        for tok in rpn_tokens:
            if tok in self.OP_PRECEDENCE:
                if tok == "NOT":
                    operand = stack.pop() if stack else set()
                    stack.append(self.universe - operand)
                else:
                    right = stack.pop() if stack else set()
                    left = stack.pop() if stack else set()
                    if tok == "AND":
                        stack.append(left & right)
                    else:
                        stack.append(left | right)
            else:
                stack.append(self._resolve_literal(tok))
        return stack[-1] if stack else set()

    def _resolve_literal(self, literal: str) -> Set[int]:
        literal = literal.strip('"')
        processed = list(self.preprocess_fn(literal))
        if not processed:
            return set()

        postings_sets = [set(self.index.get(term, {}).keys()) for term in processed]
        if not postings_sets:
            return set()

        result = postings_sets[0]
        for s in postings_sets[1:]:
            result &= s
        return result

    @staticmethod
    def _is_literal(token: str) -> bool:
        return token not in {"AND", "OR", "NOT", "(", ")"}


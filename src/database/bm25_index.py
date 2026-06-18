"""
BM25 Index — SE365 Pharma-RAG
Wraps rank_bm25.BM25Okapi to index Qdrant payloads by chunk text
and perform keyword-based retrieval as a complement to dense search.

The index is built lazily from Qdrant scroll results and cached in memory.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and normalize Unicode (NFC) for Vietnamese text."""
    return unicodedata.normalize("NFC", text.lower())


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer for Vietnamese.
    Keeps alphanumeric + Vietnamese diacritics; drops pure punctuation tokens.
    """
    text = _normalize(text)
    tokens = re.split(r"[\s,;.!?()\[\]{}/\\\"']+", text)
    return [t for t in tokens if t]


# ──────────────────────────────────────────────────────────────────────
# BM25 Index
# ──────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index built from a list of document dicts.

    Each document must have at least:
        - 'chunk_text': str  (used for BM25 scoring)
        - 'id': str          (Qdrant point ID — used for result merging)

    All other payload fields are kept and returned in search results.
    """

    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------

    def build(self, docs: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of document payload dicts.
        Each dict must contain 'chunk_text'.
        """
        self._docs = docs
        tokenized_corpus = [_tokenize(d.get("chunk_text", "")) for d in docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("[BM25] Index built with %d documents.", len(docs))

    def is_built(self) -> bool:
        return self._bm25 is not None

    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Run BM25 retrieval for a query string.

        Returns a list of dicts:
            {'id': ..., 'score': float (BM25 score), 'payload': {...}}
        Sorted descending by BM25 score, truncated to top_k.
        """
        if not self._bm25:
            raise RuntimeError("BM25 index has not been built yet. Call build() first.")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair docs with scores, sort desc
        scored = sorted(
            zip(self._docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for doc, score in scored:
            if score <= 0:
                continue   # BM25 returned 0 → no keyword overlap, skip
            results.append({
                "id": doc.get("id") or doc.get("registration_no", ""),
                "score": float(score),
                "payload": doc,
            })
        return results

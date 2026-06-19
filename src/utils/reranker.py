"""
reranker.py
-----------
Cross-Encoder Reranker module for the Pharma-RAG pipeline.

Based on PROPOSAL.md Section 5.2C: "Reranking — Cross-encoder reranker".
Model: BAAI/bge-reranker-base (multilingual, strong pharmaceutical domain baseline)

Architecture:
  - Lazy-loads the cross-encoder model on first use to save startup memory.
  - Takes a query + list of retrieved chunks (from Qdrant hybrid search).
  - Re-scores each (query, chunk_text) pair using the cross-encoder.
  - Returns top_k chunks sorted by reranker score (descending).

Role in the ablation study (PROPOSAL Section 6.4):
  - Toggle ON : Full System (reranker filters noisy Top-20 → clean Top-5)
  - Toggle OFF: Baseline without reranker (Top-K sent directly to LLM)

Usage:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, chunks, top_k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CrossEncoderReranker")


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder model for reranking.

    Supported models (from PROPOSAL.md and sentence-transformers):
      - BAAI/bge-reranker-base  (default, multilingual, ~250MB)
      - BAAI/bge-reranker-large (higher accuracy, ~500MB)
      - cross-encoder/ms-marco-MiniLM-L-6-v2  (English only, faster)

    The reranker re-scores every (query, passage) pair independently,
    unlike bi-encoders which score query and passage separately and use
    cosine similarity. Cross-encoders are more accurate but slower.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model = None  # Lazy-loaded on first rerank() call

    @property
    def model(self):
        """Lazy-load the CrossEncoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("[Reranker] Loading cross-encoder model: %s ...", self.model_name)
                print(f"[Reranker] Loading cross-encoder model: {self.model_name}...")
                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info("[Reranker] Model loaded successfully.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for the reranker. "
                    "Install it with: pip install sentence-transformers"
                )
            except Exception as exc:
                logger.error("[Reranker] Failed to load model '%s': %s", self.model_name, exc)
                raise
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-score retrieved chunks using the cross-encoder and return top_k results.

        Args:
            query:   The original user query string.
            chunks:  List of dicts from Qdrant search, each with 'id', 'score', 'payload'.
                     The payload must contain a 'chunk_text' key.
            top_k:   Number of top results to return after reranking.

        Returns:
            List of chunk dicts sorted by reranker score (descending), length <= top_k.
            Each dict has an extra key 'reranker_score' with the cross-encoder's raw score.
        """
        if not chunks:
            return chunks

        # Extract texts from chunk payloads for scoring
        pairs = []
        for chunk in chunks:
            text = chunk.get("payload", {}).get("chunk_text", "")
            pairs.append([query, text])

        try:
            scores = self.model.predict(pairs)
        except Exception as exc:
            logger.error("[Reranker] Scoring failed, returning original order: %s", exc)
            return chunks[:top_k]

        # Attach reranker scores to each chunk
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            enriched = dict(chunk)
            enriched["reranker_score"] = float(score)
            scored_chunks.append(enriched)

        # Sort by reranker score (descending) and return top_k
        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        top_results = scored_chunks[:top_k]

        logger.info(
            "[Reranker] Re-scored %d chunks → top %d. Best score: %.4f, Worst kept: %.4f",
            len(chunks),
            len(top_results),
            top_results[0]["reranker_score"] if top_results else 0,
            top_results[-1]["reranker_score"] if top_results else 0,
        )

        return top_results


# ─── Module-level lazy singleton ────────────────────────────────────────────────

_global_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> CrossEncoderReranker:
    """
    Return the global singleton CrossEncoderReranker.
    Creates it on first call with the given model_name.
    Subsequent calls return the cached instance regardless of model_name argument.
    """
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = CrossEncoderReranker(model_name=model_name)
    return _global_reranker

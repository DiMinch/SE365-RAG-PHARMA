"""
test_reranker.py
----------------
Unit tests for the CrossEncoderReranker and its integration into PharmaQdrantClient.

Tests cover:
  1. Reranker correctly re-orders chunks by cross-encoder score.
  2. Reranker handles empty input gracefully.
  3. Reranker handles model failure with fallback.
  4. PharmaQdrantClient.search() returns top_k results with use_reranker=False (baseline).
  5. PharmaQdrantClient.search() integrates reranker when use_reranker=True.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.utils.reranker import CrossEncoderReranker
from src.database.qdrant_client import PharmaQdrantClient


def _make_chunk(drug_name: str, section: str, text: str, score: float = 0.8) -> dict:
    """Helper to create a fake chunk dict matching the Qdrant payload structure."""
    return {
        "id": f"{drug_name}_{section}",
        "score": score,
        "payload": {
            "drug_name": drug_name,
            "drug_type": "WESTERN_MEDICINE",
            "registration_no": "VN-00000-22",
            "section_name": section,
            "chunk_text": text,
        }
    }


# ─── CrossEncoderReranker unit tests ──────────────────────────────────────────

class TestCrossEncoderReranker:

    def test_reranker_reorders_by_score(self):
        """Reranker should return chunks sorted by cross-encoder score descending."""
        reranker = CrossEncoderReranker()

        # Mock the CrossEncoder model
        mock_model = MagicMock()
        # Reverse-order scores: chunk at index 2 should win
        mock_model.predict.return_value = [0.1, 0.3, 0.9]
        reranker._model = mock_model

        chunks = [
            _make_chunk("DrugA", "indication", "Unrelated text"),
            _make_chunk("DrugB", "dosage", "Somewhat relevant"),
            _make_chunk("DrugC", "interaction", "Highly relevant interaction info"),
        ]

        result = reranker.rerank("drug interactions", chunks, top_k=2)

        assert len(result) == 2
        assert result[0]["payload"]["drug_name"] == "DrugC"
        assert result[0]["reranker_score"] == pytest.approx(0.9)
        assert result[1]["payload"]["drug_name"] == "DrugB"
        assert result[1]["reranker_score"] == pytest.approx(0.3)

    def test_reranker_empty_input(self):
        """Reranker should return empty list when given empty input."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("any query", [], top_k=5)
        assert result == []

    def test_reranker_model_failure_returns_fallback(self):
        """If the cross-encoder model raises an exception, should fall back to original order."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model exploded")
        reranker._model = mock_model

        chunks = [
            _make_chunk("DrugA", "indication", "Text A"),
            _make_chunk("DrugB", "dosage", "Text B"),
        ]

        result = reranker.rerank("query", chunks, top_k=2)

        # Should return same chunks in original order (fallback)
        assert len(result) == 2
        assert result[0]["payload"]["drug_name"] == "DrugA"
        assert result[1]["payload"]["drug_name"] == "DrugB"

    def test_reranker_top_k_truncation(self):
        """Reranker should return at most top_k results."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        reranker._model = mock_model

        chunks = [_make_chunk(f"Drug{i}", "indication", f"Text {i}") for i in range(5)]

        result = reranker.rerank("query", chunks, top_k=3)
        assert len(result) == 3

    def test_reranker_score_attached(self):
        """Each returned chunk should have a 'reranker_score' key."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]
        reranker._model = mock_model

        chunks = [_make_chunk("DrugA", "indication", "Some text")]
        result = reranker.rerank("query", chunks, top_k=1)

        assert "reranker_score" in result[0]
        assert result[0]["reranker_score"] == pytest.approx(0.75)


# ─── PharmaQdrantClient.search() ablation toggle tests ────────────────────────

class TestSearchWithReranker:

    @patch('src.database.qdrant_client.QdrantClient')
    @patch('src.database.qdrant_client.SentenceTransformer')
    def test_search_use_reranker_false_skips_reranking(self, mock_transformer, mock_qdrant):
        """With use_reranker=False, the reranker should never be called."""
        # Setup mocks
        mock_model = MagicMock()
        # encode() must return something with .tolist() — use a MagicMock chain
        encode_result = MagicMock()
        encode_result.tolist.return_value = [0.1] * 384
        mock_model.encode.return_value = encode_result
        mock_transformer.return_value = mock_model

        mock_qdrant_instance = MagicMock()
        fake_point = MagicMock()
        fake_point.id = "abc123"
        fake_point.score = 0.9
        fake_point.payload = {"drug_name": "DrugA", "registration_no": "VN-00000-22", "section_name": "indication", "chunk_text": "Test text"}
        mock_qdrant_instance.query_points.return_value.points = [fake_point]
        mock_qdrant.return_value = mock_qdrant_instance

        client = PharmaQdrantClient()
        client._model = mock_model

        results = client.search("test query", top_k=1, use_reranker=False, retrieval_mode="dense")

        # Reranker should not have been instantiated
        assert client._reranker is None
        assert len(results) == 1
        assert results[0]["payload"]["drug_name"] == "DrugA"
        # No reranker_score key when reranker is off
        assert "reranker_score" not in results[0]

    @patch('src.database.qdrant_client.QdrantClient')
    @patch('src.database.qdrant_client.SentenceTransformer')
    def test_search_use_reranker_true_applies_reranking(self, mock_transformer, mock_qdrant):
        """With use_reranker=True, the reranker should be invoked and scores attached."""
        # Setup mocks
        mock_model = MagicMock()
        encode_result = MagicMock()
        encode_result.tolist.return_value = [0.1] * 384
        mock_model.encode.return_value = encode_result
        mock_transformer.return_value = mock_model

        mock_qdrant_instance = MagicMock()
        fake_point = MagicMock()
        fake_point.id = "abc123"
        fake_point.score = 0.9
        fake_point.payload = {"drug_name": "DrugA", "registration_no": "VN-00000-22", "section_name": "indication", "chunk_text": "Test text"}
        mock_qdrant_instance.query_points.return_value.points = [fake_point]
        mock_qdrant.return_value = mock_qdrant_instance

        client = PharmaQdrantClient()
        client._model = mock_model

        # Mock reranker to inject directly
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [{
            "id": "abc123",
            "score": 0.9,
            "reranker_score": 0.95,
            "payload": {"drug_name": "DrugA", "registration_no": "VN-00000-22", "section_name": "indication", "chunk_text": "Test text"}
        }]
        client._reranker = mock_reranker

        results = client.search("test query", top_k=1, use_reranker=True, retrieval_mode="dense")

        assert mock_reranker.rerank.called
        assert len(results) == 1
        assert results[0]["reranker_score"] == pytest.approx(0.95)

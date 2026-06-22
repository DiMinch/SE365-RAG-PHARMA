"""
Embedding Model Comparison Script (PROPOSAL Section 5.2B)

Compares multiple embedding models for Vietnamese pharmaceutical domain
by creating separate Qdrant collections, indexing the same corpus, and
measuring retrieval quality with Recall@k, MRR, nDCG@k.

Usage:
    # Index all models (takes time — downloads models + encodes corpus):
    python -m src.evaluation.embedding_comparison index

    # Run evaluation (requires benchmark JSON):
    python -m src.evaluation.embedding_comparison evaluate --benchmark data/benchmark/pharma_qa.json

    # Run both:
    python -m src.evaluation.embedding_comparison all --benchmark data/benchmark/pharma_qa.json

    # Quick smoke-test with built-in sample queries (no benchmark needed):
    python -m src.evaluation.embedding_comparison smoke-test
"""

import os
import sys
import json
import time
import hashlib
import uuid
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

from src.utils.config import get_base_config, get_qdrant_config
from src.models.drug import Drug

logger = logging.getLogger("EmbeddingComparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")


# ─── Models to compare (PROPOSAL Section 5.2B) ──────────────────────────────
# Each model produces a separate Qdrant collection: pharma_corpus_{key}
EMBEDDING_MODELS = {
    "minilm": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Baseline — lightweight multilingual (current production model)",
    },
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "dimension": 1024,
        "description": "Strong multilingual embedding, state-of-the-art on MTEB",
    },
    "e5_large": {
        "model_name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "description": "Multilingual E5 — instruction-tuned, strong on retrieval tasks",
    },
}

# Built-in smoke-test queries for quick validation without a full benchmark
SMOKE_TEST_QUERIES = [
    {"question": "Liều Amoxicillin cho trẻ em 10kg", "expected_section": "dosage"},
    {"question": "Tương tác thuốc Azithromycin với Fluconazole", "expected_section": "interactions"},
    {"question": "Chống chỉ định của Paracetamol", "expected_section": "contraindication"},
    {"question": "Tác dụng phụ khi dùng Metformin", "expected_section": "side_effects"},
    {"question": "Augmentin có dùng cho phụ nữ mang thai không", "expected_section": "warnings"},
]


class EmbeddingComparator:
    """
    Manages multiple Qdrant collections — one per embedding model —
    and provides methods to index data and evaluate retrieval quality.
    """

    def __init__(self):
        self.qdrant_cfg = get_qdrant_config()
        self.base_cfg = get_base_config()
        conn = self.qdrant_cfg["connection"]

        # Connect to Qdrant
        if "url" in conn and "api_key" in conn:
            self.client = QdrantClient(
                url=conn["url"],
                api_key=conn["api_key"],
                timeout=conn.get("timeout", 60.0),
            )
        else:
            self.client = QdrantClient(
                host=conn.get("host", "localhost"),
                port=conn.get("port", 6333),
                timeout=conn.get("timeout", 30.0),
            )

        self.base_collection = self.qdrant_cfg["collection"]["name"]
        self._models: Dict[str, SentenceTransformer] = {}

    def _collection_name(self, model_key: str) -> str:
        """Generate collection name for a specific embedding model."""
        return f"{self.base_collection}_{model_key}"

    def _load_model(self, model_key: str) -> SentenceTransformer:
        """Lazy-load a SentenceTransformer model."""
        if model_key not in self._models:
            cfg = EMBEDDING_MODELS[model_key]
            logger.info("Loading model: %s (%s)", model_key, cfg["model_name"])

            # E5 models need a special query prefix
            model = SentenceTransformer(cfg["model_name"])
            self._models[model_key] = model
        return self._models[model_key]

    def _generate_id(self, reg_no: str, section: str, chunk_idx: int) -> str:
        """Deterministic UUID matching PharmaQdrantClient logic."""
        hash_input = f"{reg_no}_{section}_{chunk_idx}".encode("utf-8")
        return str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    # ─── Indexing ────────────────────────────────────────────────────────

    def load_drug_files(self, data_dir: str = "data/raw") -> List[Dict[str, Any]]:
        """Load all drug JSON files from data directory."""
        drugs = []
        data_path = Path(data_dir)

        if not data_path.exists():
            # Try alternative paths
            for alt in ["data/Thuốc", "data"]:
                alt_path = Path(alt)
                if alt_path.exists():
                    data_path = alt_path
                    break

        # Recursively find all JSON files
        json_files = list(data_path.rglob("*.json"))
        logger.info("Found %d JSON files in %s", len(json_files), data_path)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    # Handle both single drug and list-of-drugs format
                    if isinstance(content, list):
                        drugs.extend(content)
                    elif isinstance(content, dict):
                        if "metadata" in content and "sections" in content:
                            drugs.append(content)
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("Skipping %s: %s", jf, e)

        logger.info("Loaded %d drug records total", len(drugs))
        return drugs

    def index_model(self, model_key: str, drugs: List[Dict[str, Any]], max_drugs: int = 0):
        """
        Create a Qdrant collection for the given embedding model and index all drugs.

        Args:
            model_key: Key in EMBEDDING_MODELS dict.
            drugs: List of drug dicts with metadata + sections.
            max_drugs: Limit drugs to index (0 = all). Useful for testing.
        """
        cfg = EMBEDDING_MODELS[model_key]
        col_name = self._collection_name(model_key)
        dim = cfg["dimension"]

        # Create collection
        existing = [c.name for c in self.client.get_collections().collections]
        if col_name in existing:
            logger.info("Collection '%s' exists. Deleting for fresh index.", col_name)
            self.client.delete_collection(col_name)

        self.client.create_collection(
            collection_name=col_name,
            vectors_config=rest_models.VectorParams(
                size=dim, distance=rest_models.Distance.COSINE
            ),
        )
        # Create payload indexes
        for field in ["registration_number", "registration_no", "section_name", "drug_type"]:
            self.client.create_payload_index(
                collection_name=col_name,
                field_name=field,
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )

        model = self._load_model(model_key)
        chunk_size = self.base_cfg["chunking"]["chunk_size"]
        chunk_overlap = self.base_cfg["chunking"]["chunk_overlap"]

        subset = drugs[:max_drugs] if max_drugs > 0 else drugs
        total_points = 0
        batch_points = []
        batch_size = 64  # Upsert in batches for efficiency

        for drug_idx, drug in enumerate(subset):
            metadata = drug.get("metadata", {})
            sections = drug.get("sections", {})
            reg_no = metadata.get("registration_number", metadata.get("registration_no", ""))
            drug_name = metadata.get("name", metadata.get("drug_name", "unknown"))

            for section_name, section_content in sections.items():
                if not section_content:
                    continue

                # Normalize content
                if isinstance(section_content, dict):
                    text = section_content.get("text", "")
                elif isinstance(section_content, str):
                    text = section_content
                else:
                    continue

                if not text or len(text.strip()) < 10:
                    continue

                chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                for idx, chunk in enumerate(chunks):
                    # E5 models need "query: " prefix for queries, "passage: " for docs
                    encode_text = chunk
                    if "e5" in model_key:
                        encode_text = f"passage: {chunk}"

                    vector = model.encode(encode_text).tolist()
                    point_id = self._generate_id(reg_no, section_name, idx)

                    batch_points.append(
                        rest_models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "drug_name": drug_name,
                                "registration_no": reg_no,
                                "registration_number": reg_no,
                                "section_name": section_name,
                                "chunk_text": chunk,
                                "chunk_index": idx,
                                "drug_type": metadata.get("drug_type", ""),
                            },
                        )
                    )

                    if len(batch_points) >= batch_size:
                        self.client.upsert(collection_name=col_name, points=batch_points)
                        total_points += len(batch_points)
                        batch_points = []

            if (drug_idx + 1) % 100 == 0:
                logger.info("[%s] Indexed %d/%d drugs (%d chunks)", model_key, drug_idx + 1, len(subset), total_points)

        # Flush remaining
        if batch_points:
            self.client.upsert(collection_name=col_name, points=batch_points)
            total_points += len(batch_points)

        logger.info("[%s] Done. Total chunks indexed: %d", model_key, total_points)
        return total_points

    def index_all_models(self, max_drugs: int = 0):
        """Index drug corpus with all embedding models."""
        drugs = self.load_drug_files()
        if not drugs:
            logger.error("No drug data found! Check data/ directory.")
            return

        results = {}
        for model_key in EMBEDDING_MODELS:
            logger.info("=" * 60)
            logger.info("Indexing with model: %s", model_key)
            start = time.time()
            count = self.index_model(model_key, drugs, max_drugs)
            elapsed = time.time() - start
            results[model_key] = {
                "chunks_indexed": count,
                "time_seconds": round(elapsed, 1),
            }
            logger.info("[%s] Completed in %.1fs", model_key, elapsed)

        # Save indexing summary
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "embedding_index_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Indexing summary saved to results/embedding_index_summary.json")

    # ─── Evaluation ──────────────────────────────────────────────────────

    def search_model(
        self, model_key: str, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search a specific model's collection."""
        cfg = EMBEDDING_MODELS[model_key]
        col_name = self._collection_name(model_key)
        model = self._load_model(model_key)

        # E5 models need query prefix
        encode_text = f"query: {query}" if "e5" in model_key else query
        query_vector = model.encode(encode_text).tolist()

        response = self.client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in response.points
        ]

    def evaluate_retrieval(
        self, benchmark_path: str, top_k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all embedding models against a benchmark dataset.

        Benchmark JSON format (PharmaQA.VN):
        [
            {
                "id": "qa_0001",
                "drug_name": "Zitromax",
                "registration_no": "VN-21930-19",
                "section": "dosage",
                "question": "...",
                "ground_truth": "...",
                "evidence_span": "..."
            },
            ...
        ]

        Metrics computed:
            - Recall@k: Is the correct drug+section in top-k results?
            - MRR: Mean Reciprocal Rank of the first relevant result
            - nDCG@k: Normalized Discounted Cumulative Gain
        """
        top_k_values = top_k_values or [1, 3, 5, 10]
        max_k = max(top_k_values)

        # Load benchmark
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        logger.info("Loaded benchmark with %d questions", len(benchmark))

        all_results = {}

        for model_key in EMBEDDING_MODELS:
            logger.info("Evaluating model: %s", model_key)
            col_name = self._collection_name(model_key)

            # Check collection exists
            existing = [c.name for c in self.client.get_collections().collections]
            if col_name not in existing:
                logger.warning("Collection '%s' not found. Run `index` first.", col_name)
                continue

            metrics = {f"recall@{k}": 0.0 for k in top_k_values}
            metrics["mrr"] = 0.0
            metrics["ndcg@10"] = 0.0
            query_count = 0

            for qa in benchmark:
                question = qa.get("question", "")
                expected_reg_no = qa.get("registration_no", "")
                expected_section = qa.get("section", "")

                if not question:
                    continue

                results = self.search_model(model_key, question, top_k=max_k)
                query_count += 1

                # Check relevance: correct drug + correct section
                relevant_found_at = None
                for rank, res in enumerate(results):
                    payload = res.get("payload", {})
                    res_reg_no = payload.get("registration_no", payload.get("registration_number", ""))
                    res_section = payload.get("section_name", "")

                    is_relevant = (
                        res_reg_no == expected_reg_no and
                        res_section == expected_section
                    )
                    if is_relevant:
                        relevant_found_at = rank  # 0-indexed
                        break

                # Recall@k
                for k in top_k_values:
                    if relevant_found_at is not None and relevant_found_at < k:
                        metrics[f"recall@{k}"] += 1

                # MRR
                if relevant_found_at is not None:
                    metrics["mrr"] += 1.0 / (relevant_found_at + 1)

                # nDCG@10 (binary relevance)
                if relevant_found_at is not None and relevant_found_at < 10:
                    import math
                    metrics["ndcg@10"] += 1.0 / math.log2(relevant_found_at + 2)

            # Normalize
            if query_count > 0:
                for key in metrics:
                    metrics[key] = round(metrics[key] / query_count, 4)

            metrics["total_queries"] = query_count
            metrics["model_name"] = EMBEDDING_MODELS[model_key]["model_name"]
            metrics["description"] = EMBEDDING_MODELS[model_key]["description"]
            all_results[model_key] = metrics

            logger.info("[%s] Recall@5=%.4f  MRR=%.4f  nDCG@10=%.4f",
                        model_key, metrics.get("recall@5", 0), metrics["mrr"], metrics["ndcg@10"])

        # Save results
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "embedding_comparison_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", out_path)

        # Print summary table
        self._print_summary_table(all_results, top_k_values)
        return all_results

    def smoke_test(self):
        """Quick smoke-test with built-in queries (no benchmark needed)."""
        logger.info("Running smoke test with %d built-in queries...", len(SMOKE_TEST_QUERIES))

        for model_key in EMBEDDING_MODELS:
            col_name = self._collection_name(model_key)
            existing = [c.name for c in self.client.get_collections().collections]
            if col_name not in existing:
                logger.warning("[%s] Collection not found. Skipping.", model_key)
                continue

            logger.info("\n[%s] %s", model_key, EMBEDDING_MODELS[model_key]["description"])
            for sq in SMOKE_TEST_QUERIES:
                results = self.search_model(model_key, sq["question"], top_k=3)
                top_result = results[0] if results else None
                if top_result:
                    p = top_result["payload"]
                    section_match = "✓" if p.get("section_name") == sq["expected_section"] else "✗"
                    logger.info(
                        "  Q: %s\n    → [%s] %s | %s | score=%.4f",
                        sq["question"], section_match,
                        p.get("drug_name", "?"), p.get("section_name", "?"),
                        top_result["score"]
                    )
                else:
                    logger.info("  Q: %s\n    → No results", sq["question"])

    def _print_summary_table(self, results: Dict, top_k_values: List[int]):
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("EMBEDDING MODEL COMPARISON RESULTS")
        print("=" * 80)

        # Header
        k_headers = "  ".join(f"R@{k:<3}" for k in top_k_values)
        print(f"{'Model':<12} {k_headers}  {'MRR':<8} {'nDCG@10':<8} {'Queries'}")
        print("-" * 80)

        for key, m in results.items():
            k_values = "  ".join(f"{m.get(f'recall@{k}', 0):<5.3f}" for k in top_k_values)
            print(f"{key:<12} {k_values}  {m['mrr']:<8.4f} {m['ndcg@10']:<8.4f} {m['total_queries']}")

        print("=" * 80)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embedding Model Comparison for Pharma-RAG (PROPOSAL 5.2B)"
    )
    parser.add_argument(
        "command",
        choices=["index", "evaluate", "smoke-test", "all"],
        help="Command to run",
    )
    parser.add_argument(
        "--benchmark", "-b",
        default="data/benchmark/pharma_qa.json",
        help="Path to benchmark JSON file (for evaluate/all commands)",
    )
    parser.add_argument(
        "--max-drugs", "-m",
        type=int, default=0,
        help="Max drugs to index (0 = all). Use small number for testing.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(EMBEDDING_MODELS.keys()),
        default=list(EMBEDDING_MODELS.keys()),
        help="Which models to evaluate (default: all)",
    )

    args = parser.parse_args()
    comparator = EmbeddingComparator()

    if args.command in ("index", "all"):
        comparator.index_all_models(max_drugs=args.max_drugs)

    if args.command in ("evaluate", "all"):
        if not Path(args.benchmark).exists():
            logger.error("Benchmark file not found: %s", args.benchmark)
            logger.error("Run teammate's Q&A generation script first, or use 'smoke-test'.")
            sys.exit(1)
        comparator.evaluate_retrieval(args.benchmark)

    if args.command == "smoke-test":
        comparator.smoke_test()


if __name__ == "__main__":
    main()

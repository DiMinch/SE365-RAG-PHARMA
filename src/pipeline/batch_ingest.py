"""
batch_ingest.py — Fast batch ingestion for Pharma-RAG
Reads pre-normalized Drug JSON files, encodes vectors in batches, and
upserts large batches to Qdrant for 10-50x faster ingestion.

Usage:
    python -m src.pipeline.batch_ingest --type western
    python -m src.pipeline.batch_ingest --type traditional
    python -m src.pipeline.batch_ingest --type both
"""

import argparse
import hashlib
import json
import logging
import uuid
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Load .env BEFORE importing config (so QDRANT_URL / QDRANT_API_KEY are visible)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on system env vars

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

from src.models.drug import Drug, DrugMetadata, DrugSections, ActiveIngredient, HerbalIngredient, Manufacturer, Packaging
from src.utils.config import get_base_config, get_qdrant_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks, start, text_len = [], 0, len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += chunk_size - chunk_overlap
    return chunks

def _det_id(sdk: str, section: str, idx: int) -> str:
    h = hashlib.md5(f"{sdk}_{section}_{idx}".encode()).hexdigest()
    return str(uuid.UUID(h))


def _normalize_traditional(raw: Dict) -> Drug:
    herbal = [
        HerbalIngredient(name=h["name"], amount=h.get("amount"), role=h.get("role") or "Thành phần")
        for h in raw.get("herbal_ingredients", []) if h.get("name")
    ]
    sr = raw.get("sections", {})
    return Drug(
        metadata=DrugMetadata(
            name=raw["name"], registration_number=raw.get("registration_number") or "UNKNOWN",
            drug_type="TRADITIONAL_MEDICINE", herbal_ingredient_list=herbal,
            manufacturer=Manufacturer(name=raw.get("manufacturer") or "Unknown", country=raw.get("manufacturer_country")),
            packagings=[Packaging(unit_name=raw["dosage_form"])] if raw.get("dosage_form") else [],
        ),
        sections=DrugSections(**{k: sr.get(k) for k in DrugSections.model_fields}),
    )


def _load_drugs(directory: str, drug_type: str) -> List[Drug]:
    """Load all JSON files in a directory and return Drug objects."""
    path = Path(directory)
    files = sorted(path.glob("*.json"))
    drugs = []
    errors = 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = [data]
            for item in data:
                try:
                    if "metadata" in item and "sections" in item:
                        drug = Drug.model_validate(item)
                    elif drug_type == "TRADITIONAL_MEDICINE":
                        drug = _normalize_traditional(item)
                    else:
                        continue  # Western already normalized
                    drugs.append(drug)
                except Exception as e:
                    errors += 1
        except Exception as e:
            errors += 1

    logger.info("Loaded %d drugs from %s (%d errors)", len(drugs), directory, errors)
    return drugs


def _drug_to_chunks(drug: Drug, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Convert a Drug into a list of {text, payload, id} dicts for embedding + upsert."""
    results = []
    sections_dict = drug.sections.model_dump()

    for section_name, content in sections_dict.items():
        if not content:
            continue

        tables_data = None
        if isinstance(content, dict) and "text" in content:
            tables_data = content.get("table")
            content = content["text"]
        if not isinstance(content, str):
            continue

        chunks = _chunk_text(content, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            point_id = _det_id(drug.metadata.registration_number, section_name, idx)
            payload = {
                "drug_name": drug.metadata.name,
                "registration_number": drug.metadata.registration_number,
                "registration_no": drug.metadata.registration_number,
                "drug_type": drug.metadata.drug_type,
                "active_ingredient": (
                    ", ".join([hi.name for hi in drug.metadata.herbal_ingredient_list])
                    if drug.metadata.drug_type == "TRADITIONAL_MEDICINE"
                    else ", ".join([ai.name for ai in drug.metadata.active_ingredient_list])
                ),
                "strength": drug.metadata.strength,
                "manufacturer_name": drug.metadata.manufacturer.name,
                "manufacturer_country": drug.metadata.manufacturer.country,
                "chunk_text": chunk,
                "section_name": section_name,
                "chunk_index": idx,
            }
            if section_name == "dosage" and tables_data:
                payload["tables"] = tables_data

            results.append({"id": point_id, "text": chunk, "payload": payload})

    return results


# ──────────────────────────────────────────────────────────────────────
# Main batch ingest
# ──────────────────────────────────────────────────────────────────────

def batch_ingest(
    drugs: List[Drug],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 256,
):
    """Ingest drugs in large batches — encode then upsert."""

    # Prepare all chunks
    logger.info("Preparing chunks for %d drugs...", len(drugs))
    all_chunks: List[Dict[str, Any]] = []
    for drug in drugs:
        all_chunks.extend(_drug_to_chunks(drug, chunk_size, chunk_overlap))

    total_chunks = len(all_chunks)
    logger.info("Total chunks to ingest: %d", total_chunks)

    # Process in batches
    ingested = 0
    t0 = time.time()

    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Batch encode
        vectors = model.encode(texts, show_progress_bar=False, batch_size=batch_size)

        # Build points
        points = [
            rest_models.PointStruct(
                id=c["id"],
                vector=vec.tolist(),
                payload=c["payload"],
            )
            for c, vec in zip(batch, vectors)
        ]

        # Batch upsert
        client.upsert(collection_name=collection_name, points=points, wait=True)

        ingested += len(batch)
        elapsed = time.time() - t0
        rate = ingested / elapsed if elapsed > 0 else 0
        eta = (total_chunks - ingested) / rate if rate > 0 else 0
        logger.info(
            "[Batch %d/%d] Ingested %d/%d chunks (%.0f chunks/s, ETA %.0fs)",
            i // batch_size + 1,
            (total_chunks + batch_size - 1) // batch_size,
            ingested,
            total_chunks,
            rate,
            eta,
        )

    elapsed = time.time() - t0
    logger.info("Ingestion complete: %d chunks in %.1fs (%.0f chunks/s)", ingested, elapsed, ingested / elapsed)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast batch ingestion for Pharma-RAG")
    parser.add_argument("--type", choices=["western", "traditional", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding + upsert")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection first")
    args = parser.parse_args()

    base_cfg = get_base_config()
    qdrant_cfg = get_qdrant_config()
    conn = qdrant_cfg["connection"]

    # Connect to Qdrant (Cloud or Local, auto-detected from config/env)
    if "url" in conn and "api_key" in conn:
        client = QdrantClient(
            url=conn["url"],
            api_key=conn["api_key"],
            timeout=conn.get("timeout", 30.0),
        )
        logger.info("[Qdrant] CLOUD mode: %s", conn["url"])
    else:
        client = QdrantClient(
            host=conn.get("host", "localhost"),
            port=conn.get("port", 6333),
            timeout=conn.get("timeout", 10.0),
        )
        logger.info("[Qdrant] LOCAL mode: %s:%s", conn.get("host"), conn.get("port"))

    collection_name = qdrant_cfg["collection"]["name"]

    # Optionally recreate collection
    if args.recreate:
        logger.info("Recreating collection '%s'...", collection_name)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    # Ensure collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        vector_size = qdrant_cfg["collection"]["vector_size"]
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(
                size=vector_size,
                distance=rest_models.Distance.COSINE,
            ),
            hnsw_config=rest_models.HnswConfigDiff(
                m=qdrant_cfg["collection"]["hnsw_m"],
                ef_construct=qdrant_cfg["collection"]["hnsw_ef"],
            ),
        )
        # Create indexes
        for field in ["registration_number", "registration_no", "section_name", "drug_type"]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
        logger.info("Created collection '%s'", collection_name)

    # Load embedding model
    model_name = base_cfg["embedding"]["model_name"]
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    chunk_size = base_cfg["chunking"]["chunk_size"]
    chunk_overlap = base_cfg["chunking"]["chunk_overlap"]

    project_root = Path(__file__).parent.parent.parent

    # Ingest western
    if args.type in ("western", "both"):
        western_dir = str(project_root / "src" / "database" / "Datasets" / "thuoc" / "cleaned_json")
        drugs = _load_drugs(western_dir, "WESTERN_MEDICINE")
        if drugs:
            batch_ingest(drugs, model, client, collection_name, chunk_size, chunk_overlap, args.batch_size)

    # Ingest traditional
    if args.type in ("traditional", "both"):
        trad_dir = str(project_root / "data" / "raw" / "traditional")
        drugs = _load_drugs(trad_dir, "TRADITIONAL_MEDICINE")
        if drugs:
            batch_ingest(drugs, model, client, collection_name, chunk_size, chunk_overlap, args.batch_size)

    # Final stats
    info = client.get_collection(collection_name)
    logger.info("Collection '%s' now has %d points", collection_name, info.points_count)


if __name__ == "__main__":
    main()

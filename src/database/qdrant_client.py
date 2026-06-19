import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional, Literal
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

from src.models.drug import Drug
from src.utils.config import get_base_config, get_qdrant_config
from src.database.bm25_index import BM25Index
from src.utils.drug_synonym_resolver import get_synonym_resolver
from src.utils.reranker import CrossEncoderReranker

logger = logging.getLogger("PharmaQdrantClient")

class PharmaQdrantClient:
    """
    Qdrant database client to store and retrieve drug information.
    Handles structure-aware chunking, embedding generation, collection setup, and queries.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        # Load configurations
        self.base_cfg = get_base_config()
        self.qdrant_cfg = get_qdrant_config()
        conn = self.qdrant_cfg["connection"]

        # ── Auto-select connection mode ────────────────────────────────
        if host or port:
            # Explicit override (e.g. in tests)
            self.client = QdrantClient(
                host=host or conn.get("host", "localhost"),
                port=port or conn.get("port", 6333),
                timeout=conn.get("timeout", 10.0),
            )
            logger.info("[Qdrant] LOCAL mode: %s:%s", host, port)
        elif "url" in conn and "api_key" in conn:
            # Qdrant Cloud mode
            self.client = QdrantClient(
                url=conn["url"],
                api_key=conn["api_key"],
                timeout=conn.get("timeout", 30.0),
            )
            logger.info("[Qdrant] CLOUD mode: %s", conn["url"])
        else:
            # Local Docker default
            self.client = QdrantClient(
                host=conn.get("host", "localhost"),
                port=conn.get("port", 6333),
                timeout=conn.get("timeout", 10.0),
            )
            logger.info("[Qdrant] LOCAL mode: %s:%s", conn.get("host"), conn.get("port"))

        # Lazy load embedding model to save memory during initialization
        self._model = None
        self.collection_name = self.qdrant_cfg["collection"]["name"]

        # Lazy BM25 index — built on first hybrid query
        self._bm25: Optional[BM25Index] = None

        # Lazy synonym resolver for query expansion
        self._synonym_resolver = None

        # Lazy cross-encoder reranker (PROPOSAL.md Section 5.2C)
        self._reranker: Optional[CrossEncoderReranker] = None
        
    @property
    def model(self):
        if self._model is None:
            model_name = self.base_cfg["embedding"]["model_name"]
            print(f"[Qdrant Client] Loading embedding model: {model_name}...")
            self._model = SentenceTransformer(model_name)
        return self._model

    def create_collection_if_not_exists(self):
        """
        Creates the collection if it doesn't already exist in Qdrant.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            vector_size = self.qdrant_cfg["collection"]["vector_size"]
            distance_str = self.qdrant_cfg["collection"]["distance"].upper()
            
            # Map distance string to Qdrant Distance enum
            distance = rest_models.Distance.COSINE
            if distance_str == "EUCLIDEAN":
                distance = rest_models.Distance.EUCLID
            elif distance_str == "DOT":
                distance = rest_models.Distance.DOT
                
            print(f"[Qdrant Client] Creating collection '{self.collection_name}' with size {vector_size}...")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=vector_size,
                    distance=distance
                ),
                hnsw_config=rest_models.HnswConfigDiff(
                    m=self.qdrant_cfg["collection"]["hnsw_m"],
                    ef_construct=self.qdrant_cfg["collection"]["hnsw_ef"]
                )
            )
            
            # Create payload indexes for faster filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="registration_number",
                field_schema=rest_models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="registration_no",
                field_schema=rest_models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_name",
                field_schema=rest_models.PayloadSchemaType.KEYWORD
            )
        else:
            print(f"[Qdrant Client] Collection '{self.collection_name}' already exists.")

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits text into chunks of character size with overlap.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= text_len:
                break
            start += chunk_size - chunk_overlap
            
        return chunks

    def _generate_deterministic_id(self, registration_number: str, section: str, chunk_idx: int) -> str:
        """
        Generates a deterministic UUID based on drug registration number, section, and chunk index.
        """
        hash_input = f"{registration_number}_{section}_{chunk_idx}".encode('utf-8')
        md5_hash = hashlib.md5(hash_input).hexdigest()
        return str(uuid.UUID(md5_hash))

    def upsert_drug(self, drug: Drug) -> int:
        """
        Performs structure-aware chunking and embeds all sections of the drug, 
        then upserts them into Qdrant.
        Returns the number of upserted points.
        """
        # Ensure collection exists
        self.create_collection_if_not_exists()
        
        chunk_size = self.base_cfg["chunking"]["chunk_size"]
        chunk_overlap = self.base_cfg["chunking"]["chunk_overlap"]
        
        points = []
        
        # Iterate over all sections
        sections_dict = drug.sections.model_dump()
        for section_name, section_content in sections_dict.items():
            if not section_content:
                continue
                
            # Normalize content to string
            text_content = ""
            tables_data = []
            
            if isinstance(section_content, dict):
                text_content = section_content.get("text", "")
                tables_data = section_content.get("table", [])
            elif isinstance(section_content, str):
                text_content = section_content
            else:
                continue
                
            if not text_content:
                continue
                
            # Chunk the section text
            chunks = self._chunk_text(text_content, chunk_size, chunk_overlap)
            
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                vector = self.model.encode(chunk)
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                elif not isinstance(vector, list):
                    vector = list(vector)
                
                # Generate deterministic point ID
                point_id = self._generate_deterministic_id(
                    drug.metadata.registration_number, 
                    section_name, 
                    idx
                )
                
                # Payload definition - fully compliant with QĐ 522 names
                payload = {
                    "id": drug.metadata.id,
                    "name": drug.metadata.name,
                    "registration_number": drug.metadata.registration_number,
                    "drug_type": drug.metadata.drug_type,
                    "drug_group_id": drug.metadata.drug_group_id,
                    "active_ingredient_list": [ai.model_dump() for ai in drug.metadata.active_ingredient_list],
                    "herbal_ingredient_list": [hi.model_dump() for hi in drug.metadata.herbal_ingredient_list],
                    "strength": drug.metadata.strength,
                    "route_id": drug.metadata.route_id,
                    "prescription_status": drug.metadata.prescription_status,
                    "special_control_type": drug.metadata.special_control_type,
                    "packagings": [p.model_dump() for p in drug.metadata.packagings],
                    "manufacturer": drug.metadata.manufacturer.model_dump(),
                    "approval_date": drug.metadata.approval_date,
                    "expiry_date": drug.metadata.expiry_date,
                    "registrant": drug.metadata.registrant,
                    
                    "chunk_text": chunk,
                    "section_name": section_name,
                    "chunk_index": idx,
                    
                    # Backward-compatibility fallback keys:
                    "drug_name": drug.metadata.name,
                    "registration_no": drug.metadata.registration_number,
                    "active_ingredient": (
                        ", ".join([hi.name for hi in drug.metadata.herbal_ingredient_list])
                        if drug.metadata.drug_type == "TRADITIONAL_MEDICINE"
                        else ", ".join([ai.name for ai in drug.metadata.active_ingredient_list])
                    ),
                    "dosage_form": drug.metadata.packagings[0].unit_name if drug.metadata.packagings else None,
                }
                
                # Add table data to payload for dosage section
                if section_name == "dosage" and tables_data:
                    payload["tables"] = tables_data
                    
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
                
        # Upsert points into Qdrant
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info("[Qdrant Client] Successfully upserted %d chunks for drug: %s", len(points), drug.metadata.name)
            
        return len(points)

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        section_filter: Optional[str] = None,
        registration_no_filter: Optional[str] = None,
        drug_type_filter: Optional[str] = None,
        retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query for matches using dense, BM25, or hybrid (RRF-fused) retrieval.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            section_filter: Optionally restrict to a specific section name.
            registration_no_filter: Optionally restrict to a specific drug SDK.
            drug_type_filter: 'WESTERN_MEDICINE' | 'TRADITIONAL_MEDICINE' | None.
            retrieval_mode: 'dense' | 'bm25' | 'hybrid'.

        Returns:
            List of dicts with keys 'id', 'score', 'payload'.
            When use_reranker=True, each dict also contains 'reranker_score'.

        Ablation study toggle:
            use_reranker=False  → Baseline (no reranker)
            use_reranker=True   → Full system (Cross-Encoder reranking)
            use_reranker=None   → Read from base_config.yaml `reranker.enabled`
        """
        top_k = top_k or self.base_cfg["retrieval"]["top_k"]

        # Determine effective reranker flag
        reranker_cfg = self.base_cfg.get("reranker", {})
        if use_reranker is None:
            use_reranker = reranker_cfg.get("enabled", False)

        # When reranking, fetch more candidates to give the cross-encoder a richer pool
        pre_rerank_k = self.base_cfg["retrieval"].get("pre_rerank_k", 20) if use_reranker else top_k
        score_threshold = self.base_cfg["retrieval"]["score_threshold"]

        # ── Build Qdrant filter ────────────────────────────────────────
        must_filters = []

        if section_filter:
            must_filters.append(
                rest_models.FieldCondition(
                    key="section_name",
                    match=rest_models.MatchValue(value=section_filter)
                )
            )

        if registration_no_filter:
            must_filters.append(
                rest_models.Filter(
                    should=[
                        rest_models.FieldCondition(
                            key="registration_number",
                            match=rest_models.MatchValue(value=registration_no_filter)
                        ),
                        rest_models.FieldCondition(
                            key="registration_no",
                            match=rest_models.MatchValue(value=registration_no_filter)
                        )
                    ]
                )
            )

        if drug_type_filter:
            must_filters.append(
                rest_models.FieldCondition(
                    key="drug_type",
                    match=rest_models.MatchValue(value=drug_type_filter)
                )
            )

        query_vector = self.model.encode(query).tolist()

        qdrant_filter = rest_models.Filter(must=must_filters) if must_filters else None

        # ── Query expansion via synonym resolver ─────────────────────
        expanded_query = query
        try:
            if self._synonym_resolver is None:
                self._synonym_resolver = get_synonym_resolver()
            expanded_query = self._synonym_resolver.expand_query(query)
        except Exception as exc:
            logger.debug("Synonym expansion skipped: %s", exc)

        # Use expanded query for BM25, original embedding for dense
        # (embedding model handles semantic similarity already)

        # ── Dense retrieval ────────────────────────────────────────────
        # qdrant-client >= 1.10: search() is replaced by query_points()
        dense_results: List[Dict[str, Any]] = []
        if retrieval_mode in ("dense", "hybrid"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=pre_rerank_k,
                score_threshold=score_threshold if retrieval_mode == "dense" else 0.0,
                with_payload=True,
            )
            dense_results = [{"id": r.id, "score": r.score, "payload": r.payload} for r in response.points]

        # ── BM25 retrieval ─────────────────────────────────────────────
        bm25_results: List[Dict[str, Any]] = []
        if retrieval_mode in ("bm25", "hybrid"):
            self._ensure_bm25_built()
            if self._bm25 and self._bm25.is_built():
                # Use expanded query for BM25 (keyword matching benefits from synonyms)
                bm25_results = self._bm25.search(expanded_query, top_k=pre_rerank_k)

        # ── Reciprocal Rank Fusion (RRF) ───────────────────────────────
        if retrieval_mode == "hybrid" and bm25_results:
            candidates = self._rrf_fuse(dense_results, bm25_results, top_k=pre_rerank_k)
        elif retrieval_mode == "bm25":
            candidates = bm25_results[:pre_rerank_k]
        else:
            candidates = dense_results[:pre_rerank_k]

        # ── Cross-Encoder Reranking (PROPOSAL Section 5.2C) ────────────
        if use_reranker and candidates:
            try:
                if self._reranker is None:
                    model_name = reranker_cfg.get("model_name", "BAAI/bge-reranker-base")
                    self._reranker = CrossEncoderReranker(model_name=model_name)
                candidates = self._reranker.rerank(query, candidates, top_k=top_k)
                logger.info(
                    "[Qdrant] Reranker: %d candidates → top %d",
                    pre_rerank_k, top_k
                )
            except Exception as exc:
                logger.warning(
                    "[Qdrant] Reranker failed, falling back to pre-reranked order: %s", exc
                )
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates

    # ──────────────────────────────────────────────────────────────────
    # Hybrid Helpers
    # ──────────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: List[Dict[str, Any]],
        bm25: List[Dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.
        score_rrf(doc) = Σ 1 / (k + rank_i(doc))  for each ranking list i
        """
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}

        for rank, item in enumerate(dense):
            doc_id = str(item["id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            payloads[doc_id] = item["payload"]

        for rank, item in enumerate(bm25):
            doc_id = str(item["id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in payloads:
                payloads[doc_id] = item["payload"]

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            {"id": doc_id, "score": scores[doc_id], "payload": payloads[doc_id]}
            for doc_id in sorted_ids
        ]

    def _ensure_bm25_built(self) -> None:
        """Build BM25 index from all Qdrant chunks if not already built."""
        if self._bm25 and self._bm25.is_built():
            return

        print("[Qdrant Client] Building BM25 index from Qdrant collection...")
        try:
            all_docs = []
            offset = None
            while True:
                result, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not result:
                    break
                for point in result:
                    payload = point.payload or {}
                    payload["id"] = str(point.id)
                    all_docs.append(payload)
                if offset is None:
                    break

            self._bm25 = BM25Index()
            self._bm25.build(all_docs)
        except Exception as exc:
            print(f"[Qdrant Client] BM25 index build failed: {exc}")

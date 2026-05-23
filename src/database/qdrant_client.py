import uuid
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

from src.models.drug import Drug
from src.utils.config import get_base_config, get_qdrant_config

class PharmaQdrantClient:
    """
    Qdrant database client to store and retrieve drug information.
    Handles structure-aware chunking, embedding generation, collection setup, and queries.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        # Load configurations
        self.base_cfg = get_base_config()
        self.qdrant_cfg = get_qdrant_config()
        
        # Override connection params if provided
        host = host or self.qdrant_cfg["connection"]["host"]
        port = port or self.qdrant_cfg["connection"]["port"]
        
        # Initialize clients
        self.client = QdrantClient(
            host=host,
            port=port,
            timeout=self.qdrant_cfg["connection"]["timeout"]
        )
        
        # Lazy load embedding model to save memory during initialization
        self._model = None
        self.collection_name = self.qdrant_cfg["collection"]["name"]
        
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
                    "drug_group_id": drug.metadata.drug_group_id,
                    "active_ingredient_list": [ai.model_dump() for ai in drug.metadata.active_ingredient_list],
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
                    "active_ingredient": ", ".join([ai.name for ai in drug.metadata.active_ingredient_list]),
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
            print(f"[Qdrant Client] Successfully upserted {len(points)} chunks for drug: {drug.metadata.name}")
            
        return len(points)

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        section_filter: Optional[str] = None,
        registration_no_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Qdrant for matches, returning scores and payloads.
        Supports filtering by section name or registration number.
        """
        top_k = top_k or self.base_cfg["retrieval"]["top_k"]
        score_threshold = self.base_cfg["retrieval"]["score_threshold"]
        
        # Generate query embedding
        query_vector = self.model.encode(query).tolist()
        
        # Build filter conditions
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
            
        qdrant_filter = rest_models.Filter(must=must_filters) if must_filters else None
        
        # Search Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": res.id,
                "score": res.score,
                "payload": res.payload
            })
            
        return formatted_results

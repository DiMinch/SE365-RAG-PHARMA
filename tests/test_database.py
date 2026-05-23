import pytest
from unittest.mock import MagicMock, patch
from src.database.qdrant_client import PharmaQdrantClient
from src.models.drug import Drug, DrugMetadata, DrugSections, ActiveIngredient, Manufacturer, Packaging

def test_chunk_text():
    client = PharmaQdrantClient()
    # Test simple chunking
    text = "A" * 1200
    chunks = client._chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    assert len(chunks) == 3
    assert chunks[0] == "A" * 500
    assert chunks[1] == "A" * 500
    assert chunks[2] == "A" * 300  # remaining 1200 - (450 + 450) = 300

def test_uuid_generation():
    client = PharmaQdrantClient()
    id1 = client._generate_deterministic_id("VN-12345-20", "indication", 0)
    id2 = client._generate_deterministic_id("VN-12345-20", "indication", 0)
    id3 = client._generate_deterministic_id("VN-12345-20", "indication", 1)
    
    assert id1 == id2
    assert id1 != id3

@patch('src.database.qdrant_client.QdrantClient')
@patch('src.database.qdrant_client.SentenceTransformer')
def test_upsert_drug(mock_transformer, mock_qdrant):
    # Setup mocks
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = [0.1] * 384
    mock_transformer.return_value = mock_model_instance
    
    mock_qdrant_instance = MagicMock()
    mock_qdrant_instance.get_collections.return_value.collections = []
    mock_qdrant.return_value = mock_qdrant_instance
    
    client = PharmaQdrantClient()
    client._model = mock_model_instance
    
    drug = Drug(
        metadata=DrugMetadata(
            id="123456",
            name="Test Drug",
            registration_number="VN-99999-22",
            drug_group_id="Group A",
            active_ingredient_list=[
                ActiveIngredient(name="Test Ingredient", is_main_active_ingredient=True)
            ],
            strength="10mg",
            route_id="Oral",
            prescription_status=1,
            special_control_type=0,
            packagings=[Packaging(unit_name="Tablet", quantity=10, is_basic_unit=True)],
            manufacturer=Manufacturer(name="Factory X", country="VN"),
            approval_date="2022-01-01",
            expiry_date="2027-01-01",
            registrant="Company Y"
        ),
        sections=DrugSections(
            indication="This is a test indication for the drug.",
            contraindication="Do not take this drug if allergic."
        )
    )
    
    num_chunks = client.upsert_drug(drug)
    
    # Indication has 38 chars, contraindication has 34 chars. Both fit inside 1 chunk of size 500.
    # So 2 chunks in total.
    assert num_chunks == 2
    assert mock_qdrant_instance.upsert.called

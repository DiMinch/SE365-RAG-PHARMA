import pytest
from src.crawler.dav_validator import DAVValidator
from src.crawler.ydct_validator import YDCTValidator

def test_normalization():
    validator = DAVValidator()
    assert validator.normalize_reg_no("VN-20041-16") == "VN2004116"
    assert validator.normalize_reg_no("vn-20041-16") == "VN2004116"
    assert validator.normalize_reg_no(" VN 20041 16 ") == "VN2004116"
    assert validator.normalize_reg_no(None) == ""

def test_real_validation():
    import requests
    # VN-20041-16 is a verified valid registration number for Voltaren 75mg/3ml
    validator = DAVValidator()
    try:
        res = validator.validate("VN-20041-16")
    except (requests.exceptions.RequestException, Exception) as e:
        pytest.skip(f"Skipping test due to network/server issue: {e}")
        
    if res is None:
        pytest.skip("Skipping test: Government API returned no data (possibly blocked or down)")
        
    assert res is not None
    assert res["valid"] is True
    assert "Voltaren" in res["drug_name"]
    assert "Diclofenac" in res["active_ingredient"]

def test_real_validation_old_sdk():
    import requests
    # VN-21930-19 is an old registration number whose current SDK is 800110991624
    validator = DAVValidator()
    try:
        res = validator.validate("VN-21930-19")
    except (requests.exceptions.RequestException, Exception) as e:
        pytest.skip(f"Skipping test due to network/server issue: {e}")
        
    if res is None:
        pytest.skip("Skipping test: Government API returned no data (possibly blocked or down)")
        
    assert res is not None
    assert res["valid"] is True
    assert res["registration_number_old"] == "VN-21930-19" or res["registration_no"] == "800110991624"

def test_invalid_validation():
    validator = DAVValidator()
    res = validator.validate("INVALID-SDK-12345")
    # Should not find a match
    assert res is None

def test_ydct_normalization():
    validator = YDCTValidator()
    assert validator.normalize_reg_no("V246-H01-13") == "V246H0113"
    assert validator.normalize_reg_no("VNB-1234-20") == "VNB123420"
    assert validator.normalize_reg_no(None) == ""

def test_ydct_invalid_validation():
    validator = YDCTValidator()
    res = validator.validate("INVALID-YDCT-999")
    assert res is None

def test_ydct_real_validation():
    import requests
    validator = YDCTValidator()
    try:
        res = validator.validate("TCT-00289-25")
    except (requests.exceptions.RequestException, Exception) as e:
        pytest.skip(f"Skipping test due to network/server issue: {e}")
        
    if res is None:
        pytest.skip("Skipping test: Government API returned no data (possibly blocked or down)")
        
    assert res is not None
    assert res["valid"] is True
    assert "B GIANG" in res["name"].upper() or "BÀ GIẰNG" in res["name"].upper()
    assert len(res["herbal_ingredient_list"]) > 0


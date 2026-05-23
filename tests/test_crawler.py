import pytest
from src.crawler.dav_validator import DAVValidator

def test_normalization():
    validator = DAVValidator()
    assert validator.normalize_reg_no("VN-20041-16") == "VN2004116"
    assert validator.normalize_reg_no("vn-20041-16") == "VN2004116"
    assert validator.normalize_reg_no(" VN 20041 16 ") == "VN2004116"
    assert validator.normalize_reg_no(None) == ""

def test_real_validation():
    # VN-20041-16 is a verified valid registration number for Voltaren 75mg/3ml
    validator = DAVValidator()
    res = validator.validate("VN-20041-16")
    
    assert res is not None
    assert res["valid"] is True
    assert "Voltaren" in res["drug_name"]
    assert "Diclofenac" in res["active_ingredient"]

def test_real_validation_old_sdk():
    # VN-21930-19 is an old registration number whose current SDK is 800110991624
    validator = DAVValidator()
    res = validator.validate("VN-21930-19")
    
    assert res is not None
    assert res["valid"] is True
    assert res["registration_number_old"] == "VN-21930-19" or res["registration_no"] == "800110991624"

def test_invalid_validation():
    validator = DAVValidator()
    res = validator.validate("INVALID-SDK-12345")
    # Should not find a match
    assert res is None

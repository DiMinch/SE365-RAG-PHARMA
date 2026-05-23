import requests
import urllib3
import re
from typing import Optional, Dict, Any, List

# Disable insecure request warnings for government portal certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DAVValidator:
    """
    Validator class to check drug registration numbers with Cục Quản lý Dược (dav.gov.vn)
    via their public services portal API.
    """
    
    API_URL = "https://dichvucong.dav.gov.vn/api/services/app/soDangKy/GetAllPublicServerPaging"
    
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://dichvucong.dav.gov.vn/congbothuoc/index"
        }

    def normalize_reg_no(self, reg_no: str) -> str:
        """
        Normalize registration numbers by removing dashes, spaces and converting to uppercase
        for robust comparison.
        """
        if not reg_no:
            return ""
        return reg_no.replace("-", "").replace(" ", "").upper()

    def _parse_and_match(self, items: List[Dict[str, Any]], clean_reg_no: str) -> Optional[Dict[str, Any]]:
        """
        Scan returned items for a normalized match on soDangKy or soDangKyCu.
        """
        target_normalized = self.normalize_reg_no(clean_reg_no)
        for item in items:
            sdk = item.get("soDangKy", "")
            sdk_cu = item.get("soDangKyCu", "")
            
            if (self.normalize_reg_no(sdk) == target_normalized or 
                self.normalize_reg_no(sdk_cu) == target_normalized):
                
                # Extract dates
                reg_info = item.get("thongTinDangKyThuoc", {}) or {}
                ngay_cap = reg_info.get("ngayCapSoDangKy")
                ngay_het_han = reg_info.get("ngayHetHanSoDangKy")
                
                # Extract active ingredient, dosage and package form from basic info
                basic_info = item.get("thongTinThuocCoBan", {}) or {}
                
                raw_ingredients = basic_info.get("hoatChatChinh") or ""
                active_ingredient_list = []
                if raw_ingredients:
                    parts = [p.strip() for p in re.split(r'[;,]', raw_ingredients) if p.strip()]
                    for part in parts:
                        active_ingredient_list.append({
                            "id": None,
                            "name": part,
                            "is_main_active_ingredient": True
                        })
                
                return {
                    "valid": True,
                    "id": str(item.get("id")) if item.get("id") else None,
                    "name": item.get("tenThuoc"),
                    "registration_number": sdk,
                    "registration_number_old": sdk_cu,
                    "active_ingredient_list": active_ingredient_list,
                    "strength": basic_info.get("hamLuong"),
                    "dosage_form": basic_info.get("dangBaoChe"),
                    "packaging": basic_info.get("dongGoi"),
                    "manufacturer": (item.get("congTySanXuat") or {}).get("tenCongTySanXuat"),
                    "manufacturer_country": (item.get("congTySanXuat") or {}).get("nuocSanXuat"),
                    "registrant": (item.get("congTyDangKy") or {}).get("tenCongTyDangKy"),
                    "registrant_country": (item.get("congTyDangKy") or {}).get("nuocDangKy"),
                    "approval_date": ngay_cap,
                    "expiry_date": ngay_het_han,
                    "is_expired": item.get("isHetHan", False),
                    "decision_no": reg_info.get("soQuyetDinh"),
                    "prescription_status": 1 if item.get("keDon") else 0,
                    # Compatibility helpers
                    "drug_name": item.get("tenThuoc"),
                    "active_ingredient": raw_ingredients,
                    "registration_no": sdk,
                    "dosage": basic_info.get("hamLuong"),
                }
        return None

    def validate(self, reg_no: str) -> Optional[Dict[str, Any]]:
        """
        Query the DAV portal for a specific registration number.
        Returns a dictionary of normalized metadata if found, else None.
        """
        clean_reg_no = reg_no.strip()
        if not clean_reg_no:
            return None
            
        # Phase 1: Search via specific SoDangKy
        payload = {
            "filterText": "",
            "SoDangKyThuoc": {
                "SoDangKy": clean_reg_no,
                "TenThuoc": "",
                "HoatChatChinh": "",
                "HamLuong": "",
                "DangBaoChe": "",
                "DongGoi": "",
                "CongTyDangKy": "",
                "CongTySanXuat": ""
            },
            "KichHoat": True,
            "skipCount": 0,
            "maxResultCount": 10
        }
        
        try:
            response = requests.post(
                self.API_URL, 
                headers=self.headers, 
                json=payload, 
                verify=False, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("result", {}).get("items", [])
                match = self._parse_and_match(items, clean_reg_no)
                if match:
                    return match
        except Exception as e:
            print(f"[DAV Validator] Error in specific search for {reg_no}: {e}")
            
        # Phase 2: Fallback to general filterText (handles old registration number search)
        payload_fallback = {
            "filterText": clean_reg_no,
            "SoDangKyThuoc": {
                "SoDangKy": "",
                "TenThuoc": "",
                "HoatChatChinh": "",
                "HamLuong": "",
                "DangBaoChe": "",
                "DongGoi": "",
                "CongTyDangKy": "",
                "CongTySanXuat": ""
            },
            "KichHoat": True,
            "skipCount": 0,
            "maxResultCount": 10
        }
        
        try:
            response = requests.post(
                self.API_URL, 
                headers=self.headers, 
                json=payload_fallback, 
                verify=False, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("result", {}).get("items", [])
                match = self._parse_and_match(items, clean_reg_no)
                if match:
                    return match
                
                # If still no exact match, but items returned on filterText, return the first one parsed
                if items:
                    item = items[0]
                    sdk = item.get("soDangKy") or ""
                    reg_info = item.get("thongTinDangKyThuoc") or {}
                    basic_info = item.get("thongTinThuocCoBan") or {}
                    
                    raw_ingredients = basic_info.get("hoatChatChinh") or ""
                    active_ingredient_list = []
                    if raw_ingredients:
                        parts = [p.strip() for p in re.split(r'[;,]', raw_ingredients) if p.strip()]
                        for part in parts:
                            active_ingredient_list.append({
                                "id": None,
                                "name": part,
                                "is_main_active_ingredient": True
                            })
                    
                    return {
                        "valid": True,
                        "id": str(item.get("id")) if item.get("id") else None,
                        "name": item.get("tenThuoc"),
                        "registration_number": sdk,
                        "registration_number_old": item.get("soDangKyCu"),
                        "active_ingredient_list": active_ingredient_list,
                        "strength": basic_info.get("hamLuong"),
                        "dosage_form": basic_info.get("dangBaoChe"),
                        "packaging": basic_info.get("dongGoi"),
                        "manufacturer": (item.get("congTySanXuat") or {}).get("tenCongTySanXuat"),
                        "manufacturer_country": (item.get("congTySanXuat") or {}).get("nuocSanXuat"),
                        "registrant": (item.get("congTyDangKy") or {}).get("tenCongTyDangKy"),
                        "registrant_country": (item.get("congTyDangKy") or {}).get("nuocDangKy"),
                        "approval_date": reg_info.get("ngayCapSoDangKy"),
                        "expiry_date": reg_info.get("ngayHetHanSoDangKy"),
                        "is_expired": item.get("isHetHan", False),
                        "decision_no": reg_info.get("soQuyetDinh"),
                        "prescription_status": 1 if item.get("keDon") else 0,
                        # Compatibility helpers
                        "drug_name": item.get("tenThuoc"),
                        "active_ingredient": raw_ingredients,
                        "registration_no": sdk,
                        "dosage": basic_info.get("hamLuong"),
                    }
                    
        except Exception as e:
            print(f"[DAV Validator] Error in fallback search for {reg_no}: {e}")
            
        return None

if __name__ == "__main__":
    import sys
    test_no = "VN-21930-19" if len(sys.argv) < 2 else sys.argv[1]
    print(f"Testing validation for: {test_no}")
    validator = DAVValidator()
    res = validator.validate(test_no)
    if res:
        print("Validation Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        print("Invalid or not found registration number.")

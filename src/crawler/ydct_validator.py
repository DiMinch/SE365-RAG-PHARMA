import requests
import urllib3
import ssl
import re
from typing import Optional, Dict, Any, List
from requests.adapters import HTTPAdapter

# Disable insecure request warnings for government portal certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SSLContextAdapter(HTTPAdapter):
    """
    Custom HTTP Adapter to allow legacy/weak DH keys by setting SECLEVEL=1.
    This is required to bypass 'dh key too small' SSL errors on some legacy government websites.
    """
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

class YDCTValidator:
    """
    Validator class to check traditional medicine registration numbers with Cục Quản lý Y, Dược Cổ truyền
    (ydct-dichvucong.moh.gov.vn) via their public services portal API.
    """
    API_URL = "https://ydct-dichvucong.moh.gov.vn/api/services/app/soDangKy/GetAllPublicServerPaging"
    
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://ydct-dichvucong.moh.gov.vn/congbothuoc/index"
        }
        # Set up session with custom SSL adapter to bypass weak DH key constraints
        self.session = requests.Session()
        self.session.mount('https://', SSLContextAdapter())

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
                
                # Extract basic info
                basic_info = item.get("thongTinThuocCoBan", {}) or {}
                
                # Extract herbs from hoatChatChinh which is a text representation of the formula
                raw_herbs = basic_info.get("hoatChatChinh") or ""
                herbal_ingredient_list = []
                if raw_herbs:
                    # Split by common delimiters in traditional recipes
                    parts = [p.strip() for p in re.split(r'[;,+]', raw_herbs) if p.strip()]
                    for part in parts:
                        # Attempt to extract weights/amounts like 0.5g, 10ml, etc.
                        amount_match = re.search(r'([\d.,]+\s*(?:g|mg|ml|%|lượng)?)', part, re.IGNORECASE)
                        amount = amount_match.group(1) if amount_match else None
                        name = part.replace(amount, "").strip() if amount else part
                        
                        herbal_ingredient_list.append({
                            "name": name.strip(" ()"),
                            "amount": amount,
                            "role": "Thành phần chính"
                        })
                
                return {
                    "valid": True,
                    "id": str(item.get("id")) if item.get("id") else None,
                    "name": item.get("tenThuoc"),
                    "registration_no": sdk,
                    "registration_number_old": sdk_cu,
                    "drug_type": "TRADITIONAL_MEDICINE",
                    "herbal_ingredient_list": herbal_ingredient_list,
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
                    "prescription_status": 0,
                    # Compatibility helpers
                    "drug_name": item.get("tenThuoc"),
                    "active_ingredient": raw_herbs,
                    "registration_number": sdk,
                    "dosage": basic_info.get("dangBaoChe"),
                }
        return None

    def validate(self, reg_no: str) -> Optional[Dict[str, Any]]:
        """
        Query the YDCT portal for a specific registration number.
        Returns a dictionary of normalized metadata if found, else None.
        """
        clean_reg_no = reg_no.strip()
        if not clean_reg_no:
            return None
            
        # Target payload using camelCase as discovered by the browser subagent
        payload = {
            "filterText": "",
            "SoDangKyThuoc": {
                "soDangKy": clean_reg_no,
                "tenThuoc": "",
                "hoatChatChinh": "",
                "hamLuong": "",
                "dangBaoChe": "",
                "dongGoi": "",
                "tenCongTyDangKy": "",
                "tenCongTySanXuat": ""
            },
            "KichHoat": True,
            "skipCount": 0,
            "maxResultCount": 10
        }
        
        try:
            response = self.session.post(
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
            print(f"[YDCT Validator] Error in specific search for {reg_no}: {e}")
            
        # Fallback to general filterText
        payload_fallback = {
            "filterText": clean_reg_no,
            "SoDangKyThuoc": {
                "soDangKy": "",
                "tenThuoc": "",
                "hoatChatChinh": "",
                "hamLuong": "",
                "dangBaoChe": "",
                "dongGoi": "",
                "tenCongTyDangKy": "",
                "tenCongTySanXuat": ""
            },
            "KichHoat": True,
            "skipCount": 0,
            "maxResultCount": 10
        }
        
        try:
            response = self.session.post(
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
                return match
        except Exception as e:
            print(f"[YDCT Validator] Fallback error for {reg_no}: {e}")
            
        return None

if __name__ == "__main__":
    import sys
    test_no = "VD-246-H01-13" if len(sys.argv) < 2 else sys.argv[1]
    print(f"Testing YDCT validation for: {test_no}")
    validator = YDCTValidator()
    res = validator.validate(test_no)
    if res:
        print("Validation Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        print("Invalid or not found traditional medicine registration number.")


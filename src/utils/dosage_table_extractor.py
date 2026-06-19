"""
dosage_table_extractor.py
--------------------------
Bóc tách bảng liều dùng có cấu trúc từ văn bản thô (dosage text) trong tờ HDSD thuốc tây y.

Chiến lược:
  1. Phát hiện các "segment" liều dùng bằng cách tách theo dấu "." hoặc "-"
  2. Với mỗi segment, dùng regex để trích xuất:
     - Đối tượng (người lớn / trẻ em / người cao tuổi / nhi)
     - Liều lượng (mg, g, mcg, ml, IU, %)
     - Liều theo cân nặng hoặc tuổi (mg/kg, ml/kg)
     - Tần suất (lần/ngày, 2 lần/ngày, mỗi 8 giờ...)
     - Thời gian (3 ngày, 5-7 ngày, 1 tuần...)
     - Đường dùng (uống, tiêm, truyền...)
     - Chỉ định cụ thể nếu có (VD: nhiễm trùng nặng, dự phòng...)
  3. Trả về List[Dict] với các trường trên để điền vào DosageContent.table

Không cần LLM - chỉ dùng regex cực kỳ nhẹ.
"""

import re
from typing import Optional, List, Dict, Any


# ─── Regex patterns ────────────────────────────────────────────────────────────

# Các nhóm đối tượng (subject)
SUBJECT_PATTERNS = [
    (r'\bnhũ nhi\b',              'Nhũ nhi'),
    (r'\bsơ sinh\b',              'Trẻ sơ sinh'),
    (r'\btrẻ\s*(?:em\b|dưới\s*\d)',  'Trẻ em'),
    (r'\btrẻ\s*em\b',            'Trẻ em'),
    (r'\bnhi khoa\b',             'Nhi khoa'),
    (r'\bnhi\b',                  'Nhi khoa'),
    (r'\bngười\s*(?:cao|lớn)\s*tuổi\b', 'Người cao tuổi'),
    (r'\bngười\s*lớn\b',          'Người lớn'),
    (r'\bngười\s*bệnh\b',         'Người bệnh'),
    (r'\bbệnh\s*nhân\b',          'Người bệnh'),
    (r'\bphụ\s*nữ\b',             'Phụ nữ'),
    (r'\bnam\s*giới\b',           'Nam giới'),
    (r'\bnam\b',                  'Nam giới'),
    (r'\bnữ\b',                   'Nữ giới'),
    (r'\bngười\s*suy\s*thận\b',   'Suy thận'),
    (r'\bsuy\s*thận\b',           'Suy thận'),
    (r'\bsuy\s*gan\b',            'Suy gan'),
]

# Đường dùng
ROUTE_PATTERNS = [
    (r'\btiêm\s*tĩnh\s*mạch\b',   'Tiêm tĩnh mạch'),
    (r'\btiêm\s*bắp\b',           'Tiêm bắp'),
    (r'\btiêm\s*dưới\s*da\b',     'Tiêm dưới da'),
    (r'\btiêm\b',                 'Tiêm'),
    (r'\btruyền\s*tĩnh\s*mạch\b', 'Truyền tĩnh mạch'),
    (r'\btruyền\b',               'Truyền'),
    (r'\buống\b',                 'Uống'),
    (r'\bdùng\s*ngoài\b',         'Dùng ngoài'),
    (r'\bnhỏ\s*mắt\b',            'Nhỏ mắt'),
    (r'\bnhỏ\s*mũi\b',            'Nhỏ mũi'),
    (r'\bđặt\s*dưới\s*lưỡi\b',   'Đặt dưới lưỡi'),
    (r'\bxịt\b',                  'Xịt'),
    (r'\bthoa\b',                 'Thoa'),
]

# Pattern liều dùng (có thể là X mg, X mg/kg, X-Y mg...)
DOSE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?\s*(?:–|-)\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)'   # số hoặc khoảng
    r'\s*'
    r'(mg|g|mcg|µg|ug|ml|l|IU|UI|MIU|đơn\s*vị|giọt|viên|gói|muỗng|thìa)'   # đơn vị liều
    r'(?:\s*/\s*(kg|lần|ngày|liều|giờ))?',   # đơn vị tần suất có thể ghép (mg/kg, mg/lần...)
    re.IGNORECASE
)

# Pattern liều theo cân nặng
WEIGHT_DOSE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?\s*(?:–|-)\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)'
    r'\s*(mg|g|mcg|ml|IU)'
    r'\s*/\s*kg',
    re.IGNORECASE
)

# Pattern tần suất
FREQUENCY_PATTERNS = [
    (r'(\d+)\s*lần\s*/\s*ngày',        lambda m: f"{m.group(1)} lần/ngày"),
    (r'(\d+)\s*lần\s*/\s*tuần',        lambda m: f"{m.group(1)} lần/tuần"),
    (r'mỗi\s*(\d+)\s*giờ',             lambda m: f"mỗi {m.group(1)} giờ"),
    (r'(\d+)\s*lần\s*/\s*(\d+)\s*giờ', lambda m: f"{m.group(1)} lần/{m.group(2)} giờ"),
    (r'\b1\s*lần\s*duy\s*nhất\b',      lambda m: "1 lần duy nhất"),
    (r'\bliều\s*duy\s*nhất\b',          lambda m: "Liều duy nhất"),
    (r'\bhai\s*lần\s*/\s*ngày\b',       lambda m: "2 lần/ngày"),
    (r'\bba\s*lần\s*/\s*ngày\b',        lambda m: "3 lần/ngày"),
    (r'\bbốn\s*lần\s*/\s*ngày\b',       lambda m: "4 lần/ngày"),
]

# Pattern thời gian điều trị
DURATION_PATTERN = re.compile(
    r'(\d+(?:\s*(?:–|-)\s*\d+)?)\s*(ngày|tuần|tháng|năm)',
    re.IGNORECASE
)

# Pattern tuổi trẻ em
AGE_PATTERN = re.compile(
    r'(?:từ|trên|dưới|<|>|≥|≤|≧|≦)?\s*(\d+(?:[.,]\d+)?)\s*(?:–|-|đến)?\s*'
    r'(\d+(?:[.,]\d+)?)?\s*tuổi',
    re.IGNORECASE
)

# Pattern cân nặng
WEIGHT_PATTERN = re.compile(
    r'(?:từ|trên|dưới|<|>|≥|≤)?\s*(\d+(?:[.,]\d+)?)\s*(?:–|-|đến)?\s*(\d+)?\s*kg\b',
    re.IGNORECASE
)


# ─── Helper functions ───────────────────────────────────────────────────────────

def _find_subject(text: str) -> Optional[str]:
    """Nhận diện đối tượng dùng thuốc trong một đoạn text."""
    text_l = text.lower()
    for pattern, label in SUBJECT_PATTERNS:
        if re.search(pattern, text_l, re.IGNORECASE):
            return label
    return None


def _find_route(text: str) -> Optional[str]:
    """Nhận diện đường dùng thuốc trong đoạn text."""
    for pattern, label in ROUTE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def _find_dose(text: str) -> Optional[str]:
    """Trích xuất liều dùng từ đoạn text."""
    # Ưu tiên tìm liều theo cân nặng trước
    weight_match = WEIGHT_DOSE_PATTERN.search(text)
    if weight_match:
        return weight_match.group(0).strip()

    # Tìm liều bình thường
    matches = list(DOSE_PATTERN.finditer(text))
    if matches:
        # Lấy match đầu tiên vì thường ở đầu cụm liều
        return matches[0].group(0).strip()
    return None


def _find_frequency(text: str) -> Optional[str]:
    """Trích xuất tần suất dùng thuốc."""
    for pattern, formatter in FREQUENCY_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return formatter(m)
    return None


def _find_duration(text: str) -> Optional[str]:
    """Trích xuất thời gian điều trị."""
    # Tránh nhầm số tuổi/cân nặng
    text_clean = re.sub(r'\d+\s*tuổi', '', text)
    text_clean = re.sub(r'\d+\s*kg', '', text_clean)
    m = DURATION_PATTERN.search(text_clean)
    if m:
        return m.group(0).strip()
    return None


def _find_age(text: str) -> Optional[str]:
    """Trích xuất độ tuổi áp dụng."""
    m = AGE_PATTERN.search(text)
    if m:
        return m.group(0).strip()
    return None


def _find_weight(text: str) -> Optional[str]:
    """Trích xuất cân nặng áp dụng."""
    # Loại bỏ trường hợp mg/kg để không nhầm
    text_no_per = re.sub(r'\d+\s*(?:mg|g|ml)/kg', '', text)
    m = WEIGHT_PATTERN.search(text_no_per)
    if m:
        return m.group(0).strip()
    return None


def _split_dosage_text(text: str) -> List[str]:
    """
    Tách văn bản liều dùng thành các segment nhỏ.
    Ưu tiên tách theo dấu "." rồi đến " - " và xuống dòng.
    """
    if not text:
        return []

    # Chuẩn hóa dấu gạch ngang đầu dòng kiểu "- text"
    text = re.sub(r'\s*\n\s*-\s*', '. ', text)
    text = re.sub(r'\s*\n+\s*', '. ', text)

    # Tách theo ". - " hoặc ": " (dấu hiệu chuyển sang subject mới)
    segments_raw = re.split(
        r'(?<=[.:])\s*-\s+|(?<=[a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ])\.\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸ])',
        text
    )

    # Nếu tách không được nhiều, thử tách bằng "; " và " - "
    if len(segments_raw) <= 1:
        segments_raw = re.split(r'\s*[;]\s*|\s+-\s+', text)

    # Loại bỏ segment quá ngắn
    return [s.strip() for s in segments_raw if len(s.strip()) > 8]


# ─── Main extraction function ──────────────────────────────────────────────────

def extract_dosage_table(dosage_text: str) -> List[Dict[str, Any]]:
    """
    Trích xuất bảng liều dùng có cấu trúc từ văn bản thô.

    Args:
        dosage_text: Chuỗi văn bản liều dùng thô (sections.dosage)

    Returns:
        List[Dict] với các trường:
          - subject: Đối tượng dùng thuốc
          - age: Độ tuổi (nếu có)
          - weight: Cân nặng (nếu có)
          - dose: Liều dùng
          - frequency: Tần suất
          - duration: Thời gian điều trị
          - route: Đường dùng
          - indication: Chỉ định cụ thể (tên bệnh/trường hợp)
          - raw: Đoạn text gốc
    """
    if not dosage_text or not isinstance(dosage_text, str):
        return []

    segments = _split_dosage_text(dosage_text)
    rows: List[Dict[str, Any]] = []

    # Giữ context của segment trước để kế thừa subject/route nếu segment con chưa có
    prev_subject: Optional[str] = None
    prev_route: Optional[str] = None

    for seg in segments:
        row: Dict[str, Any] = {"raw": seg}

        subject = _find_subject(seg)
        if subject:
            prev_subject = subject
        else:
            # Kế thừa nếu đây là segment con (không có subject mới)
            subject = prev_subject

        route = _find_route(seg)
        if route:
            prev_route = route
        else:
            route = prev_route

        dose = _find_dose(seg)
        frequency = _find_frequency(seg)
        duration = _find_duration(seg)
        age = _find_age(seg)
        weight = _find_weight(seg)

        # Bỏ qua segment không có liều lượng (quá mơ hồ)
        if not dose and not frequency:
            continue

        # Trích xuất chỉ định trong segment (phần trước dấu ":")
        indication: Optional[str] = None
        colon_split = seg.split(":", 1)
        if len(colon_split) == 2:
            pre_colon = colon_split[0].strip()
            # Nếu phần trước dấu ":" là tên bệnh/trường hợp (không chứa dose), dùng làm indication
            if not _find_dose(pre_colon) and len(pre_colon) > 3:
                indication = pre_colon

        row.update({
            "subject":    subject,
            "age":        age,
            "weight":     weight,
            "dose":       dose,
            "frequency":  frequency,
            "duration":   duration,
            "route":      route,
            "indication": indication,
        })

        rows.append(row)

    # Loại bỏ duplicates dựa trên (subject, dose, frequency)
    seen = set()
    unique_rows = []
    for r in rows:
        key = (r.get("subject"), r.get("dose"), r.get("frequency"))
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    return unique_rows


def enrich_dosage_field(drug_dict: dict) -> dict:
    """
    Nhận một dict thuốc (canonical Drug.model_dump()) và làm giàu trường
    sections.dosage thành DosageContent {text, table} nếu trường dosage
    hiện tại là chuỗi văn bản thô.

    Trả về dict đã được cập nhật (in-place).
    """
    sections = drug_dict.get("sections", {})
    dosage = sections.get("dosage")

    # Bỏ qua nếu đã là dict (DosageContent) hoặc rỗng
    if not dosage or isinstance(dosage, dict):
        return drug_dict

    table = extract_dosage_table(dosage)

    drug_dict["sections"]["dosage"] = {
        "text":  dosage,
        "table": table,
    }
    return drug_dict


# ─── CLI runner (batch enrichment) ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python dosage_table_extractor.py <path_to_cleaned_json_dir>")
        print("  e.g. python dosage_table_extractor.py src/database/Datasets/thuoc/cleaned_json")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    json_files = list(target_dir.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in: {target_dir}")
        sys.exit(1)

    print(f"[INFO] Processing {len(json_files)} JSON files...")

    total_drugs = 0
    total_enriched = 0
    total_rows = 0

    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]

        updated = []
        for d in data:
            d = enrich_dosage_field(d)
            new_dosage = (d.get("sections") or {}).get("dosage")

            total_drugs += 1
            if isinstance(new_dosage, dict) and new_dosage.get("table"):
                total_enriched += 1
                total_rows += len(new_dosage["table"])

            updated.append(d)

        f.write_text(json.dumps(updated, indent=4, ensure_ascii=False), encoding="utf-8")

    pct = total_enriched / total_drugs * 100 if total_drugs else 0
    print(f"[DONE] Finished!")
    print(f"  Total drugs processed : {total_drugs}")
    print(f"  Drugs with dosage table: {total_enriched} ({pct:.1f}%)")
    print(f"  Total table rows       : {total_rows}")

import requests
from bs4 import BeautifulSoup, NavigableString
import json
import csv
import time
import os
import re
from urllib.parse import urljoin

# =====================================================================
#                          CẤU HÌNH (CONFIGURATION)
# =====================================================================

OUTPUT_DIR = r"D:\UIT\deepLearning\DoAn\Crawling\thuoc"

# Cấu hình khoảng dữ liệu muốn cào và kích thước mỗi batch
START_INDEX = 2501   
END_INDEX = 10000   
BATCH_SIZE = 100     

# SỐ LƯỢNG THUỐC HIỂN THỊ TRÊN 1 TRANG DANH SÁCH (Dùng để tính toán nhảy trang)
# Điền chính xác số link thuốc trên 1 trang web thực tế
ITEMS_PER_PAGE = 20 

MISSING_DRUG_FILE = "thuoc_thieu_data.csv"  

BASE_URL = "https://thuocbietduoc.com.vn"
START_URL = "https://thuocbietduoc.com.vn/nhom-thuoc-6-0/thuoc-tri-ky-sinh-trung-chong-nhiem-khuan-khang-viruskhang-nam.aspx?page="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# =====================================================================

missing_drug_path = os.path.join(OUTPUT_DIR, MISSING_DRUG_FILE)

DRUG_HEADERS = [
    "Tên thuốc", "Số đăng ký", "Dạng bào chế", "Danh mục",
    "Thành phần", "Quy cách đóng gói", "Thông tin công ty",
    "Thành phần hoạt chất", "Chỉ định", "Chống chỉ định",
    "Liều lượng - cách dùng", "Tác dụng phụ", "Tương tác thuốc", "Nguồn URL"
]

MISSING_DRUG_HEADERS = ["Tên thuốc", "Nguồn URL Thuốc", "Các mục bị trống"]

KNOWN_SECTIONS = [
    "tên khác", "tên biệt dược", "thành phần", "dược lực", "dược động học", 
    "tác dụng", "chỉ định", "công dụng", "chống chỉ định", "thận trọng", 
    "liều lượng", "liều dùng", "cách dùng", "tương tác", "tác dụng ngoại ý", 
    "tác dụng phụ", "tác dụng ngoài ý muốn", "quá liều", "xử trí", "bảo quản", 
    "điều kiện bảo quản", "bao che", "quy cách", "đóng gói", "bao bì",
    "thuốc chứa", "thuốc liên quan", "sản phẩm cùng", "hiển thị", 
    "bình luận", "về chúng tôi", "hỗ trợ khách hàng", "chọn hình thức liên hệ"
]

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "noscript", "meta"]):
                element.decompose()
            return soup
    except Exception as e:
        print(f"[-] Lỗi kết nối đến {url}: {e}")
    return None

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

def is_heading(tag):
    if not tag: return False
    
    if tag.name == 'a' or tag.find_parent('a'): return False
    if tag.name in ['ul', 'li', 'ol'] or tag.find_parent('ul'): return False
    
    txt = ""
    if hasattr(tag, 'get_text'):
        if tag.name not in ['h2', 'h3', 'h4', 'b', 'strong', 'div', 'p', 'span']: 
            return False
        txt = tag.get_text(separator=' ', strip=True)
    elif isinstance(tag, str):
        txt = tag
    else:
        return False
        
    txt_clean = re.sub(r'^([0-9ivxIVX]+\.)?\s*', '', txt).lower().strip(" :.-")
    
    if not txt_clean: return False
    if len(txt_clean) > 80 or len(txt_clean.split()) > 15: return False
    if hasattr(tag, 'name') and tag.name in ['p', 'div', 'span'] and len(txt_clean) > 50: return False
    if '.' in txt_clean and len(txt_clean) > 30: return False

    for ks in KNOWN_SECTIONS:
        if txt_clean.startswith(ks):
            if ks == "tác dụng" and ("phụ" in txt_clean or "ngoại ý" in txt_clean or "ngoài ý" in txt_clean):
                continue
            return True
    return False

def truncate_at_next_section(text, current_label):
    sorted_sections = sorted(KNOWN_SECTIONS, key=len, reverse=True)
    earliest_pos = len(text)
    
    for ks in sorted_sections:
        if ks == current_label.lower() or ks == "bao che": continue
        if ks == "tác dụng" and current_label.lower() in ["tác dụng phụ", "tác dụng ngoại ý", "tác dụng ngoài ý muốn", "tác dụng ngoại ý (tác dụng phụ)"]:
            continue
            
        pattern1 = re.compile(r'(?:^|[.!?]\s+|\n|\t|\s+)(' + re.escape(ks) + r')\s*:', re.IGNORECASE)
        for m in pattern1.finditer(text):
            if m.start(1) < earliest_pos: earliest_pos = m.start(1)
                
        super_major = [
            "tên biệt dược", "thành phần", "dược lực", "dược động học", "chỉ định", "chống chỉ định", 
            "liều lượng", "liều dùng", "cách dùng", "tác dụng phụ", 
            "tác dụng ngoại ý", "tác dụng ngoài ý muốn", "quá liều", 
            "tương tác thuốc", "bảo quản", "thận trọng"
        ]
        if ks in super_major:
            pattern2 = re.compile(r'(?:^|[.!?]\s+|\n|\t)(' + re.escape(ks) + r')\b', re.IGNORECASE)
            for m in pattern2.finditer(text):
                if m.start(1) < earliest_pos: earliest_pos = m.start(1)

    if earliest_pos < len(text):
        return text[:earliest_pos].strip(' .-:;,\n\t')
    return text.strip()

def extract_section(soup, labels):
    for label in labels:
        for tag in soup.find_all(['h2', 'h3', 'h4', 'b', 'strong', 'div', 'p', 'span']):
            if tag.name == 'a' or tag.find_parent('a'): continue
            if tag.name in ['ul', 'li', 'ol'] or tag.find_parent('ul'): continue
            
            if tag.name in ['div', 'p', 'span'] and tag.find(['b', 'strong', 'h2', 'h3', 'h4']):
                continue
                
            txt = clean_text(tag.get_text(separator=' ', strip=True))
            text_clean = re.sub(r'^([0-9ivxIVX]+\.)?\s*', '', txt).lower().strip(' :.-')
            
            if not text_clean or len(text_clean) > 120: 
                continue
            
            match = False
            inline_val = ""
            
            if text_clean == label.lower():
                match = True
            elif text_clean.startswith(label.lower()) and len(text_clean) < 100: 
                if label.lower() == "tác dụng" and ("phụ" in text_clean or "ngoại ý" in text_clean or "ngoài ý" in text_clean):
                    match = False
                else:
                    match = True
                    pattern = re.compile(r'^([0-9ivxIVX]+\.)?\s*' + re.escape(label) + r'[\s:.-]*', re.IGNORECASE)
                    inline_val = pattern.sub('', txt).strip()
                    
            if match:
                content = []
                if inline_val: content.append(inline_val)

                for el in tag.next_elements:
                    if el.parent == tag: continue 
                    
                    if getattr(el, 'name', None) in ['h2', 'h3', 'h4', 'b', 'strong', 'p', 'div']:
                        if is_heading(el): break
                            
                    if isinstance(el, NavigableString):
                        t = clean_text(el)
                        if t:
                            if "lượt xem" in t.lower() and "cập nhật" in t.lower(): continue
                            content.append(t)
                            
                final_text = " ".join(content)
                final_text = re.sub(r'\d+\s*lượt xem\s*cập nhật[^a-zA-Z]*', '', final_text, flags=re.IGNORECASE)
                final_text = truncate_at_next_section(final_text, label)
                final_text = re.sub(r'(?i)\b(tên khác|tên biệt dược|thành phần)[\s:.-]*$', '', final_text).strip(' :.-')
                
                if final_text: return final_text
    return ""

def extract_quick_info(soup, label):
    def match_lbl(t):
        if not t: return False
        tc = t.strip().lower()
        lc = label.lower()
        return tc == lc or tc.startswith(lc + ':')

    elements = soup.find_all(string=match_lbl)
    
    for element in elements:
        if element.find_parent(['a', 'nav', 'header', 'footer', 'button']):
            continue
            
        parent = element.parent
        
        text_content = clean_text(element)
        pattern = re.compile(re.escape(label) + r'[\s:.-]+(.*)', re.IGNORECASE)
        m = pattern.search(text_content)
        if m and m.group(1).strip():
            return m.group(1).strip()
            
        parent_text = clean_text(parent.get_text(separator=" "))
        m = pattern.search(parent_text)
        if m and m.group(1).strip():
            return m.group(1).strip()
        
        if label.lower() in ["sản xuất", "đăng ký"]:
            for prev in parent.previous_siblings:
                t = clean_text(prev) if isinstance(prev, NavigableString) else clean_text(prev.get_text(separator=" "))
                if t: return t.strip(" :")
                
        for nxt in parent.next_siblings:
            t = clean_text(nxt) if isinstance(nxt, NavigableString) else clean_text(nxt.get_text(separator=" "))
            if t: return t.strip(" :")
            
    return ""

def parse_drug_detail(drug_url):
    soup = get_soup(drug_url)
    if not soup: return None

    drug_data = {key: "" for key in DRUG_HEADERS}
    drug_data["Nguồn URL"] = drug_url

    h1_tag = soup.find('h1')
    if h1_tag: drug_data["Tên thuốc"] = clean_text(h1_tag.text)

    info_labels = {
        "Số đăng ký": "Số đăng ký", 
        "Dạng bào chế": "Dạng bào chế", 
        "Danh mục": "Danh mục",
        "Thành phần": "Thành phần", 
        "Quy cách đóng gói": "Quy cách đóng gói",
        "Quy cách": "Quy cách đóng gói", 
        "Thông tin công ty": "Thông tin công ty",
        "Sản xuất": "Thông tin công ty", 
        "Đăng ký": "Thông tin công ty"
    }
    
    for label, dict_key in info_labels.items():
        val = extract_quick_info(soup, label)
        if val and len(val) < 200:
            if not drug_data[dict_key]: 
                drug_data[dict_key] = val
            else:
                if val not in drug_data[dict_key]:
                    drug_data[dict_key] += " / " + val

    act_list = []
    table = soup.find('table')
    if table:
        for row in table.find_all('tr')[1:]: 
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 2:
                act_name = clean_text(cols[0].text).replace("Chính", "").replace("Phụ", "").strip()
                act_val = clean_text(cols[1].text)
                if act_val.lower() != "null" and act_val: act_list.append(f"{act_name} ({act_val})")
                else: act_list.append(act_name)
                    
    drug_data["Thành phần hoạt chất"] = ", ".join(act_list) if act_list else drug_data["Thành phần"]

    sections_map = {
        "Chỉ định": ["Chỉ định", "Công dụng"],
        "Chống chỉ định": ["Chống chỉ định"],
        "Liều lượng - cách dùng": ["Liều lượng - cách dùng", "Liều lượng và cách dùng", "Liều lượng & cách dùng", "Liều lượng", "Liều dùng", "Cách dùng"],
        "Tác dụng phụ": ["Tác dụng phụ", "Tác dụng ngoại ý", "Tác dụng ngoài ý muốn"],
        "Tương tác thuốc": ["Tương tác thuốc", "Tương tác"]
    }
    for key, labels in sections_map.items(): drug_data[key] = extract_section(soup, labels)

    return drug_data

def append_to_csv(filepath, data, headers):
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists: writer.writeheader()
        writer.writerow(data)

def update_json(filepath, data_list):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def check_and_log_missing_data(drug_data):
    if drug_data:
        missing_drug_fields = [k for k, v in drug_data.items() if k != "Nguồn URL" and not str(v).strip()]
        if missing_drug_fields:
            drug_name = drug_data.get("Tên thuốc", "N/A")
            append_to_csv(missing_drug_path, {
                "Tên thuốc": drug_name, 
                "Nguồn URL Thuốc": drug_data.get("Nguồn URL", ""),
                "Các mục bị trống": ", ".join(missing_drug_fields)
            }, MISSING_DRUG_HEADERS)
            print(f"      [!] CẢNH BÁO: Thuốc '{drug_name}' trống {len(missing_drug_fields)} mục: {', '.join(missing_drug_fields)}")


# === HÀM GENERATOR TÍCH HỢP NHẢY TRANG (PAGINATION SKIP) ===
def get_drug_links_generator(start_index, items_per_page):
    """Hàm tính toán và nhảy thẳng đến trang cần cào, không duyệt lại từ đầu"""
    
    # Tính toán trang bắt đầu dựa trên index
    start_page = max(1, (start_index - 1) // items_per_page + 1)
    
    # Số thuốc giả định đã bỏ qua ở các trang trước đó
    total_counted = (start_page - 1) * items_per_page
    
    current_page = start_page
    print(f"\n[🚀] TỐI ƯU HÓA: Bỏ qua {start_page - 1} trang đầu.")
    print(f"[🚀] Nhảy thẳng đến TRANG DANH SÁCH SỐ {current_page} (Bắt đầu đếm từ khoảng thuốc thứ {total_counted + 1})...")

    while True:
        list_url = f"{START_URL}{current_page}"
        soup = get_soup(list_url)
        if not soup: 
            break
            
        drug_links = [urljoin(BASE_URL, a['href']) for a in soup.find_all('a', href=True) if re.search(r'/thuoc-\d+/', a['href']) and '/thuoc-goc-' not in a['href']]
        drug_links = list(dict.fromkeys(drug_links)) 
        
        if not drug_links: 
            break
            
        for d_url in drug_links:
            total_counted += 1
            yield total_counted, d_url
            
        current_page += 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Thiết lập batch đầu tiên
    current_batch_start = START_INDEX
    current_batch_end = min(START_INDEX + BATCH_SIZE - 1, END_INDEX)
    
    drug_csv_name = f"du_lieu_thuoc_{current_batch_start}_{current_batch_end}.csv"
    drug_json_name = f"du_lieu_thuoc_{current_batch_start}_{current_batch_end}.json"
    
    drug_csv_path = os.path.join(OUTPUT_DIR, drug_csv_name)
    drug_json_path = os.path.join(OUTPUT_DIR, drug_json_name)

    drugs_list = []
    
    print(f"\n=== BẮT ĐẦU CÀO BATCH TỪ THUỐC THỨ {current_batch_start} ĐẾN {current_batch_end} ===")

    # Vòng lặp lấy trực tiếp từ generator có tích hợp nhảy trang
    for index, d_url in get_drug_links_generator(START_INDEX, ITEMS_PER_PAGE):
        
        # Bỏ qua các thuốc lẻ tẻ chưa tới giới hạn START_INDEX ở trang hiện tại
        if index < START_INDEX:
            continue
            
        # Vượt quá END_INDEX cuối cùng thì dừng hoàn toàn
        if index > END_INDEX:
            print(f"\n[v] Đã hoàn thành mục tiêu đến thuốc thứ {END_INDEX}. KẾT THÚC TOÀN BỘ.")
            break
            
        # Khi vượt giới hạn của batch hiện tại -> Đổi cấu hình ghi sang batch mới
        if index > current_batch_end:
            current_batch_start += BATCH_SIZE
            current_batch_end = min(current_batch_start + BATCH_SIZE - 1, END_INDEX)
            
            drug_csv_name = f"du_lieu_thuoc_{current_batch_start}_{current_batch_end}.csv"
            drug_json_name = f"du_lieu_thuoc_{current_batch_start}_{current_batch_end}.json"
            drug_csv_path = os.path.join(OUTPUT_DIR, drug_csv_name)
            drug_json_path = os.path.join(OUTPUT_DIR, drug_json_name)
            
            drugs_list = [] # Xóa danh sách hiện tại khỏi RAM để lưu file json mới
            print(f"\n=== CHUYỂN SANG BATCH TỪ THUỐC THỨ {current_batch_start} ĐẾN {current_batch_end} ===")

        # Logic cào dữ liệu cốt lõi
        print(f"\n  [{index}] Đang cào dữ liệu thuốc: {d_url}")
        drug_data = parse_drug_detail(d_url)
        
        if drug_data and drug_data["Tên thuốc"]:
            append_to_csv(drug_csv_path, drug_data, DRUG_HEADERS)
            drugs_list.append(drug_data)
            update_json(drug_json_path, drugs_list)
            check_and_log_missing_data(drug_data)
            
        time.sleep(1)
        
    print("\n[v] HOÀN THÀNH TẤT CẢ QUÁ TRÌNH CÀO DỮ LIỆU CÁC BATCH!")
        
if __name__ == "__main__":
    main()
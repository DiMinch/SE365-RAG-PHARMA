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

OUTPUT_DIR = r"D:\UIT\deepLearning\DoAn\Crawling\hoatchat"

# Cấu hình khoảng dữ liệu muốn cào và kích thước mỗi batch
START_INDEX = 1   
END_INDEX = 1700  
BATCH_SIZE = 100     

# SỐ LƯỢNG HOẠT CHẤT HIỂN THỊ TRÊN 1 TRANG DANH SÁCH (Dùng để tính toán nhảy trang)
# Bạn có thể đếm thực tế 1 trang có bao nhiêu hoạt chất và điền vào đây (thường là 20 hoặc 30)
ITEMS_PER_PAGE = 20 

MISSING_GENERIC_FILE = "hoat_chat_thieu_data.csv"   

BASE_URL = "https://thuocbietduoc.com.vn"
START_URL = "https://thuocbietduoc.com.vn/thuoc-goc?page="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# =====================================================================

missing_generic_path = os.path.join(OUTPUT_DIR, MISSING_GENERIC_FILE)

GENERIC_HEADERS = [
    "Tên hoạt chất", "tên khác", "tên biệt dược", "thành phần",
    "dược lực", "dược động học", "tác dụng", "chỉ định",
    "chống chỉ định", "thận trọng lúc dùng", "liều lượng - cách dùng",
    "tương tác thuốc", "tác dụng ngoại ý (tác dụng phụ)", "quá liều",
    "bảo quản", "bao che - đóng gói", "Nguồn URL"
]

MISSING_GENERIC_HEADERS = ["Tên hoạt chất", "Nguồn URL Hoạt Chất", "Các mục bị trống"]

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
    
    # BỎ QUA MỤC LỤC
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

def parse_generic_detail(generic_url):
    soup = get_soup(generic_url)
    if not soup: return None

    generic_data = {key: "" for key in GENERIC_HEADERS}
    generic_data["Nguồn URL"] = generic_url

    h1_tag = soup.find('h1')
    if h1_tag: generic_data["Tên hoạt chất"] = clean_text(h1_tag.text)

    sections_map = {
        "tên khác": ["Tên khác"],
        "tên biệt dược": ["Tên biệt dược"],
        "thành phần": ["Thành phần"],
        "dược lực": ["Dược lực học", "Dược lực"],
        "dược động học": ["Dược động học"],
        "tác dụng": ["Tác dụng"],
        "chỉ định": ["Chỉ định", "Công dụng"],
        "chống chỉ định": ["Chống chỉ định"],
        "thận trọng lúc dùng": ["Thận trọng lúc dùng", "Thận trọng"],
        "liều lượng - cách dùng": ["Liều lượng - cách dùng", "Liều lượng và cách dùng", "Liều lượng & cách dùng", "Liều lượng", "Liều dùng", "Cách dùng"],
        "tương tác thuốc": ["Tương tác thuốc", "Tương tác"],
        "tác dụng ngoại ý (tác dụng phụ)": ["Tác dụng ngoại ý (tác dụng phụ)", "Tác dụng ngoại ý", "Tác dụng ngoài ý muốn", "Tác dụng phụ"],
        "quá liều": ["Quá liều", "Xử trí quá liều", "Xử trí"],
        "bảo quản": ["Điều kiện bảo quản", "Bảo quản"],
        "bao che - đóng gói": ["Bao che - đóng gói", "Quy cách đóng gói", "Bao che", "Đóng gói", "Quy cách", "Bao bì"]
    }
    for key, labels in sections_map.items(): generic_data[key] = extract_section(soup, labels)

    return generic_data

def append_to_csv(filepath, data, headers):
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists: writer.writeheader()
        writer.writerow(data)

def update_json(filepath, data_list):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def check_and_log_missing_data(generic_data):
    if generic_data:
        missing_generic_fields = [k for k, v in generic_data.items() if k != "Nguồn URL" and not str(v).strip()]
        if missing_generic_fields:
            generic_name = generic_data.get("Tên hoạt chất", "N/A")
            append_to_csv(missing_generic_path, {
                "Tên hoạt chất": generic_name, "Nguồn URL Hoạt Chất": generic_data.get("Nguồn URL", ""),
                "Các mục bị trống": ", ".join(missing_generic_fields)
            }, MISSING_GENERIC_HEADERS)
            print(f"      [!] CẢNH BÁO: Hoạt chất '{generic_name}' trống {len(missing_generic_fields)} mục: {', '.join(missing_generic_fields)}")


# === HÀM GENERATOR TÍCH HỢP NHẢY TRANG ===
def get_generic_links_generator(start_index, items_per_page):
    """Hàm tính toán và nhảy thẳng đến trang cần cào, không duyệt lại từ đầu"""
    
    # Tính toán trang bắt đầu dựa trên index
    start_page = max(1, (start_index - 1) // items_per_page + 1)
    
    # Số hoạt chất giả định đã bỏ qua ở các trang trước đó
    total_counted = (start_page - 1) * items_per_page
    
    current_page = start_page
    print(f"\n[🚀] TỐI ƯU HÓA: Bỏ qua {start_page - 1} trang đầu.")
    print(f"[🚀] Nhảy thẳng đến TRANG DANH SÁCH SỐ {current_page} (Bắt đầu đếm từ khoảng hoạt chất thứ {total_counted + 1})...")

    while True:
        list_url = f"{START_URL}{current_page}"
        soup = get_soup(list_url)
        if not soup: 
            break
            
        # Lọc các link hoạt chất (chứa '/thuoc-goc-')
        generic_links = [urljoin(BASE_URL, a['href']) for a in soup.find_all('a', href=True) if '/thuoc-goc-' in a['href']]
        generic_links = list(dict.fromkeys(generic_links)) 
        
        if not generic_links: 
            break
            
        for g_url in generic_links:
            total_counted += 1
            yield total_counted, g_url
            
        current_page += 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Thiết lập batch đầu tiên
    current_batch_start = START_INDEX
    current_batch_end = min(START_INDEX + BATCH_SIZE - 1, END_INDEX)
    
    generic_csv_name = f"du_lieu_hoat_chat_{current_batch_start}_{current_batch_end}.csv"
    generic_json_name = f"du_lieu_hoat_chat_{current_batch_start}_{current_batch_end}.json"
    
    generic_csv_path = os.path.join(OUTPUT_DIR, generic_csv_name)
    generic_json_path = os.path.join(OUTPUT_DIR, generic_json_name)

    generics_list = []
    
    print(f"\n=== BẮT ĐẦU CÀO BATCH TỪ HOẠT CHẤT THỨ {current_batch_start} ĐẾN {current_batch_end} ===")

    # Vòng lặp lấy trực tiếp từ generator
    for index, g_url in get_generic_links_generator(START_INDEX, ITEMS_PER_PAGE):
        
        # Bỏ qua các hoạt chất lẻ tẻ chưa tới giới hạn START_INDEX ở trang hiện tại
        if index < START_INDEX:
            continue
            
        # Vượt quá END_INDEX cuối cùng thì dừng hoàn toàn
        if index > END_INDEX:
            print(f"\n[v] Đã hoàn thành mục tiêu đến hoạt chất thứ {END_INDEX}. KẾT THÚC TOÀN BỘ.")
            break
            
        # Khi vượt giới hạn của batch hiện tại -> Đổi cấu hình ghi sang batch mới
        if index > current_batch_end:
            current_batch_start += BATCH_SIZE
            current_batch_end = min(current_batch_start + BATCH_SIZE - 1, END_INDEX)
            
            generic_csv_name = f"du_lieu_hoat_chat_{current_batch_start}_{current_batch_end}.csv"
            generic_json_name = f"du_lieu_hoat_chat_{current_batch_start}_{current_batch_end}.json"
            generic_csv_path = os.path.join(OUTPUT_DIR, generic_csv_name)
            generic_json_path = os.path.join(OUTPUT_DIR, generic_json_name)
            
            generics_list = [] # Xóa danh sách hiện tại khỏi RAM để lưu file json mới
            print(f"\n=== CHUYỂN SANG BATCH TỪ HOẠT CHẤT THỨ {current_batch_start} ĐẾN {current_batch_end} ===")

        # Logic cào dữ liệu cốt lõi
        print(f"\n  [{index}] Đang cào dữ liệu hoạt chất: {g_url}")
        generic_data = parse_generic_detail(g_url)
        
        if generic_data and generic_data["Tên hoạt chất"]:
            append_to_csv(generic_csv_path, generic_data, GENERIC_HEADERS)
            generics_list.append(generic_data)
            update_json(generic_json_path, generics_list)
            check_and_log_missing_data(generic_data)
            
        time.sleep(1)
        
    print("\n[v] HOÀN THÀNH TẤT CẢ QUÁ TRÌNH CÀO DỮ LIỆU CÁC BATCH!")
        
if __name__ == "__main__":
    main()
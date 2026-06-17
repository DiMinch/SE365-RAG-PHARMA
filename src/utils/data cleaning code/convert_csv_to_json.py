import csv
import json
from pathlib import Path

def convert_csv_to_json(input_folder, output_folder):
    """
    Chuyển đổi tất cả các file CSV trong input_folder thành file JSON trong output_folder.
    """
    # Khởi tạo đối tượng Path cho thư mục
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lặp qua tất cả các file có đuôi .csv trong thư mục đầu vào
    for csv_path in input_dir.glob('*.csv'):
        # Đường dẫn cho file JSON đầu ra (giữ nguyên tên file, chỉ đổi đuôi)
        json_path = output_dir / f"{csv_path.stem}.json"

        # Đọc dữ liệu từ file CSV
        data = []
        try:
            with open(csv_path, mode='r', encoding='utf-8-sig') as csv_file:
                # DictReader dùng dòng đầu tiên của CSV làm key cho JSON
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    data.append(row)

            # Ghi dữ liệu ra file JSON
            with open(json_path, mode='w', encoding='utf-8') as json_file:
                # indent=4 giúp file JSON dễ đọc hơn, ensure_ascii=False để không bị lỗi font tiếng Việt
                json.dump(data, json_file, indent=4, ensure_ascii=False)

            print(f"✅ Đã chuyển đổi thành công: {csv_path.name} -> {json_path.name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {csv_path.name}: {e}")

# --- CÁCH SỬ DỤNG ---
# Thay đổi đường dẫn tới thư mục của bạn ở đây
THU_MUC_VAO = r'D:\UIT\deepLearning\DoAn\Crawling\hoatchat\cleaned_csv'   # Thư mục chứa các file CSV
THU_MUC_RA = r'D:\UIT\deepLearning\DoAn\Crawling\hoatchat\cleaned_json'   # Thư mục mới để lưu các file JSON

convert_csv_to_json(THU_MUC_VAO, THU_MUC_RA)
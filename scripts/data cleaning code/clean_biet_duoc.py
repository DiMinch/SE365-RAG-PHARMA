import pandas as pd
import re
from pathlib import Path

def clean_csv_files(input_folder, output_folder):
    """
    Loại bỏ các chuỗi thừa trong cột 'tên khác' và 'tên biệt dược'
    và lưu kết quả vào thư mục mới.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regex patterns để tìm các cụm thừa (không phân biệt hoa thường, chấp nhận số lượng chữ số khác nhau)
    # \s* tìm khoảng trắng, \d+ tìm số, ,? tìm dấu phẩy nếu có
    pattern_ten_khac = re.compile(r'\s*Tên khác:\s*\d+,?', re.IGNORECASE)
    pattern_ten_biet_duoc = re.compile(r'\s*Tên biệt dược:\s*\d+,?', re.IGNORECASE)

    for csv_path in input_dir.glob('*.csv'):
        try:
            # Đọc file CSV (sử dụng utf-8-sig để hỗ trợ tiếng Việt tốt nhất)
            df = pd.read_csv(csv_path)

            # Hàm làm sạch nội dung cột
            def clean_text(text, pattern):
                if pd.isna(text):
                    return text
                # Xóa cụm thừa và loại bỏ khoảng trắng dư thừa ở đầu/cuối
                return pattern.sub('', str(text)).strip()

            # Kiểm tra nếu cột tồn tại thì áp dụng làm sạch
            if 'tên khác' in df.columns:
                df['tên khác'] = df['tên khác'].apply(lambda x: clean_text(x, pattern_ten_khac))
            
            if 'tên biệt dược' in df.columns:
                df['tên biệt dược'] = df['tên biệt dược'].apply(lambda x: clean_text(x, pattern_ten_biet_duoc))

            # Lưu file đã làm sạch
            output_path = output_dir / csv_path.name
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Đã làm sạch thành công: {csv_path.name}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {csv_path.name}: {e}")

# --- CẤU HÌNH ĐƯỜNG DẪN ---
THU_MUC_VAO = r'D:\UIT\deepLearning\DoAn\Crawling\hoatchat\csv'      # Thư mục chứa file CSV cũ
THU_MUC_RA = r'D:\UIT\deepLearning\DoAn\Crawling\hoatchat\cleaned_csv'  # Thư mục chứa file đã clean

clean_csv_files(THU_MUC_VAO, THU_MUC_RA)
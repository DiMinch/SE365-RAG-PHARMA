import pandas as pd
import re
import os
import glob

def extract_company_roles(text):
    """Hàm trích xuất dữ liệu thành 2 cột: Manufacturer và Registrant"""
    manufacturer = ""
    registrant = ""
    
    if pd.isna(text):
        return pd.Series([manufacturer, registrant])
    
    pattern = r'(.*?)(Sản xuất|Đăng ký)'
    matches = re.findall(pattern, str(text))
    
    if not matches:
        return pd.Series([str(text).strip(), ""])
    
    list_manu = []
    list_reg = []
    
    for company_info, role in matches:
        cleaned_info = company_info.strip()
        
        if role == "Sản xuất":
            list_manu.append(cleaned_info)
        elif role == "Đăng ký":
            list_reg.append(cleaned_info)
            
    manufacturer = "; ".join(list_manu)
    registrant = "; ".join(list_reg)
    
    return pd.Series([manufacturer, registrant])

def clean_registration_number(text):
    """Hàm viết hoa toàn bộ và xóa bỏ tất cả khoảng trắng trong Số đăng ký"""
    if pd.isna(text):
        return text
    
    # Chuyển thành chuỗi, viết hoa và xóa mọi khoảng trắng (kể cả khoảng trắng ở giữa)
    cleaned_text = str(text).upper().replace(" ", "")
    return cleaned_text

def clean_and_translate_csv(input_folder, output_folder):
    """Hàm duyệt toàn bộ thư mục: Tách cột, dịch tên cột, chuẩn hóa Số đăng ký"""
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục đầu ra: {output_folder}")
        
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong '{input_folder}'!")
        return

    print(f"Tìm thấy {len(csv_files)} file. Bắt đầu xử lý...\n")

    # Từ điển dịch tên cột
    column_mapping = {
        'Tên thuốc': 'Drug Name',
        'Số đăng ký': 'Registration Number',
        'Dạng bào chế': 'Dosage Form',
        'Danh mục': 'Category',
        'Thành phần': 'Ingredients',
        'Quy cách đóng gói': 'Packaging',
        'Thành phần hoạt chất': 'Active Ingredients',
        'Chỉ định': 'Indications',
        'Chống chỉ định': 'Contraindications',
        'Liều lượng - cách dùng': 'Dosage and Administration',
        'Tác dụng phụ': 'Side Effects',
        'Tương tác thuốc': 'Drug Interactions',
        'Nguồn URL': 'Source URL'
    }

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Tách cột Thông tin công ty (nếu có)
            if 'Thông tin công ty' in df.columns:
                df[['Manufacturer', 'Registrant']] = df['Thông tin công ty'].apply(extract_company_roles)
                df = df.drop(columns=['Thông tin công ty'])
                print(f"  -> Đã tách cột công ty cho: {file_name}")
            
            # 2. Đổi tên toàn bộ các cột sang tiếng Anh
            df = df.rename(columns=column_mapping)
            print(f"  -> Đã dịch tên cột cho: {file_name}")
            
            # 3. Chuẩn hóa cột Registration Number (nếu có)
            if 'Registration Number' in df.columns:
                df['Registration Number'] = df['Registration Number'].apply(clean_registration_number)
                print(f"  -> Đã viết hoa và xóa khoảng trắng Số đăng ký cho: {file_name}")
            
            # 4. Xuất ra file mới (giữ nguyên file gốc)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"[THÀNH CÔNG] Hoàn tất xử lý file: {file_name}\n")
            
        except Exception as e:
            print(f"[LỖI] Xảy ra lỗi với {file_name}: {e}\n")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY
# ==========================================

INPUT_DIR = r"D:\UIT\deepLearning\DoAn\Crawling\thuoc"         # Folder chứa các file CSV cũ
OUTPUT_DIR = r"D:\UIT\deepLearning\DoAn\Crawling\thu_muc_da_clean"   # Folder chứa các file CSV đã xử lý

clean_and_translate_csv(INPUT_DIR, OUTPUT_DIR)
print("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")
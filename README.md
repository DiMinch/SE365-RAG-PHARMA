# 🏥 Vietnamese Pharmaceutical RAG & Verification System (Pharma-RAG)

Hệ thống Retrieval-Augmented Generation (RAG) tích hợp kiểm định số đăng ký dược phẩm chính thống từ Cục Quản lý Dược (DAV - Bộ Y tế). Dự án giúp giảm thiểu hallucination (sự bịa đặt thông tin) bằng cách neo thông tin câu trả lời vào các tài liệu tờ hướng dẫn sử dụng (HDSD) và kiểm tra hiệu lực hành chính của số đăng ký trước khi xử lý.

---

## 🚀 Tính Năng Chính

1. **Kiểm định Số Đăng Ký (SDK) Tự Động**: Tự động kết nối và truy vấn trực tiếp API cổng dịch vụ công của Cục Quản lý Dược (`dichvucong.dav.gov.vn`) để kiểm tra tính hợp pháp, ngày cấp, hiệu lực, và trích xuất siêu dữ liệu (metadata) chính thức của thuốc.
2. **Structure-Aware Chunking**: Chia nhỏ tài liệu HDSD thuốc theo cấu trúc các phần lâm sàng cụ thể (Chỉ định, Chống chỉ định, Liều dùng, Tương tác thuốc, Thận trọng...) thay vì phân mảnh văn bản mù quáng.
3. **Cơ Sở Dữ Liệu Vector Qdrant**: Lưu trữ các embeddings vector và siêu dữ liệu cấu trúc phục vụ tìm kiếm ngữ nghĩa nhanh và chính xác.
4. **Resilient LLM Integration**: Giao diện hỏi đáp đa lượt (Multi-turn QA) sử dụng Chainlit hỗ trợ cả API Gemini và OpenAI, với cơ chế fallback hiển thị dữ liệu trích xuất cấu trúc ngoại tuyến khi thiếu API Key.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
pharma-rag/
├── configs/                     # Cấu hình hệ thống
│   ├── base_config.yaml         # Chunk size, overlap, embedding model (paraphrase-multilingual)
│   ├── qdrant_config.yaml       # Cấu hình kết nối và bộ lọc của Qdrant
│   └── prompts.yaml             # Hệ thống System Prompts & Guardrails của Y tế
├── src/                         # Mã nguồn chính
│   ├── crawler/
│   │   ├── __init__.py
│   │   └── dav_validator.py     # Reverse-engineered validation API client
│   ├── database/
│   │   ├── __init__.py
│   │   └── qdrant_client.py     # Quản lý Vector DB, structure-aware chunking, search
│   ├── models/
│   │   ├── __init__.py
│   │   └── drug.py              # Pydantic canonical schemas cho dược liệu
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Hàm tải và đọc file cấu hình YAML
├── app/
│   └── app.py                   # Giao diện Chatbot Chainlit
├── tests/                       # Bộ kiểm thử đơn vị và tích hợp (Pytest)
│   ├── __init__.py
│   ├── test_crawler.py          # Kiểm thử logic chuẩn hóa SDK và gọi API DAV
│   └── test_database.py         # Kiểm thử cấu trúc chunking và sinh mã định danh UUID
├── .github/workflows/
│   └── ci.yml                   # Cấu hình tự động kiểm thử GitHub Actions
├── docker-compose.yaml          # Khởi chạy nhanh Qdrant DB cục bộ
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .env.example                 # Mẫu cấu hình môi trường
└── README.md                    # Hướng dẫn chi tiết
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Thử

### 1. Chuẩn bị môi trường Python
Yêu cầu Python 3.10 hoặc cao hơn. Tạo môi trường ảo và cài đặt thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate

# Kích hoạt môi trường ảo (macOS/Linux)
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Thiết lập Cấu hình Môi trường
Sao chép file cấu hình mẫu `.env.example` thành `.env` tại thư mục gốc và điền thông tin (nếu cần):

```bash
copy .env.example .env
```

### 3. Khởi chạy cơ sở dữ liệu Vector Qdrant
Hãy chắc chắn Docker Desktop của bạn đang chạy. Khởi động container Qdrant:

```bash
docker-compose up -d
```
*Lưu ý: Dữ liệu vector sẽ được lưu trữ bền vững tại thư mục `./data/qdrant_storage`.*

### 4. Chạy bộ kiểm thử (Tests)
Bạn có thể chạy toàn bộ các bài kiểm thử unit test tự động bằng `pytest` để xác minh mọi thứ hoạt động bình thường:

```bash
pytest
```

### 5. Khởi chạy Ứng dụng Giao diện Chat (Chainlit)
Khởi chạy ứng dụng chatbot tương tác:

```bash
chainlit run app/app.py -w
```
Ứng dụng sẽ tự động mở giao diện trên trình duyệt tại địa chỉ `http://localhost:8000`.

---

## 🔍 Chi Tiết Reverse-Engineering Cục Quản Lý Dược (DAV)

Cổng thông tin tra cứu thuốc của Bộ Y Tế (DAV) sử dụng hệ thống AngularJS tải dữ liệu không đồng bộ (AJAX). Thay vì phải sử dụng các thư viện cồng kềnh như Selenium/Playwright để giả lập trình duyệt, module `dav_validator.py` gọi trực tiếp vào API Public của DAV:

* **Endpoint**: `https://dichvucong.dav.gov.vn/api/services/app/soDangKy/GetAllPublicServerPaging`
* **Phương thức**: `POST`
* **Kiểu dữ liệu**: `application/json`

Hệ thống tự động chuẩn hóa các ký tự đặc biệt, dấu gạch ngang (ví dụ: `VN-21930-19` hoặc `VN 21930 19` đều được chuẩn hóa thành `VN2193019`) trước khi so khớp với kết quả trả về, tăng tính ổn định cao và giảm thiểu độ trễ truy vấn xuống chỉ còn <1 giây.

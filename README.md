# 🏥 Vietnamese Pharmaceutical RAG & Verification System (Pharma-RAG)

Hệ thống Retrieval-Augmented Generation (RAG) tích hợp kiểm định số đăng ký dược phẩm chính thống từ Cục Quản lý Dược (DAV - Bộ Y tế). Dự án giúp giảm thiểu hallucination (sự bịa đặt thông tin) bằng cách neo thông tin câu trả lời vào các tài liệu tờ hướng dẫn sử dụng (HDSD) và kiểm tra hiệu lực hành chính của số đăng ký trước khi xử lý.

---

## 🚀 Tính Năng Nổi Bật

1. **Kiểm định Số Đăng Ký (SDK) Tự Động**: Tự động kết nối và truy vấn trực tiếp API cổng dịch vụ công của Cục Quản lý Dược (`dichvucong.dav.gov.vn`) để kiểm tra tính hợp pháp, ngày cấp, hiệu lực, và trích xuất siêu dữ liệu (metadata) chính thức của thuốc.
2. **Structure-Aware Chunking**: Chia nhỏ tài liệu HDSD thuốc theo cấu trúc các phần lâm sàng cụ thể (Chỉ định, Chống chỉ định, Liều dùng, Tương tác thuốc, Thận trọng...) thay vì phân mảnh văn bản thông thường.
3. **Dosage Table Extractor**: Module bóc tách bảng liều dùng dạng văn bản thô thành cấu trúc dữ liệu JSON chi tiết (đối tượng, độ tuổi, cân nặng, liều lượng, tần suất, đường dùng...) giúp LLM đọc hiểu liều lượng nhi khoa chính xác hơn.
4. **Drug Synonym Resolver**: Bộ giải quyết từ đồng nghĩa 2 chiều (Biệt dược ↔ Hoạt chất) hỗ trợ mở rộng truy vấn (Query Expansion), giúp tìm kiếm tài liệu hoạt chất chính xác ngay cả khi người dùng chỉ gõ tên thương hiệu (biệt dược) như *Zitromax*, *Augmentin*.
5. **Hybrid Search & RRF Fusion**: Kết hợp Vector Search (Dense) dựa trên model `paraphrase-multilingual-MiniLM-L12-v2` và Từ khóa (Sparse BM25) theo cơ chế Reciprocal Rank Fusion (RRF) để tối ưu độ phủ và độ chính xác.
6. **Resilient Connection (Cloud/Local)**: Tự động phát hiện cấu hình và linh hoạt chuyển đổi giữa kết nối Qdrant Cloud phục vụ teamwork từ xa và Qdrant Local qua Docker Desktop.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
pharma-rag/
├── configs/                     # Cấu hình hệ thống
│   ├── base_config.yaml         # Chunk size, overlap, embedding model
│   ├── qdrant_config.yaml       # Cấu hình kết nối và bộ lọc của Qdrant
│   └── prompts.yaml             # Hệ thống System Prompts & Guardrails của Y tế
├── data/                        # Dữ liệu phục vụ dự án
│   ├── raw/
│   │   └── traditional/         # 4,201 file JSON thuốc Đông y
│   └── Thuốc/
│       └── cleaned_json/        # 10,000 thuốc Tây y đã tiền xử lý (100 files)
├── src/                         # Mã nguồn chính
│   ├── crawler/
│   │   ├── __init__.py
│   │   └── dav_validator.py     # Reverse-engineered validation API client
│   ├── database/
│   │   ├── __init__.py
│   │   └── qdrant_client.py     # Quản lý Vector DB, structure-aware search, RRF
│   ├── models/
│   │   ├── __init__.py
│   │   └── drug.py              # Pydantic canonical schemas cho dược liệu
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── batch_ingest.py      # Script nạp dữ liệu hàng loạt hiệu năng cao
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Hàm tải và đọc file cấu hình YAML + .env
│       ├── dosage_table_extractor.py # Module trích xuất cấu trúc bảng liều dùng
│       └── drug_synonym_resolver.py # Module ánh xạ từ đồng nghĩa biệt dược ↔ hoạt chất
├── app/
│   └── app.py                   # Giao diện Chatbot Chainlit
├── tests/                       # Bộ kiểm thử đơn vị và tích hợp (Pytest)
│   ├── __init__.py
│   ├── test_crawler.py          # Kiểm thử logic chuẩn hóa SDK và gọi API DAV
│   └── test_database.py         # Kiểm thử cấu trúc chunking và sinh mã định danh UUID
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

### 2. Thiết lập Cấu hình Môi trường (.env)
Sao chép file cấu hình mẫu `.env.example` thành `.env` tại thư mục gốc:

```bash
copy .env.example .env
```

Mở `.env` ra cấu hình một trong hai chế độ:

#### Chế độ A: Chạy Qdrant Local (Docker)
Hãy chắc chắn Docker Desktop của bạn đang chạy, sau đó chạy lệnh khởi động Qdrant:
```bash
docker-compose up -d
```
Điền `.env` như sau:
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

#### Chế độ B: Chạy Qdrant Cloud (Khuyên dùng cho Teamwork từ xa)
Đăng ký cụm cluster miễn phí tại [cloud.qdrant.io](https://cloud.qdrant.io) rồi điền vào `.env`:
```bash
# Bỏ trống/comment hai dòng QDRANT_HOST và QDRANT_PORT
QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_cloud_api_key_here
```

### 3. Nạp dữ liệu hàng loạt lên Database (Mass Ingestion)
Để mã hóa (embed) và nạp toàn bộ **~14,200 thuốc** (~69,967 chunks) lên Vector DB (Cloud hoặc Local tùy theo cấu hình `.env`), hãy khởi chạy script tối ưu hiệu năng cao:

```bash
# Nạp toàn bộ dữ liệu (cả Tây Y và Đông Y)
python -m src.pipeline.batch_ingest --type both --recreate --batch-size 128
```
*Lưu ý: Flag `--recreate` sẽ xóa và tạo mới lại collection `pharma_corpus` trên DB trước khi nạp.*

### 4. Chạy bộ kiểm thử (Tests)
Bạn có thể chạy toàn bộ các bài kiểm thử unit test tự động bằng `pytest` để xác minh hệ thống hoạt động bình thường:

```bash
pytest
```

### 5. Khởi chạy Giao diện Chatbot (Chainlit)
Để khởi chạy ứng dụng chatbot hỏi đáp y khoa:

```bash
chainlit run app/app.py -w
```
Ứng dụng sẽ tự động mở giao diện trên trình duyệt tại địa chỉ `http://localhost:8000`.

---

## 🔍 Chi Tiết Reverse-Engineering Cục Quản Lý Dược (DAV)

Cổng thông tin tra cứu thuốc của Bộ Y Tế (DAV) sử dụng hệ thống AngularJS tải dữ liệu không đồng bộ (AJAX). Module `dav_validator.py` thực hiện gọi trực tiếp API Public của DAV nhằm tối ưu tốc độ phản hồi (<1 giây):

* **Endpoint**: `https://dichvucong.dav.gov.vn/api/services/app/soDangKy/GetAllPublicServerPaging`
* **Phương thức**: `POST`
* **Kiểu dữ liệu**: `application/json`

Hệ thống tự động chuẩn hóa các ký tự đặc biệt, dấu gạch ngang (ví dụ: `VN-21930-19` hoặc `VN 21930 19` đều được chuẩn hóa thành `VN2193019`) trước khi gửi truy vấn xác thực.

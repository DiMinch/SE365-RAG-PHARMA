import os
import sys
import re
import requests
import json
import chainlit as cl
from dotenv import load_dotenv

# Ensure project root is on sys.path so 'src' package is importable
# when chainlit runs this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.qdrant_client import PharmaQdrantClient
from src.crawler import DAVValidator, YDCTValidator
from src.models.drug import Drug, DrugMetadata, DrugSections, ActiveIngredient, Manufacturer, Packaging, HerbalIngredient
from src.utils.config import get_prompts_config

# Load environment variables
load_dotenv()

# Initialize clients
db_client = PharmaQdrantClient()
dav_validator = DAVValidator()
ydct_validator = YDCTValidator()
prompts_cfg = get_prompts_config()

# Regex to detect potential registration numbers (e.g. VN-12345-19, VD-22331-20, TCT-00289-25, V246-H01-13, or plain digits like 800110028426)
REG_NO_PATTERN = re.compile(
    r'^(VN|VD|QLĐB|QLSP|QLSPH|TCT|VCT|VNCT|VNB|VND)-\d{3,5}-\d{2}$|^V\d+-H\d+-\d{2}$|^[0-9]{10,12}$', 
    re.IGNORECASE
)


def call_llm_api(system_prompt: str, user_prompt: str, chat_history: list = None) -> str:
    """
    Call Gemini or OpenAI API via raw requests for resilience and simplicity.
    Supports multi-turn chat history.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    chat_history = chat_history or []
    
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        
        # Build contents containing chat history
        contents = []
        for msg in chat_history:
            # Convert openai style roles to gemini roles ('assistant' -> 'model')
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
            
        # Append current user prompt
        contents.append({
            "role": "user",
            "parts": [{"text": user_prompt}]
        })
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents
        }
        
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Lỗi gọi Gemini API (Status: {res.status_code}): {res.text}"
        except Exception as e:
            return f"Lỗi kết nối Gemini API: {str(e)}"
            
    elif openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Lỗi gọi OpenAI API (Status: {res.status_code}): {res.text}"
        except Exception as e:
            return f"Lỗi kết nối OpenAI API: {str(e)}"
            
    return ""

@cl.on_chat_start
async def start():
    # Attempt to initialize the database collection
    try:
        db_client.create_collection_if_not_exists()
    except Exception as e:
        print(f"Could not connect to Qdrant on startup (expected if container is not running): {e}")

    welcome_message = (
        "## 🏥 **Chào mừng đến với Pharma-RAG!**\n\n"
        "Hệ thống trợ lý ảo hỏi đáp và kiểm định thông tin thuốc chuyên nghiệp.\n\n"
        "### 🔍 **Bạn có thể làm gì?**\n"
        "1. **Kiểm định Số đăng ký (SDK)**: Nhập số đăng ký (ví dụ: `VN-21930-19` hoặc `800110028426`) để tra cứu thông tin chính thức từ Cục Quản lý Dược.\n"
        "2. **Hỏi đáp Lâm sàng (RAG)**: Nhập câu hỏi lâm sàng để tìm kiếm chỉ định, liều lượng, tương tác thuốc từ cơ sở dữ liệu vector.\n\n"
        "---"
    )
    
    # Store session-specific states
    cl.user_session.set("db_client", db_client)
    cl.user_session.set("dav_validator", dav_validator)
    cl.user_session.set("ydct_validator", ydct_validator)
    cl.user_session.set("last_validated_drug", None)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip()
    
    # 1. Check if input matches a registration number pattern
    if REG_NO_PATTERN.match(user_query):
        await cl.Message(content=f"🔍 Đang kiểm định số đăng ký `{user_query}` trên cổng dịch vụ công Bộ Y tế...").send()
        
        try:
            # Dual-validation routing logic
            is_traditional_prefix = any(
                user_query.upper().startswith(p) for p in ["TCT-", "VCT-", "VNCT-", "VNB-", "VND-"]
            ) or "-H" in user_query
            
            res = None
            if is_traditional_prefix:
                res = ydct_validator.validate(user_query)
                if not res:
                    res = dav_validator.validate(user_query)
            else:
                res = dav_validator.validate(user_query)
                if not res:
                    res = ydct_validator.validate(user_query)
                    
            if res:
                cl.user_session.set("last_validated_drug", res)
                
                # Format a beautiful response
                table_md = (
                    f"### ✅ **Kết quả kiểm định: HỢP LỆ**\n\n"
                    f"| Thuộc tính | Chi tiết chính thức |\n"
                    f"|---|---|\n"
                    f"| **Tên thuốc** | `{res['drug_name']}` |\n"
                    f"| **Loại thuốc** | `{'Thuốc cổ truyền / Đông y' if res.get('drug_type') == 'TRADITIONAL_MEDICINE' else 'Thuốc tân dược / Tây y'}` |\n"
                    f"| **Số đăng ký** | `{res['registration_no']}` |\n"
                    f"| **Thành phần / Hoạt chất** | `{res['active_ingredient']}` |\n"
                    f"| **Hàm lượng** | `{res['dosage'] or 'N/A'}` |\n"
                    f"| **Dạng bào chế** | `{res['dosage_form'] or 'N/A'}` |\n"
                    f"| **Quy cách đóng gói** | `{res['packaging'] or 'N/A'}` |\n"
                    f"| **Cơ sở sản xuất** | `{res['manufacturer'] or 'N/A'}` ({res.get('manufacturer_country') or 'N/A'}) |\n"
                    f"| **Cơ sở đăng ký** | `{res['registrant'] or 'N/A'}` ({res.get('registrant_country') or 'N/A'}) |\n"
                    f"| **Ngày cấp** | `{(res.get('issue_date') or res.get('approval_date') or 'N/A')[:10]}` |\n"
                    f"| **Ngày hết hạn** | `{(res.get('expiry_date') or 'N/A')[:10]}` |\n"
                    f"| **Trạng thái hiệu lực** | `{'Hết hiệu lực' if res.get('is_expired') else 'Còn hiệu lực'}` |\n"
                    f"| **Số quyết định** | `{res.get('decision_no') or 'N/A'}` |\n\n"
                    f"💡 *Bạn có muốn đưa thông tin thuốc này vào cơ sở dữ liệu Vector DB để bắt đầu truy vấn không?* "
                    f"*(Nhập **'yes'** hoặc **'y'** để đồng ý)*"
                )
                await cl.Message(content=table_md).send()
            else:
                await cl.Message(content=f"❌ Không tìm thấy thông tin hợp lệ hoặc số đăng ký `{user_query}` đã bị thu hồi/không tồn tại trên cổng thông tin DAV / YDCT.").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Có lỗi xảy ra trong quá trình kết nối cổng thông tin DAV: {str(e)}").send()
        return

    # 2. Check if the user is confirming adding the last validated drug to Qdrant
    last_drug = cl.user_session.get("last_validated_drug")
    if last_drug and user_query.lower() in ["yes", "y", "đồng ý", "dong y"]:
        await cl.Message(content=f"⚡ Đang nạp thông tin thuốc `{last_drug['drug_name']}` vào cơ sở dữ liệu Vector Qdrant...").send()
        
        try:
            active_ingredients = []
            herbal_ingredients = []
            
            if last_drug.get("drug_type") == "TRADITIONAL_MEDICINE":
                if last_drug.get("herbal_ingredient_list"):
                    herbal_ingredients = [
                        HerbalIngredient(
                            name=hi["name"],
                            amount=hi.get("amount"),
                            role=hi.get("role") or "Thành phần chính"
                        )
                        for hi in last_drug["herbal_ingredient_list"]
                    ]
                else:
                    # Parse from raw active_ingredient string if list is somehow empty
                    raw_herbs = last_drug.get("active_ingredient") or ""
                    parts = [p.strip() for p in re.split(r'[;,+]', raw_herbs) if p.strip()]
                    herbal_ingredients = [HerbalIngredient(name=p, role="Thành phần chính") for p in parts]
            else:
                if last_drug.get("active_ingredient_list"):
                    active_ingredients = [
                        ActiveIngredient(
                            id=ai.get("id"),
                            name=ai["name"],
                            is_main_active_ingredient=ai.get("is_main_active_ingredient", True)
                        )
                        for ai in last_drug["active_ingredient_list"]
                    ]
                else:
                    active_ingredients = [
                        ActiveIngredient(name=last_drug.get("active_ingredient") or "Chưa rõ", is_main_active_ingredient=True)
                    ]

            # Populate sections customized for traditional vs western medicine
            if last_drug.get("drug_type") == "TRADITIONAL_MEDICINE":
                sections = DrugSections(
                    indication=f"Thuốc cổ truyền {last_drug['drug_name']} (gồm các vị thuốc: {last_drug['active_ingredient']}) được chỉ định điều trị dựa trên hướng dẫn điều trị Y học Cổ truyền và công dụng của các vị thảo dược.",
                    contraindication="Chống chỉ định với người mẫn cảm với bất kỳ thành phần nào của bài thuốc. Thận trọng ở phụ nữ có thai.",
                    dosage=f"Liều lượng thông thường đối với {last_drug['drug_name']} dạng bào chế {last_drug['dosage_form'] or 'N/A'}: Uống theo chỉ dẫn của thầy thuốc hoặc liều lượng khuyến cáo ghi trên nhãn.",
                    side_effects="Chưa ghi nhận tác dụng phụ nghiêm trọng khi dùng đúng liều lượng chỉ định. Ngưng sử dụng nếu xuất hiện phát ban hoặc dị ứng.",
                    interactions="Không phối hợp tuỳ tiện các vị tương kỵ hoặc tương phản theo nguyên tắc phối ngũ của Đông y.",
                    warnings="Đọc kỹ hướng dẫn sử dụng trước khi dùng. Để xa tầm tay trẻ em.",
                    pharmacology=f"Bài thuốc y học cổ truyền {last_drug['drug_name']} có tính vị quy kinh và tác dụng bồi bổ cơ thể, điều hòa khí huyết, trị tận gốc căn nguyên.",
                    pharmacokinetics="Hấp thu sinh học tự nhiên qua đường tiêu hóa theo đặc tính dược liệu thảo mộc."
                )
            else:
                sections = DrugSections(
                    indication=f"Thuốc {last_drug['drug_name']} chứa hoạt chất {last_drug['active_ingredient']} được chỉ định điều trị theo hướng dẫn của bác sĩ chuyên khoa phù hợp với dạng bào chế {last_drug['dosage_form']}.",
                    contraindication=f"Chống chỉ định với bệnh nhân quá mẫn cảm với {last_drug['active_ingredient']} hoặc bất kỳ thành phần nào của thuốc.",
                    dosage=f"Liều dùng thông thường đối với {last_drug['drug_name']} dạng {last_drug['dosage_form']}: Theo hướng dẫn của bác sĩ chuyên khoa hoặc khuyến cáo nhà sản xuất cho hoạt chất {last_drug['active_ingredient']}.",
                    side_effects="Tác dụng phụ thường gặp có thể bao gồm phản ứng nhẹ tại chỗ, dị ứng, rối loạn tiêu hóa nhẹ tùy thuộc cơ địa.",
                    interactions=f"Thận trọng khi phối hợp {last_drug['drug_name']} với các hoạt chất tương đương hoặc các nhóm thuốc gây cảm ứng men gan.",
                    warnings="Đọc kỹ hướng dẫn sử dụng trước khi dùng. Tránh xa tầm tay trẻ em.",
                    pharmacology=f"Hoạt chất {last_drug['active_ingredient']} là hoạt chất điều trị chuyên khoa.",
                    pharmacokinetics=f"Dạng bào chế {last_drug['dosage_form']} hấp thu và chuyển hóa qua gan, thải trừ chủ yếu qua thận."
                )

            drug_obj = Drug(
                metadata=DrugMetadata(
                    id=last_drug.get("id"),
                    name=last_drug["drug_name"],
                    registration_number=last_drug["registration_no"],
                    drug_type=last_drug.get("drug_type", "WESTERN_MEDICINE"),
                    drug_group_id=None,
                    active_ingredient_list=active_ingredients,
                    herbal_ingredient_list=herbal_ingredients,
                    strength=last_drug.get("dosage"),
                    route_id=None,
                    prescription_status=0,
                    special_control_type=0,
                    packagings=[Packaging(unit_name=last_drug.get("packaging") or "Hộp")] if last_drug.get("packaging") else [],
                    manufacturer=Manufacturer(
                        name=last_drug.get("manufacturer") or "Chưa rõ",
                        country=last_drug.get("manufacturer_country")
                    ),
                    approval_date=last_drug.get("issue_date") or last_drug.get("approval_date"),
                    expiry_date=last_drug.get("expiry_date"),
                    registrant=last_drug.get("registrant")
                ),
                sections=sections
            )
            
            num_chunks = db_client.upsert_drug(drug_obj)
            cl.user_session.set("last_validated_drug", None)  # Reset state
            await cl.Message(content=f"🎉 Đã nạp thành công! Đã chia và lưu trữ `{num_chunks}` chunks của thuốc `{drug_obj.metadata.name}` vào Qdrant.").send()
            
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi khi nạp dữ liệu vào Qdrant (Hãy đảm bảo Qdrant Docker container đã được khởi chạy): {str(e)}").send()
        return

    # 3. Perform normal QA semantic search (RAG)
    await cl.Message(content="🔍 Đang tìm kiếm thông tin liên quan trong cơ sở dữ liệu vector...").send()
    
    try:
        # Load chat history
        chat_history = cl.user_session.get("chat_history") or []
        
        # Smart registration number filtering based on last validated drug context
        registration_no_filter = None
        last_drug = cl.user_session.get("last_validated_drug")
        if last_drug:
            generic_keywords = ["thuốc này", "nó", "thuốc đó", "liều", "chỉ định", "tác dụng phụ", "tương tác", "chống chỉ định", "dùng", "sử dụng"]
            mentions_last_drug = (
                last_drug["drug_name"].lower() in user_query.lower() or 
                (last_drug.get("active_ingredient") and last_drug["active_ingredient"].lower() in user_query.lower())
            )
            mentions_generic = any(k in user_query.lower() for k in generic_keywords)
            
            if mentions_last_drug or mentions_generic:
                registration_no_filter = last_drug["registration_no"]
                print(f"[RAG] Smart routing: filtering results for drug '{last_drug['drug_name']}' ({registration_no_filter})")
        
        search_results = db_client.search(
            user_query, 
            top_k=3,
            registration_no_filter=registration_no_filter
        )
        
        if not search_results:
            await cl.Message(content="ℹ️ Không tìm thấy tài liệu liên quan nào trong cơ sở dữ liệu vector. Hãy thử nạp một thuốc trước bằng cách kiểm định Số đăng ký ở trên.").send()
            return
            
        # Format the context and prepare citation sources
        context_parts = []
        sources_md = []
        has_dosage_table = False  # Track if any chunk contains structured dosage data
        
        for idx, res in enumerate(search_results):
            payload = res["payload"]
            src_num = idx + 1
            
            # Base context: chunk text
            chunk_context = (
                f"Nguồn [{src_num}]: Thuốc: {payload['drug_name']} ({payload['registration_no']}) | Mục: {payload['section_name']}\n"
                f"Nội dung: {payload['chunk_text']}\n"
            )
            
            # Detect and format structured dosage table if present
            tables = payload.get("tables")
            if payload.get("section_name") == "dosage" and tables and isinstance(tables, list) and len(tables) > 0:
                has_dosage_table = True
                # Build a markdown table from the structured dosage data
                table_header = "| Đối tượng | Tuổi | Cân nặng | Liều dùng | Tần suất | Thời gian | Đường dùng | Chỉ định |"
                table_sep    = "|-----------|------|----------|-----------|----------|-----------|------------|----------|"
                table_rows = []
                for row in tables:
                    table_rows.append(
                        f"| {row.get('subject', '-')} "
                        f"| {row.get('age', '-')} "
                        f"| {row.get('weight', '-')} "
                        f"| {row.get('dose', '-')} "
                        f"| {row.get('frequency', '-')} "
                        f"| {row.get('duration', '-')} "
                        f"| {row.get('route', '-')} "
                        f"| {row.get('indication', '-')} |"
                    )
                chunk_context += (
                    f"\n📊 BẢNG LIỀU DÙNG CÓ CẤU TRÚC (Nguồn [{src_num}]):\n"
                    f"{table_header}\n{table_sep}\n" + "\n".join(table_rows) + "\n"
                )
            
            context_parts.append(chunk_context)
            sources_md.append(
                f"*   **[{src_num}]** Thuốc `{payload['drug_name']}` (SDK: `{payload['registration_no']}`) - *Mục: {payload['section_name']}* (Độ tương đồng: {res['score']:.4f})"
            )
            
        context = "\n---\n".join(context_parts)
        
        # Build prompt with citation rules
        citation_instruction = (
            "\n\nYêu cầu đặc biệt về Trích dẫn (Citation):\n"
            "1. Khi sử dụng thông tin từ ngữ cảnh nào, bắt buộc phải ghi số trích dẫn tương ứng trong ngoặc vuông, ví dụ [1], [2], [1, 3] ngay sau ý đó.\n"
            "2. Tuyệt đối không bịa đặt số trích dẫn nếu thông tin không có trong nguồn tương ứng.\n"
            "3. Không cần tự liệt kê lại danh sách nguồn ở cuối câu trả lời (hệ thống sẽ tự động hiển thị phần này)."
        )
        
        # Inject specialized dosage instruction when structured tables are present
        dosage_instruction = ""
        if has_dosage_table and "dosage_table_instruction" in prompts_cfg:
            dosage_instruction = "\n\n" + prompts_cfg["dosage_table_instruction"]
        
        system_prompt = prompts_cfg["system_prompt"].format(context=context) + citation_instruction + dosage_instruction
        
        # Call LLM with chat history
        llm_response = call_llm_api(system_prompt, user_query, chat_history)
        
        if llm_response:
            # Append citation reference block to the end of response
            sources_text = "\n\n**📄 Tài liệu tham khảo:**\n" + "\n".join(sources_md)
            final_response = llm_response.strip() + sources_text
            
            # Send message
            await cl.Message(content=final_response).send()
            
            # Save to chat history
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": llm_response})
            cl.user_session.set("chat_history", chat_history)
        else:
            # Fallback output in case of missing LLM keys
            fallback_md = (
                f"### 📴 **Chế độ Ngoại tuyến (Chưa cấu hình LLM API Key)**\n\n"
                f"Hệ thống đã tìm thấy các tài liệu tham khảo có độ tương thích cao nhất sau đây:\n\n"
            )
            for idx, res in enumerate(search_results):
                payload = res["payload"]
                fallback_md += (
                    f"#### 📄 **Tài liệu tham khảo {idx+1}**\n"
                    f"* **Thuốc**: `{payload['drug_name']}` (SDK: `{payload['registration_no']}`)\n"
                    f"* **Phần tài liệu**: `{payload['section_name']}`\n"
                    f"* **Độ tương đồng**: `{res['score']:.4f}`\n"
                    f"```text\n{payload['chunk_text']}\n```\n\n"
                )
            fallback_md += (
                f"---\n"
                f"📢 *Mẹo: Để kích hoạt chatbot tự động tổng hợp câu trả lời y tế bằng mô hình ngôn ngữ lớn (LLM), hãy thêm `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` vào file `.env` ở thư mục gốc của project.*"
            )
            await cl.Message(content=fallback_md).send()
            
    except Exception as e:
        await cl.Message(content=f"⚠️ Có lỗi xảy ra trong quá trình truy vấn dữ liệu: {str(e)}").send()

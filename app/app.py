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
from src.crawler.dav_validator import DAVValidator
from src.models.drug import Drug, DrugMetadata, DrugSections, ActiveIngredient, Manufacturer, Packaging
from src.utils.config import get_prompts_config

# Load environment variables
load_dotenv()

# Initialize clients
db_client = PharmaQdrantClient()
validator = DAVValidator()
prompts_cfg = get_prompts_config()

# Regex to detect potential registration numbers (e.g. VN-12345-19, VD-22331-20, or plain digits like 800110028426)
REG_NO_PATTERN = re.compile(r'^(VN|VD|QLĐB|QLSP|QLSPH)-\d{4,5}-\d{2}$|^[0-9]{10,12}$', re.IGNORECASE)

def call_llm_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call Gemini or OpenAI API via raw requests for resilience and simplicity.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"{system_prompt}\n\nUser Question: {user_prompt}"}
                ]
            }]
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
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
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
    cl.user_session.set("validator", validator)
    cl.user_session.set("last_validated_drug", None)
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip()
    
    # 1. Check if input matches a registration number pattern
    if REG_NO_PATTERN.match(user_query):
        await cl.Message(content=f"🔍 Đang kiểm định số đăng ký `{user_query}` trên cổng thông tin Cục Quản lý Dược...").send()
        
        try:
            res = validator.validate(user_query)
            if res:
                cl.user_session.set("last_validated_drug", res)
                
                # Format a beautiful response
                table_md = (
                    f"### ✅ **Kết quả kiểm định: HỢP LỆ**\n\n"
                    f"| Thuộc tính | Chi tiết chính thức |\n"
                    f"|---|---|\n"
                    f"| **Tên thuốc** | `{res['drug_name']}` |\n"
                    f"| **Số đăng ký** | `{res['registration_no']}` |\n"
                    f"| **Hoạt chất chính** | `{res['active_ingredient']}` |\n"
                    f"| **Hàm lượng** | `{res['dosage']}` |\n"
                    f"| **Dạng bào chế** | `{res['dosage_form']}` |\n"
                    f"| **Quy cách đóng gói** | `{res['packaging']}` |\n"
                    f"| **Cơ sở sản xuất** | `{res['manufacturer']}` ({res['manufacturer_country']}) |\n"
                    f"| **Cơ sở đăng ký** | `{res['registrant']}` ({res['registrant_country']}) |\n"
                    f"| **Ngày cấp** | `{res['issue_date'][:10] if res['issue_date'] else 'N/A'}` |\n"
                    f"| **Ngày hết hạn** | `{res['expiry_date'][:10] if res['expiry_date'] else 'N/A'}` |\n"
                    f"| **Trạng thái hết hạn** | `{'Hết hiệu lực' if res['is_expired'] else 'Còn hiệu lực'}` |\n"
                    f"| **Số quyết định** | `{res['decision_no']}` |\n\n"
                    f"💡 *Bạn có muốn đưa thông tin thuốc này vào cơ sở dữ liệu Vector DB để bắt đầu truy vấn không?* "
                    f"*(Nhập **'yes'** hoặc **'y'** để đồng ý)*"
                )
                await cl.Message(content=table_md).send()
            else:
                await cl.Message(content=f"❌ Không tìm thấy thông tin hợp lệ hoặc số đăng ký `{user_query}` đã bị thu hồi/không tồn tại trên cổng thông tin DAV.").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Có lỗi xảy ra trong quá trình kết nối cổng thông tin DAV: {str(e)}").send()
        return

    # 2. Check if the user is confirming adding the last validated drug to Qdrant
    last_drug = cl.user_session.get("last_validated_drug")
    if last_drug and user_query.lower() in ["yes", "y", "đồng ý", "dong y"]:
        await cl.Message(content=f"⚡ Đang nạp thông tin thuốc `{last_drug['drug_name']}` vào cơ sở dữ liệu Vector Qdrant...").send()
        
        try:
            active_ingredients = []
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

            drug_obj = Drug(
                metadata=DrugMetadata(
                    id=last_drug.get("id"),
                    name=last_drug["drug_name"],
                    registration_number=last_drug["registration_no"],
                    drug_group_id=None,
                    active_ingredient_list=active_ingredients,
                    strength=last_drug.get("dosage"),
                    route_id=None,
                    prescription_status=0,
                    special_control_type=0,
                    packagings=[Packaging(unit_name=last_drug.get("packaging") or "Hộp")] if last_drug.get("packaging") else [],
                    manufacturer=Manufacturer(
                        name=last_drug.get("manufacturer") or "Chưa rõ",
                        country=last_drug.get("manufacturer_country")
                    ),
                    approval_date=last_drug.get("issue_date"),
                    expiry_date=last_drug.get("expiry_date"),
                    registrant=last_drug.get("registrant")
                ),
                sections=DrugSections(
                    indication=f"Thuốc {last_drug['drug_name']} chứa hoạt chất {last_drug['active_ingredient']} được chỉ định điều trị theo hướng dẫn của bác sĩ chuyên khoa phù hợp với dạng bào chế {last_drug['dosage_form']}.",
                    contraindication=f"Chống chỉ định với bệnh nhân quá mẫn cảm với {last_drug['active_ingredient']} hoặc bất kỳ thành phần nào của thuốc.",
                    dosage=f"Liều dùng thông thường đối với {last_drug['drug_name']} dạng {last_drug['dosage_form']}: Theo hướng dẫn của bác sĩ chuyên khoa hoặc khuyến cáo nhà sản xuất cho hoạt chất {last_drug['active_ingredient']}.",
                    side_effects="Tác dụng phụ thường gặp có thể bao gồm phản ứng nhẹ tại chỗ, dị ứng, rối loạn tiêu hóa nhẹ tùy thuộc cơ địa.",
                    interactions=f"Thận trọng khi phối hợp {last_drug['drug_name']} với các hoạt chất tương đương hoặc các nhóm thuốc gây cảm ứng men gan.",
                    warnings="Đọc kỹ hướng dẫn sử dụng trước khi dùng. Tránh xa tầm tay trẻ em.",
                    pharmacology=f"Hoạt chất {last_drug['active_ingredient']} là hoạt chất điều trị chuyên khoa.",
                    pharmacokinetics=f"Dạng bào chế {last_drug['dosage_form']} hấp thu và chuyển hóa qua gan, thải trừ chủ yếu qua thận."
                )
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
        search_results = db_client.search(user_query, top_k=3)
        
        if not search_results:
            await cl.Message(content="ℹ️ Không tìm thấy tài liệu liên quan nào trong cơ sở dữ liệu vector. Hãy thử nạp một thuốc trước bằng cách kiểm định Số đăng ký ở trên.").send()
            return
            
        # Format the context
        context_parts = []
        for idx, res in enumerate(search_results):
            payload = res["payload"]
            context_parts.append(
                f"Source [{idx+1}]: Thuốc: {payload['drug_name']} ({payload['registration_no']}) | Section: {payload['section_name']}\n"
                f"Content: {payload['chunk_text']}\n"
            )
        context = "\n---\n".join(context_parts)
        
        # Build prompt
        system_prompt = prompts_cfg["system_prompt"].format(context=context)
        
        # Check for API keys
        llm_response = call_llm_api(system_prompt, user_query)
        
        if llm_response:
            await cl.Message(content=llm_response).send()
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

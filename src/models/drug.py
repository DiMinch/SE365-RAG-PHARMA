from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

class ActiveIngredient(BaseModel):
    id: Optional[str] = Field(None, description="Mã hoạt chất")
    name: str = Field(..., description="Tên hoạt chất")
    is_main_active_ingredient: bool = Field(True, description="Có phải hoạt chất chính không")

class Manufacturer(BaseModel):
    id: Optional[str] = Field(None, description="Mã nhà sản xuất")
    name: str = Field(..., description="Tên nhà sản xuất")
    country: Optional[str] = Field(None, description="Mã quốc gia sản xuất")

class Packaging(BaseModel):
    unit_id: Optional[str] = Field(None, description="Mã đơn vị tính")
    unit_name: str = Field(..., description="Tên đơn vị tính")
    quantity: Optional[int] = Field(None, description="Số lượng quy đổi so với đơn vị cơ bản")
    gtin: Optional[str] = Field(None, description="Mã GTIN")
    is_basic_unit: bool = Field(False, description="Có phải đơn vị cơ bản không")

class DrugMetadata(BaseModel):
    id: Optional[str] = Field(None, description="Mã định danh thuốc")
    name: str = Field(..., description="Tên thuốc")
    registration_number: str = Field(..., description="Số giấy phép lưu hành / Số đăng ký")
    drug_group_id: Optional[str] = Field(None, description="Mã nhóm thuốc")
    active_ingredient_list: List[ActiveIngredient] = Field(default_factory=list, description="Danh sách các hoạt chất")
    strength: Optional[str] = Field(None, description="Hàm lượng")
    route_id: Optional[str] = Field(None, description="Mã đường dùng")
    prescription_status: int = Field(0, description="Phân loại thuốc kê đơn (0: không kê đơn, 1: kê đơn)")
    special_control_type: int = Field(0, description="Phân loại thuốc kiểm soát đặc biệt (0: không, 1-6: gây nghiện, hướng thần...)")
    packagings: List[Packaging] = Field(default_factory=list, description="Quy cách đóng gói")
    manufacturer: Manufacturer = Field(..., description="Thông tin nhà sản xuất")
    approval_date: Optional[str] = Field(None, description="Ngày cấp số giấy phép lưu hành")
    expiry_date: Optional[str] = Field(None, description="Ngày hết hiệu lực lưu hành")
    # Trường mở rộng
    registrant: Optional[str] = Field(None, description="Cơ sở đăng ký")

class DosageContent(BaseModel):
    text: str = Field(..., description="Mô tả liều dùng dạng văn bản")
    table: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Bảng liều dùng trích xuất")

class DrugSections(BaseModel):
    indication: Optional[str] = Field(None, description="Chỉ định")
    contraindication: Optional[str] = Field(None, description="Chống chỉ định")
    dosage: Optional[Union[str, DosageContent]] = Field(None, description="Liều dùng")
    side_effects: Optional[str] = Field(None, description="Tác dụng phụ")
    interactions: Optional[str] = Field(None, description="Tương tác thuốc")
    warnings: Optional[str] = Field(None, description="Thận trọng / Cảnh báo")
    pharmacology: Optional[str] = Field(None, description="Dược lực học")
    pharmacokinetics: Optional[str] = Field(None, description="Dược động học")

class Drug(BaseModel):
    metadata: DrugMetadata
    sections: DrugSections

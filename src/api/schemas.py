"""
Pydantic Schemas cho API đặt lịch dịch vụ và tính giá thời gian thực.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PriceEstimateRequest(BaseModel):
    services: List[str] = Field(..., description="Danh sách mã dịch vụ đã chọn (VD: ['tam', 'kham_benh'])")
    weight_kg: Optional[float] = Field(None, ge=0.1, le=150.0, description="Cân nặng thú cưng (kg)")
    duration_days: Optional[int] = Field(1, ge=1, le=365, description="Số ngày lưu trú nếu có chọn dịch vụ lưu trú 24h")


class PriceItemResponse(BaseModel):
    service_type: str = Field(..., description="Mã loại dịch vụ")
    service_name: str = Field(..., description="Tên hiển thị dịch vụ")
    price: Optional[int] = Field(None, description="Đơn giá (VND), trả về null nếu là dịch vụ ẩn giá như Khám bệnh")
    formatted_price: str = Field(..., description="Giá tiền đã format (VD: 150.000đ hoặc 'Báo giá sau khi khám')")
    hide_price: bool = Field(False, description="Cờ đánh dấu có ẩn giá hay không")
    note: Optional[str] = Field(None, description="Ghi chú chi tiết cho dịch vụ này")
    base_price_per_day: Optional[int] = Field(None, description="Đơn giá gốc/ngày (nếu là lưu trú)")
    num_days: Optional[int] = Field(None, description="Số ngày lưu trú")
    discount_pct: Optional[int] = Field(None, description="% Giảm giá lưu trú")
    discount_amount: Optional[int] = Field(None, description="Số tiền giảm giá (VND)")
    free_bath: Optional[bool] = Field(None, description="Tặng tắm free nếu lưu trú > 10 ngày")


class PriceEstimateResponse(BaseModel):
    items: List[PriceItemResponse] = Field(..., description="Chi tiết giá từng dịch vụ")
    subtotal: Optional[int] = Field(None, description="Tổng giá gốc trước giảm")
    discount_amount: int = Field(0, description="Tổng số tiền giảm giá")
    total_estimated_price: Optional[int] = Field(None, description="Tổng chi phí ước tính (null nếu chỉ khám bệnh)")
    total_formatted_price: str = Field(..., description="Tổng tiền format hiển thị cho người dùng")
    has_unpriced_service: bool = Field(False, description="Có dịch vụ ẩn giá/chưa niêm yết (như Khám bệnh) hay không")
    free_bath: bool = Field(False, description="Có được tặng suất tắm miễn phí hay không")
    notes: List[str] = Field(..., description="Các lưu ý và hướng dẫn liên quan đến báo giá")


class AvailableSlotsResponse(BaseModel):
    date: str = Field(..., description="Ngày kiểm tra (YYYY-MM-DD)")
    is_open: bool = Field(..., description="Trạng thái cửa hàng có mở cửa nhận khách hay không")
    slots: List[str] = Field(..., description="Danh sách các khung giờ còn đặt được (HH:MM)")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")


class BookingCreateRequest(BaseModel):
    customer_name: str = Field(..., min_length=2, max_length=100, description="Họ và tên khách hàng")
    customer_phone: str = Field(..., description="Số điện thoại liên hệ (10 số di động VN)")
    pet_name: Optional[str] = Field(None, max_length=50, description="Tên thú cưng")
    pet_type: str = Field("dog", description="Loài thú cưng: 'dog', 'cat', hoặc 'other'")
    weight_kg: Optional[float] = Field(None, ge=0.1, le=150.0, description="Cân nặng thú cưng (kg)")
    services: List[str] = Field(..., min_length=1, description="Danh sách mã dịch vụ đăng ký")
    booking_date: str = Field(..., description="Ngày hẹn (YYYY-MM-DD)")
    booking_time: str = Field(..., description="Khung giờ hẹn (HH:MM)")
    duration_days: Optional[int] = Field(1, ge=1, le=365, description="Số ngày lưu trú")
    notes: Optional[str] = Field(None, max_length=500, description="Ghi chú thêm cho nhân viên/bác sĩ")
    session_id: Optional[str] = Field(None, description="ID phiên chat nếu đặt từ luồng tư vấn")


class BookingResponse(BaseModel):
    id: int = Field(..., description="ID bản ghi trong database")
    booking_code: str = Field(..., description="Mã định danh đặt lịch độc nhất (VD: BK-20260827-XXXX)")
    session_id: Optional[str] = Field(None, description="ID phiên chat")
    customer_name: str = Field(..., description="Tên khách hàng")
    customer_phone: str = Field(..., description="Số điện thoại")
    pet_name: Optional[str] = Field(None, description="Tên thú cưng")
    pet_type: str = Field(..., description="Loài thú cưng")
    weight_kg: Optional[float] = Field(None, description="Cân nặng (kg)")
    services: List[str] = Field(..., description="Danh sách dịch vụ đã đặt")
    booking_date: str = Field(..., description="Ngày hẹn")
    booking_time: str = Field(..., description="Khung giờ hẹn")
    duration_days: int = Field(..., description="Số ngày lưu trú")
    estimated_price: Optional[int] = Field(None, description="Chi phí ước tính")
    discount_amount: int = Field(0, description="Số tiền giảm giá")
    has_unpriced_service: bool = Field(..., description="Đơn có bao gồm dịch vụ ẩn giá (Khám bệnh) không")
    price_breakdown: Optional[Dict[str, Any]] = Field(None, description="Chi tiết báo giá từng dịch vụ")
    notes: Optional[str] = Field(None, description="Ghi chú")
    status: str = Field(..., description="Trạng thái đơn: PENDING, CONFIRMED, CANCELLED, COMPLETED")
    created_at: str = Field(..., description="Thời gian tạo đơn (ISO format)")
    updated_at: Optional[str] = Field(None, description="Thời gian cập nhật gần nhất")


class BookingListResponse(BaseModel):
    total: int = Field(..., description="Tổng số lượng bản ghi thỏa mãn điều kiện lọc")
    limit: int = Field(..., description="Số lượng bản ghi trên một trang")
    offset: int = Field(..., description="Vị trí bắt đầu lấy")
    items: List[BookingResponse] = Field(..., description="Danh sách chi tiết các đơn đặt lịch")


class BookingStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="Trạng thái mới: CONFIRMED, CANCELLED, hoặc COMPLETED")

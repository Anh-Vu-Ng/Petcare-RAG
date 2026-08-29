"""
Automated Test Suite cho Booking Backend & FastAPI Endpoints.
Bao gồm kiểm thử logic tính giá (ẩn giá cho Khám bệnh), kiểm tra giờ mở cửa,
xác thực số điện thoại và các API CRUD.
"""

import os
import sys
import datetime
import pytest

# Thêm thư mục root vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thiết lập DATABASE_URL sang SQLite local cho test trước khi import DB modules
os.environ["DATABASE_URL"] = "sqlite:///data/petcare_test.db"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi.testclient import TestClient

import src.db.database as db_module
from src.service_db import ServiceDB
from src.booking_service import BookingService
from src.api.main import app
import src.api.main as api_main

# Ghi đè engine & session sang SQLite test database
TEST_DB_URL = "sqlite:///data/petcare_test.db"
TEST_ASYNC_DB_URL = "sqlite+aiosqlite:///data/petcare_test.db"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
test_async_engine = create_async_engine(TEST_ASYNC_DB_URL, connect_args={"check_same_thread": False})

TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
TestAsyncSessionLocal = async_sessionmaker(bind=test_async_engine, class_=AsyncSession, expire_on_commit=False)

db_module.engine = test_engine
db_module.async_engine = test_async_engine
db_module.SessionLocal = TestSessionLocal
db_module.AsyncSessionLocal = TestAsyncSessionLocal

# Update imported modules
import src.booking_service as bs_module
bs_module.SessionLocal = TestSessionLocal
bs_module.AsyncSessionLocal = TestAsyncSessionLocal

import src.service_db as sdb_module
sdb_module.SessionLocal = TestSessionLocal
sdb_module.AsyncSessionLocal = TestAsyncSessionLocal


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Khởi tạo database schema và import dữ liệu giá vào SQLite test db."""
    db_module.Base.metadata.drop_all(bind=test_engine)
    db_module.Base.metadata.create_all(bind=test_engine)
    sdb = ServiceDB()
    sdb.import_from_csv(force=True)
    
    # Khởi tạo booking_service cho test client
    api_main.service_db = sdb
    api_main.booking_service = BookingService(sdb)
    yield
    # Cleanup sau khi test
    db_module.Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def booking_service():
    sdb = ServiceDB()
    return BookingService(sdb)


@pytest.fixture
def client():
    return TestClient(app)


# ==========================================
# 1. UNIT TESTS: VALIDATION & SLOTS
# ==========================================

def test_phone_validation():
    """Kiểm tra validation số điện thoại Việt Nam."""
    assert BookingService.validate_phone_number("0912345678") is True
    assert BookingService.validate_phone_number("0389998888") is True
    assert BookingService.validate_phone_number("0701234567") is True
    assert BookingService.validate_phone_number("0861112222") is True
    assert BookingService.validate_phone_number("0567891234") is True
    assert BookingService.validate_phone_number("091 234 5678") is True
    assert BookingService.validate_phone_number("091-234-5678") is True

    # Số không hợp lệ
    assert BookingService.validate_phone_number("12345") is False
    assert BookingService.validate_phone_number("0123456789") is False # Đầu 01 cũ 11 số
    assert BookingService.validate_phone_number("02412345678") is False # Số bàn
    assert BookingService.validate_phone_number("abcdefghij") is False
    assert BookingService.validate_phone_number("") is False


def test_datetime_validation():
    """Kiểm tra validation ngày giờ theo giờ hoạt động của Petcare."""
    future_date = (datetime.date.today() + datetime.timedelta(days=7))
    
    # Tìm ngày trong tuần (T2 - T7) và ngày Chủ nhật trong tương lai
    days_ahead = 0
    weekday_date = None
    sunday_date = None
    while weekday_date is None or sunday_date is None:
        d = future_date + datetime.timedelta(days=days_ahead)
        if d.weekday() < 6 and weekday_date is None:
            weekday_date = d.isoformat()
        elif d.weekday() == 6 and sunday_date is None:
            sunday_date = d.isoformat()
        days_ahead += 1

    # Ngày trong quá khứ -> Lỗi
    past_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    valid, msg = BookingService.validate_booking_datetime(past_date, "09:00")
    assert valid is False
    assert "quá khứ" in msg

    # Ngày thường: 09:00 (Sáng) -> Hợp lệ
    valid, _ = BookingService.validate_booking_datetime(weekday_date, "09:00")
    assert valid is True

    # Ngày thường: 15:30 (Chiều) -> Hợp lệ
    valid, _ = BookingService.validate_booking_datetime(weekday_date, "15:30")
    assert valid is True

    # Ngày thường: 12:30 (Nghỉ trưa) -> Không hợp lệ
    valid, msg = BookingService.validate_booking_datetime(weekday_date, "12:30")
    assert valid is False

    # Ngày thường: 20:00 (Sau giờ đóng cửa) -> Không hợp lệ
    valid, msg = BookingService.validate_booking_datetime(weekday_date, "20:00")
    assert valid is False

    # Chủ nhật: 09:30 (Sáng) -> Hợp lệ
    valid, _ = BookingService.validate_booking_datetime(sunday_date, "09:30")
    assert valid is True

    # Chủ nhật: 15:00 (Chiều) -> Chủ nhật chỉ mở buổi sáng -> Không hợp lệ
    valid, msg = BookingService.validate_booking_datetime(sunday_date, "15:00")
    assert valid is False
    assert "Chủ nhật" in msg


def test_available_slots_generation():
    """Kiểm tra sinh khung giờ hẹn cho ngày thường và Chủ nhật."""
    future_date = datetime.date.today() + datetime.timedelta(days=10)
    # Nếu là Chủ nhật
    if future_date.weekday() == 6:
        slots_info = BookingService.get_available_time_slots(future_date.isoformat())
        assert slots_info["is_open"] is True
        assert "08:00" in slots_info["slots"]
        assert "11:30" in slots_info["slots"]
        assert "14:00" not in slots_info["slots"] # CN không mở buổi chiều
    else:
        slots_info = BookingService.get_available_time_slots(future_date.isoformat())
        assert slots_info["is_open"] is True
        assert "08:00" in slots_info["slots"]
        assert "11:30" in slots_info["slots"]
        assert "14:00" in slots_info["slots"]
        assert "18:30" in slots_info["slots"]


# ==========================================
# 2. UNIT TESTS: PRICE CALCULATION & "KHÁM BỆNH"
# ==========================================

def test_price_quote_kham_benh_only(booking_service):
    """
    KIỂM THỬ ĐẶC THÙ: Khi chỉ chọn 'Khám bệnh', giá tiền phải bị ẩn hoàn toàn:
    - price: None
    - hide_price: True
    - total_estimated_price: None
    - has_unpriced_service: True
    """
    quote = booking_service.calculate_booking_quote(
        services=["kham_benh"],
        weight_kg=5.0
    )

    assert len(quote["items"]) == 1
    item = quote["items"][0]
    assert item["service_type"] == "kham_benh"
    assert item["price"] is None
    assert item["hide_price"] is True
    assert item["formatted_price"] == "Báo giá sau khi khám"

    assert quote["total_estimated_price"] is None
    assert quote["has_unpriced_service"] is True
    assert quote["total_formatted_price"] == "Báo giá sau khi khám"
    assert any("Bác sĩ" in n for n in quote["notes"])


def test_price_quote_standard_services(booking_service):
    """Kiểm thử tính giá dịch vụ niêm yết theo cân nặng."""
    quote = booking_service.calculate_booking_quote(
        services=["tam", "cat_mai_mong"],
        weight_kg=5.0
    )

    assert len(quote["items"]) == 2
    assert quote["has_unpriced_service"] is False
    assert quote["total_estimated_price"] is not None
    assert quote["total_estimated_price"] > 0
    assert all(item["hide_price"] is False for item in quote["items"])


def test_price_quote_mixed_services(booking_service):
    """
    KIỂM THỬ DỊCH VỤ HỖN HỢP: Tắm + Khám bệnh.
    - Dịch vụ Tắm có giá cụ thể.
    - Dịch vụ Khám bệnh ẩn giá (`hide_price: True`, `price: None`).
    - Tổng tiền tính trên dịch vụ Tắm, có ghi chú rõ chưa bao gồm tiền khám.
    """
    quote = booking_service.calculate_booking_quote(
        services=["tam", "kham_benh"],
        weight_kg=5.0
    )

    assert len(quote["items"]) == 2
    
    # Tìm item tắm và khám bệnh
    tam_item = next(i for i in quote["items"] if i["service_type"] == "tam")
    kham_item = next(i for i in quote["items"] if i["service_type"] == "kham_benh")

    assert tam_item["price"] is not None
    assert tam_item["hide_price"] is False

    assert kham_item["price"] is None
    assert kham_item["hide_price"] is True

    assert quote["has_unpriced_service"] is True
    assert quote["total_estimated_price"] == tam_item["price"]
    assert any("CHƯA bao gồm tiền khám bệnh" in n for n in quote["notes"])


def test_price_quote_boarding_discount(booking_service):
    """Kiểm thử chiết khấu lưu trú dài ngày (12 ngày -> giảm 15% + tặng tắm)."""
    quote = booking_service.calculate_booking_quote(
        services=["luu_tru_24h"],
        weight_kg=5.0,
        duration_days=12
    )

    item = quote["items"][0]
    assert item["discount_pct"] == 15
    assert item["free_bath"] is True
    assert quote["free_bath"] is True
    assert quote["discount_amount"] > 0


# ==========================================
# 3. FASTAPI INTEGRATION TESTS
# ==========================================

def test_api_estimate_price_kham_benh(client):
    """Test endpoint POST /api/v1/bookings/estimate-price với dịch vụ khám bệnh."""
    response = client.post(
        "/api/v1/bookings/estimate-price",
        json={
            "services": ["kham_benh"],
            "weight_kg": 4.5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_unpriced_service"] is True
    assert data["total_estimated_price"] is None
    assert data["items"][0]["hide_price"] is True


def test_api_available_slots(client):
    """Test endpoint GET /api/v1/bookings/available-slots."""
    future_date = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
    response = client.get(f"/api/v1/bookings/available-slots?date={future_date}")
    assert response.status_code == 200
    data = response.json()
    assert data["is_open"] is True
    assert len(data["slots"]) > 0


def test_api_create_booking_success(client):
    """Test tạo đơn đặt lịch thành công qua POST /api/v1/bookings."""
    # Tìm ngày thường hợp lệ
    target_date = datetime.date.today() + datetime.timedelta(days=3)
    while target_date.weekday() == 6:
        target_date += datetime.timedelta(days=1)

    booking_payload = {
        "customer_name": "Nguyễn Văn An",
        "customer_phone": "0987654321",
        "pet_name": "Mimi",
        "pet_type": "cat",
        "weight_kg": 3.5,
        "services": ["tam", "kham_benh"],
        "booking_date": target_date.isoformat(),
        "booking_time": "09:30",
        "duration_days": 1,
        "notes": "Bé hơi nhát người lạ",
        "session_id": "test_session_123"
    }

    response = client.post("/api/v1/bookings", json=booking_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["booking_code"].startswith("BK-")
    assert data["customer_name"] == "Nguyễn Văn An"
    assert data["customer_phone"] == "0987654321"
    assert data["status"] == "PENDING"
    assert data["has_unpriced_service"] is True

    booking_code = data["booking_code"]
    booking_id = data["id"]

    # 1. Tra cứu lại theo mã booking_code
    get_res = client.get(f"/api/v1/bookings/{booking_code}")
    assert get_res.status_code == 200
    assert get_res.json()["booking_code"] == booking_code

    # 2. Cập nhật trạng thái thành CONFIRMED
    patch_res = client.patch(
        f"/api/v1/bookings/{booking_id}/status",
        json={"status": "CONFIRMED"}
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "CONFIRMED"

    # 3. Lấy danh sách booking
    list_res = client.get("/api/v1/bookings?search=Nguyễn+Văn+An")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(b["booking_code"] == booking_code for b in list_data["items"])


def test_api_create_booking_invalid_phone(client):
    """Test từ chối tạo booking nếu số điện thoại không hợp lệ."""
    target_date = (datetime.date.today() + datetime.timedelta(days=3)).isoformat()
    response = client.post(
        "/api/v1/bookings",
        json={
            "customer_name": "Test User",
            "customer_phone": "12345", # Sai số
            "services": ["tam"],
            "booking_date": target_date,
            "booking_time": "09:00",
        }
    )
    assert response.status_code == 400
    assert "Số điện thoại không hợp lệ" in response.json()["detail"]

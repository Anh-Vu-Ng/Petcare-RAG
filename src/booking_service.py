"""
Module BookingService: Quản lý toàn bộ nghiệp vụ đặt lịch dịch vụ Petcare.
Hỗ trợ tính giá thời gian thực, ẩn giá cho dịch vụ 'Khám bệnh', kiểm tra giờ mở cửa và tương tác CSDL.
"""

import json
import re
import uuid
import datetime
from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy import text, select, func, or_, and_, desc
from src.db.database import SessionLocal, AsyncSessionLocal, engine
from src.db import models
from src.service_db import ServiceDB, SERVICE_NAME_MAP as BASE_SERVICE_NAME_MAP
from src.tools import calculate_final_price

# Mở rộng danh mục dịch vụ với dịch vụ Khám bệnh
SERVICE_NAME_MAP = {
    **BASE_SERVICE_NAME_MAP,
    "kham_benh": "Khám bệnh",
}

# Hằng số cấu hình giờ làm việc
WORKING_HOURS = {
    "weekday": [
        ("08:00", "12:00"),
        ("14:00", "19:00"),
    ],
    "sunday": [
        ("08:00", "12:00"),
    ]
}

VALID_STATUSES = {"PENDING", "CONFIRMED", "CANCELLED", "COMPLETED"}


class BookingService:
    """Service layer xử lý nghiệp vụ đặt lịch và thao tác với bảng bookings."""

    def __init__(self, service_db: Optional[ServiceDB] = None):
        self.service_db = service_db or ServiceDB()

    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """Kiểm tra số điện thoại Việt Nam hợp lệ (10 chữ số: 03x, 05x, 07x, 08x, 09x)."""
        if not phone:
            return False
        clean_phone = re.sub(r'[\s\.\-\(\)]', '', phone.strip())
        pattern = r'^(03|05|07|08|09)\d{8}$'
        return bool(re.match(pattern, clean_phone))

    @staticmethod
    def validate_booking_datetime(booking_date_str: str, booking_time_str: str) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của ngày và giờ hẹn theo quy định hoạt động của Petcare:
        - Ngày hẹn không được trong quá khứ.
        - Giờ hẹn:
          * Thứ 2 - Thứ 7: 08:00 - 12:00 và 14:00 - 19:00
          * Chủ nhật: 08:00 - 12:00
        """
        try:
            target_date = datetime.date.fromisoformat(booking_date_str.strip())
        except ValueError:
            return False, f"Định dạng ngày không hợp lệ ({booking_date_str}). Vui lòng dùng định dạng YYYY-MM-DD."

        today = datetime.date.today()
        if target_date < today:
            return False, f"Ngày hẹn ({booking_date_str}) không thể ở trong quá khứ."

        try:
            target_time = datetime.time.fromisoformat(booking_time_str.strip())
        except ValueError:
            return False, f"Định dạng giờ không hợp lệ ({booking_time_str}). Vui lòng dùng định dạng HH:MM."

        # Kiểm tra nếu đặt cùng ngày hôm nay thì giờ hẹn không được trôi qua
        if target_date == today:
            now_time = datetime.datetime.now().time()
            if target_time <= now_time:
                return False, f"Giờ hẹn ({booking_time_str}) đã qua so với thời gian hiện tại."

        # Kiểm tra thứ trong tuần (0: Thứ 2, ..., 6: Chủ nhật)
        weekday = target_date.weekday()
        target_minutes = target_time.hour * 60 + target_time.minute

        if weekday == 6:  # Chủ nhật
            # 08:00 (480) - 12:00 (720)
            if not (8 * 60 <= target_minutes <= 12 * 60):
                return False, "Chủ nhật Petcare chỉ mở cửa buổi sáng từ 08:00 đến 12:00. Vui lòng chọn khung giờ khác."
        else:  # Thứ 2 đến Thứ 7
            # Sáng: 08:00 - 12:00 | Chiều: 14:00 - 19:00
            in_morning = (8 * 60 <= target_minutes <= 12 * 60)
            in_afternoon = (14 * 60 <= target_minutes <= 19 * 60)
            if not (in_morning or in_afternoon):
                return False, (
                    "Khung giờ làm việc từ Thứ 2 đến Thứ 7 là: "
                    "Sáng 08:00 - 12:00 và Chiều 14:00 - 19:00. Vui lòng chọn khung giờ phù hợp."
                )

        return True, "Hợp lệ"

    @staticmethod
    def get_available_time_slots(booking_date_str: str) -> Dict[str, Any]:
        """Trả về danh sách các slot thời gian đặt hẹn hợp lệ trong ngày (bước nhảy 30 phút)."""
        try:
            target_date = datetime.date.fromisoformat(booking_date_str.strip())
        except ValueError:
            return {
                "date": booking_date_str,
                "is_open": False,
                "slots": [],
                "error": "Định dạng ngày không hợp lệ. Vui lòng dùng YYYY-MM-DD."
            }

        today = datetime.date.today()
        if target_date < today:
            return {
                "date": booking_date_str,
                "is_open": False,
                "slots": [],
                "error": "Ngày hẹn không được ở trong quá khứ."
            }

        weekday = target_date.weekday()
        slots = []

        def _gen_slots(start_h, start_m, end_h, end_m):
            cur = start_h * 60 + start_m
            end = end_h * 60 + end_m
            res = []
            while cur <= end:
                h = cur // 60
                m = cur % 60
                res.append(f"{h:02d}:{m:02d}")
                cur += 30
            return res

        if weekday == 6:  # Chủ nhật
            slots = _gen_slots(8, 0, 11, 30)
        else:  # Thứ 2 đến Thứ 7
            morning = _gen_slots(8, 0, 11, 30)
            afternoon = _gen_slots(14, 0, 18, 30)
            slots = morning + afternoon

        # Lọc bỏ các slot đã trôi qua nếu đặt cho ngày hôm nay
        if target_date == today:
            now_time = datetime.datetime.now().time()
            now_minutes = now_time.hour * 60 + now_time.minute
            filtered_slots = []
            for slot_str in slots:
                h, m = map(int, slot_str.split(":"))
                if h * 60 + m > now_minutes:
                    filtered_slots.append(slot_str)
            slots = filtered_slots

        return {
            "date": booking_date_str,
            "is_open": len(slots) > 0,
            "slots": slots
        }

    @staticmethod
    def generate_booking_code() -> str:
        """Sinh mã đặt lịch duy nhất theo định dạng: BK-YYYYMMDD-XXXX."""
        date_part = datetime.datetime.now().strftime("%Y%m%d")
        rand_part = uuid.uuid4().hex[:4].upper()
        return f"BK-{date_part}-{rand_part}"

    def calculate_booking_quote(
        self,
        services: List[str],
        weight_kg: Optional[float] = None,
        duration_days: int = 1,
    ) -> Dict[str, Any]:
        """
        Tính toán chi tiết giá dự kiến cho danh sách dịch vụ đã chọn.
        
        Quy tắc đặc biệt:
        - Dịch vụ "kham_benh": Ẩn giá tiền (price: None, hide_price: True), ghi chú báo giá sau khi khám.
        - Dịch vụ niêm yết: Tra cứu theo cân nặng thú cưng.
        - Dịch vụ "luu_tru_24h": Áp dụng chiết khấu theo số ngày và tặng tắm nếu > 10 ngày.
        """
        if not services:
            return {
                "items": [],
                "subtotal": 0,
                "discount_amount": 0,
                "total_estimated_price": 0,
                "has_unpriced_service": False,
                "free_bath": False,
                "notes": ["Chưa chọn dịch vụ nào."]
            }

        items = []
        priced_subtotal = 0
        discount_total = 0
        has_unpriced_service = False
        free_bath_awarded = False
        notes = []

        has_kham_benh = "kham_benh" in services

        for stype in services:
            stype_clean = stype.strip()
            service_name = SERVICE_NAME_MAP.get(stype_clean, stype_clean)

            # Trường hợp 1: Dịch vụ Khám Bệnh (Ẩn giá cố định)
            if stype_clean == "kham_benh":
                has_unpriced_service = True
                items.append({
                    "service_type": "kham_benh",
                    "service_name": service_name,
                    "price": None,
                    "formatted_price": "Báo giá sau khi khám",
                    "hide_price": True,
                    "note": "Chi phí sẽ được Bác sĩ thú y báo giá trực tiếp sau khi khám lâm sàng cho bé."
                })
                notes.append(
                    "Dịch vụ Khám bệnh: Bác sĩ thú y sẽ thăm khám trực tiếp và báo chi phí cụ thể tùy thuộc vào tình trạng sức khỏe và đơn thuốc của bé."
                )

            # Trường hợp 2: Dịch vụ Lưu trú 24h (Có chiết khấu theo ngày)
            elif stype_clean == "luu_tru_24h":
                if weight_kg is None:
                    items.append({
                        "service_type": "luu_tru_24h",
                        "service_name": service_name,
                        "price": None,
                        "formatted_price": "Cần cung cấp cân nặng để tính giá",
                        "hide_price": True,
                        "note": "Vui lòng nhập cân nặng của bé để tính tiền lưu trú."
                    })
                    has_unpriced_service = True
                else:
                    base_info = self.service_db.lookup_price("luu_tru_24h", weight_kg)
                    if base_info:
                        base_price_day = base_info["price"]
                        calc_result = calculate_final_price(
                            base_price_per_day=base_price_day,
                            num_days=max(1, duration_days),
                            service_type="luu_tru_24h",
                            weight_kg=weight_kg,
                            db=self.service_db
                        )
                        priced_subtotal += calc_result["total_before_discount"]
                        discount_total += calc_result["discount_amount"]
                        if calc_result["free_bath"]:
                            free_bath_awarded = True

                        formatted_str = f"{calc_result['final_price']:,}đ".replace(",", ".")
                        items.append({
                            "service_type": "luu_tru_24h",
                            "service_name": f"{service_name} ({duration_days} ngày)",
                            "price": calc_result["final_price"],
                            "formatted_price": formatted_str,
                            "hide_price": False,
                            "base_price_per_day": base_price_day,
                            "num_days": duration_days,
                            "discount_pct": calc_result["discount_pct"],
                            "discount_amount": calc_result["discount_amount"],
                            "free_bath": calc_result["free_bath"],
                            "note": calc_result["note"]
                        })
                    else:
                        items.append({
                            "service_type": "luu_tru_24h",
                            "service_name": service_name,
                            "price": None,
                            "formatted_price": "Liên hệ Petcare",
                            "hide_price": True,
                            "note": f"Không tìm thấy bảng giá lưu trú cho mức cân nặng {weight_kg}kg."
                        })
                        has_unpriced_service = True

            # Trường hợp 3: Dịch vụ niêm yết thông thường (Tắm, Cạo lông, Cắt móng, Vệ sinh tai, Nặn tuyến hôi)
            else:
                if weight_kg is None:
                    items.append({
                        "service_type": stype_clean,
                        "service_name": service_name,
                        "price": None,
                        "formatted_price": "Cần cung cấp cân nặng",
                        "hide_price": True,
                        "note": "Cần cân nặng để tra cứu giá chính xác."
                    })
                    has_unpriced_service = True
                else:
                    price_info = self.service_db.lookup_price(stype_clean, weight_kg)
                    if price_info:
                        item_price = price_info["price"]
                        priced_subtotal += item_price
                        items.append({
                            "service_type": stype_clean,
                            "service_name": service_name,
                            "price": item_price,
                            "formatted_price": f"{item_price:,}đ".replace(",", "."),
                            "hide_price": False,
                            "note": f"Đơn giá cho bé {weight_kg}kg"
                        })
                    else:
                        items.append({
                            "service_type": stype_clean,
                            "service_name": service_name,
                            "price": None,
                            "formatted_price": "Liên hệ Petcare",
                            "hide_price": True,
                            "note": f"Không tìm thấy bảng giá cho mức cân nặng {weight_kg}kg."
                        })
                        has_unpriced_service = True

        total_final = max(0, priced_subtotal - discount_total)

        # Xử lý hiển thị tổng giá
        if has_kham_benh and len(services) == 1:
            total_estimated_price = None
            total_formatted = "Báo giá sau khi khám"
        elif has_unpriced_service and priced_subtotal == 0:
            total_estimated_price = None
            total_formatted = "Báo giá khi check-in"
        else:
            total_estimated_price = total_final
            total_formatted = f"{total_final:,}đ".replace(",", ".")
            if has_kham_benh:
                notes.append("Lưu ý: Tổng chi phí tạm tính trên CHƯA bao gồm tiền khám bệnh và thuốc điều trị.")

        if free_bath_awarded:
            notes.append("🎁 Bạn được TẶNG KÈM 1 suất Tắm miễn phí cho gói lưu trú trên 10 ngày!")

        return {
            "items": items,
            "subtotal": priced_subtotal if priced_subtotal > 0 else (None if has_kham_benh and len(services) == 1 else 0),
            "discount_amount": discount_total,
            "total_estimated_price": total_estimated_price,
            "total_formatted_price": total_formatted,
            "has_unpriced_service": has_unpriced_service,
            "free_bath": free_bath_awarded,
            "notes": notes
        }

    # --- ASYNCHRONOUS DATABASE OPERATIONS ---
    async def create_booking_async(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo mới một yêu cầu đặt lịch trong database (Async)."""
        phone = booking_data.get("customer_phone", "").strip()
        if not self.validate_phone_number(phone):
            raise ValueError(f"Số điện thoại không hợp lệ: '{phone}'. Vui lòng nhập đúng 10 số di động Việt Nam.")

        b_date = booking_data.get("booking_date", "").strip()
        b_time = booking_data.get("booking_time", "").strip()
        is_valid_time, time_msg = self.validate_booking_datetime(b_date, b_time)
        if not is_valid_time:
            raise ValueError(time_msg)

        raw_services = booking_data.get("services", [])
        if isinstance(raw_services, str):
            try:
                services_list = json.loads(raw_services)
            except Exception:
                services_list = [s.strip() for s in raw_services.split(",") if s.strip()]
        else:
            services_list = list(raw_services)

        if not services_list:
            raise ValueError("Vui lòng chọn ít nhất một dịch vụ.")

        weight_kg = booking_data.get("weight_kg")
        if weight_kg is not None:
            try:
                weight_kg = float(weight_kg)
                if weight_kg <= 0 or weight_kg > 150:
                    raise ValueError("Cân nặng thú cưng phải lớn hơn 0 và nhỏ hơn 150kg.")
            except (ValueError, TypeError):
                raise ValueError("Cân nặng không hợp lệ.")

        duration_days = int(booking_data.get("duration_days", 1) or 1)

        # Tính toán báo giá
        quote = self.calculate_booking_quote(
            services=services_list,
            weight_kg=weight_kg,
            duration_days=duration_days
        )

        booking_code = self.generate_booking_code()

        async with AsyncSessionLocal() as session:
            new_booking = models.BookingModel(
                booking_code=booking_code,
                session_id=booking_data.get("session_id"),
                customer_name=booking_data.get("customer_name", "").strip(),
                customer_phone=phone,
                pet_name=booking_data.get("pet_name", "").strip() if booking_data.get("pet_name") else None,
                pet_type=booking_data.get("pet_type", "dog").strip(),
                weight_kg=weight_kg,
                services=json.dumps(services_list, ensure_ascii=False),
                booking_date=b_date,
                booking_time=b_time,
                duration_days=duration_days,
                estimated_price=quote["total_estimated_price"],
                discount_amount=quote["discount_amount"],
                has_unpriced_service=quote["has_unpriced_service"],
                price_breakdown=json.dumps(quote, ensure_ascii=False),
                notes=booking_data.get("notes", "").strip() if booking_data.get("notes") else None,
                status="PENDING"
            )
            session.add(new_booking)
            await session.commit()
            await session.refresh(new_booking)

            return self._model_to_dict(new_booking)

    async def get_booking_by_code_async(self, booking_code: str) -> Optional[Dict[str, Any]]:
        """Tra cứu thông tin đặt lịch theo booking_code (Async)."""
        async with AsyncSessionLocal() as session:
            stmt = select(models.BookingModel).where(
                models.BookingModel.booking_code == booking_code.strip().upper()
            )
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            return self._model_to_dict(item) if item else None

    async def get_booking_by_id_async(self, booking_id: int) -> Optional[Dict[str, Any]]:
        """Tra cứu thông tin đặt lịch theo ID (Async)."""
        async with AsyncSessionLocal() as session:
            stmt = select(models.BookingModel).where(models.BookingModel.id == booking_id)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            return self._model_to_dict(item) if item else None

    async def list_bookings_async(
        self,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Lấy danh sách các lượt đặt lịch có phân trang và bộ lọc (Async)."""
        async with AsyncSessionLocal() as session:
            filters = []

            if status and status.upper() in VALID_STATUSES:
                filters.append(models.BookingModel.status == status.upper())

            if date_from:
                filters.append(models.BookingModel.booking_date >= date_from)

            if date_to:
                filters.append(models.BookingModel.booking_date <= date_to)

            if search:
                search_term = f"%{search.strip()}%"
                filters.append(
                    or_(
                        models.BookingModel.booking_code.ilike(search_term),
                        models.BookingModel.customer_phone.ilike(search_term),
                        models.BookingModel.customer_name.ilike(search_term),
                        models.BookingModel.pet_name.ilike(search_term),
                    )
                )

            # Đếm tổng số
            count_stmt = select(func.count(models.BookingModel.id))
            if filters:
                count_stmt = count_stmt.where(and_(*filters))
            total_result = await session.execute(count_stmt)
            total = total_result.scalar_one()

            # Lấy danh sách phân trang
            query_stmt = (
                select(models.BookingModel)
                .order_by(desc(models.BookingModel.created_at))
                .limit(limit)
                .offset(offset)
            )
            if filters:
                query_stmt = query_stmt.where(and_(*filters))

            items_result = await session.execute(query_stmt)
            items = items_result.scalars().all()

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "items": [self._model_to_dict(b) for b in items]
            }

    async def update_booking_status_async(self, booking_id: int, new_status: str) -> Optional[Dict[str, Any]]:
        """Cập nhật trạng thái đặt lịch: PENDING, CONFIRMED, CANCELLED, COMPLETED (Async)."""
        status_clean = new_status.strip().upper()
        if status_clean not in VALID_STATUSES:
            raise ValueError(f"Trạng thái không hợp lệ: '{new_status}'. Chỉ chấp nhận: {', '.join(VALID_STATUSES)}.")

        async with AsyncSessionLocal() as session:
            stmt = select(models.BookingModel).where(models.BookingModel.id == booking_id)
            result = await session.execute(stmt)
            booking = result.scalar_one_or_none()

            if not booking:
                return None

            booking.status = status_clean
            await session.commit()
            await session.refresh(booking)
            return self._model_to_dict(booking)

    # --- SYNCHRONOUS DATABASE OPERATIONS ---
    def create_booking(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo mới một yêu cầu đặt lịch trong database (Sync)."""
        phone = booking_data.get("customer_phone", "").strip()
        if not self.validate_phone_number(phone):
            raise ValueError(f"Số điện thoại không hợp lệ: '{phone}'. Vui lòng nhập đúng 10 số di động Việt Nam.")

        b_date = booking_data.get("booking_date", "").strip()
        b_time = booking_data.get("booking_time", "").strip()
        is_valid_time, time_msg = self.validate_booking_datetime(b_date, b_time)
        if not is_valid_time:
            raise ValueError(time_msg)

        raw_services = booking_data.get("services", [])
        if isinstance(raw_services, str):
            try:
                services_list = json.loads(raw_services)
            except Exception:
                services_list = [s.strip() for s in raw_services.split(",") if s.strip()]
        else:
            services_list = list(raw_services)

        if not services_list:
            raise ValueError("Vui lòng chọn ít nhất một dịch vụ.")

        weight_kg = booking_data.get("weight_kg")
        if weight_kg is not None:
            try:
                weight_kg = float(weight_kg)
                if weight_kg <= 0 or weight_kg > 150:
                    raise ValueError("Cân nặng thú cưng phải lớn hơn 0 và nhỏ hơn 150kg.")
            except (ValueError, TypeError):
                raise ValueError("Cân nặng không hợp lệ.")

        duration_days = int(booking_data.get("duration_days", 1) or 1)

        quote = self.calculate_booking_quote(
            services=services_list,
            weight_kg=weight_kg,
            duration_days=duration_days
        )

        booking_code = self.generate_booking_code()

        with SessionLocal() as session:
            new_booking = models.BookingModel(
                booking_code=booking_code,
                session_id=booking_data.get("session_id"),
                customer_name=booking_data.get("customer_name", "").strip(),
                customer_phone=phone,
                pet_name=booking_data.get("pet_name", "").strip() if booking_data.get("pet_name") else None,
                pet_type=booking_data.get("pet_type", "dog").strip(),
                weight_kg=weight_kg,
                services=json.dumps(services_list, ensure_ascii=False),
                booking_date=b_date,
                booking_time=b_time,
                duration_days=duration_days,
                estimated_price=quote["total_estimated_price"],
                discount_amount=quote["discount_amount"],
                has_unpriced_service=quote["has_unpriced_service"],
                price_breakdown=json.dumps(quote, ensure_ascii=False),
                notes=booking_data.get("notes", "").strip() if booking_data.get("notes") else None,
                status="PENDING"
            )
            session.add(new_booking)
            session.commit()
            session.refresh(new_booking)
            return self._model_to_dict(new_booking)

    def get_booking_by_code(self, booking_code: str) -> Optional[Dict[str, Any]]:
        """Tra cứu thông tin đặt lịch theo booking_code (Sync)."""
        with SessionLocal() as session:
            item = session.query(models.BookingModel).filter(
                models.BookingModel.booking_code == booking_code.strip().upper()
            ).first()
            return self._model_to_dict(item) if item else None

    def update_booking_status(self, booking_id: int, new_status: str) -> Optional[Dict[str, Any]]:
        """Cập nhật trạng thái đặt lịch (Sync)."""
        status_clean = new_status.strip().upper()
        if status_clean not in VALID_STATUSES:
            raise ValueError(f"Trạng thái không hợp lệ: '{new_status}'.")

        with SessionLocal() as session:
            booking = session.query(models.BookingModel).filter(models.BookingModel.id == booking_id).first()
            if not booking:
                return None
            booking.status = status_clean
            session.commit()
            session.refresh(booking)
            return self._model_to_dict(booking)

    @staticmethod
    def _model_to_dict(item: Optional[models.BookingModel]) -> Optional[Dict[str, Any]]:
        """Chuyển đổi BookingModel thành Dictionary tiện dụng."""
        if not item:
            return None

        try:
            services_data = json.loads(item.services) if item.services else []
        except Exception:
            services_data = [item.services]

        try:
            breakdown_data = json.loads(item.price_breakdown) if item.price_breakdown else None
        except Exception:
            breakdown_data = None

        return {
            "id": item.id,
            "booking_code": item.booking_code,
            "session_id": item.session_id,
            "customer_name": item.customer_name,
            "customer_phone": item.customer_phone,
            "pet_name": item.pet_name,
            "pet_type": item.pet_type,
            "weight_kg": item.weight_kg,
            "services": services_data,
            "booking_date": item.booking_date,
            "booking_time": item.booking_time,
            "duration_days": item.duration_days,
            "estimated_price": item.estimated_price,
            "discount_amount": item.discount_amount,
            "has_unpriced_service": item.has_unpriced_service,
            "price_breakdown": breakdown_data,
            "notes": item.notes,
            "status": item.status,
            "created_at": item.created_at.isoformat() if hasattr(item.created_at, "isoformat") else str(item.created_at),
            "updated_at": item.updated_at.isoformat() if hasattr(item.updated_at, "isoformat") and item.updated_at else None,
        }

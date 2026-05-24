"""
PostgreSQL Database module cho bảng giá dịch vụ Petcare (Sử dụng SQLAlchemy).
"""

import csv
from typing import List, Dict, Optional, Any
from sqlalchemy import text
from src.config import CSV_PRICING_PATH
from src.db.database import engine, SessionLocal
from src.db import models

# Mapping từ tên cột CSV sang tên hiển thị tiếng Việt
SERVICE_NAME_MAP = {
    "luu_tru_24h": "Lưu trú 24h",
    "tam": "Tắm",
    "cao_long": "Cạo lông",
    "cat_mai_mong": "Cắt mài móng",
    "ve_sinh_tai": "Vệ sinh tai",
    "nan_tuyen_hoi": "Nặn tuyến hôi",
}

class ServiceDB:
    """Quản lý kết nối PostgreSQL cho bảng giá dịch vụ Petcare."""

    def __init__(self, db_path: str = None):
        # db_path không còn được dùng nhưng vẫn giữ tham số để tránh lỗi chữ ký hàm
        self.init_db()

    def init_db(self):
        """Khởi tạo các bảng nếu chưa tồn tại và tạo index"""
        models.Base.metadata.create_all(bind=engine)
        # Tạo index nếu chưa tồn tại để tối ưu hiệu năng cho CSDL hiện tại
        with engine.connect() as conn:
            try:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_services_service_type ON services (service_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_services_weight_kg ON services (weight_kg)"))
                conn.commit()
            except Exception as e:
                print(f"⚠️ Không thể tạo index tự động: {e}")

    def _model_to_dict(self, item: models.ServiceModel) -> Optional[Dict[str, Any]]:
        if not item:
            return None
        return {
            "id": item.id,
            "weight_kg": item.weight_kg,
            "service_type": item.service_type,
            "service_name": item.service_name,
            "price": item.price,
            "created_at": item.created_at,
        }

    def import_from_csv(self, csv_path: str = CSV_PRICING_PATH, force: bool = False):
        """Import dữ liệu từ CSV vào Database"""
        with SessionLocal() as db:
            try:
                # Kiểm tra xem có dữ liệu chưa
                services_count = db.query(models.ServiceModel).count()
                if services_count > 0 and not force:
                    print(f"✅ Database đã có {services_count} records. Bỏ qua import (dùng force=True để ghi đè).")
                    return

                if force:
                    # Xóa sạch bảng
                    db.query(models.ServiceModel).delete()
                    db.commit()

                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows_inserted = 0

                    for row in reader:
                        weight = int(row["trong_luong_kg"])

                        for service_type, service_name in SERVICE_NAME_MAP.items():
                            price = int(row[service_type])
                            self._add_service_internal(
                                db=db,
                                weight_kg=weight,
                                service_type=service_type,
                                service_name=service_name,
                                price=price
                            )
                            rows_inserted += 1

                print(f"✅ Đã import {rows_inserted} records từ CSV vào PostgreSQL.")
            except Exception as e:
                db.rollback()
                raise e

    def _add_service_internal(self, db, weight_kg: int, service_type: str, service_name: str, price: int) -> models.ServiceModel:
        """Thao tác INSERT OR REPLACE nội bộ, sử dụng db session có sẵn"""
        service = db.query(models.ServiceModel).filter(
            models.ServiceModel.weight_kg == weight_kg,
            models.ServiceModel.service_type == service_type
        ).first()

        if service:
            service.service_name = service_name
            service.price = price
        else:
            service = models.ServiceModel(
                weight_kg=weight_kg,
                service_type=service_type,
                service_name=service_name,
                price=price
            )
            db.add(service)
        
        db.commit()
        db.refresh(service)
        return service

    def lookup_price(self, service_type: str, weight_kg: float) -> Optional[Dict[str, Any]]:
        with SessionLocal() as db:
            # Tìm mức cân nặng gần nhất >= weight_kg
            service = db.query(models.ServiceModel).filter(
                models.ServiceModel.service_type == service_type,
                models.ServiceModel.weight_kg >= weight_kg
            ).order_by(models.ServiceModel.weight_kg.asc()).first()

            # Nếu không có, lấy mức cao nhất
            if not service:
                service = db.query(models.ServiceModel).filter(
                    models.ServiceModel.service_type == service_type
                ).order_by(models.ServiceModel.weight_kg.desc()).first()

            return self._model_to_dict(service)

    def search_services(self, query: str) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            query_lower = query.lower()
            matching_types = []
            
            for stype, sname in SERVICE_NAME_MAP.items():
                if query_lower in sname.lower() or query_lower in stype.lower():
                    matching_types.append(stype)

            if not matching_types:
                # Fallback: LIKE trên service_name
                results = db.query(models.ServiceModel).filter(
                    models.ServiceModel.service_name.ilike(f"%{query}%")
                ).order_by(models.ServiceModel.weight_kg).all()
            else:
                results = db.query(models.ServiceModel).filter(
                    models.ServiceModel.service_type.in_(matching_types)
                ).order_by(models.ServiceModel.weight_kg).all()

            return [self._model_to_dict(r) for r in results]

    def get_all_services(self) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            results = db.query(models.ServiceModel).order_by(
                models.ServiceModel.service_type, 
                models.ServiceModel.weight_kg
            ).all()
            return [self._model_to_dict(r) for r in results]

    def get_price_table_for_service(self, service_type: str) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            results = db.query(models.ServiceModel).filter(
                models.ServiceModel.service_type == service_type
            ).order_by(models.ServiceModel.weight_kg).all()
            return [self._model_to_dict(r) for r in results]

    def get_service_types(self) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            results = db.query(
                models.ServiceModel.service_type, 
                models.ServiceModel.service_name
            ).distinct().order_by(models.ServiceModel.service_type).all()
            
            return [{"type": r.service_type, "name": r.service_name} for r in results]

    def add_service(self, weight_kg: int, service_type: str, service_name: str, price: int):
        with SessionLocal() as db:
            self._add_service_internal(db, weight_kg, service_type, service_name, price)

    def update_service(self, service_id: int, price: int):
        with SessionLocal() as db:
            service = db.query(models.ServiceModel).filter(models.ServiceModel.id == service_id).first()
            if service:
                service.price = price
                db.commit()

    def delete_service(self, service_id: int) -> bool:
        with SessionLocal() as db:
            service = db.query(models.ServiceModel).filter(models.ServiceModel.id == service_id).first()
            if service:
                db.delete(service)
                db.commit()
                return True
            return False

    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """Format kết quả tra cứu thành text dễ đọc cho LLM."""
        if not results:
            return "Không tìm thấy dịch vụ phù hợp trong bảng giá."

        lines = ["📋 BẢNG GIÁ DỊCH VỤ PETCARE:", ""]

        # Group theo service_name
        grouped = {}
        for r in results:
            sname = r["service_name"]
            if sname not in grouped:
                grouped[sname] = []
            grouped[sname].append(r)

        for service_name, items in grouped.items():
            lines.append(f"🔹 {service_name}:")
            for item in items:
                price_formatted = f"{item['price']:,}đ".replace(",", ".")
                lines.append(f"   • {item['weight_kg']}kg: {price_formatted}")
            lines.append("")

        lines.append("📌 Lưu ý: Giá lưu trú đã bao gồm dịch vụ ăn uống.")
        return "\n".join(lines)

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy lịch sử hội thoại của session từ Supabase."""
        with SessionLocal() as db:
            results = db.query(models.ChatHistoryModel).filter(
                models.ChatHistoryModel.session_id == session_id
            ).order_by(models.ChatHistoryModel.created_at.desc()).limit(limit).all()
            # Reverse order to return chronological history
            results = results[::-1]
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "role": r.role,
                    "content": r.content,
                    "created_at": r.created_at,
                }
                for r in results
            ]

    def save_chat_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """Lưu tin nhắn mới vào lịch sử hội thoại của session trên Supabase."""
        with SessionLocal() as db:
            msg = models.ChatHistoryModel(
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return {
                "id": msg.id,
                "session_id": msg.session_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
            }

    def clear_chat_history(self, session_id: str):
        """Xóa lịch sử hội thoại của session trên Supabase."""
        with SessionLocal() as db:
            db.query(models.ChatHistoryModel).filter(
                models.ChatHistoryModel.session_id == session_id
            ).delete()
            db.commit()

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from .database import Base

class ServiceModel(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, index=True)
    weight_kg = Column(Integer, nullable=False, index=True)
    service_type = Column(String, nullable=False, index=True)
    service_name = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('weight_kg', 'service_type', name='uq_weight_service'),
    )

class ChatHistoryModel(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' hoặc 'assistant'
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class BookingModel(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    booking_code = Column(String(32), unique=True, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=True)
    customer_name = Column(String(100), nullable=False)
    customer_phone = Column(String(20), index=True, nullable=False)
    pet_name = Column(String(50), nullable=True)
    pet_type = Column(String(20), nullable=False)  # 'dog', 'cat', 'other'
    weight_kg = Column(Float, nullable=True)
    services = Column(String, nullable=False)       # JSON string list: ["tam", "kham_benh"]
    booking_date = Column(String(10), nullable=False) # YYYY-MM-DD
    booking_time = Column(String(10), nullable=False) # HH:MM
    duration_days = Column(Integer, default=1)
    estimated_price = Column(Integer, nullable=True) # None nếu chỉ khám bệnh
    discount_amount = Column(Integer, default=0)
    has_unpriced_service = Column(Boolean, default=False)
    price_breakdown = Column(String, nullable=True) # JSON string chi tiết giá
    notes = Column(String(500), nullable=True)
    status = Column(String(20), default="PENDING", index=True) # PENDING, CONFIRMED, CANCELLED, COMPLETED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

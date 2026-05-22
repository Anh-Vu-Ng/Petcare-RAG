from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint
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

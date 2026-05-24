from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Any, Optional
from . import models

def init_db(engine):
    """Tạo tất cả các bảng nếu chưa tồn tại"""
    models.Base.metadata.create_all(bind=engine)

def get_all_services(db: Session) -> List[models.ServiceModel]:
    return db.query(models.ServiceModel).order_by(
        models.ServiceModel.service_type, 
        models.ServiceModel.weight_kg
    ).all()

def lookup_price(db: Session, service_type: str, weight_kg: float) -> Optional[models.ServiceModel]:
    """Tìm mức cân nặng gần nhất >= weight_kg. Nếu không có, lấy mức max."""
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

    return service

def search_services(db: Session, query: str, service_name_map: Dict[str, str]) -> List[models.ServiceModel]:
    """Tìm kiếm dịch vụ theo query"""
    query_lower = query.lower()
    matching_types = []
    
    for stype, sname in service_name_map.items():
        if query_lower in sname.lower() or query_lower in stype.lower():
            matching_types.append(stype)

    if not matching_types:
        # Fallback: LIKE trên service_name
        return db.query(models.ServiceModel).filter(
            models.ServiceModel.service_name.ilike(f"%{query}%")
        ).order_by(models.ServiceModel.weight_kg).all()
    else:
        return db.query(models.ServiceModel).filter(
            models.ServiceModel.service_type.in_(matching_types)
        ).order_by(models.ServiceModel.weight_kg).all()

def get_price_table_for_service(db: Session, service_type: str) -> List[models.ServiceModel]:
    return db.query(models.ServiceModel).filter(
        models.ServiceModel.service_type == service_type
    ).order_by(models.ServiceModel.weight_kg).all()

def get_service_types(db: Session) -> List[Dict[str, Any]]:
    # Dùng distinct trên service_type, service_name
    results = db.query(
        models.ServiceModel.service_type, 
        models.ServiceModel.service_name
    ).distinct().order_by(models.ServiceModel.service_type).all()
    
    return [{"type": r.service_type, "name": r.service_name} for r in results]

def add_service(db: Session, weight_kg: int, service_type: str, service_name: str, price: int) -> models.ServiceModel:
    # INSERT OR REPLACE logic
    # Tìm xem đã có service_type + weight_kg chưa
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

def update_service(db: Session, service_id: int, price: int) -> Optional[models.ServiceModel]:
    service = db.query(models.ServiceModel).filter(models.ServiceModel.id == service_id).first()
    if service:
        service.price = price
        db.commit()
        db.refresh(service)
    return service

def delete_service(db: Session, service_id: int) -> bool:
    service = db.query(models.ServiceModel).filter(models.ServiceModel.id == service_id).first()
    if service:
        db.delete(service)
        db.commit()
        return True
    return False

# --- Chat History CRUD ---
def save_chat_message(db: Session, session_id: str, role: str, content: str) -> models.ChatHistoryModel:
    msg = models.ChatHistoryModel(
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_chat_history(db: Session, session_id: str, limit: int = 50) -> List[models.ChatHistoryModel]:
    results = db.query(models.ChatHistoryModel).filter(
        models.ChatHistoryModel.session_id == session_id
    ).order_by(models.ChatHistoryModel.created_at.desc()).limit(limit).all()
    return results[::-1]


def clear_chat_history(db: Session, session_id: str):
    db.query(models.ChatHistoryModel).filter(
        models.ChatHistoryModel.session_id == session_id
    ).delete()
    db.commit()

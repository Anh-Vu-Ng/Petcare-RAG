from typing import Dict, Any, Optional, List
from src.service_db import ServiceDB, SERVICE_NAME_MAP


def lookup_service_price(
    db: ServiceDB,
    query: str,
    service_type: str = None,
    weight_kg: float = None,
) -> str:
    """
    Tra cứu bảng giá dịch vụ Petcare từ database.

    Args:
        db: ServiceDB instance.
        query: Mô tả dịch vụ cần tra (VD: "tắm", "grooming").
        service_type: Loại dịch vụ cụ thể (VD: "tam", "luu_tru_24h").
        weight_kg: Cân nặng thú cưng (kg).

    Returns:
        Bảng giá format text cho LLM.
    """
    # Nếu có service_type + weight_kg cụ thể → lookup chính xác
    if service_type and weight_kg:
        result = db.lookup_price(service_type, weight_kg)
        if result:
            return db.format_for_llm([result])
    
    # Nếu có service_type nhưng không có weight → lấy toàn bộ bảng giá dịch vụ đó
    if service_type:
        results = db.get_price_table_for_service(service_type)
        if results:
            return db.format_for_llm(results)

    # Fallback: tìm kiếm theo query text
    if query:
        # Thử map query text sang service_type
        resolved_type = _resolve_service_type(query)
        if resolved_type:
            if weight_kg:
                result = db.lookup_price(resolved_type, weight_kg)
                if result:
                    return db.format_for_llm([result])
            else:
                results = db.get_price_table_for_service(resolved_type)
                if results:
                    return db.format_for_llm(results)

        # Tìm bằng LIKE
        results = db.search_services(query)
        if results:
            return db.format_for_llm(results)

    return "Không tìm thấy dịch vụ phù hợp. Vui lòng liên hệ Petcare để được tư vấn cụ thể."


def calculate_final_price(
    base_price_per_day: int,
    num_days: int,
    service_type: str = "luu_tru_24h",
    weight_kg: float = None,
    db: ServiceDB = None,
) -> Dict[str, Any]:
    """
    Tính giá cuối cùng cho dịch vụ lưu trú với discount.

    Chính sách discount:
      - ≤ 3 ngày:  0%
      - 4-5 ngày:  5%
      - 6-10 ngày: 10%
      - > 10 ngày: 15% + tắm miễn phí

    Giá lưu trú đã bao gồm ăn uống (không phát sinh thêm chi phí).

    Args:
        base_price_per_day: Giá gốc 1 ngày (VND).
        num_days: Số ngày lưu trú.
        service_type: Loại dịch vụ (mặc định "luu_tru_24h").
        weight_kg: Cân nặng (dùng để tra giá tắm free nếu có).
        db: ServiceDB instance (dùng để tra giá tắm free).

    Returns:
        Dict chứa chi tiết tính giá.
    """
    total_before_discount = base_price_per_day * num_days

    if num_days > 10:
        discount_pct = 15
        free_bath = True
    elif num_days > 5:
        discount_pct = 10
        free_bath = False
    elif num_days > 3:
        discount_pct = 5
        free_bath = False
    else:
        discount_pct = 0
        free_bath = False

    discount_amount = int(total_before_discount * discount_pct / 100)
    final_price = total_before_discount - discount_amount

    result = {
        "base_price_per_day": base_price_per_day,
        "num_days": num_days,
        "total_before_discount": total_before_discount,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "final_price": final_price,
        "free_bath": free_bath,
        "free_bath_price": 0,
        "note": "Giá lưu trú đã bao gồm ăn uống, không phát sinh thêm chi phí.",
    }

    # Nếu được tắm free, tra giá tắm để hiển thị
    if free_bath and db and weight_kg:
        bath_info = db.lookup_price("tam", weight_kg)
        if bath_info:
            result["free_bath_price"] = bath_info["price"]

    return result


def format_final_price_for_llm(price_result: Dict[str, Any]) -> str:
    """
    Format kết quả calculate_final_price thành text cho LLM.

    Args:
        price_result: Dict trả về từ calculate_final_price().

    Returns:
        String mô tả chi tiết giá, sẵn sàng đưa vào prompt.
    """
    lines = [
        "💰 CHI TIẾT TÍNH GIÁ LƯU TRÚ:",
        "",
        f"• Giá gốc/ngày: {price_result['base_price_per_day']:,}đ".replace(",", "."),
        f"• Số ngày lưu trú: {price_result['num_days']} ngày",
        f"• Tổng trước giảm: {price_result['total_before_discount']:,}đ".replace(",", "."),
    ]

    if price_result["discount_pct"] > 0:
        lines.append(
            f"• Giảm giá: {price_result['discount_pct']}% "
            f"(-{price_result['discount_amount']:,}đ)".replace(",", ".")
        )

    lines.append(f"• 💵 TỔNG THANH TOÁN: {price_result['final_price']:,}đ".replace(",", "."))

    if price_result["free_bath"]:
        bath_price = price_result.get("free_bath_price", 0)
        bath_text = f" (trị giá {bath_price:,}đ)".replace(",", ".") if bath_price else ""
        lines.append(f"• 🎁 TẶNG KÈM: Tắm miễn phí{bath_text}")

    lines.append(f"• 📌 {price_result['note']}")

    return "\n".join(lines)


def _resolve_service_type(query: str) -> Optional[str]:
    """
    Map query text sang service_type dựa trên keywords.

    Args:
        query: Query text từ user.

    Returns:
        service_type string hoặc None.
    """
    query_lower = query.lower()

    keyword_map = {
        "luu_tru_24h": [
            "lưu trú", "gửi", "ở lại", "nội trú", "boarding", "lưu trú 24h", "hotel",
            "lưu trữ", "khách sạn", "pet hotel", "gửi chó", "gửi mèo", "ở nhờ", "ở tạm", # synonyms & LLM rewrites
            "đi vắng", "vắng nhà", "chủ đi vắng", "chủ vắng nhà", "đi chơi", "đi du lịch", "công tác",
            "luu tru", "gui", "o lai", "noi tru", "luu tru 24h", "khach san", "di vang", "vang nha", "chu di vang", "chu vang nha", "o tam", # không dấu
        ],
        "tam": [
            "tắm", "tắm rửa", "bath", "spa",
            "tam",  # không dấu
        ],
        "cao_long": [
            "cạo lông", "cạo", "cắt lông", "grooming", "trim",
            "cao long", "cat long",  # không dấu
        ],
        "cat_mai_mong": [
            "cắt móng", "mài móng", "móng", "nail",
            "cat mong", "mai mong", "mong",  # không dấu
        ],
        "ve_sinh_tai": [
            "vệ sinh tai", "tai", "ear",
            "ve sinh tai",  # không dấu
        ],
        "nan_tuyen_hoi": [
            "nặn tuyến hôi", "tuyến hôi", "tuyến", "gland",
            "nan tuyen hoi", "tuyen hoi", "tuyen",  # không dấu
        ],
    }

    for stype, keywords in keyword_map.items():
        for kw in keywords:
            if kw in query_lower:
                return stype

    return None


def _resolve_all_service_types(query: str) -> List[str]:
    """
    Tìm tất cả các loại dịch vụ có trong query.

    Args:
        query: Query text từ user.

    Returns:
        List chứa các service_type string.
    """
    query_lower = query.lower()
    found_types = []

    keyword_map = {
        "luu_tru_24h": [
            "lưu trú", "gửi", "ở lại", "nội trú", "boarding", "lưu trú 24h", "hotel",
            "lưu trữ", "khách sạn", "pet hotel", "gửi chó", "gửi mèo", "ở nhờ", "ở tạm",
            "đi vắng", "vắng nhà", "chủ đi vắng", "chủ vắng nhà", "đi chơi", "đi du lịch", "công tác",
            "luu tru", "gui", "o lai", "noi tru", "luu tru 24h", "khach san", "di vang", "vang nha", "chu di vang", "chu vang nha", "o tam",
        ],
        "tam": [
            "tắm", "tắm rửa", "bath", "spa",
            "tam",
        ],
        "cao_long": [
            "cạo lông", "cạo", "cắt lông", "grooming", "trim",
            "cao long", "cat long",
        ],
        "cat_mai_mong": [
            "cắt móng", "mài móng", "móng", "nail",
            "cat mong", "mai mong", "mong",
        ],
        "ve_sinh_tai": [
            "vệ sinh tai", "tai", "ear",
            "ve sinh tai",
        ],
        "nan_tuyen_hoi": [
            "nặn tuyến hôi", "tuyến hôi", "tuyến", "gland",
            "nan tuyen hoi", "tuyen hoi", "tuyen",
        ],
    }

    for stype, keywords in keyword_map.items():
        for kw in keywords:
            if kw in query_lower:
                if stype not in found_types:
                    found_types.append(stype)
                break

    return found_types

import os
import sys
import time
import asyncio
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Thêm thư mục root của dự án vào python path để import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_chain import build_conversational_rag_chain
from src.service_db import ServiceDB
from src.booking_service import BookingService
from src.api.schemas import (
    PriceEstimateRequest,
    PriceEstimateResponse,
    AvailableSlotsResponse,
    BookingCreateRequest,
    BookingResponse,
    BookingListResponse,
    BookingStatusUpdateRequest,
)

class MockAgenticRAGPipeline:
    """Giả lập AgenticRAGPipeline phục vụ load test cô lập."""
    async def invoke(self, inputs: dict) -> dict:
        # 1. Giả lập thời gian LLM xử lý (Random từ 3s - 30s)
        delay = random.uniform(3.0, 30.0)
        await asyncio.sleep(delay)
        
        # 2. Trả về kết quả JSON giả lập cấu trúc giống RAG thật
        return {
            "answer": f"Đây là câu trả lời giả lập cho câu hỏi: '{inputs.get('input')}' (Thời gian giả lập xử lý: {delay:.2f} giây).",
            "context": [],
            "from_cache": False,
            "standalone_query": inputs.get("input"),
            "intent": "KNOWLEDGE",
            "price_data": "",
            "timing": {
                "mock_delay": delay,
                "rewrite_query": 0.10,
                "intent_router": 0.15,
                "qa_generation": round(delay - 0.25, 3)
            }
        }

# Biến toàn cục giữ instance của pipeline, service_db và booking_service
pipeline = None
service_db = None
booking_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi tạo và giải phóng tài nguyên của FastAPI."""
    global pipeline, service_db, booking_service
    
    mock_rag_env = os.getenv("MOCK_RAG", "false").lower() == "true"
    
    if mock_rag_env:
        print("[Lifespan] Initializing Mock Conversational RAG Chain (Load Test Mode)...")
    else:
        print("[Lifespan] Initializing Conversational RAG Chain and Database Wrapper...")
        
    try:
        if mock_rag_env:
            pipeline = MockAgenticRAGPipeline()
        else:
            pipeline = build_conversational_rag_chain()
        service_db = ServiceDB()
        # Khởi tạo các bảng và index bất đồng bộ
        await service_db.init_db_async()
        booking_service = BookingService(service_db)
        print("[Lifespan] Initialization successful!")
    except Exception as e:
        print(f"[Lifespan] Critical error during initialization: {e}")
        raise e
    yield
    print("[Lifespan] Shutting down application...")
    if pipeline is not None and hasattr(pipeline, "aclose"):
        await pipeline.aclose()

app = FastAPI(
    title="Petcare Agentic RAG API",
    description="REST API Backend cho chatbot tư vấn dịch vụ Petcare sử dụng Agentic RAG và Supabase",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS bảo mật
origins = [
    "https://petcare-seven-beta.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https://petcare-fe-iota.*\\.vercel\\.app",  # Cho phép tất cả các preview URL của Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID định danh phiên chat của client")
    message: str = Field(..., description="Tin nhắn từ người dùng gửi lên chatbot")

class MessageResponse(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")
    created_at: str = Field(..., description="Thời gian tin nhắn được lưu")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời của AI")
    intent: str = Field(..., description="Ý định câu hỏi được classify (KNOWLEDGE hoặc TOOL)")
    from_cache: bool = Field(..., description="Câu trả lời có được lấy từ Semantic Cache hay không")
    elapsed_time: float = Field(..., description="Tổng thời gian xử lý phản hồi (giây)")
    num_docs: int = Field(..., description="Số lượng tài liệu tham chiếu từ RAG")
    context_docs: List[Dict[str, Any]] = Field(..., description="Danh sách chi tiết tài liệu tham chiếu")
    price_data: Optional[str] = Field(None, description="Bảng giá dịch vụ thô trả về từ tool (nếu có)")
    timing: Optional[Dict[str, Any]] = Field(None, description="Chi tiết thời gian chạy của từng thành phần pipeline")
    standalone_query: Optional[str] = Field(None, description="Câu hỏi độc lập được rút gọn từ lịch sử chat")

# --- Endpoints ---
@app.get("/", include_in_schema=False)
async def index():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Kiểm tra tình trạng hoạt động của API và kết nối Database."""
    if pipeline is None or service_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống chưa khởi tạo hoàn tất hoặc đang gặp lỗi."
        )
    return {"status": "healthy", "database": "connected"}

@app.get("/api/services", status_code=status.HTTP_200_OK)
async def get_services():
    """Lấy danh sách toàn bộ dịch vụ và bảng giá từ database."""
    try:
        services = await service_db.get_all_services_async()
        return services
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi truy vấn danh sách dịch vụ: {str(e)}"
        )

@app.get("/api/chat/history/{session_id}", response_model=List[MessageResponse])
async def get_history(session_id: str):
    """Lấy lại lịch sử chat của một session đã lưu trong database."""
    try:
        history_records = await service_db.get_chat_history_async(session_id)
        # Convert từ format lưu trữ DB ('human'/'ai') sang format hiển thị FE ('user'/'assistant')
        response = []
        for r in history_records:
            role = "user" if r["role"] == "human" else "assistant"
            # Format datetime sang string ISO 8601
            created_at_str = r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"])
            response.append(
                MessageResponse(
                    role=role,
                    content=r["content"],
                    created_at=created_at_str
                )
            )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đọc lịch sử chat: {str(e)}"
        )

@app.post("/api/chat/send", response_model=ChatResponse)
async def send_chat(payload: ChatRequest):
    """Gửi câu hỏi tới RAG chatbot. Xử lý RAG và lưu lịch sử chat vào database."""
    t_start = asyncio.get_event_loop().time()
    
    session_id = payload.session_id.strip()
    message = payload.message.strip()
    
    if not session_id or not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id và message không được để trống."
        )
    
    try:
        mock_db = os.getenv("MOCK_DB", "false").lower() == "true"
        
        # 1. Tải lịch sử chat hiện tại từ DB để đưa vào LangChain
        if mock_db:
            history_records = []
        else:
            history_records = await service_db.get_chat_history_async(session_id)
            
        chat_history = []
        for r in history_records:
            role = "human" if r["role"] in ["human", "user"] else "ai"
            chat_history.append((role, r["content"]))
            
        # Giới hạn số lượng hội thoại truyền vào context để tránh quá tải token (khoảng 5 lượt gần nhất)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        # 2. Gọi RAG Pipeline xử lý câu hỏi
        mock_rag_env = os.getenv("MOCK_RAG", "false").lower() == "true"
        if mock_rag_env:
            # Pipeline mock là hàm async
            response = await pipeline.invoke({
                "chat_history": chat_history,
                "input": message
            })
        else:
            # Pipeline thật chạy hoàn toàn async
            response = await pipeline.ainvoke({
                "chat_history": chat_history,
                "input": message
            })
        
        answer = response.get("answer", "")
        intent = response.get("intent", "KNOWLEDGE")
        from_cache = response.get("from_cache", False)
        context = response.get("context", [])
        price_data = response.get("price_data", "")
        timing = response.get("timing", {})
        standalone_query = response.get("standalone_query", "")
        
        # 3. Lưu tin nhắn mới của user và bot vào Database (chạy song song để tối ưu độ trễ)
        if not mock_db:
            await asyncio.gather(
                service_db.save_chat_message_async(session_id, "human", message),
                service_db.save_chat_message_async(session_id, "ai", answer)
            )
        
        # 4. Trả kết quả về
        elapsed = asyncio.get_event_loop().time() - t_start
        
        context_docs = []
        for doc in context:
            source = doc.metadata.get("source", "N/A")
            # Trích xuất tên file từ đường dẫn đầy đủ cho gọn
            if "/" in source or "\\" in source:
                source = os.path.basename(source)
            context_docs.append({
                "source": source,
                "content": doc.page_content[:500]  # Giới hạn ký tự docs trả về để tiết kiệm băng thông
            })
            
        return ChatResponse(
            answer=answer,
            intent=intent,
            from_cache=from_cache,
            elapsed_time=round(elapsed, 3),
            num_docs=len(context),
            context_docs=context_docs,
            price_data=price_data,
            timing=timing,
            standalone_query=standalone_query
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý chatbot RAG: {str(e)}"
        )

@app.delete("/api/chat/history/{session_id}", status_code=status.HTTP_200_OK)
async def clear_history(session_id: str):
    """Xóa toàn bộ lịch sử chat của một session."""
    try:
        await service_db.clear_chat_history_async(session_id)
        return {"status": "success", "message": f"Đã xóa lịch sử của session {session_id}."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa lịch sử chat: {str(e)}"
        )


# ==========================================
# --- BOOKING & PRICE ESTIMATION ENDPOINTS ---
# ==========================================

@app.post("/api/v1/bookings/estimate-price", response_model=PriceEstimateResponse, tags=["Bookings"])
async def estimate_price(payload: PriceEstimateRequest):
    """
    Tính toán chi tiết báo giá thời gian thực cho các dịch vụ đã chọn.
    
    Quy tắc nghiệp vụ đặc biệt:
    - Nếu chọn 'kham_benh': Giá mục này sẽ bị ẩn (`hide_price: True`), hiển thị 'Báo giá sau khi khám', và cờ `has_unpriced_service = True`.
    - Nếu chọn 'luu_tru_24h': Áp dụng chiết khấu giảm giá theo số ngày lưu trú và tặng tắm nếu > 10 ngày.
    """
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    try:
        quote = booking_service.calculate_booking_quote(
            services=payload.services,
            weight_kg=payload.weight_kg,
            duration_days=payload.duration_days or 1
        )
        return PriceEstimateResponse(**quote)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi tính toán báo giá: {str(e)}"
        )


@app.get("/api/v1/bookings/available-slots", response_model=AvailableSlotsResponse, tags=["Bookings"])
async def get_available_slots(date: str):
    """
    Kiểm tra và lấy danh sách các khung giờ đặt hẹn hợp lệ trong ngày.
    
    - Thứ 2 - Thứ 7: Sáng 08:00 - 12:00, Chiều 14:00 - 19:00 (bước 30 phút).
    - Chủ nhật: Sáng 08:00 - 12:00.
    """
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    slots_info = booking_service.get_available_time_slots(date)
    return AvailableSlotsResponse(**slots_info)


@app.post("/api/v1/bookings", response_model=BookingResponse, status_code=status.HTTP_201_CREATED, tags=["Bookings"])
async def create_booking(payload: BookingCreateRequest):
    """
    Tạo mới một yêu cầu đặt lịch dịch vụ.
    
    - Kiểm tra định dạng số điện thoại Việt Nam (10 số).
    - Kiểm tra ngày giờ hẹn theo khung giờ làm việc của Petcare.
    - Tự động sinh mã định danh duy nhất BK-YYYYMMDD-XXXX và lưu vào cơ sở dữ liệu.
    """
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    try:
        booking_record = await booking_service.create_booking_async(payload.model_dump())
        return BookingResponse(**booking_record)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lưu đặt lịch: {str(e)}"
        )


@app.get("/api/v1/bookings", response_model=BookingListResponse, tags=["Bookings"])
async def list_bookings(
    status_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Lấy danh sách các đơn đặt lịch (dành cho Quản trị viên / Nhân viên).
    
    Hỗ trợ:
    - Lọc theo trạng thái: PENDING, CONFIRMED, CANCELLED, COMPLETED.
    - Lọc theo khoảng ngày hẹn: date_from, date_to (YYYY-MM-DD).
    - Tìm kiếm theo SĐT, Tên khách, Tên pet hoặc Mã đơn đặt lịch.
    - Phân trang: limit, offset.
    """
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    try:
        result = await booking_service.list_bookings_async(
            status=status_filter,
            date_from=date_from,
            date_to=date_to,
            search=search,
            limit=limit,
            offset=offset,
        )
        return BookingListResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi truy vấn danh sách đặt lịch: {str(e)}"
        )


@app.get("/api/v1/bookings/{booking_code}", response_model=BookingResponse, tags=["Bookings"])
async def get_booking_detail(booking_code: str):
    """Tra cứu chi tiết một đơn đặt lịch theo mã booking_code (VD: BK-20260827-A8F2)."""
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    try:
        booking = await booking_service.get_booking_by_code_async(booking_code)
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy đơn đặt lịch với mã: '{booking_code}'"
            )
        return BookingResponse(**booking)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tra cứu đặt lịch: {str(e)}"
        )


@app.patch("/api/v1/bookings/{booking_id}/status", response_model=BookingResponse, tags=["Bookings"])
async def update_booking_status(booking_id: int, payload: BookingStatusUpdateRequest):
    """
    Cập nhật trạng thái đơn đặt lịch (Duyệt đơn, Hủy đơn, Hoàn tất).
    
    Trạng thái chấp nhận: CONFIRMED, CANCELLED, COMPLETED.
    """
    if booking_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Booking chưa sẵn sàng."
        )
    try:
        updated = await booking_service.update_booking_status_async(booking_id, payload.status)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy đơn đặt lịch với ID: {booking_id}"
            )
        return BookingResponse(**updated)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật trạng thái đặt lịch: {str(e)}"
        )



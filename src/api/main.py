import os
import sys
import time
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

# Biến toàn cục giữ instance của pipeline và service_db
pipeline = None
service_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi tạo và giải phóng tài nguyên của FastAPI."""
    global pipeline, service_db
    print("[Lifespan] Đang khởi tạo Conversational RAG Chain và Database Wrapper...")
    try:
        pipeline = build_conversational_rag_chain()
        service_db = ServiceDB()
        print("[Lifespan] Khởi tạo thành công!")
    except Exception as e:
        print(f"[Lifespan] Lỗi nghiêm trọng khi khởi tạo: {e}")
        raise e
    yield
    print("[Lifespan] Đang tắt ứng dụng...")

app = FastAPI(
    title="Petcare Agentic RAG API",
    description="REST API Backend cho chatbot tư vấn dịch vụ Petcare sử dụng Agentic RAG và Supabase",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS bảo mật
origins = [
    "https://petcare-fe-iota.vercel.app",
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

# --- Endpoints ---
@app.get("/", include_in_schema=False)
def index():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/api/health", status_code=status.HTTP_200_OK)
def health_check():
    """Kiểm tra tình trạng hoạt động của API và kết nối Database."""
    if pipeline is None or service_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống chưa khởi tạo hoàn tất hoặc đang gặp lỗi."
        )
    return {"status": "healthy", "database": "connected"}

@app.get("/api/services", status_code=status.HTTP_200_OK)
def get_services():
    """Lấy danh sách toàn bộ dịch vụ và bảng giá từ database."""
    try:
        services = service_db.get_all_services()
        return services
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi truy vấn danh sách dịch vụ: {str(e)}"
        )

@app.get("/api/chat/history/{session_id}", response_model=List[MessageResponse])
def get_history(session_id: str):
    """Lấy lại lịch sử chat của một session đã lưu trong database."""
    try:
        history_records = service_db.get_chat_history(session_id)
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
def send_chat(payload: ChatRequest):
    """Gửi câu hỏi tới RAG chatbot. Xử lý RAG và lưu lịch sử chat vào database."""
    t_start = time.time()
    
    session_id = payload.session_id.strip()
    message = payload.message.strip()
    
    if not session_id or not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id và message không được để trống."
        )
    
    try:
        # 1. Tải lịch sử chat hiện tại từ DB để đưa vào LangChain
        history_records = service_db.get_chat_history(session_id)
        chat_history = []
        for r in history_records:
            role = "human" if r["role"] in ["human", "user"] else "ai"
            chat_history.append((role, r["content"]))
            
        # Giới hạn số lượng hội thoại truyền vào context để tránh quá tải token (khoảng 5 lượt gần nhất)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        # 2. Gọi RAG Pipeline xử lý câu hỏi
        response = pipeline.invoke({
            "chat_history": chat_history,
            "input": message
        })
        
        answer = response.get("answer", "")
        intent = response.get("intent", "KNOWLEDGE")
        from_cache = response.get("from_cache", False)
        context = response.get("context", [])
        price_data = response.get("price_data", "")
        timing = response.get("timing", {})
        
        # 3. Lưu tin nhắn mới của user và bot vào Database
        service_db.save_chat_message(session_id, "human", message)
        service_db.save_chat_message(session_id, "ai", answer)
        
        # 4. Trả kết quả về
        elapsed = time.time() - t_start
        
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
            timing=timing
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý chatbot RAG: {str(e)}"
        )

@app.delete("/api/chat/history/{session_id}", status_code=status.HTTP_200_OK)
def clear_history(session_id: str):
    """Xóa toàn bộ lịch sử chat của một session."""
    try:
        service_db.clear_chat_history(session_id)
        return {"status": "success", "message": f"Đã xóa lịch sử của session {session_id}."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa lịch sử chat: {str(e)}"
        )

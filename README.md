# 🛍️ Petcare RAG Hybrid — Hệ thống Tư vấn Chăm sóc Thú cưng

> Project này được xây dựng để tạo một hệ thống chatbot tư vấn chăm sóc khách hàng tự động, sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** kết hợp **Hybrid Search** (BM25 + FAISS), thuật toán **Reciprocal Rank Fusion (RRF)** và **Jina Reranker** để trả lời chính xác dựa trên nguồn dữ liệu Petcare.

---

## 📑 Mục lục

- [Tổng quan](#-tổng-quan)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 🔍 Tổng quan

Petcare RAG Hybrid là một hệ thống **Conversational RAG** được thiết kế để đóng vai trò là một nhân viên tư vấn cho khách hàng về sản phẩm và dịch vụ. Hệ thống hoạt động theo nguyên lý:

1. **Thu thập dữ liệu** từ file PDF và các URL của website Petcare.
2. **Chia nhỏ** văn bản thành các chunk có kích thước phù hợp.
3. **Lập chỉ mục kép**: FAISS (tìm kiếm ngữ nghĩa) + BM25 (tìm kiếm từ khóa).
4. **Hỗ trợ hội thoại đa lượt** với cơ chế viết lại câu hỏi (Query Rewriting) dựa trên lịch sử chat để tạo câu hỏi độc lập.
5. **Kiểm tra bộ nhớ đệm (Semantic Cache)**: Sử dụng câu hỏi độc lập để đánh giá độ tương đồng ngữ nghĩa, trả về ngay câu trả lời nếu đã từng được hỏi.
6. **Truy xuất lai (Hybrid Retrieval)** kết hợp kết quả từ cả hai hệ thống qua thuật toán RRF nếu không tìm thấy trong Cache.
7. **Đánh giá lại (Reranking)** sử dụng mô hình Jina Reranker để chọn lọc các tài liệu phù hợp nhất.
8. **Sinh câu trả lời** bằng LLM (qua OpenRouter API) dựa trên ngữ cảnh đã truy xuất và lưu trữ lại vào Cache.

---


## 🚀 Các bước Cài đặt & Chạy hệ thống

Dự án sử dụng trình quản lý gói `uv` của Astral để quản lý môi trường và dependencies.

### 1. Yêu cầu hệ thống (Prerequisites)

- **Hệ điều hành**: Windows, macOS, hoặc Linux
- **Python**: Phiên bản `>= 3.13`
- **Công cụ**: Cài đặt sẵn trình quản lý gói `uv` ([Hướng dẫn cài đặt uv](https://docs.astral.sh/uv/getting-started/installation/)).

### 2. Hướng dẫn cài đặt chi tiết

**Bước 1: Clone mã nguồn**
```bash
git clone https://github.com/Bugold/Rag-using-hybrid-search-and-reranking.git
cd Rag-using-hybrid-search-and-reranking
```

**Bước 2: Cài đặt thư viện bằng `uv`**
Chạy lệnh sau để đồng bộ và tự động tạo môi trường ảo (virtual environment) chứa đầy đủ các dependencies cần thiết cho dự án:
```bash
uv sync
```

**Bước 3: Cấu hình biến môi trường**
Tạo file `.env` và điền API Key:
```bash
# Đối với Windows PowerShell
echo "OPENROUTER_API_KEY=sk-or-v1-xxx-key-cua-ban-xxx" > .env
echo "GROQ_API_KEY=gsk_xxx-key-cua-ban-xxx" >> .env
echo "JINA_API_KEY=jina_xxx-key-cua-ban-xxx" >> .env
```

**Bước 4: Nạp dữ liệu đầu vào**
- Chuẩn bị file PDF hướng dẫn về Petcare và đặt tại đường dẫn: `data/rag_docs.pdf`.
- Thêm các đường dẫn URL cần thiết vào file `data/url.txt`.

### 3. Khởi chạy hệ thống
**Chạy giao diện Web bằng Streamlit**
```bash
uv run python -m streamlit run app.py
```
*Truy cập hệ thống tại: `http://localhost:8501`*

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------|
| **Framework** | LangChain | Orchestration cho RAG pipeline |
| **QA LLM** | OpenRouter API | Sinh câu trả lời |
| **Rewriter LLM** | Groq API | Viết lại câu hỏi hội thoại |
| **Embedding** | jina-embeddings-v5-text-small | Chuyển text → vector |
| **Reranker** | jina-reranker-v3 | Đánh giá lại độ liên quan |
| **Vector DB** | FAISS | Tìm kiếm ngữ nghĩa (dense retrieval) |
| **Keyword Search** | BM25 (rank-bm25) | Tìm kiếm từ khóa (sparse retrieval) |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Kết hợp kết quả từ 2 retriever |
| **PDF Parser** | PyMuPDF (fitz) | Trích xuất text từ PDF |
| **Web Crawler** | Requests + BeautifulSoup | Crawl nội dung từ URLs |
| **Frontend** | Streamlit | Giao diện chat tương tác |
| **Package Manager** | uv | Quản lý dependencies nhanh |

---

<p align="center">
  Made with ❤️ for Petcare
</p>

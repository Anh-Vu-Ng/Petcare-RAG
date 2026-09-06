# 🐾 Petcare RAG — Smart Petcare Consulting System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626?style=flat&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Evaluation-DeepEval-8A2BE2?style=flat" alt="DeepEval" />
</p>

> An advanced **RAG** system designed for automated petcare consulting. It features **LLM-Based Intent Routing**, **Semantic Caching**, **Hybrid Search** (Dense Jina-v5-embedding-text-small + Sparse FastEmbed BM25), **Parent Document Chunking**, **Jina-Reranker-v3**, and **Tool Integration** with Supabase/PostgreSQL and SQLite to provide accurate, context-aware veterinary guidance and service quotations.


---

### Evaluation Framework

![Evaluation Framework](docs/images/evaluation_framework.png)

The system is rigorously benchmarked using **DeepEval** with **GPT-4o-mini** acting as an LLM Judge (via OpenRouter) across production test scenarios (`data/test_scenario_pro.json`), achieving a **100% pass rate (50/50 test cases)** across core RAG metrics:
- **Faithfulness**: `0.90` (clinical accuracy without hallucination)
- **Answer Relevancy**: `0.97` (direct, highly pertinent answers)
- **Contextual Precision**: `0.97` (optimal ranking of retrieved medical context)
- **Contextual Recall**: `1.00` (complete retrieval of necessary reference material)

🔗 [Detailed Evaluation Results](outputs/deepeval_results.json) | [DeepEval Test Suite](tests/deep_eval/deepeval.py)

#### 1. Retrieval & Chunking Optimization (Grid Search)

To eliminate retrieval bottlenecks and determine optimal context granularity, an automated Grid Search was conducted on the veterinary medical dataset using **jina-embeddings-v5-text-small**. The search was evaluated on a fixed set of 150 veterinary-related questions across three core hyperparameters:
- `child_size`: Text chunk size for child documents. An initial range of 200-500 tokens was explored, after which the search space was narrowed to approximately 300-320 tokens based on preliminary results
- `overlap`: Overlap between adjacent chunks (10 or 20 tokens) to preserve contextual boundaries.
- `top_k`: Number of retrieved child chunks (6 or 8 chunks).

![Retrieval Tuning - Child Size Grid Search](docs/images/gridsearch_childsize.png)

> **Selected Configuration:** **`child_size=310, overlap=20, top_k=8`** was deployed for production, achieving a top **Hit Rate of 98.67%** and **MRR Score of 0.9406** to guarantee reliable context retrieval for generation. (Raw data: [`data/pdr_tuning_results.csv`](data/pdr_tuning_results.csv))

#### 2. Intent Router Benchmarks & Prompt Optimization

The Intent Router sits at the system gateway to classify incoming queries into `GREETING`, `TOOL` (pricing calculator for 6 listed services), or `KNOWLEDGE` (veterinary QA). 

To benchmark model robustness, I evaluated 4 LLMs on **49 challenge test cases** containing real-world customer behaviors: conversational rambling, preamble chit-chat, and compound intents.

##### A. Baseline Benchmark (Before Prompt Optimization)

With the baseline prompt, smaller models were vulnerable to conversational preamble and noise (e.g., misclassifying tool pricing inquiries as chit-chat greetings).

![Intent Router Benchmark - Before Optimization](docs/images/router_bf.png)


##### B. Optimized Benchmark (After Anti-Noise & Few-Shot Prompt Tuning)

I refined the prompt by introducing an explicit anti-noise instruction (*read full context and focus on the final intent*) along with targeted few-shot trap examples ([docs/adjust_intent_router.md](docs/adjust_intent_router.md)).

![Intent Router Benchmark - After Optimization](docs/images/router_after.png)


> **Key Takeaway:** Prompt optimization significantly boosted open-source models, enabling `qwen3-8b` to match `gpt-4o-mini` at a perfect **100% accuracy**, while `llama-3.1-8b` gained **+16.33%**.

---

### Demo

🌐 [Live Web Application (Vercel)](https://petcare-seven-beta.vercel.app/)

![Live Demo Preview](docs/images/demo_preview.png)
<!-- 📸 Gợi ý ảnh: Ảnh chụp giao diện Web UI trò chuyện trực tiếp với Petcare Assistant -->

---

#### Chatbot Architecture

![Chatbot Architecture](docs/images/system_architecture.png)
<!-- 📸 Gợi ý ảnh: Sơ đồ kiến trúc luồng dữ liệu (Deployment Architecture hoặc RAG Pipeline) -->

> Detailed technical diagrams and sequence flows are documented in [architecture_diagram.md](architecture_diagram.md).

#### +The chatbot can retrieve pet medical knowledge and answer health questions
#### +It can dynamically calculate service pricing & discounts via Tool Calling
#### +It can also handle casual conversations & repeated queries instantly using Intent Router & Semantic Cache

---

#### Send Message Endpoint

```http
POST /api/chat/send
Content-Type: application/json
```

**Request Body:**

```json
{
  "session_id": "pet_owner_001",
  "message": "Bé poodle nhà mình 4kg, mình muốn hỏi giá gói tắm vệ sinh và dịch vụ khách sạn thú cưng 3 ngày"
}
```

**Response:**

```json
{
  "answer": "**Bảng giá cho bé Poodle 4 kg**\n\n| Dịch vụ | Trọng lượng | Giá (đ) |\n|---|---|---|\n| Lưu trú 24h (3 ngày) | 3 – 5 kg | 100.000đ/ngày → **300.000đ** cho 3 ngày (đã bao gồm ăn uống) |\n| Tắm vệ sinh | 3 – 5 kg | **90.000đ – 110.000đ** |\n\n- Giá lưu trú đã bao gồm dịch vụ ăn uống, không phát sinh thêm chi phí.  \n- Giá tắm là mức giá tham khảo trong khoảng 90.000đ – 110.000đ; cụ thể sẽ được xác nhận khi đặt lịch.\n\n**Đặt lịch:** Dạ anh/chị có thể sử dụng cổng đặt lịch trực tuyến của Petcare để chọn ngày giờ và dịch vụ phù hợp, hoặc liên hệ qua Zalo nhé: https://zalo.me/3900819148490236884.",
  "intent": "TOOL",
  "from_cache": false,
  "elapsed_time": 9.411,
  "num_docs": 0,
  "context_docs": [],
  "price_data": "📋 BẢNG GIÁ DỊCH VỤ PETCARE:\n\n🔹 Lưu trú 24h:\n   • 3kg - 5kg: 70.000đ - 100.000đ\n\n📌 Lưu ý: Giá lưu trú đã bao gồm dịch vụ ăn uống.\n\n💰 CHI TIẾT TÍNH GIÁ LƯU TRÚ:\n\n• Giá gốc/ngày: 100.000đ\n• Số ngày lưu trú: 3 ngày\n• Tổng trước giảm: 300.000đ\n• 💵 TỔNG THANH TOÁN: 300.000đ\n• 📌 Giá lưu trú đã bao gồm ăn uống, không phát sinh thêm chi phí.\n\n---\n\n📋 BẢNG GIÁ DỊCH VỤ PETCARE:\n\n🔹 Tắm:\n   • 3kg - 5kg: 90.000đ - 110.000đ\n\n📌 Lưu ý: Giá lưu trú đã bao gồm dịch vụ ăn uống.",
  "timing": {
    "rewrite_query": 0.0000011920928955078125,
    "cache_lookup": 1.6653532981872559,
    "intent_router": 1.2279555797576904,
    "tool_lookup": 1.0454261302947998,
    "qa_generation": 3.8152318000793457
  },
  "standalone_query": "Bé poodle nhà mình 4kg, mình muốn hỏi giá gói tắm vệ sinh và dịch vụ khách sạn thú cưng 3 ngày"
}
```

---

### Setup

#### 1. Installation

Requires **Python >= 3.13**. This project uses [Astral uv](https://docs.astral.sh/uv/) for lightning-fast package management.

```bash
# Clone repository
git clone https://github.com/Anh-Vu-Ng/Petcare-RAG.git
cd Petcare-RAG

# Sync virtual environment and dependencies
uv sync
```

#### 2. Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Inference (OpenRouter / LiteLLM)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx

# Embedding & Reranker (Jina AI)
JINA_API_KEY=jina_xxxxxxxxxxxxxxxx

# Vector Database (Qdrant Cloud)
URL_QDRANT=https://your-cluster-id.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# Relational Database (Supabase PostgreSQL / SQLite fallback)
# Leave empty to automatically fallback to local SQLite (data/petcare_services.db)
DATABASE_URL=postgresql://postgres:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

# DeepEval Evaluation (Optional)
CONFIDENT_API_KEY=your_confident_api_key
```

---

#### 3. Data Preparation

Initialize the database schema, generate table indexes, and import petcare service catalogues and vector embeddings:

```bash
uv run python import_db.py
```

> Make sure your Qdrant cluster is accessible. The script will automatically configure collections with hybrid vector search (Dense 1024-dim + Sparse BM25).

---

#### 4. Customize Your Prompt

You can customize the expert consulting persona and system instructions in `src/rag_chain.py` or `src/intent_router.py`:

```python
PETCARE_CONSULTANT_PROMPT = """
Bạn là Chuyên gia Tư vấn Chăm sóc Thú cưng tận tâm và chuyên nghiệp.
Nhiệm vụ của bạn là giải đáp thắc mắc về sức khỏe, dinh dưỡng và báo giá dịch vụ chính xác.

Câu hỏi của khách hàng: {query}
Thông tin tài liệu chuyên ngành:
{context}

Nguyên tắc:
1. Đưa ra lời khuyên y tế thận trọng, luôn khuyến nghị đưa bé đến phòng khám nếu có triệu chứng khẩn cấp.
2. Với câu hỏi dịch vụ, trích xuất thông tin giá và ưu đãi rõ ràng, minh bạch.
"""
```

---

#### 5. Run the Server

##### Local FastAPI Backend (with Hot-Reload)

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

##### Local Streamlit Dashboard (Web UI)

```bash
uv run streamlit run app.py
```

##### Production Deployment via Docker Compose

```bash
docker compose up -d --build
```

---

#### 6. Test the API

![Swagger UI](docs/images/sw_ui.png)

Interactive Swagger documentation is available locally at:
🔗 [http://localhost:8000/docs](http://localhost:8000/docs)

Quick test using `curl`:

```bash
curl -X POST "http://localhost:8000/api/chat/send" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test_user", "message": "Triệu chứng mèo bị nôn khan là do đâu?"}'
```

---

#### 7. Run Evaluation & Tests

Run **DeepEval evaluation pipeline**:

```bash
uv run python tests/deep_eval/run_deepeval.py
```

Run **all unit tests**:

```bash
uv run pytest tests/unit_test/ -v
```

Run **Retrieval & Rerank parameter tuning**:

```bash
uv run python tests/unit_test/tune_pdr.py
```

**Evaluation & Test Hierarchy:**

- **Integration Evaluation (DeepEval)**
  - LLM Answer Quality
    - Answer Relevancy Metric
    - Faithfulness Metric (Groundedness / Anti-Hallucination)
    - Contextual Precision Metric
    - Contextual Recall Metric
- **Retrieval & Reranker Tests**
  - Hit@K & MRR Benchmarks (Dense vs. Sparse BM25 vs. Hybrid)
  - nDCG@K (Jina Reranker v3 evaluation)
  - Parent-Child Chunking Retrieval verification
- **Agentic Logic & Functional Tests**
  - Intent Router Classification accuracy (`KNOWLEDGE`, `TOOL`, `GREETING`)
  - Tool Calculations (Pricing, stay duration discounts, slot booking)
  - Extreme greeting handling & Semantic Cache hit-rate

---

### 🛠️ Technologies Used

| Component | Technology | Role & Technical Specifications |
| :--- | :--- | :--- |
| **Orchestration** | LangChain 0.3+ | RAG Pipeline orchestration, LCEL chains, and dynamic tool calling |
| **Backend API** | FastAPI + Uvicorn | High-performance asynchronous RESTful API with validated Pydantic schemas |
| **Agentic Logic** | Custom Intent Router | Real-time query intent classification (`KNOWLEDGE`, `TOOL`, `GREETING`) |
| **Database** | Supabase (PostgreSQL) / SQLite | Storage for service catalogues, pricing tiers, appointments, and conversation history |
| **Vector Database** | Qdrant Cloud | Cloud vector database powering Hybrid Search and Semantic Caching |
| **Sparse Retrieval** | FastEmbed (BM25) | Keyword-based sparse vector generation for hybrid retrieval |
| **Dense Embeddings** | Jina Embeddings v5 | High-dimensional dense vector embeddings (1024-dim) for clinical petcare texts |
| **Reranker** | Jina Reranker v3 | Cross-encoder semantic reranking to optimize Top-K retrieved context precision |
| **LLM Inference** | OpenRouter (LiteLLM) | Multi-model routing and inference serving (GPT-4o, GPT-4o-mini, Qwen, Llama 3.1) |
| **Evaluation** | DeepEval | Automated LLM-as-a-Judge evaluation framework for RAG metric benchmarking |
| **UI Client** | Streamlit & Next.js | Interactive consultation interfaces and administrative service dashboards |
| **Deployment** | Docker & Docker Compose | Full-stack containerization for streamlined VPS and cloud deployment |

---

<p align="center">
  Built with ❤️ for the Petcare Community
</p>
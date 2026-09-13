# 🐾 Petcare RAG — Smart Petcare Consulting System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626?style=flat&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Evaluation-DeepEval-8A2BE2?style=flat" alt="DeepEval" />
</p>

> An advanced **Agentic RAG** system designed for automated, highly reliable petcare consulting. It combines **LLM-Based Intent Routing**, **Qdrant Semantic Caching**, **Hybrid Search** (Dense Jina-v5 + Sparse FastEmbed BM25), **Parent-Child Document Chunking**, **Jina-Reranker-v3**, and **Dynamic Tool Integration** with Supabase/PostgreSQL and SQLite to deliver accurate veterinary guidance and instant service quotations.

---

### ✨ Key Capabilities

- 🩺 **Evidence-Grounded Veterinary QA (`KNOWLEDGE`)**: Delivers accurate clinical, nutritional, and health guidance grounded strictly in veterinary reference material. Employs **Parent-Child Chunking** and **Jina Reranker v3** cross-encoder scoring to eliminate hallucinations (validated by a **0.90 Faithfulness** and **1.00 Contextual Recall** on DeepEval).
- 💰 **Deterministic Pricing & Quotation Engine (`TOOL`)**: Automatically parses pet attributes (species, weight, duration) and queries the PostgreSQL/SQLite database to calculate tiered service pricing, multi-day boarding discounts, and generate transparent quotes via LangChain Tool Calling.
- 💬 **Multi-Turn Context Tracking & Etiquette (`GREETING`)**: Retains conversation history across turns (e.g., accumulating pet weight, breed, and combined services) via an intelligent Query Rewriter, handling polite greetings and customer inquiries naturally.
- ⚡ **Semantic Caching & Business Intent Routing**: Integrated Qdrant-based Semantic Cache to serve repeated queries instantaneously while significantly cutting LLM token usage, paired with an Intent Router to dispatch requests into domain-specific business workflows.

---

### 🌐 Live Demo & Preview

🔗 **Live Web Application**: [https://petcare-seven-beta.vercel.app/](https://petcare-seven-beta.vercel.app/)

![Live Demo Preview](docs/images/demo_preview.png)

---

### 🏗️ System Architecture

The pipeline orchestrates query rewriting, intent classification, semantic caching, hybrid vector retrieval, cross-encoder reranking, and dynamic tool execution:

![Chatbot Architecture](docs/images/system_architecture.png)


---

### 📊 Empirical Evaluation & Benchmarks

#### 1. End-to-End RAG Benchmarking (DeepEval)

![Evaluation Framework](docs/images/evaluation_framework.png)

The system is rigorously benchmarked using **DeepEval** with **GPT-4o-mini** acting as an LLM Judge (via OpenRouter) across 50 production test scenarios (`data/test_scenario_pro.json`), achieving a **100% pass rate (50/50 test cases)** across core RAG metrics:

| Metric | Score | Assessment |
| :--- | :---: | :--- |
| **Faithfulness** | `0.90` | High clinical accuracy with strict adherence to medical context |
| **Answer Relevancy** | `0.97` | Pertinent, focused answers directly addressing the user's need |
| **Contextual Precision** | `0.97` |  Optimal context prioritization by Jina-Reranker-v3 |
| **Contextual Recall** | `1.00` |  Zero retrieval omissions across all reference ground-truth items |

🔗 [Detailed Evaluation JSON Report](outputs/deepeval_results.json) | [DeepEval Test Suite](tests/deep_eval/deepeval.py)

#### 2. Retrieval & Chunking Optimization (Grid Search)

To eliminate retrieval bottlenecks and determine optimal context granularity, an automated Grid Search was conducted on the veterinary medical dataset using **jina-embeddings-v5-text-small** across 150 domain-specific questions:
- `child_size`: Explored from 200–500 tokens, narrowed down to 300–320 tokens.
- `overlap`: Overlap between adjacent chunks (10 vs. 20 tokens) to preserve contextual boundaries.
- `top_k`: Number of retrieved child chunks (6 vs. 8 chunks).

![Retrieval Tuning - Child Size Grid Search](docs/images/gridsearch_childsize.png)

> **Selected Production Configuration:** **`child_size=310, overlap=20, top_k=8`**, achieving a top **Hit Rate of 98.67%** and **MRR Score of 0.9406** to guarantee reliable context retrieval. (Raw data: [`data/pdr_tuning_results.csv`](data/pdr_tuning_results.csv))

#### 3. Intent Router Benchmarks & Prompt Tuning

The Intent Router sits at the system gateway to classify incoming queries into `GREETING`, `TOOL` (pricing calculator for 6 listed services), or `KNOWLEDGE` (veterinary QA). Models were benchmarked against **49 challenge test cases** containing real-world customer noise, conversational preamble, and compound requests.

##### Baseline Benchmark (Before Prompt Optimization)
Smaller models were vulnerable to preamble chit-chat (e.g., misclassifying pricing inquiries as greetings):

![Intent Router Benchmark - Before Optimization](docs/images/router_bf.png)

##### Optimized Benchmark (After Anti-Noise & Few-Shot Tuning)
By introducing explicit anti-noise system instructions along with targeted few-shot trap examples ([docs/adjust_intent_router.md](docs/adjust_intent_router.md)):

![Intent Router Benchmark - After Optimization](docs/images/router_after.png)

> **Key Takeaway:** Prompt optimization enabled open-source `qwen3-8b` to match `gpt-4o-mini` at a perfect **100% accuracy**, while `llama-3.1-8b` achieved a **+16.33%** gain.

---

### 🚀 Setup & Getting Started

#### 1. Installation

Requires **Python >= 3.13**. This project uses [Astral uv](https://docs.astral.sh/uv/) for fast package and environment management.

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

#### 3. Data Preparation & Database Initialization

Initialize the service pricing database (PostgreSQL/SQLite) and import the catalogue:

```bash
uv run python import_db.py
```

> **Note on Vector Storage:** Qdrant collections (`petcare_knowledge_base`, `petcare_parent_documents`, `petcare_semantic_cache`) are automatically initialized and populated with hybrid vector embeddings on the first run of the application via `src/vector_store.py`.

#### 4. Customizing Prompts & Personas

All system instructions, personas, and prompts are centralized in [`src/prompts.py`](src/prompts.py):
- `qa_system_prompt`: Core consulting persona, guidelines, and context grounding.
- `contextualize_q_system_prompt`: History-aware query rewriter instructions.
- `router_prompt`: Intent classification rules and few-shot examples.

```python
# src/prompts.py
qa_system_prompt = """
<system_role>
You are Petcare Assistant, a cute, polite, and helpful customer service representative for Petcare.
Your PRIMARY OBJECTIVE is to answer customer queries accurately, relying EXCLUSIVELY on the provided <context>.
</system_role>
...
"""
```

#### 5. Run the Application

##### Option A: Interactive Terminal CLI
Test the RAG pipeline directly inside your console with cache inspection commands (`cache`, `clear`):

```bash
uv run python main.py
```

##### Option B: FastAPI Backend Server (with Hot-Reload)
```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

##### Option C: Streamlit Web UI
```bash
uv run streamlit run app.py
```

##### Option D: Production Deployment via Docker Compose
```bash
docker compose up -d --build
```

---

### 🔌 API Documentation & Usage

Interactive Swagger documentation is available at:
🔗 [http://localhost:8000/docs](http://localhost:8000/docs)

![Swagger UI](docs/images/sw_ui.png)

#### Send Message Endpoint

```http
POST /api/chat/send
Content-Type: application/json
```

**Sample Request:**

```json
{
  "session_id": "pet_owner_001",
  "message": "Bé poodle nhà mình 4kg, mình muốn hỏi giá gói tắm vệ sinh và dịch vụ khách sạn thú cưng 3 ngày"
}
```

**Sample Response:**

```json
{
  "answer": "**Bảng giá cho bé Poodle 4 kg**\n\n| Dịch vụ | Trọng lượng | Giá (đ) |\n|---|---|---|\n| Lưu trú 24h (3 ngày) | 3 – 5 kg | 100.000đ/ngày → **300.000đ** cho 3 ngày (đã bao gồm ăn uống) |\n| Tắm vệ sinh | 3 – 5 kg | **90.000đ – 110.000đ** |\n\n- Giá lưu trú đã bao gồm dịch vụ ăn uống, không phát sinh thêm chi phí.  \n- Giá tắm là mức giá tham khảo trong khoảng 90.000đ – 110.000đ; cụ thể sẽ được xác nhận khi đặt lịch.\n\n**Đặt lịch:** Dạ anh/chị có thể sử dụng cổng đặt lịch trực tuyến của Petcare để chọn ngày giờ và dịch vụ phù hợp, hoặc liên hệ qua Zalo nhé: https://zalo.me/3900819148490236884.",
  "intent": "TOOL",
  "from_cache": false,
  "elapsed_time": 9.41,
  "num_docs": 0,
  "context_docs": [],
  "price_data": "📋 BẢNG GIÁ DỊCH VỤ PETCARE:\n\n🔹 Lưu trú 24h:\n   • 3kg - 5kg: 70.000đ - 100.000đ\n\n📌 Lưu ý: Giá lưu trú đã bao gồm dịch vụ ăn uống.\n\n💰 CHI TIẾT TÍNH GIÁ LƯU TRÚ:\n\n• Giá gốc/ngày: 100.000đ\n• Số ngày lưu trú: 3 ngày\n• Tổng trước giảm: 300.000đ\n• 💵 TỔNG THANH TOÁN: 300.000đ\n• 📌 Giá lưu trú đã bao gồm ăn uống, không phát sinh thêm chi phí.\n\n---\n\n📋 BẢNG GIÁ DỊCH VỤ PETCARE:\n\n🔹 Tắm:\n   • 3kg - 5kg: 90.000đ - 110.000đ\n\n📌 Lưu ý: Giá lưu trú đã bao gồm dịch vụ ăn uống.",
  "timing": {
    "rewrite_query": 0.0001,
    "cache_lookup": 1.6654,
    "intent_router": 1.2280,
    "tool_lookup": 1.0454,
    "qa_generation": 3.8152
  },
  "standalone_query": "Bé poodle nhà mình 4kg, mình muốn hỏi giá gói tắm vệ sinh và dịch vụ khách sạn thú cưng 3 ngày"
}
```

---

### 🧪 Automated Testing & Evaluation Suite

Run **DeepEval evaluation pipeline**:
```bash
uv run python -m tests.deep_eval.run_deepeval
```

Run **all unit & integration tests**:
```bash
uv run pytest tests/unit_test/ -v
```

Run **Retrieval & Rerank parameter tuning**:
```bash
uv run python tests/unit_test/tune_pdr.py
```

---

### 🛠️ Technologies Used

| Component | Technology | Role & Technical Specifications |
| :--- | :--- | :--- |
| **Orchestration** | LangChain 0.3+ | RAG Pipeline orchestration, LCEL chains, and dynamic tool calling |
| **Backend API** | FastAPI + Uvicorn | High-performance asynchronous RESTful API with validated Pydantic schemas |
| **Agentic Logic** | Custom Intent Router | Real-time query intent classification (`KNOWLEDGE`, `TOOL`, `GREETING`) |
| **Relational DB** | Supabase (PostgreSQL) / SQLite | Storage for service catalogues, pricing tiers, appointments, and conversation history |
| **Vector Database** | Qdrant Cloud | Cloud vector database powering Hybrid Search and Semantic Caching |
| **Sparse Retrieval** | FastEmbed (BM25) | Keyword-based sparse vector generation for hybrid retrieval |
| **Dense Embeddings** | Jina-Embeddings-v5-text-small | High-dimensional dense vector embeddings (1024-dim) for clinical petcare texts |
| **Reranker** | Jina-Reranker-v3 | Cross-encoder semantic reranking to optimize Top-K retrieved context precision |
| **LLM Inference** | OpenRouter (LiteLLM) | Multi-model routing and inference serving(gpt-oss-120b, gpt-oss-20b, qwen3 8b, llama 3.1) |
| **Evaluation** | DeepEval | Automated LLM-as-a-Judge evaluation framework for RAG metric benchmarking |
| **UI Client** | Streamlit & Next.js | Interactive consultation interfaces and administrative service dashboards |
| **Deployment** | Docker & Docker Compose | Full-stack containerization for streamlined VPS and cloud deployment |

---

<p align="center">
  Built with ❤️ for the Petcare Community
</p>
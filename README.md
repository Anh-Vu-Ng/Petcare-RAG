# 🛍️ Petcare Agentic RAG — Smart Petcare Consulting System

> An advanced **Agentic RAG** system designed for automated petcare consulting. It features **Intent Routing**, **Selective Semantic Caching**, **Hybrid Search**, and **Dynamic Tool Calling** (connecting directly to Supabase/PostgreSQL) to provide accurate, real-time assistance.

---

## 📑 Table of Contents

- [🛍️ Petcare Agentic RAG — Smart Petcare Consulting System](#️-petcare-agentic-rag--smart-petcare-consulting-system)
  - [📑 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [🧭 System Architecture](#-system-architecture)
  - [🚀 Installation \& Running](#-installation--running)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Setup Steps](#2-setup-steps)
    - [3. Execution](#3-execution)
  - [🛠️ Technologies Used](#️-technologies-used)

---

## 🔍 Overview

Petcare Agentic RAG is an intelligent chatbot system acting as a professional consultant for pet owners. Transitioning from traditional RAG, this system utilizes an **Agentic** approach: an Intent Router classifies user queries on-the-fly to decide whether to retrieve general knowledge from document vectors or execute specific database-driven Tools for real-time pricing and discount calculations.

---

## 🧭 System Architecture



## 🚀 Installation & Running

The project utilizes Astral's `uv` for package and virtual environment management, delivering near-instant installations and clean builds.

### 1. Prerequisites
*   **Python**: `>= 3.13`
*   **uv**: Make sure [uv](https://docs.astral.sh/uv/) is installed on your system.

### 2. Setup Steps
**Step 1: Clone the repository**
```bash
git clone https://github.com/Anh-Vu-Ng/Petcare-RAG.git
cd Petcare-RAG
```

**Step 2: Sync dependencies**
```bash
uv sync
```

**Step 3: Configuration**
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=sk-or-v1-xxx
JINA_API_KEY=jina_xxx
# Database Configuration (Optional)
# Use the Connection Pooler URI (port 6543) from Supabase for Production Cloud Database.
# If left empty, the application will automatically fallback to local SQLite (data/petcare_services.db).
DATABASE_URL=postgresql://postgres:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

**Step 4: Initialize and Seed Database**
Generate the database schema, apply table indexes, and import pricing items from the CSV file:
```bash
uv run python import_db.py
```

### 3. Execution

**Run the FastAPI Backend:**
```bash
uv run uvicorn src.api.main:app --reload
```
The REST API will be available at: `http://localhost:8000`

**Run the Streamlit Dashboard (Web UI):**
```bash
uv run streamlit run app.py
```
Open your browser at: `http://localhost:8501`

*(To run the CLI Terminal mode instead: `uv run python main.py`)*

---

## 🛠️ Technologies Used

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | LangChain | RAG Pipeline & Tool management |
| **Backend API** | FastAPI | RESTful endpoints for client communication |
| **Agentic Logic** | Custom Intent Router | Smart classification (`KNOWLEDGE` vs `TOOL`) |
| **Database** | Supabase (PostgreSQL) / SQLite (Fallback) | Real-time pricing & service data storage |
| **ORM** | SQLAlchemy | Secure Database Abstraction Layer |
| **Vector DB** | FAISS | Dense semantic vector database |
| **Keyword Search** | rank-bm25 | Local sparse retrieval engine |
| **Embedding** | jina-embeddings-v5 | High-quality text embeddings |
| **Reranker** | jina-reranker-v3 | Semantic reranking for document precision |
| **QA LLM** | GPT-OSS-120B | Final answer generator model |
| **Router LLM** | GPT-OSS-20B | Lightweight and fast routing model |
| **Caching** | Semantic Cache (FAISS) | Low-latency response caching |
| **UI Framework** | Streamlit | Chat and service management dashboard |

---

<p align="center">
  Built with ❤️ for the Petcare Community
</p>
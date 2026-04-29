# 🛍️ Petcare RAG Hybrid — Petcare Consulting System

> This project is built to create an automated customer care chatbot system, using **Retrieval-Augmented Generation (RAG)** techniques combined with **Hybrid Search** (BM25 + FAISS), **Reciprocal Rank Fusion (RRF)** algorithms, and **Jina Reranker** to accurately answer based on Petcare data sources.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Installation & Running](#-installation--running)
- [Technologies Used](#-technologies-used)

---

## 🔍 Overview

Petcare RAG Hybrid is a **Conversational RAG** system designed to act as a consultant for customers regarding products and services. The system operates on the following principles:

1. **Data Collection** from PDF files and URLs of the Petcare website.
2. **Chunking** text into appropriately sized chunks.
3. **Dual Indexing**: FAISS (semantic search) + BM25 (keyword search).
4. **Multi-turn Conversation Support** with Query Rewriting mechanism based on chat history to create standalone questions.
5. **Semantic Cache Checking**: Uses standalone questions to evaluate semantic similarity, returning answers immediately if previously asked.
6. **Hybrid Retrieval** combining results from both systems via the RRF algorithm if not found in Cache.
7. **Reranking** using the Jina Reranker model to filter the most relevant documents.
8. **Answer Generation** by LLM (via OpenRouter API) based on retrieved context and saving it back to Cache.

---


## 🚀 Installation & Running Steps

The project uses Astral's `uv` package manager to manage environments and dependencies.

### 1. Prerequisites

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version `>= 3.13`
- **Tools**: Pre-installed `uv` package manager ([uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

### 2. Detailed Installation Guide

**Step 1: Clone the source code**
```bash
git clone https://github.com/Anh-Vu-Ng/Petcare-RAG.git
cd Petcare-RAG
```

**Step 2: Install libraries using `uv`**
Run the following command to sync and automatically create a virtual environment containing all necessary dependencies for the project:
```bash
uv sync
```

**Step 3: Configure environment variables**
Create a `.env` file and fill in your API Keys:
```bash
# For Windows PowerShell
echo "OPENROUTER_API_KEY=sk-or-v1-xxx-your-key-xxx" > .env
echo "GROQ_API_KEY=gsk_xxx-your-key-xxx" >> .env
echo "JINA_API_KEY=jina_xxx-your-key-xxx" >> .env
```

**Step 4: Load input data**
- Prepare a Petcare guide PDF file and place it at the path: `data/rag_docs.pdf`.
- Add necessary URLs to the `data/url.txt` file.

### 3. Start the system
**Run Web Interface using Streamlit**
```bash
uv run python -m streamlit run app.py
```
*Access the system at: `http://localhost:8501`*

---

## 🛠️ Technologies Used

| Component | Technology | Role |
|------------|-----------|---------|
| **Framework** | LangChain | Orchestration for RAG pipeline |
| **QA LLM** | OpenRouter API | Answer Generation |
| **Rewriter LLM** | Groq API | Rewrite conversational queries |
| **Embedding** | jina-embeddings-v5-text-small | Convert text → vector |
| **Reranker** | jina-reranker-v3 | Re-evaluate relevance |
| **Vector DB** | FAISS | Semantic Search (dense retrieval) |
| **Keyword Search** | BM25 (rank-bm25) | Keyword Search (sparse retrieval) |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combine results from 2 retrievers |
| **PDF Parser** | PyMuPDF (fitz) | Extract text from PDF |
| **Web Crawler** | Requests + BeautifulSoup | Crawl content from URLs |
| **Frontend** | Streamlit | Interactive chat interface |
| **Package Manager** | uv | Fast dependency management |

---

<p align="center">
  Made with ❤️ for Petcare
</p>

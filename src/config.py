EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
EMBEDDING_DIM = 1024
LLM_MODEL = "openai/gpt-oss-120b:free"
REWRITER_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_FINAL = 10
RRF_K = 60
URL_FILE = "data/url.txt"
PDF_FILE = "data/rag_docs.pdf"
INDEX_DIR = "data/faiss_index"
BM25_INDEX_PATH = "data/faiss_index/bm25_index.pkl"
CHAT_HISTORY_WINDOW = 5

# Semantic Cache
CACHE_SIMILARITY_THRESHOLD = 0.85  t
CACHE_INDEX_PATH = "data/cache_index"  # Thư mục lưu FAISS index của cache
CACHE_MAX_SIZE = 500  # Số lượng tối đa cache entries

# Jina Reranker
RERANKER_MODEL = "jina-reranker-v3"
TOP_K_RERANK = 5  # Số lượng documents giữ lại sau reranking
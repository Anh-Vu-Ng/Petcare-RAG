# Database Config
import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
EMBEDDING_DIM = 1024

LLM_MODEL = "openai/gpt-oss-120b"
ROUTER_MODEL = "qwen/qwen3-8b"
GREETINGS_MODEL = "meta-llama/llama-3.1-8b-instruct"
REWRITE_MODEL = "openai/gpt-oss-20b:nitro"
PARENT_CHUNK_SIZE = 1400
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 310
CHILD_CHUNK_OVERLAP = 20

TOP_K = 20
TOP_K_FINAL = 8

URL_FILE = "data/url.txt"
PDF_FILE = "data/rag_docs.pdf"

# Qdrant Config
URL_QDRANT = os.getenv("URL_QDRANT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_KB_COLLECTION = "petcare_knowledge_base"
QDRANT_PARENT_COLLECTION = "petcare_parent_documents"
QDRANT_CACHE_COLLECTION = "petcare_semantic_cache"

CHAT_HISTORY_WINDOW = 4

# Semantic Cache
CACHE_SIMILARITY_THRESHOLD = 0.95  
CACHE_MAX_SIZE = 500 

# Jina Reranker
RERANKER_MODEL = "jina-reranker-v3"
TOP_K_RERANK = 3

# Fallback sang SQLite local nếu DATABASE_URL chưa được cấu hình
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/petcare_services.db")
CSV_PRICING_PATH = "data/petcare_pricing_data.csv"
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import string
import pickle
import uuid
import logging
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from src.jina_embeddings import JinaEmbeddings
from src.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONSTANTS ---
TEST_DATASET_PATH = "data/test_finetune_chunks.json"
RAW_DOCS_CACHE_PATH = "data/raw_documents_cache.pkl"
EMBEDDING_CACHE_PATH = "data/eval_embedding_cache.pkl"
RESULTS_CSV_PATH = "data/pdr_tuning_results.csv"
DEFAULT_RETRIEVE_K = 20


class CachedEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper caching embeddings to a local pickle file.
    Guarantees each unique chunk/query text is only embedded once.
    """
    def __init__(self, base_embeddings: Embeddings, cache_path: str = EMBEDDING_CACHE_PATH):
        self.base_embeddings = base_embeddings
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"[{self.__class__.__name__}] Loaded {len(self.cache)} cached embeddings.")
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Error loading cache: {e}. Starting fresh.")
        else:
            logger.info(f"[{self.__class__.__name__}] No existing cache found at {self.cache_path}.")

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error saving cache: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        uncached = [text for text in texts if text not in self.cache]
        if uncached:
            logger.info(f"[{self.__class__.__name__}] Calling API for {len(uncached)} new documents...")
            new_embs = self.base_embeddings.embed_documents(uncached)
            for text, emb in zip(uncached, new_embs):
                self.cache[text] = emb
            self._save_cache()
        return [self.cache[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        cache_key = f"query:{text}"
        if cache_key not in self.cache:
            logger.info(f"[{self.__class__.__name__}] Calling API for query: {text[:40]}...")
            emb = self.base_embeddings.embed_query(text)
            self.cache[cache_key] = emb
            self._save_cache()
        return self.cache[cache_key]


def get_raw_documents() -> List[Document]:
    """
    Loads raw documents from local cache or runs load_all_docs() and caches them.
    """
    if os.path.exists(RAW_DOCS_CACHE_PATH):
        try:
            with open(RAW_DOCS_CACHE_PATH, "rb") as f:
                docs = pickle.load(f)
            logger.info(f"Loaded {len(docs)} documents from cache: {RAW_DOCS_CACHE_PATH}")
            return docs
        except Exception as e:
            logger.error(f"Error loading raw documents cache: {e}")

    # Fallback to loading from source
    from src.data_loader import load_all_docs
    logger.info("Loading documents from PDF and URLs...")
    docs = load_all_docs()
    
    if not docs:
        raise ValueError("Loaded documents list is empty! Ensure data files are present.")
        
    try:
        os.makedirs(os.path.dirname(RAW_DOCS_CACHE_PATH), exist_ok=True)
        with open(RAW_DOCS_CACHE_PATH, "wb") as f:
            pickle.dump(docs, f)
        logger.info(f"Saved {len(docs)} documents to local cache.")
    except Exception as e:
        logger.error(f"Error saving raw documents cache: {e}")
        
    return docs


def split_parent_child_param(
    documents: List[Document], 
    child_size: int, 
    child_overlap: int, 
    parent_size: int = PARENT_CHUNK_SIZE, 
    parent_overlap: int = PARENT_CHUNK_OVERLAP
) -> Tuple[Dict[str, Document], List[Document]]:
    """
    Splits documents into parent and child chunks with dynamic parameters.
    """
    if not documents:
        return {}, []

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=parent_size, chunk_overlap=parent_overlap, separators=["\n\n", "\n", ".", " ", ""]
    )
    
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=child_size, chunk_overlap=child_overlap, separators=["\n\n", "\n", ".", " ", ""]
    )

    parent_docs = {}
    child_docs = []
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"), ("##", "Header 2"), 
            ("###", "Header 3"), ("####", "Header 4")
        ],
        strip_headers=False
    )

    for doc in documents:
        source = doc.metadata.get("source", "")
        doc_parents = []
        
        # Handle Markdown Splitting for URLs
        if source.startswith("https://"):
            try:
                md_chunks = markdown_splitter.split_text(doc.page_content)
                for chunk in md_chunks:
                    chunk.metadata.update(doc.metadata)  # Merge parent metadata
                    doc_parents.extend(parent_splitter.split_documents([chunk]))
            except Exception as e:
                logger.warning(f"Markdown splitting failed for {source}: {e}. Fallback to recursive.")
                doc_parents.extend(parent_splitter.split_documents([doc]))
        else:
            doc_parents.extend(parent_splitter.split_documents([doc]))

        # Process Unique Parents & Generate Children
        seen_parent_contents = set()
        for p in doc_parents:
            if p.page_content not in seen_parent_contents:
                seen_parent_contents.add(p.page_content)
                
                parent_id = str(uuid.uuid4())
                parent_docs[parent_id] = p
                
                sub_children = child_splitter.split_documents([p])
                for c in sub_children:
                    c.metadata["parent_id"] = parent_id
                    child_docs.append(c)

    # Filter unique children
    unique_children = []
    seen_child_contents = set()
    for c in child_docs:
        if c.page_content not in seen_child_contents:
            seen_child_contents.add(c.page_content)
            unique_children.append(c)

    return parent_docs, unique_children


def reciprocal_rank_fusion_param(
    dense_results: List[Document], 
    sparse_results: List[Document], 
    rrf_k: int = 60, 
    top_k_final: int = 8
) -> List[Document]:
    """
    RRF combining dense and sparse results.
    """
    rrf_scores = defaultdict(lambda: {"score": 0.0, "doc": None})

    def add_to_scores(results: List[Document]):
        for rank, doc in enumerate(results):
            content = doc.page_content
            rrf_scores[content]["doc"] = doc
            rrf_scores[content]["score"] += 1.0 / (rank + rrf_k + 1)

    add_to_scores(dense_results)
    add_to_scores(sparse_results)

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    final_docs = []
    for item in sorted_results[:top_k_final]:
        doc = item["doc"]
        copied_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": item["score"]}
        )
        final_docs.append(copied_doc)
        
    return final_docs


def expand_to_parent(child_docs: List[Document], parent_docs: Dict[str, Document]) -> List[Document]:
    """
    Maps child documents to their original parent documents.
    """
    parent_docs_list = []
    seen_parent_ids = set()
    
    for doc in child_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id in parent_docs:
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                
                orig_parent = parent_docs[parent_id]
                parent_meta = orig_parent.metadata.copy()
                if "rrf_score" in doc.metadata:
                    parent_meta["rrf_score"] = doc.metadata["rrf_score"]
                    
                parent_docs_list.append(Document(
                    page_content=orig_parent.page_content,
                    metadata=parent_meta
                ))
        else:
            parent_docs_list.append(doc)
            
    return parent_docs_list


def normalize_text(text: str) -> str:
    """Standard text normalization for evaluation."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Tính độ tương đồng Cosine giữa hai vector."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def evaluate_combination(
    eval_set: List[Dict[str, str]],
    hybrid_retriever,
    cached_embeddings: CachedEmbeddings,
    semantic_threshold: float = 0.55
) -> Tuple[float, float, float]:
    """
    Đánh giá cấu hình sử dụng độ tương đồng ngữ nghĩa.
    - KPI Chính: Hit Rate và MRR (đo lường trực tiếp hiệu quả Retrieval).
    - Chỉ số ngầm: debug_sim (Best Match Similarity) dùng để tham khảo phổ điểm.
    """
    hit_count = 0
    total_mrr = 0.0
    total_debug_sim = 0.0
    total = len(eval_set)

    if total == 0:
        return 0.0, 0.0, 0.0

    # 1. Thu thập tất cả các đoạn văn bản cần tạo embedding
    queries_data = []
    texts_to_embed = []
    
    for item in eval_set:
        query = item["question"]
        gt = item["ground_truth"]
        
        # Lấy trực tiếp từ hybrid retriever (child chunks)
        fused_docs = hybrid_retriever.invoke(query)
        doc_contents = [doc.page_content for doc in fused_docs]
            
        queries_data.append({
            "gt": gt,
            "doc_contents": doc_contents
        })
        
        texts_to_embed.append(gt)
        texts_to_embed.extend(doc_contents)

    # 2. Loại bỏ trùng lặp văn bản và sinh batch embedding
    unique_texts = list(set(texts_to_embed))
    unique_embeddings = cached_embeddings.embed_documents(unique_texts)
    text_to_emb = dict(zip(unique_texts, unique_embeddings))
    
    # 3. Tính toán độ tương đồng và điểm số
    for q_data in queries_data:
        gt = q_data["gt"]
        gt_emb = text_to_emb[gt]
        
        best_sim = 0.0
        mrr_score = 0.0
        
        for rank, doc_content in enumerate(q_data["doc_contents"], start=1):
            doc_emb = text_to_emb[doc_content]
            doc_sim = cosine_similarity(gt_emb, doc_emb)
            
            # Lưu lại điểm similarity cao nhất (để debug phổ điểm)
            if doc_sim > best_sim:
                best_sim = doc_sim
                
            # Tính MRR cho chunk ĐẦU TIÊN vượt ngưỡng
            if doc_sim >= semantic_threshold and mrr_score == 0.0:
                mrr_score = 1.0 / rank
                
        total_debug_sim += best_sim
        total_mrr += mrr_score
        
        # Hit rate: Chỉ cần có ít nhất 1 chunk vượt ngưỡng
        if best_sim >= semantic_threshold:
            hit_count += 1
            
    # Trả về: (hit_rate, mrr_score, debug_sim)
    return hit_count / total, total_mrr / total, total_debug_sim / total

def main():
    logger.info("=" * 60)
    logger.info("🚀 Running PDR Parameter Tuning (Grid Search)")
    logger.info("=" * 60)

    # 1. Load test dataset
    if not os.path.exists(TEST_DATASET_PATH):
        logger.error(f"Test dataset mock file not found at {TEST_DATASET_PATH}!")
        sys.exit(1)
        
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    logger.info(f"Loaded {len(eval_set)} queries from test dataset.")

    # 2. Load documents & setup embeddings
    raw_docs = get_raw_documents()
    jina_emb = JinaEmbeddings()
    cached_emb = CachedEmbeddings(jina_emb)

    # 3. Parameter Grid
    child_sizes = [300, 310, 320]
    child_overlaps = [10, 20]
    top_k_children = [6, 8]
    
    results = []
    total_runs = len(child_sizes) * len(child_overlaps) * len(top_k_children)
    current_run = 0

    # 4. Grid Search
    for c_size in child_sizes:
        for c_overlap in child_overlaps:
            logger.info(f"\n--- Splitting documents: child_size={c_size}, child_overlap={c_overlap} ---")
            parent_docs, child_docs = split_parent_child_param(raw_docs, c_size, c_overlap)
            logger.info(f"Split raw docs into {len(parent_docs)} parents and {len(child_docs)} child chunks.")

            if not child_docs:
                logger.warning("No child documents generated. Skipping evaluation for this config.")
                continue

            # BUILD INDICES ONLY ONCE PER CHUNK CONFIGURATION
            logger.info("Building Qdrant (in-memory) hybrid index...")
            qclient = QdrantClient(location=":memory:")
            from qdrant_client.http import models
            qclient.create_collection(
                collection_name="eval_collection",
                vectors_config=models.VectorParams(
                    size=1024, # Jina v5 embedding size
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "langchain-sparse": models.SparseVectorParams()
                }
            )
            sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")
            vectorstore = QdrantVectorStore(
                client=qclient,
                collection_name="eval_collection",
                embedding=cached_emb,
                sparse_embedding=sparse_emb,
                retrieval_mode=RetrievalMode.HYBRID,
            )
            vectorstore.add_documents(child_docs)

            for k in top_k_children:
                current_run += 1
                logger.info(f"[{current_run}/{total_runs}] Testing combo: top_k={k}")
                
                # Khởi tạo lại retriever bên trong lặp để đảm bảo tham số `k` ăn khớp
                hybrid_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                
                try:
                    hit_rate, mrr_score, debug_sim = evaluate_combination(
                        eval_set=eval_set,
                        hybrid_retriever=hybrid_retriever,
                        cached_embeddings=cached_emb,
                        semantic_threshold=0.55 
                    )
                    logger.info(f" => Hit Rate: {hit_rate:.2%} | MRR: {mrr_score:.2%} | (Debug Sim: {debug_sim:.2%})")
                    
                    # Cập nhật lại các key được lưu vào list
                    results.append({
                        "child_size": c_size,
                        "child_overlap": c_overlap,
                        "top_k": k,
                        "hit_rate": hit_rate,
                        "mrr_score": mrr_score,
                        "debug_sim": debug_sim
                    })
                    time.sleep(0.5) 
                except Exception as e:
                    logger.error(f" => Error evaluating combo: {e}")

    # 5. Report results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["mrr_score", "hit_rate", "debug_sim"], ascending=False).reset_index(drop=True)

    logger.info("\n" + "=" * 60)
    logger.info("📊 Grid Search Results Summary")
    logger.info("=" * 60)
    print(df_results.to_markdown(index=False))
    
    # 6. Save to CSV
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    logger.info(f"💾 Results successfully saved to {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main()
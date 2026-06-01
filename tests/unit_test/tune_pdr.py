import os
import sys
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
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from src.jina_embeddings import JinaEmbeddings
from src.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, RRF_K

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Ensure stdout uses UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
    rrf_k: int = RRF_K, 
    top_k_final: int = 8
) -> List[Document]:
    """
    RRF combining dense and sparse results.
    """
    # Using defaultdict simplifies the initialization logic
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


def evaluate_combination(
    eval_set: List[Dict[str, str]],
    parent_docs: Dict[str, Document],
    dense_retriever,
    sparse_retriever,
    top_k: int
) -> Tuple[float, float]:
    """
    Retrieves and evaluates metrics for a specific top_k configuration using pre-built retrievers.
    """
    hit_count = 0
    total_overlap = 0.0
    total = len(eval_set)

    if total == 0:
        return 0.0, 0.0

    for item in eval_set:
        query = item["question"]
        gt = item["ground_truth"]

        dense_docs = dense_retriever.invoke(query)
        sparse_docs = sparse_retriever.invoke(query)

        fused_docs = reciprocal_rank_fusion_param(dense_docs, sparse_docs, top_k_final=top_k)
        retrieved_docs = expand_to_parent(fused_docs, parent_docs)

        full_text = " ".join([doc.page_content for doc in retrieved_docs])
        
        norm_gt_words = set(normalize_text(gt).split())
        norm_text_words = set(normalize_text(full_text).split())
        
        overlap_ratio = len(norm_gt_words.intersection(norm_text_words)) / len(norm_gt_words) if norm_gt_words else 0.0
        total_overlap += overlap_ratio
        
        if overlap_ratio >= 0.75:
            hit_count += 1

    return hit_count / total, total_overlap / total


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
    child_sizes = [200, 300, 350, 400, 450]
    child_overlaps = [30, 40, 50, 60]
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

            # BUILD INDICES ONLY ONCE PER CHUNK CONFIGURATION (HUGE PERFORMANCE BOOST)
            logger.info("Building FAISS and BM25 indices...")
            vectorstore = FAISS.from_documents(child_docs, cached_emb)
            dense_retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVE_K})
            sparse_retriever = BM25Retriever.from_documents(child_docs, k=DEFAULT_RETRIEVE_K)

            for k in top_k_children:
                current_run += 1
                logger.info(f"[{current_run}/{total_runs}] Testing combo: top_k={k}")
                
                try:
                    hit_rate, avg_overlap = evaluate_combination(
                        eval_set=eval_set,
                        parent_docs=parent_docs,
                        dense_retriever=dense_retriever,
                        sparse_retriever=sparse_retriever,
                        top_k=k
                    )
                    logger.info(f" => Hit Rate: {hit_rate:.2%} | Avg Overlap: {avg_overlap:.2%}")
                    results.append({
                        "child_size": c_size,
                        "child_overlap": c_overlap,
                        "top_k": k,
                        "hit_rate": hit_rate,
                        "average_overlap": avg_overlap
                    })
                except Exception as e:
                    logger.error(f" => Error evaluating combo: {e}")

    # 5. Report results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["hit_rate", "average_overlap"], ascending=False).reset_index(drop=True)

    logger.info("\n" + "=" * 60)
    logger.info("📊 Grid Search Results Summary")
    logger.info("=" * 60)
    print(df_results.to_markdown(index=False)) # Print bảng markdown ra console cho dễ đọc
    
    # 6. Save to CSV
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    logger.info(f"💾 Results successfully saved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
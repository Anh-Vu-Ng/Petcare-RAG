import os
import time
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import (
    URL_QDRANT,
    QDRANT_API_KEY,
    QDRANT_KB_COLLECTION,
    QDRANT_PARENT_COLLECTION,
    EMBEDDING_DIM,
    TOP_K
)
from src.data_loader import load_all_docs
from src.text_processor import split_parent_child
from src.jina_embeddings import JinaEmbeddings

def get_embeddings():
    """Khởi tạo Jina Embeddings (gọi API)."""
    return JinaEmbeddings()

def _safe_recreate_collection(client, collection_name, vectors_config, sparse_vectors_config=None):
    """
    Xóa và tạo mới collection trên Qdrant Cloud một cách an toàn.
    Giải quyết vấn đề bất đồng bộ khi xóa của cụm phân tán (eventual consistency)
    bằng cách poll/chờ cho đến khi collection thực sự biến mất trước khi tạo lại.
    """
    try:
        if client.collection_exists(collection_name):
            print(f"🗑️ Deleting collection '{collection_name}'...")
            client.delete_collection(collection_name)
            # Chờ tối đa 10 giây để collection thực sự bị xóa
            for _ in range(20):
                if not client.collection_exists(collection_name):
                    break
                time.sleep(0.5)
    except Exception as e:
        print(f"⚠️ Warning khi xóa collection '{collection_name}': {e}")
        
    print(f"🏗️ Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config
    )
    # Chờ tối đa 10 giây để đảm bảo collection đã sẵn sàng
    for _ in range(20):
        if client.collection_exists(collection_name):
            break
        time.sleep(0.5)

def get_qdrant_retriever():
    """Load Qdrant index từ cloud hoặc khởi tạo mới nếu chưa có."""
    embeddings = get_embeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    if not URL_QDRANT or not QDRANT_API_KEY:
        raise ValueError("Chưa thiết lập URL_QDRANT hoặc QDRANT_API_KEY trong file .env.")
        
    client = QdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
    
    # Kiểm tra xem cả hai Collections đã tồn tại và có dữ liệu chưa
    kb_exists = False
    parent_exists = False
    try:
        kb_info = client.get_collection(QDRANT_KB_COLLECTION)
        if kb_info.points_count > 0:
            kb_exists = True
    except Exception:
        pass
        
    try:
        parent_info = client.get_collection(QDRANT_PARENT_COLLECTION)
        if parent_info.points_count > 0:
            parent_exists = True
    except Exception:
        pass
        
    if kb_exists and parent_exists:
        print(f"✅ Found Qdrant collections: '{QDRANT_KB_COLLECTION}' and '{QDRANT_PARENT_COLLECTION}'. Loading retriever...")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_KB_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    else:
        print(f"Building Qdrant collections and loading data...")
        docs = load_all_docs()
        
        if not docs:
            raise ValueError("No input data found. Data Ingestion pipeline is broken!")
            
        parent_docs, child_docs = split_parent_child(docs)
        
        # 1. Recreate & populate Parent Collection (vectorless)
        _safe_recreate_collection(
            client=client,
            collection_name=QDRANT_PARENT_COLLECTION,
            vectors_config={}
        )
        
        parent_points = []
        for parent_id, parent_doc in parent_docs.items():
            parent_points.append(
                models.PointStruct(
                    id=parent_id,
                    vector={},
                    payload={
                        "page_content": parent_doc.page_content,
                        "metadata": parent_doc.metadata
                    }
                )
            )
        
        print(f"Uploading {len(parent_points)} parent documents to Qdrant...")
        client.upsert(
            collection_name=QDRANT_PARENT_COLLECTION,
            points=parent_points
        )
        
        # 2. Recreate & populate Knowledge Base (dense + sparse configs)
        _safe_recreate_collection(
            client=client,
            collection_name=QDRANT_KB_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE
            ),
            sparse_vectors_config={
                "langchain-sparse": models.SparseVectorParams()
            }
        )
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_KB_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        vectorstore.add_documents(child_docs)
        print(f"🚀 Qdrant collections built and saved.")

    # Trả về đối tượng retriever để cắm vào RAG Chain
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

import os
import sys
import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import get_embeddings

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def test_tasks():
    load_dotenv()
    embeddings = get_embeddings()
    
    q1 = "xin chào"
    q2 = "xin chào"
    q3 = "chào shop"
    
    print("--- Test 1: Mismatched Tasks (retrieval.query vs retrieval.passage) ---")
    v1_q = embeddings._call_api([q1], task="retrieval.query")[0]
    v2_p = embeddings._call_api([q2], task="retrieval.passage")[0]
    v3_p = embeddings._call_api([q3], task="retrieval.passage")[0]
    
    print(f"Cosine Similarity('{q1}' as query vs '{q2}' as passage): {cosine_similarity(v1_q, v2_p):.6f}")
    print(f"Cosine Similarity('{q1}' as query vs '{q3}' as passage): {cosine_similarity(v1_q, v3_p):.6f}")
    
    print("\n--- Test 2: Matching Tasks (retrieval.query vs retrieval.query) ---")
    v1_q_q = embeddings._call_api([q1], task="retrieval.query")[0]
    v2_q_q = embeddings._call_api([q2], task="retrieval.query")[0]
    v3_q_q = embeddings._call_api([q3], task="retrieval.query")[0]
    
    print(f"Cosine Similarity('{q1}' vs '{q2}'): {cosine_similarity(v1_q_q, v2_q_q):.6f}")
    print(f"Cosine Similarity('{q1}' vs '{q3}'): {cosine_similarity(v1_q_q, v3_q_q):.6f}")

    print("\n--- Test 3: Matching Tasks (text-matching vs text-matching) ---")
    v1_tm = embeddings._call_api([q1], task="text-matching")[0]
    v2_tm = embeddings._call_api([q2], task="text-matching")[0]
    v3_tm = embeddings._call_api([q3], task="text-matching")[0]
    
    print(f"Cosine Similarity('{q1}' vs '{q2}'): {cosine_similarity(v1_tm, v2_tm):.6f}")
    print(f"Cosine Similarity('{q1}' vs '{q3}'): {cosine_similarity(v1_tm, v3_tm):.6f}")

if __name__ == "__main__":
    test_tasks()

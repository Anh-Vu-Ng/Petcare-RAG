from typing import Any, List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.config import RRF_K, TOP_K_FINAL

def reciprocal_rank_fusion(dense_results: List[Document], spare_results: List[Document], k: int=RRF_K) ->List[Document]:
    rrf_scores = {}

    def add_to_scores(results: List[Document]):
        for rank, doc in enumerate(results):
            doc_content = doc.page_content
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {"score": 0, "doc": doc}
            rrf_scores[doc_content]["score"] += 1.0/(rank+k+1)
    add_to_scores(dense_results)
    add_to_scores(spare_results)

    #sap xep lai diem giam dan
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    final_docs = []
    for item in sorted_results[:TOP_K_FINAL]:
        doc = item["doc"]
        doc.metadata["rrf_score"] = item["score"]
        final_docs.append(doc)
    return final_docs

class HybridRetriever(BaseRetriever):
    dense_retriever: Any = None
    sparse_retriever: Any = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        return reciprocal_rank_fusion(dense_docs, sparse_docs)
"""
Intent Router cho Agentic RAG.

Phân loại query thành KNOWLEDGE hoặc TOOL trước khi quyết định
có dùng Semantic Cache hay không (Selective Caching).

- KNOWLEDGE: Câu hỏi kiến thức tĩnh → Semantic Cache + Hybrid RAG
- TOOL: Câu hỏi về giá/dịch vụ → Bypass cache, gọi tools
"""

from langchain_core.prompts import ChatPromptTemplate
from src.prompts import router_prompt


class IntentRouter:
    """
    Bộ phân loại ý định sử dụng LLM
    Output: "KNOWLEDGE" hoặc "TOOL"
    """

    def __init__(self, llm):
        self.chain = router_prompt | llm

    def classify(self, query: str) -> str:
        """
        Phân loại query thành KNOWLEDGE hoặc TOOL.

        Args:
            query: Standalone query (đã qua rewriter).

        Returns:
            "KNOWLEDGE" hoặc "TOOL"
        """
        try:
            response = self.chain.invoke({"query": query})
            result = response.content.strip().upper()

            # Chỉ chấp nhận 2 giá trị hợp lệ
            if "TOOL" in result:
                return "TOOL"
            return "KNOWLEDGE"
        except Exception as e:
            print(f"[IntentRouter] Lỗi classify, fallback KNOWLEDGE: {e}")
            return "KNOWLEDGE"

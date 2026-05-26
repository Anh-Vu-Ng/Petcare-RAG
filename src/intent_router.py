from langchain_core.prompts import ChatPromptTemplate
from src.prompts import router_prompt


class IntentRouter:
    def __init__(self, llm):
        self.chain = router_prompt | llm

    def classify(self, query: str) -> str:
        """
        Phân loại query thành GREETING, KNOWLEDGE hoặc TOOL.

        Args:
            query: Standalone query (đã qua rewriter).

        Returns:
            "GREETING", "KNOWLEDGE" hoặc "TOOL"
        """
        try:
            response = self.chain.invoke({"query": query})
            result = response.content.strip().upper()

            if "TOOL" in result:
                return "TOOL"
            elif "GREETING" in result:
                return "GREETING"
            return "KNOWLEDGE"
        except Exception as e:
            print(f"[IntentRouter] Lỗi classify, fallback KNOWLEDGE: {e}")
            return "KNOWLEDGE"

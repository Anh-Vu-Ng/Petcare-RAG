"""
Tự động sinh bộ dữ liệu Q&A (question + ground_truth) từ kho tài liệu Petcare.
Dùng LLM qua OpenRouter để đọc từng chunk tài liệu và tạo câu hỏi tình huống thực tế.

Output format tương thích với ragas_eval.py:
    [{"question": str, "ground_truth": str, "source": str}, ...]

Usage:
    python tests/generate_qa.py                        # Random 50 chunks, 2 câu/chunk 
    python tests/generate_qa.py --sample 30            # Random 30 chunks
    python tests/generate_qa.py --sample 0             # Chạy toàn bộ chunks (không random)
    python tests/generate_qa.py --num-questions 5      # Sinh 5 câu/chunk
    python tests/generate_qa.py --seed 123             # Đổi seed random
"""

import os
import sys
import json
import random
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ── Path setup ──────────────────────────────────────────────────────────────
# Trỏ đường dẫn về project root để import src.* hoạt động đúng.
# Đồng thời chdir về root để các đường dẫn tương đối trong src/config.py
# (PDF_FILE = "data/rag_docs.pdf", v.v.) được resolve chính xác.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.data_loader import load_all_docs

# ── Environment ─────────────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Lỗi: Không tìm thấy OPENROUTER_API_KEY trong file .env")
    sys.exit(1)

# ── LLM cho QA Generation ──────────────────────────────────────────────────
QA_GEN_MODEL = "openai/gpt-4o-mini"

# ── Pydantic schema (ép LLM tuân thủ) ──────────────────────────────────────

class QAItem(BaseModel):
    question: str = Field(
        description=(
            "Một tình huống thực tế, phức tạp của người nuôi thú cưng đang lo lắng. "
            "KHÔNG dùng trực tiếp từ khóa chuyên môn hay tên bệnh. Độ dài từ 2-3 câu."
        )
    )
    ground_truth: str = Field(
        description="Câu trả lời chuẩn xác, mang tính tư vấn y khoa chỉ dựa trên tài liệu được cung cấp."
    )


class QADataset(BaseModel):
    items: list[QAItem]


# ── Prompt template ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là một Data Scientist chuyên tạo tập dữ liệu Ground Truth để đánh giá hệ thống RAG.
Nhiệm vụ: Trích xuất chính xác {num_questions} cặp QA từ CHUNK DỮ LIỆU được cung cấp.

LUẬT SINH CÂU HỎI (Question):
- Đóng vai: Chủ vật nuôi đang hoảng hốt, lo lắng.
- Văn phong: Ngôn ngữ nói đời thường, lủng củng, mô tả triệu chứng qua góc nhìn người thường.
- CẤM: Tuyệt đối không dùng thuật ngữ y khoa chuyên ngành (ví dụ: Parvo, viêm ruột, v.v.).

LUẬT SINH CÂU TRẢ LỜI (Ground Truth):
- Đóng vai: Bác sĩ thú y chuyên môn cao.
- Nội dung: Đưa ra chẩn đoán hoặc hướng dẫn CHỈ DỰA TRÊN thông tin có trong CHUNK DỮ LIỆU.
- Cho phép dùng thuật ngữ chuyên môn.

RÀNG BUỘC HỆ THỐNG:
- BẮT BUỘC phải tạo ra đủ {num_questions} câu hỏi.
- Tuyệt đối không tự bịa (hallucinate) thông tin ngoài.
"""


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "TÀI LIỆU:\n{context}"),
])


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sinh bộ dữ liệu Q&A cho Petcare-RAG")
    parser.add_argument(
        "--sample", type=int, default=50,
        help="Số lượng chunk random để xử lý. Đặt 0 để chạy toàn bộ (mặc định: 50)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=2,
        help="Số câu hỏi sinh ra cho mỗi chunk (mặc định: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed cho random sampling, đảm bảo kết quả có thể tái tạo (mặc định: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Đường dẫn file output (mặc định: data/test_scenario_pro.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 Khởi chạy tạo dữ liệu Test tự động bằng LLM")
    print("=" * 60)

    # ── 1. Load tài liệu ────────────────────────────────────────────────
    print("\n⏳ Đang load tài liệu từ src.data_loader...")
    raw_docs = load_all_docs()

    if not raw_docs:
        print("❌ Dữ liệu trống rỗng. Hãy kiểm tra lại thư mục data/.")
        sys.exit(1)

    print(f"✅ Đã load {len(raw_docs)} tài liệu gốc.")

    # ── 2. Chunking ─────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunked_docs = splitter.split_documents(raw_docs)
    print(f"✅ Đã cắt thành {len(chunked_docs)} chunk văn bản.")

    # ── 3. Cấu hình LLM ────────────────────────────────────────────────
    print(f"🤖 Đang kết nối với OpenRouter (model: {QA_GEN_MODEL})...")

    llm = ChatOpenAI(
        model=QA_GEN_MODEL,
        temperature=0.7,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    # Bọc LLM lại để ép trả về chuẩn format QADataset
    structured_llm = llm.with_structured_output(QADataset)
    chain = QA_PROMPT | structured_llm

    # ── 4. Random sampling & Sinh Q&A ────────────────────────────────────
    final_dataset = []

    if args.sample > 0 and args.sample < len(chunked_docs):
        random.seed(args.seed)
        chunks_to_process = random.sample(chunked_docs, args.sample)
        print(f"🎲 Random chọn {args.sample}/{len(chunked_docs)} chunks (seed={args.seed})")
    else:
        chunks_to_process = chunked_docs
        print(f"📄 Xử lý toàn bộ {len(chunked_docs)} chunks")

    total_chunks = len(chunks_to_process)
    expected_total = total_chunks * args.num_questions

    print(f"\n📝 Bắt đầu sinh Q&A: {total_chunks} chunks "
          f"× {args.num_questions} câu = ~{expected_total} câu hỏi dự kiến\n")

    for i, doc in enumerate(chunks_to_process):
        print(f"🔄 Đang xử lý chunk {i + 1}/{total_chunks}...", end=" ")
        try:
            res = chain.invoke({
                "context": doc.page_content,
                "num_questions": args.num_questions,
            })
            if res and res.items:
                for item in res.items:
                    final_dataset.append({
                        "question": item.question,
                        "ground_truth": item.ground_truth,
                        "source": doc.metadata.get("source", "unknown"),
                    })
                print(f"✅ +{len(res.items)} câu")
            else:
                print("⚠️ LLM trả về rỗng")
        except Exception as e:
            print(f"⚠️ Lỗi: {e}")

    # ── 5. Lưu kết quả ─────────────────────────────────────────────────
    output_path = args.output or os.path.join(project_root, "data", "test_scenario_pro.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 60)
    print(f"🎯 HOÀN TẤT! Đã sinh thành công {len(final_dataset)} câu hỏi tình huống.")
    print(f"💾 File được lưu tại: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
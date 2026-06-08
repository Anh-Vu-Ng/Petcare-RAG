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
sys.stdout.reconfigure(encoding="utf-8")
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
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
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
QUESTION_MODEL = "deepseek/deepseek-v4-flash"
GROUND_TRUTH_MODEL = "openai/gpt-4o-mini"

# ── Pydantic schema (ép LLM tuân thủ) ──────────────────────────────────────
class QuestionItem(BaseModel):
    quote_from_chunk: str = Field(
        description="Trích dẫn NGUYÊN VĂN đoạn text trong CHUNK chứa thông tin sẽ dùng. Trích dẫn BẮT BUỘC phải chứa các triệu chứng vật lý cụ thể mà bạn định dùng để đặt câu hỏi."
    )
    question: str = Field(
        description=(
            "Câu hỏi thực tế của người nuôi. "
            "BẮT BUỘC chứa các triệu chứng/hành động vật lý cụ thể mô tả trong 'quote_from_chunk'. "
            "KHÔNG dùng từ ngữ trừu tượng (như 'mệt mỏi', 'bỏ ăn') nếu chunk có chi tiết cụ thể hơn."
        )
    )

class QuestionDataset(BaseModel):
    items: list[QuestionItem]

class GroundTruthItem(BaseModel):
    ground_truth: str = Field(
        description=(
            "Câu trả lời chính xác, được viết lại CHỈ DỰA TRÊN 'quote_from_chunk'. "
            "CẤM tuyệt đối việc tự thêm các phương pháp điều trị, vitamin, thuốc hay chẩn đoán nằm ngoài trích dẫn."
        )
    )


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_QUESTION = """Bạn là một Data Engineer cực kỳ khắt khe, làm việc như một cỗ máy trích xuất thông tin y khoa tự động.
Nhiệm vụ: Phân tích CHUNK DỮ LIỆU và tạo ra tối đa {num_questions} cặp [quote_from_chunk, question] chất lượng cao để làm dữ liệu nền tảng.

QUY TRÌNH THỰC HIỆN BẮT BUỘC:
1. Đọc kỹ CHUNK văn bản đầu vào.
2. Quét tìm cụm thông tin có giá trị. BẮT BUỘC phải thoả mãn ĐIỀU KIỆN CHỌN QUOTE bên dưới.
3. Trích xuất NGUYÊN VĂN toàn bộ cụm thông tin đó vào trường `quote_from_chunk`. 
   *LƯU Ý CỰC KỲ QUAN TRỌNG: Trích dẫn phải đủ dài và bao gồm đầy đủ các từ mô tả triệu chứng vật lý mà bạn định sử dụng trong câu hỏi.*
4. Sinh `question` dựa vào `quote_from_chunk` vừa trích.

ĐIỀU KIỆN CHỌN QUOTE (QUAN TRỌNG):
- Cụm thông tin trích xuất BẮT BUỘC phải là một mệnh đề hoặc đoạn hoàn chỉnh chứa CẢ triệu chứng vật lý cụ thể VÀ giải pháp/chẩn đoán tương ứng.
- TUYỆT ĐỐI KHÔNG trích xuất những câu lửng lơ chỉ mô tả chung chung như "PETCARE điều trị viêm tai" mà bỏ qua phần mô tả triệu chứng vật lý (lắc đầu, hôi tai...) có trong chunk gốc. Hãy trích xuất cả đoạn bao gồm các triệu chứng đó.

CƠ CHẾ TỪ CHỐI (BẮT BUỘC BỎ QUA):
- Nếu CHUNK chỉ chứa: Địa chỉ, danh sách chi nhánh, số điện thoại, thông tin liên hệ, mục lục, bảng giá, câu chào hỏi xã giao, hoặc không chứa bất kỳ cặp [Triệu chứng y khoa -> Giải pháp/Chẩn đoán] rõ ràng nào -> KHÔNG ĐƯỢC PHÉP sinh Q&A. Hãy trả về danh sách items rỗng []. Đừng cố tạo ra dữ liệu rác.

RÀNG BUỘC CHO 'QUESTION':
Hãy chọn NGẪU NHIÊN một trong các kịch bản người dùng sau đây để đặt câu hỏi cho 'quote_from_chunk':
1. Người đang hoảng loạn: Hỏi dồn dập, dùng nhiều dấu chấm than, lo lắng. 
2. Người thiếu kiên nhẫn: Hỏi cộc lốc, ngắn gọn, chỉ mô tả đúng triệu chứng. 
3. Người kể chuyện rườm rà: Kể lể bối cảnh không liên quan rồi mới chốt lại triệu chứng ở cuối. 
-> Yêu cầu cốt lõi: Câu hỏi BẮT BUỘC phải sử dụng triệu chứng vật lý có mặt NGUYÊN VĂN (hoặc đồng nghĩa rất sát) trong `quote_from_chunk`. TUYỆT ĐỐI KHÔNG tự bịa ra triệu chứng mới không xuất hiện trong `quote_from_chunk`.
"""

SYSTEM_PROMPT_GT = """Bạn là Data Engineer cực kỳ khắt khe, làm việc như một cỗ máy trích xuất thông tin y khoa tự động.
Nhiệm vụ: Viết câu trả lời chính xác (ground_truth) cho câu hỏi dựa vào đoạn trích dẫn được cung cấp.

RÀNG BUỘC TUYỆT ĐỐI (ZERO-HALLUCINATION):
1. Câu trả lời của bạn CHỈ ĐƯỢC PHÉP dựa trên thông tin trong đoạn TRÍCH DẪN.
2. CẤM tuyệt đối việc tự ý kết nối các triệu chứng ngoài đời với chẩn đoán nếu đoạn TRÍCH DẪN không ghi rõ mối quan hệ đó.
   - Ví dụ: Nếu câu hỏi hỏi "Chó lắc đầu có phải viêm tai không?" nhưng đoạn TRÍCH DẪN chỉ ghi "PETCARE điều trị viêm tai" (không nói gì về triệu chứng lắc đầu) -> Bạn KHÔNG ĐƯỢC viết: "lắc đầu là triệu chứng viêm tai...". Bạn chỉ được viết: "PETCARE khám và điều trị viêm tai ở chó. Bạn nên đưa bé đến PETCARE để được chẩn đoán chính xác."
3. CẤM tuyệt đối việc tự ý thêm thắt các phương pháp điều trị, vitamin, thuốc hay chẩn đoán nằm ngoài trích dẫn.
4. Trả lời ngắn gọn, trực diện, đi thẳng vào vấn đề, trung thành 100% với văn bản trích dẫn.
"""

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_QUESTION),
    ("human", "TÀI LIỆU:\n{context}"),
])

GT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_GT),
    ("human", "TRÍCH DẪN:\n{quote}\n\nCÂU HỎI:\n{question}"),
])


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sinh bộ dữ liệu Q&A cho Petcare-RAG")
    parser.add_argument(
        "--test-size", type=int, default=50,
        help="Số lượng câu hỏi cho tập test hệ thống (mặc định: 50)",
    )
    parser.add_argument(
        "--tune-size", type=int, default=150,
        help="Số lượng câu hỏi cho tập grid search tuning (mặc định: 150)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=1,
        help="Số câu hỏi sinh ra cho mỗi chunk (mặc định: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=10,
        help="Seed cho random sampling, đảm bảo kết quả có thể tái tạo",
    )
    parser.add_argument(
        "--output-test", type=str, default=None,
        help="Đường dẫn file test dataset",
    )
    parser.add_argument(
        "--output-tune", type=str, default=None,
        help="Đường dẫn file tuning dataset",
    )
    return parser.parse_args()


def generate_qa_until_target(q_chain, gt_chain, chunk_pool, target_size, num_questions_per_chunk):
    """Gọi LLM sinh cặp QA cho đến khi đạt đủ số lượng câu hỏi mục tiêu."""
    final_dataset = []
    consecutive_errors = 0
    max_consecutive_errors = 5
    processed_chunks = 0

    print(f"📝 Bắt đầu sinh Q&A: Mục tiêu {target_size} câu. Pool hiện tại có: {len(chunk_pool)} chunks.")

    while len(final_dataset) < target_size and chunk_pool:
        doc = chunk_pool.pop(0)
        processed_chunks += 1
        current_len = len(final_dataset)
        print(f"  🔄 Đang xử lý chunk {processed_chunks} (Pool còn lại: {len(chunk_pool)})... [{current_len}/{target_size} câu]", end=" ", flush=True)
        try:
            # Bước 1: Dùng DeepSeek-V4-Flash để tạo câu hỏi và trích dẫn
            res_q = q_chain.invoke({
                "context": doc.page_content,
                "num_questions": num_questions_per_chunk,
            })
            consecutive_errors = 0
            
            if res_q and res_q.items:
                added_this_chunk = 0
                for item in res_q.items:
                    # Bước 2: Dùng GPT-4o-mini để tạo ground_truth dựa trên câu hỏi và trích dẫn
                    try:
                        res_gt = gt_chain.invoke({
                            "quote": item.quote_from_chunk,
                            "question": item.question,
                        })
                        if res_gt and res_gt.ground_truth:
                            final_dataset.append({
                                "question": item.question,
                                "ground_truth": res_gt.ground_truth,
                                "source": doc.metadata.get("source", "unknown"),
                                "quote_from_chunk": item.quote_from_chunk,
                            })
                            added_this_chunk += 1
                    except Exception as e_gt:
                        print(f"\n    ⚠️ Lỗi khi sinh Ground Truth từ GPT-4o-mini: {e_gt}")
                
                if added_this_chunk > 0:
                    print(f"✅ +{added_this_chunk} câu (Tổng cộng: {len(final_dataset)}/{target_size})")
                else:
                    print("⚠️ Không tạo được QA nào hợp lệ.")
            else:
                print("⚠️ DeepSeek từ chối (trả về rỗng)")
        except Exception as e:
            consecutive_errors += 1
            print(f"⚠️ Lỗi sinh câu hỏi: {e}")
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n❌ Tiến trình bị dừng đột ngột do gặp {max_consecutive_errors} lỗi API liên tiếp.")
                break

    # Cắt bớt phần dư thừa để đạt chính xác size yêu cầu
    return final_dataset[:target_size]


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 Khởi chạy tạo dữ liệu Test và Tuning tự động bằng LLM (2-Model Pipeline)")
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
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunked_docs = splitter.split_documents(raw_docs)
    print(f"✅ Đã cắt thành {len(chunked_docs)} chunk văn bản.")

    # ── 3. Cấu hình các LLM ────────────────────────────────────────────────
    print(f"🤖 Đang kết nối với OpenRouter...")
    print(f"  - Model sinh câu hỏi: {QUESTION_MODEL}")
    print(f"  - Model sinh Ground Truth: {GROUND_TRUTH_MODEL}")

    q_llm = ChatOpenAI(
        model=QUESTION_MODEL,
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    gt_llm = ChatOpenAI(
        model=GROUND_TRUTH_MODEL,
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # Bọc các LLM để ép trả về chuẩn format Pydantic
    structured_q_llm = q_llm.with_structured_output(QuestionDataset)
    q_chain = QUESTION_PROMPT | structured_q_llm

    structured_gt_llm = gt_llm.with_structured_output(GroundTruthItem)
    gt_chain = GT_PROMPT | structured_gt_llm

    # ── 4. Phân chia chunks không giao nhau ──────────────────────────────
    random.seed(args.seed)
    chunk_pool = list(chunked_docs)
    random.shuffle(chunk_pool)

    print(f"🎲 Tổng số chunks sẵn có: {len(chunk_pool)}")
    print(f"👉 Tập Test (Mục tiêu: {args.test_size} câu)")
    print(f"👉 Tập Tuning (Mục tiêu: {args.tune_size} câu)")

    # ── 5a. Sinh dữ liệu tập Test ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("🎯 BẮT ĐẦU TẠO DỮ LIỆU TẬP TEST (HỆ THỐNG)")
    print("=" * 50)
    test_dataset = generate_qa_until_target(q_chain, gt_chain, chunk_pool, args.test_size, args.num_questions)
    
    output_test_path = args.output_test or os.path.join(project_root, "data", "test_scenario_pro.json")
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    with open(output_test_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu thành công {len(test_dataset)} câu hỏi tập Test vào: {output_test_path}")

    # ── 5b. Sinh dữ liệu tập Tuning ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("⚙️ BẮT ĐẦU TẠO DỮ LIỆU TẬP TUNING (GRID SEARCH)")
    print("=" * 50)
    tune_dataset = generate_qa_until_target(q_chain, gt_chain, chunk_pool, args.tune_size, args.num_questions)
    
    output_tune_path = args.output_tune or os.path.join(project_root, "data", "test_finetune_chunks.json")
    os.makedirs(os.path.dirname(output_tune_path), exist_ok=True)
    with open(output_tune_path, "w", encoding="utf-8") as f:
        json.dump(tune_dataset, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu thành công {len(tune_dataset)} câu hỏi tập Tuning vào: {output_tune_path}")

    print("\n" + "=" * 60)
    print("🎯 HOÀN TẤT QUÁ TRÌNH TẠO DỮ LIỆU THÀNH CÔNG!")
    print(f"  - Tập Test: {len(test_dataset)} câu hỏi -> {output_test_path}")
    print(f"  - Tập Tuning: {len(tune_dataset)} câu hỏi -> {output_tune_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
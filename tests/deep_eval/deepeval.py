"""
DeepEval Evaluation Module for Petcare-RAG using:
- AnswerRelevancyMetric
- FaithfulnessMetric
- ContextualPrecisionMetric
- ContextualRecallMetric
Judge LLM: GPT-4o-mini via OpenRouter (litellm).
Embeddings: Jina Embedding v5-text-small (cho AnswerRelevancy).
"""
import sys
import langchain_core.messages

sys.modules['langchain.schema'] = langchain_core.messages
import os
import json
import time
from typing import Optional, List

from pydantic import BaseModel
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel

from src.jina_embeddings import JinaEmbeddings
from src.rag_chain import build_conversational_rag_chain

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

EVAL_LLM_MODEL = "openrouter/openai/gpt-4o-mini"
DEFAULT_TEST_DATA_PATH = "data/test_scenario_pro.json"
DEFAULT_RESULTS_PATH = "outputs/deepeval_results.json"
METRIC_THRESHOLD = 0.5

# ── Custom LLM (litellm + OpenRouter) ───────────────────────────────────────

class OpenRouterLLM(DeepEvalBaseLLM):
    """
    Wrap litellm completion cho DeepEval, sử dụng OpenRouter làm provider.
    Hỗ trợ structured output qua instructor để tương thích với các metrics
    yêu cầu JSON schema (G-Eval, Faithfulness, v.v.).
    """

    def __init__(self, model_name: str = EVAL_LLM_MODEL):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        import instructor
        from litellm import completion

        if schema:
            client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
            return client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=0,
                max_tokens=4096
            )
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel | str:
        import instructor
        from litellm import acompletion

        if schema:
            client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
            return await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=0,
                max_tokens=4096
            )
        response = await acompletion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        return response.choices[0].message.content


# ── Custom Embeddings (Jina) ────────────────────────────────────────────────

class JinaDeepEvalEmbeddings(DeepEvalBaseEmbeddingModel):
    """
    Wrap JinaEmbeddings (jina-embeddings-v5-text-small) cho DeepEval.
    Dùng bởi AnswerRelevancyMetric để tính cosine similarity.
    """

    def __init__(self):
        self._model = None

    def load_model(self):
        if self._model is None:
            self._model = JinaEmbeddings()
        return self._model

    def get_model_name(self) -> str:
        return "jina-embeddings-v5-text-small"

    def embed_text(self, text: str) -> List[float]:
        model = self.load_model()
        return model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self.load_model()
        return model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)


# ── Pipeline Runner ─────────────────────────────────────────────────────────

def run_pipeline_for_eval(pipeline, question: str) -> dict:
    """
    Chạy RAG pipeline cho 1 câu hỏi (chỉ nhánh KNOWLEDGE).
    Bypass cache để đảm bảo retrieval thực tế.

    Returns:
        {
            "response": str,                # câu trả lời LLM
            "retrieved_contexts": list[str], # nội dung các docs đã retrieve
            "intent": str,                   # KNOWLEDGE hoặc TOOL
        }
    """
    # Tạm disable cache để force retrieval
    original_threshold = pipeline.semantic_cache.threshold
    pipeline.semantic_cache.threshold = 2.0  # Score > 1.0 = never match

    try:
        result = pipeline.invoke({
            "input": question,
            "chat_history": [],
        })
    finally:
        # Khôi phục cache threshold
        pipeline.semantic_cache.threshold = original_threshold

    # Trích xuất context strings từ retrieved documents
    context_docs = result.get("context", [])
    retrieved_contexts = [doc.page_content for doc in context_docs]

    return {
        "response": result.get("answer", ""),
        "retrieved_contexts": retrieved_contexts,
        "intent": result.get("intent", "KNOWLEDGE"),
    }


# ── Test Case Builder ───────────────────────────────────────────────────────

def build_test_cases(
    pipeline,
    test_data: list[dict],
    verbose: bool = True,
) -> tuple[list[LLMTestCase], list[dict]]:
    """
    Chạy pipeline trên toàn bộ test data, tạo list[LLMTestCase].

    Chỉ xử lý câu hỏi KNOWLEDGE (có retrieved context).
    Câu hỏi bị route sang TOOL sẽ bị skip.

    Returns:
        (test_cases, raw_results)
    """
    test_cases = []
    raw_results = []
    skipped = 0

    total = len(test_data)
    for idx, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        if verbose:
            print(f"\n[{idx}/{total}] 🔄 Processing: {question[:60]}...")

        t0 = time.time()
        result = run_pipeline_for_eval(pipeline, question)
        elapsed = time.time() - t0

        # Skip câu hỏi bị route sang TOOL
        if result["intent"] == "TOOL":
            if verbose:
                print(f"  ⏭️  Skipped (TOOL intent) — {elapsed:.2f}s")
            skipped += 1
            continue

        # Nếu không có context nào được retrieve, vẫn giữ lại nhưng cảnh báo
        if not result["retrieved_contexts"]:
            if verbose:
                print(f"  ⚠️  No context retrieved — {elapsed:.2f}s")
            result["retrieved_contexts"] = [""]

        response = result.get("response", "")
        if not response or not response.strip():
            response = "Không có câu trả lời từ hệ thống RAG."

        test_case = LLMTestCase(
            input=question,
            actual_output=response,
            expected_output=ground_truth,
            retrieval_context=result["retrieved_contexts"],
        )
        test_cases.append(test_case)

        raw_results.append({
            "input": question,
            "expected_output": ground_truth,
            "actual_output": result["response"],
            "retrieval_context": result["retrieved_contexts"],
            "intent": result["intent"],
            "elapsed_seconds": round(elapsed, 2),
        })

        if verbose:
            ctx_count = len(result["retrieved_contexts"])
            resp_preview = result["response"][:80].replace("\n", " ")
            print(f"  ✅ Done — {ctx_count} contexts, {elapsed:.2f}s")
            print(f"     Response: {resp_preview}...")

    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {len(test_cases)} test cases built, {skipped} skipped (TOOL)")
        print(f"{'='*60}")

    return test_cases, raw_results


# ── Main Evaluation ─────────────────────────────────────────────────────────

def evaluate_pregenerated_answers(
    answers_path: str,
    results_path: str = DEFAULT_RESULTS_PATH,
    verbose: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """
    Chạy evaluation từ file câu trả lời đã được pre-generate.

    1. Load pre-generated answers từ JSON.
    2. Build LLMTestCase list từ data đã load.
    3. Gọi deepeval.evaluate() với 4 metrics.
    4. Xuất kết quả ra console + JSON file.
    """
    if verbose:
        print(f"📂 Loading pre-generated answers from {answers_path}...")

    with open(answers_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    if verbose:
        print(f"   Loaded {len(raw_results)} answers.")

    if limit is not None and limit > 0:
        raw_results = raw_results[:limit]
        if verbose:
            print(f"   ⚡ Limited to {len(raw_results)} samples (--limit {limit})")

    # Build test cases from pre-generated results
    test_cases = []
    skipped_tool = 0
    for item in raw_results:
        # Bỏ qua các câu hỏi route sang TOOL (không có retrieval context thực tế)
        if item.get("intent") == "TOOL":
            skipped_tool += 1
            continue

        # Nếu không có context nào, gán chuỗi rỗng
        retrieval_context = item.get("retrieval_context", [])
        if not retrieval_context:
            retrieval_context = [""]

        actual_output = item.get("actual_output", "")
        if not actual_output or not actual_output.strip():
            actual_output = "Không có câu trả lời từ hệ thống RAG."

        test_case = LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)

    if verbose and skipped_tool > 0:
        print(f"   ⏭️  Skipped {skipped_tool} samples with TOOL intent.")

    if not test_cases:
        print("❌ No test cases to evaluate.")
        return {"error": "No test cases to evaluate"}

    # ── DeepEval Evaluate ─────────────────────────────────────────────
    if verbose:
        print("\n🧪 Running DeepEval evaluation...")
        print(f"   Judge LLM: {EVAL_LLM_MODEL} (via OpenRouter)")
        print(f"   Embeddings: jina-embeddings-v5-text-small")
        print(f"   Metrics: AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall")

    eval_llm = OpenRouterLLM(EVAL_LLM_MODEL)

    metrics = [
        AnswerRelevancyMetric(
            model=eval_llm,
            threshold=METRIC_THRESHOLD,
        ),
        FaithfulnessMetric(model=eval_llm, threshold=METRIC_THRESHOLD),
        ContextualPrecisionMetric(model=eval_llm, threshold=METRIC_THRESHOLD),
        ContextualRecallMetric(model=eval_llm, threshold=METRIC_THRESHOLD),
    ]

    from deepeval.evaluate import AsyncConfig, DisplayConfig

    t0 = time.time()

    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=verbose, inspect_after_run=False),
    )
    eval_elapsed = time.time() - t0

    # ── Format & export results ───────────────────────────────────────
    # Metric name mapping (DeepEval uses display names like "Answer Relevancy")
    METRIC_KEY_MAP = {
        "Answer Relevancy": "answer_relevancy",
        "Faithfulness": "faithfulness",
        "Contextual Precision": "contextual_precision",
        "Contextual Recall": "contextual_recall",
    }

    per_sample_scores = []
    aggregate_sums = {v: 0.0 for v in METRIC_KEY_MAP.values()}
    aggregate_counts = {v: 0 for v in METRIC_KEY_MAP.values()}

    for i, test_result in enumerate(eval_result.test_results):
        sample_data = {
            "question": test_result.input,
        }

        for metric_data in test_result.metrics_data:
            key = METRIC_KEY_MAP.get(metric_data.name, metric_data.name.lower().replace(" ", "_"))
            score = metric_data.score

            if score is not None:
                sample_data[key] = round(score, 4)
                aggregate_sums[key] = aggregate_sums.get(key, 0) + score
                aggregate_counts[key] = aggregate_counts.get(key, 0) + 1
            else:
                sample_data[key] = None

            sample_data[f"{key}_reason"] = metric_data.reason

        # Merge raw_results nếu có
        if i < len(raw_results):
            sample_data["ground_truth"] = raw_results[i].get("expected_output", "")
            sample_data["response_preview"] = raw_results[i].get("actual_output", "")[:200]
            sample_data["num_contexts"] = len(raw_results[i].get("retrieval_context", []))
            sample_data["pipeline_time_s"] = raw_results[i].get("elapsed_seconds", 0)

        per_sample_scores.append(sample_data)

    # Aggregate scores (mean)
    aggregate_scores = {}
    for key in METRIC_KEY_MAP.values():
        if aggregate_counts.get(key, 0) > 0:
            aggregate_scores[key] = round(
                aggregate_sums[key] / aggregate_counts[key], 4
            )

    output = {
        "metadata": {
            "eval_framework": "deepeval",
            "eval_model": EVAL_LLM_MODEL,
            "embedding_model": "jina-embeddings-v5-text-small",
            "answers_path": answers_path,
            "num_samples_evaluated": len(test_cases),
            "eval_duration_seconds": round(eval_elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "aggregate_scores": aggregate_scores,
        "per_sample_scores": per_sample_scores,
    }

    # Save to JSON
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Console output ───────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"{'DEEPEVAL EVALUATION RESULTS':^60}")
        print(f"{'='*60}")

        print(f"\n📊 Aggregate Scores:")
        for metric, score in aggregate_scores.items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {bar} {metric}: {score:.4f}")

        print(f"\n📋 Per-Sample Breakdown:")
        print(f"   {'#':<4} {'A.Relev.':<10} {'Faith.':<10} {'C.Prec.':<10} {'C.Recall':<10} {'Question':<40}")
        print(f"   {'─'*84}")

        for idx, s in enumerate(per_sample_scores, 1):
            ar = f"{s['answer_relevancy']:.4f}" if s.get("answer_relevancy") is not None else "N/A"
            ff = f"{s['faithfulness']:.4f}" if s.get("faithfulness") is not None else "N/A"
            cp = f"{s['contextual_precision']:.4f}" if s.get("contextual_precision") is not None else "N/A"
            cr = f"{s['contextual_recall']:.4f}" if s.get("contextual_recall") is not None else "N/A"
            q = s["question"][:38]
            print(f"   {idx:<4} {ar:<10} {ff:<10} {cp:<10} {cr:<10} {q}")

        print(f"\n⏱️  Evaluation time: {eval_elapsed:.2f}s")
        print(f"💾 Results saved to: {results_path}")
        print(f"{'='*60}")

    return output


def evaluate_rag(
    test_data_path: str = DEFAULT_TEST_DATA_PATH,
    results_path: str = DEFAULT_RESULTS_PATH,
    verbose: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """
    Hàm chính tương thích ngược — chạy toàn bộ flow evaluation DeepEval.
    Tự động sinh câu trả lời tạm thời rồi chạy evaluation.
    """
    # ── 1. Load test data ────────────────────────────────────────────────
    if verbose:
        print("📂 Loading test data...")

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if verbose:
        print(f"   Loaded {len(test_data)} test samples from {test_data_path}")

    # Giới hạn số câu hỏi nếu có --limit
    if limit is not None and limit > 0:
        test_data = test_data[:limit]
        if verbose:
            print(f"   ⚡ Limited to {len(test_data)} samples (--limit {limit})")

    # ── 2. Build pipeline ────────────────────────────────────────────────
    if verbose:
        print("\n🔧 Building RAG pipeline...")

    pipeline = build_conversational_rag_chain()

    if verbose:
        print("   Pipeline ready.")

    # ── 3. Build test cases ──────────────────────────────────────────────
    if verbose:
        print("\n🔄 Running pipeline on test data...")

    test_cases, raw_results = build_test_cases(
        pipeline, test_data, verbose=verbose
    )

    if not test_cases:
        print("❌ No test cases to evaluate (all skipped or errored).")
        return {"error": "No test cases to evaluate"}

    # ── 4. Save temporary answers & evaluate ─────────────────────────────
    temp_answers_path = results_path + ".temp_answers.json"
    os.makedirs(os.path.dirname(temp_answers_path) or ".", exist_ok=True)
    with open(temp_answers_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    try:
        return evaluate_pregenerated_answers(
            answers_path=temp_answers_path,
            results_path=results_path,
            verbose=verbose,
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_answers_path):
            try:
                os.remove(temp_answers_path)
            except Exception:
                pass


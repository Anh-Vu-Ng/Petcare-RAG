import os
# Tăng thời gian chờ (timeout) của DeepEval lên 10 phút (600s) để tránh lỗi khi OpenRouter phản hồi chậm hoặc bị nghẽn
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from tests.deep_eval.deepeval import evaluate_pregenerated_answers, DEFAULT_RESULTS_PATH


def main():
    parser = argparse.ArgumentParser(
        description="🧪 DeepEval Evaluation từ câu trả lời có sẵn cho Petcare-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m tests.deep_eval.run_deepeval
  uv run python -m tests.deep_eval.run_deepeval --answers-file outputs/generated_answers.json --output outputs/deepeval_results.json
  uv run python -m tests.deep_eval.run_deepeval --limit 5
  uv run python -m tests.deep_eval.run_deepeval --quiet
        """,
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="outputs/generated_answers.json",
        help="Đường dẫn tới file câu trả lời đã sinh (default: outputs/generated_answers.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Đường dẫn lưu kết quả evaluation (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Chỉ in kết quả tổng hợp, không in chi tiết từng sample",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ đánh giá N câu hỏi đầu tiên (VD: --limit 5 để test nhanh)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        🧪 DeepEval Evaluation — Petcare-RAG             ║")
    print("║        Metrics: AnswerRelevancy, Faithfulness,           ║")
    print("║                 ContextualPrecision, ContextualRecall    ║")
    print("║        Judge: openai/gpt-4o-mini                         ║")
    print("║        Embedding: Jina v5-text-small                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    try:
        result = evaluate_pregenerated_answers(
            answers_path=args.answers_file,
            results_path=args.output,
            verbose=not args.quiet,
            limit=args.limit,
        )

        if "error" in result:
            print(f"\n❌ Evaluation failed: {result['error']}")
            sys.exit(1)

        # Nếu quiet mode, chỉ in aggregate scores
        if args.quiet:
            print("\n📊 Aggregate Scores:")
            for metric, score in result.get("aggregate_scores", {}).items():
                print(f"   {metric}: {score:.4f}")
            print(f"\n💾 Full results saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"   Kiểm tra lại đường dẫn file câu trả lời: {args.answers_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


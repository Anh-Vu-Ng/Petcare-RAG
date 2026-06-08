import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import argparse
from typing import Optional

from dotenv import load_dotenv

from tests.deep_eval.deepeval import (
    DEFAULT_TEST_DATA_PATH,
    run_pipeline_for_eval
)
from src.rag_chain import build_conversational_rag_chain

load_dotenv()

DEFAULT_ANSWERS_PATH = "outputs/generated_answers.json"


def main():
    parser = argparse.ArgumentParser(
        description="🏃 Chạy RAG pipeline để sinh câu trả lời cho test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=DEFAULT_TEST_DATA_PATH,
        help=f"Đường dẫn tới file test dataset JSON (default: {DEFAULT_TEST_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_ANSWERS_PATH,
        help=f"Đường dẫn lưu câu trả lời (default: {DEFAULT_ANSWERS_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ sinh câu trả lời cho N câu hỏi đầu tiên",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Bắt đầu lại từ đầu, không dùng lại câu trả lời đã sinh trước đó",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        🏃 RAG Answer Generator — Petcare-RAG             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 1. Load test data
    print(f"📂 Loading test data from: {args.test_data}")
    if not os.path.exists(args.test_data):
        print(f"❌ Test data file not found at: {args.test_data}")
        sys.exit(1)

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"   Loaded {len(test_data)} samples.")

    # Limit if specified
    if args.limit is not None and args.limit > 0:
        test_data = test_data[:args.limit]
        print(f"   ⚡ Limited to {len(test_data)} samples.")

    # 2. Check for existing answers to resume
    existing_answers = {}
    if not args.no_resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    # Map input to its answer object
                    existing_answers = {item["input"]: item for item in loaded_data if "input" in item}
                    print(f"🔄 Found existing progress. Loaded {len(existing_answers)} generated answers.")
                else:
                    print("⚠️ Existing output file format is invalid. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Failed to read existing output file ({e}). Starting fresh.")

    # 3. Build RAG pipeline (only if there are questions to process)
    questions_to_run = [item for item in test_data if item["question"] not in existing_answers]
    
    if not questions_to_run:
        print("\n✨ All questions are already answered! Nothing to do.")
        print(f"💾 File: {args.output}")
        return

    print(f"\n🔧 Building RAG pipeline (need to run {len(questions_to_run)}/{len(test_data)} questions)...")
    pipeline = build_conversational_rag_chain()
    print("   Pipeline ready.")

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 4. Generate answers
    results = []
    skipped = 0
    total = len(test_data)

    print("\n🔄 Running pipeline on test data...")
    for idx, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Re-use existing answer if available
        if question in existing_answers:
            results.append(existing_answers[question])
            print(f"[{idx}/{total}] ⏭️  Reused existing answer for: {question[:50]}...")
            continue

        print(f"[{idx}/{total}] 🔄 Processing: {question[:50]}...")
        t0 = time.time()
        try:
            result = run_pipeline_for_eval(pipeline, question)
            elapsed = time.time() - t0

            # If no context retrieved, warning but keep it
            if result["intent"] != "TOOL" and not result["retrieved_contexts"]:
                print(f"  ⚠️  No context retrieved — {elapsed:.2f}s")
                result["retrieved_contexts"] = [""]

            answer_item = {
                "input": question,
                "expected_output": ground_truth,
                "actual_output": result["response"],
                "retrieval_context": result["retrieved_contexts"],
                "intent": result["intent"],
                "elapsed_seconds": round(elapsed, 2),
            }
            results.append(answer_item)

            if result["intent"] == "TOOL":
                print(f"  ⏭️  Processed (TOOL intent) — {elapsed:.2f}s")
                skipped += 1
            else:
                ctx_count = len(result["retrieved_contexts"])
                resp_preview = result["response"][:60].replace("\n", " ")
                print(f"  ✅ Done — {ctx_count} contexts, {elapsed:.2f}s")
                print(f"     Response: {resp_preview}...")

            # Save progress incrementally to prevent losing data
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  ❌ Error processing question: {e}")
            print("💾 Saving current progress and exiting...")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🎉 Answer generation completed!")
    print(f"💾 Saved to: {args.output}")
    print(f"📊 Total processed: {len(results)}, Skipped (TOOL): {skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

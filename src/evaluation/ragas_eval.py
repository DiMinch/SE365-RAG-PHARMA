"""
RAGAS Evaluation Pipeline (PROPOSAL Section 6.1–6.4)

Runs the full RAG pipeline against PharmaQA.VN benchmark and computes:
  - Retrieval metrics: Recall@k, MRR, Context Precision
  - QA metrics: Faithfulness, Answer Relevancy (via RAGAS)
  - Token-level F1 and ROUGE-L (custom)

Supports ablation study: toggle components via CLI flags to compare
different system configurations (PROPOSAL Section 6.4).

Usage:
    # Full evaluation with default settings:
    python -m src.evaluation.ragas_eval --benchmark data/benchmark/pharma_qa.json

    # Ablation: baseline (BM25 only, no reranker):
    python -m src.evaluation.ragas_eval --benchmark data/benchmark/pharma_qa.json \\
        --retrieval-mode bm25 --no-reranker --no-hallucination-ctrl

    # Ablation: dense only:
    python -m src.evaluation.ragas_eval --benchmark data/benchmark/pharma_qa.json \\
        --retrieval-mode dense --no-reranker

    # Run all ablation configs automatically:
    python -m src.evaluation.ragas_eval --benchmark data/benchmark/pharma_qa.json --ablation
"""

import os
import sys
import json
import time
import logging
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv()

from src.database.qdrant_client import PharmaQdrantClient
from src.utils.config import get_base_config, get_prompts_config

logger = logging.getLogger("RAGASEval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")


# ─── LLM API caller (standalone, no chainlit dependency) ─────────────────────

def call_llm_api(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini or OpenAI API (extracted from app.py for standalone use)."""
    import requests

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        }
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if res.status_code == 200:
                return res.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"[LLM Error {res.status_code}]"
        except Exception as e:
            return f"[LLM Error: {e}]"

    elif openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            return f"[LLM Error {res.status_code}]"
        except Exception as e:
            return f"[LLM Error: {e}]"

    return "[No LLM API key configured]"


# ─── Custom Metrics (no RAGAS dependency needed) ─────────────────────────────

def tokenize_vi(text: str) -> List[str]:
    """Simple Vietnamese tokenizer (whitespace + lowercase)."""
    return re.findall(r'\w+', text.lower())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = tokenize_vi(prediction)
    truth_tokens = tokenize_vi(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L (Longest Common Subsequence)."""
    pred_tokens = tokenize_vi(prediction)
    truth_tokens = tokenize_vi(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == truth_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_faithfulness_llm(answer: str, context: str) -> Tuple[float, str]:
    """
    Use LLM-as-judge to evaluate faithfulness (RAGAS-style).
    Returns (score, verdict).
    """
    prompt = f"""Bạn là một dược sĩ kiểm duyệt. Đánh giá xem câu trả lời có trung thành với ngữ cảnh không.

NGỮ CẢNH:
{context[:3000]}

CÂU TRẢ LỜI:
{answer[:2000]}

Chấm điểm FAITHFULNESS từ 0.0 đến 1.0:
- 1.0: Mọi thông tin trong câu trả lời đều có căn cứ trong ngữ cảnh
- 0.5: Phần lớn đúng nhưng có 1-2 chi tiết không có trong ngữ cảnh
- 0.0: Câu trả lời bịa đặt thông tin không có trong ngữ cảnh

Trả về ĐÚNG format:
SCORE: <số từ 0.0 đến 1.0>
REASON: <1 câu ngắn>"""

    result = call_llm_api("Bạn là hệ thống chấm điểm. Chỉ trả về SCORE và REASON.", prompt)

    score = 0.5  # default
    try:
        for line in result.split("\n"):
            if "SCORE:" in line.upper():
                score_str = line.split(":")[-1].strip()
                score = float(score_str)
                score = max(0.0, min(1.0, score))
                break
    except (ValueError, IndexError):
        pass

    return score, result


def compute_answer_relevancy_llm(answer: str, question: str) -> Tuple[float, str]:
    """
    Use LLM-as-judge to evaluate answer relevancy.
    Returns (score, verdict).
    """
    prompt = f"""Đánh giá xem câu trả lời có liên quan và trả lời đúng câu hỏi không.

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
{answer[:2000]}

Chấm điểm RELEVANCY từ 0.0 đến 1.0:
- 1.0: Câu trả lời trực tiếp và đầy đủ cho câu hỏi
- 0.5: Có liên quan nhưng không trả lời trực tiếp hoặc thiếu thông tin
- 0.0: Không liên quan đến câu hỏi

Trả về ĐÚNG format:
SCORE: <số từ 0.0 đến 1.0>
REASON: <1 câu ngắn>"""

    result = call_llm_api("Bạn là hệ thống chấm điểm. Chỉ trả về SCORE và REASON.", prompt)

    score = 0.5
    try:
        for line in result.split("\n"):
            if "SCORE:" in line.upper():
                score_str = line.split(":")[-1].strip()
                score = float(score_str)
                score = max(0.0, min(1.0, score))
                break
    except (ValueError, IndexError):
        pass

    return score, result


# ─── Ablation Configurations (PROPOSAL Section 6.4) ─────────────────────────

ABLATION_CONFIGS = [
    {
        "name": "naive_rag",
        "description": "BM25 only, no reranker, no hallucination control",
        "retrieval_mode": "bm25",
        "use_reranker": False,
        "hallucination_ctrl": False,
    },
    {
        "name": "dense_only",
        "description": "Dense retrieval only, no reranker",
        "retrieval_mode": "dense",
        "use_reranker": False,
        "hallucination_ctrl": False,
    },
    {
        "name": "hybrid",
        "description": "BM25 + Dense (RRF), no reranker",
        "retrieval_mode": "hybrid",
        "use_reranker": False,
        "hallucination_ctrl": False,
    },
    {
        "name": "hybrid_reranker",
        "description": "Hybrid + Cross-Encoder Reranker",
        "retrieval_mode": "hybrid",
        "use_reranker": True,
        "hallucination_ctrl": False,
    },
    {
        "name": "hybrid_reranker_halluc",
        "description": "Hybrid + Reranker + Hallucination Control",
        "retrieval_mode": "hybrid",
        "use_reranker": True,
        "hallucination_ctrl": True,
    },
    {
        "name": "full_system",
        "description": "Full system (all components enabled)",
        "retrieval_mode": "hybrid",
        "use_reranker": True,
        "hallucination_ctrl": True,
    },
]


# ─── Main Evaluator ─────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates the RAG pipeline against PharmaQA.VN benchmark.
    """

    def __init__(self):
        self.db_client = PharmaQdrantClient()
        self.prompts_cfg = get_prompts_config()
        self.base_cfg = get_base_config()

    def run_single_query(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a single RAG query and return results for evaluation.

        Returns dict with:
            - question, answer, contexts, retrieved_sections
        """
        # Retrieve
        search_results = self.db_client.search(
            query=question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
        )

        if not search_results:
            return {
                "question": question,
                "answer": "[No results found]",
                "contexts": [],
                "retrieved_sections": [],
                "best_score": 0.0,
            }

        # Build context
        context_parts = []
        retrieved_sections = []
        for idx, res in enumerate(search_results):
            payload = res["payload"]
            context_parts.append(
                f"Nguồn [{idx+1}]: {payload.get('drug_name', '')} | {payload.get('section_name', '')}\n"
                f"{payload.get('chunk_text', '')}"
            )
            retrieved_sections.append({
                "drug_name": payload.get("drug_name", ""),
                "registration_no": payload.get("registration_no", payload.get("registration_number", "")),
                "section_name": payload.get("section_name", ""),
                "score": res["score"],
            })

        context = "\n---\n".join(context_parts)
        best_score = max(res["score"] for res in search_results)

        # Generate answer
        system_prompt = self.prompts_cfg["system_prompt"].format(context=context)
        answer = call_llm_api(system_prompt, question)

        return {
            "question": question,
            "answer": answer,
            "contexts": context_parts,
            "context_joined": context,
            "retrieved_sections": retrieved_sections,
            "best_score": best_score,
        }

    def evaluate(
        self,
        benchmark_path: str,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        hallucination_ctrl: bool = True,
        use_llm_judge: bool = True,
        config_name: str = "default",
        max_queries: int = 0,
    ) -> Dict[str, Any]:
        """
        Run full evaluation against benchmark.

        Args:
            benchmark_path: Path to PharmaQA.VN JSON.
            retrieval_mode: 'dense', 'bm25', or 'hybrid'.
            use_reranker: Whether to use cross-encoder reranker.
            hallucination_ctrl: Whether to enable abstention mechanism.
            use_llm_judge: Whether to run LLM-as-judge for Faithfulness & Relevancy.
            config_name: Name for this evaluation run.
            max_queries: Limit queries (0 = all). Useful for testing.

        Returns:
            Dict with aggregated metrics.
        """
        # Load benchmark
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        logger.info("Loaded %d benchmark questions", len(benchmark))

        if max_queries > 0:
            benchmark = benchmark[:max_queries]
            logger.info("Limited to %d queries (--max-queries)", max_queries)

        # Metrics accumulators
        retrieval_metrics = {
            "recall@1": 0, "recall@3": 0, "recall@5": 0,
            "mrr": 0.0, "context_precision": 0.0,
        }
        qa_metrics = {
            "f1": 0.0, "exact_match": 0.0, "rouge_l": 0.0,
            "faithfulness": 0.0, "answer_relevancy": 0.0,
        }
        abstention_count = 0
        query_count = 0
        detailed_results = []

        abstention_threshold = self.base_cfg.get("hallucination_reduction", {}).get(
            "abstention_threshold", 0.55
        )

        for i, qa in enumerate(benchmark):
            question = qa.get("question", "")
            ground_truth = qa.get("ground_truth", "")
            expected_reg_no = qa.get("registration_no", "")
            expected_section = qa.get("section", "")

            if not question:
                continue

            query_count += 1
            logger.info("[%d/%d] Q: %s", i + 1, len(benchmark), question[:60])

            # Run RAG
            result = self.run_single_query(
                question,
                retrieval_mode=retrieval_mode,
                use_reranker=use_reranker,
            )

            # Abstention check
            if hallucination_ctrl and result["best_score"] < abstention_threshold:
                abstention_count += 1
                result["abstained"] = True
                detailed_results.append(result)
                continue

            result["abstained"] = False

            # ── Retrieval Metrics ────────────────────────────────────────
            relevant_rank = None
            for rank, sec in enumerate(result["retrieved_sections"]):
                if (sec["registration_no"] == expected_reg_no and
                        sec["section_name"] == expected_section):
                    relevant_rank = rank
                    break

            if relevant_rank is not None:
                for k in [1, 3, 5]:
                    if relevant_rank < k:
                        retrieval_metrics[f"recall@{k}"] += 1
                retrieval_metrics["mrr"] += 1.0 / (relevant_rank + 1)
                # Context precision: is the relevant doc in position 1?
                if relevant_rank == 0:
                    retrieval_metrics["context_precision"] += 1

            # ── QA Metrics ───────────────────────────────────────────────
            answer = result["answer"]
            if ground_truth:
                qa_metrics["f1"] += compute_f1(answer, ground_truth)
                qa_metrics["exact_match"] += compute_exact_match(answer, ground_truth)
                qa_metrics["rouge_l"] += compute_rouge_l(answer, ground_truth)

            # LLM-as-judge metrics (expensive — 2 extra API calls per query)
            if use_llm_judge and result["context_joined"]:
                faith_score, _ = compute_faithfulness_llm(answer, result["context_joined"])
                relev_score, _ = compute_answer_relevancy_llm(answer, question)
                qa_metrics["faithfulness"] += faith_score
                qa_metrics["answer_relevancy"] += relev_score
                result["faithfulness"] = faith_score
                result["answer_relevancy"] = relev_score

            result["ground_truth"] = ground_truth
            detailed_results.append(result)

            # Rate limiting for LLM API
            if use_llm_judge:
                time.sleep(1)

        # ── Aggregate ────────────────────────────────────────────────────
        answered = query_count - abstention_count
        if answered > 0:
            for key in retrieval_metrics:
                retrieval_metrics[key] = round(retrieval_metrics[key] / answered, 4)
            for key in qa_metrics:
                qa_metrics[key] = round(qa_metrics[key] / answered, 4)

        summary = {
            "config_name": config_name,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "hallucination_ctrl": hallucination_ctrl,
            "total_queries": query_count,
            "answered": answered,
            "abstained": abstention_count,
            "abstention_rate": round(abstention_count / query_count, 4) if query_count > 0 else 0,
            "retrieval": retrieval_metrics,
            "qa": qa_metrics,
        }

        # Save results
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"eval_{config_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "detailed": detailed_results}, f,
                      indent=2, ensure_ascii=False, default=str)
        logger.info("Results saved to %s", out_path)

        self._print_summary(summary)
        return summary

    def run_ablation_study(self, benchmark_path: str, use_llm_judge: bool = True, max_queries: int = 0):
        """Run all ablation configurations and produce comparison table."""
        all_summaries = []

        for config in ABLATION_CONFIGS:
            logger.info("\n" + "=" * 60)
            logger.info("Ablation: %s — %s", config["name"], config["description"])
            logger.info("=" * 60)

            summary = self.evaluate(
                benchmark_path=benchmark_path,
                retrieval_mode=config["retrieval_mode"],
                use_reranker=config["use_reranker"],
                hallucination_ctrl=config["hallucination_ctrl"],
                use_llm_judge=use_llm_judge,
                config_name=config["name"],
                max_queries=max_queries,
            )
            all_summaries.append(summary)

        # Save combined results
        out_dir = Path("results")
        combined_path = out_dir / "ablation_study_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        logger.info("\nAblation study complete. Combined results: %s", combined_path)

        # Print comparison table
        self._print_ablation_table(all_summaries)
        return all_summaries

    def _print_summary(self, summary: Dict):
        """Print single evaluation summary."""
        print(f"\n{'─' * 60}")
        print(f"Config: {summary['config_name']}")
        print(f"Mode: {summary['retrieval_mode']} | Reranker: {summary['use_reranker']} | Halluc: {summary['hallucination_ctrl']}")
        print(f"Queries: {summary['total_queries']} | Answered: {summary['answered']} | Abstained: {summary['abstained']}")
        print(f"{'─' * 60}")
        r = summary["retrieval"]
        q = summary["qa"]
        print(f"  Recall@1: {r['recall@1']:.4f}  Recall@3: {r['recall@3']:.4f}  Recall@5: {r['recall@5']:.4f}")
        print(f"  MRR: {r['mrr']:.4f}  Context Precision: {r['context_precision']:.4f}")
        print(f"  F1: {q['f1']:.4f}  EM: {q['exact_match']:.4f}  ROUGE-L: {q['rouge_l']:.4f}")
        print(f"  Faithfulness: {q['faithfulness']:.4f}  Answer Relevancy: {q['answer_relevancy']:.4f}")
        print(f"{'─' * 60}\n")

    def _print_ablation_table(self, summaries: List[Dict]):
        """Print formatted ablation comparison table."""
        print("\n" + "=" * 100)
        print("ABLATION STUDY RESULTS (PROPOSAL Section 6.4)")
        print("=" * 100)
        print(f"{'Config':<25} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6} {'F1':>6} {'Faith':>6} {'Relev':>6} {'Abst%':>6}")
        print("-" * 100)

        for s in summaries:
            r, q = s["retrieval"], s["qa"]
            print(
                f"{s['config_name']:<25} "
                f"{r['recall@1']:>5.3f} {r['recall@3']:>5.3f} {r['recall@5']:>5.3f} "
                f"{r['mrr']:>6.4f} {q['f1']:>6.4f} "
                f"{q['faithfulness']:>6.4f} {q['answer_relevancy']:>6.4f} "
                f"{s['abstention_rate']:>6.2%}"
            )

        print("=" * 100)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Pipeline for Pharma-RAG (PROPOSAL 6.1–6.4)"
    )
    parser.add_argument(
        "--benchmark", "-b", required=True,
        help="Path to PharmaQA.VN benchmark JSON",
    )
    parser.add_argument(
        "--retrieval-mode", choices=["dense", "bm25", "hybrid"], default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--no-reranker", action="store_true",
        help="Disable cross-encoder reranker",
    )
    parser.add_argument(
        "--no-hallucination-ctrl", action="store_true",
        help="Disable abstention mechanism",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM-as-judge metrics (saves API calls)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run full ablation study (all 6 configs)",
    )
    parser.add_argument(
        "--max-queries", "-m", type=int, default=0,
        help="Max queries to evaluate (0 = all)",
    )
    parser.add_argument(
        "--config-name", default="default",
        help="Name for this evaluation run",
    )

    args = parser.parse_args()

    if not Path(args.benchmark).exists():
        logger.error("Benchmark not found: %s", args.benchmark)
        logger.error("Waiting for PharmaQA.VN from teammate.")
        sys.exit(1)

    evaluator = RAGEvaluator()

    if args.ablation:
        evaluator.run_ablation_study(
            benchmark_path=args.benchmark,
            use_llm_judge=not args.no_llm_judge,
            max_queries=args.max_queries,
        )
    else:
        evaluator.evaluate(
            benchmark_path=args.benchmark,
            retrieval_mode=args.retrieval_mode,
            use_reranker=not args.no_reranker,
            hallucination_ctrl=not args.no_hallucination_ctrl,
            use_llm_judge=not args.no_llm_judge,
            config_name=args.config_name,
            max_queries=args.max_queries,
        )


if __name__ == "__main__":
    main()

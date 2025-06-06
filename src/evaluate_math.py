#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_math.py
===============================
Multi-dataset evaluation of autoregressive LLMs via vLLM.

Key changes:
- Instantiate LLM client only once.
- Save combined metrics for all datasets.
- Save combined inference results for all datasets.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import datasets
import transformers
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from vllm import LLM, SamplingParams

try:
    from prompts import PROMPTS
except ImportError:
    PROMPTS: dict = {}

PASS_K = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FIXED_32_DATASETS: Set[str] = {"aime24", "aime25", "amc23"}
LOG = logging.getLogger("benchmark_eval")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument(
        "--prompt_type", choices=list(PROMPTS.keys()), default=None,
        help="template key in prompts.PROMPTS; omit for raw problem"
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument(
        "--num_samples", type=int, default=max(PASS_K),
        help="default number of completions per problem"
    )
    ap.add_argument(
        "--benchmark_path", type=str, required=True,
        help="path to on-disk combined benchmark with 'dataset','problem','answer'"
    )
    ap.add_argument("--output_path", required=True)
    return ap.parse_args()

# --------------------------------------------------------------------------- #
# Load combined benchmarks
# --------------------------------------------------------------------------- #
def load_composed_benchmarks(path: str) -> Dict[str, List[Dict]]:
    ds = datasets.load_from_disk(path)
    data: Dict[str, List[Dict]] = {}
    for alias in set(ds["dataset"]):
        subset = ds.filter(lambda x, alias=alias: x["dataset"] == alias)
        data[alias] = [
            {"problem": row["problem"], "answer": row["answer"]}
            for row in subset
        ]
    return data

# --------------------------------------------------------------------------- #
# Prompt conversion
# --------------------------------------------------------------------------- #
def build_chat_prompt(
    problem: str,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt_type: Optional[str]
) -> str:
    user_content = (
        PROMPTS[prompt_type].format(question=problem)
        if prompt_type else problem
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

# --------------------------------------------------------------------------- #
# Correctness checker
# --------------------------------------------------------------------------- #
def is_correct(pred: str, gold: str) -> Optional[float]:
    gold_parsed = parse(gold, extraction_mode="first_match")
    if not gold_parsed:
        return None
    ans_parsed = parse(
        pred,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False
            )
        ],
        extraction_mode="first_match"
    )
    try:
        return float(verify(gold_parsed, ans_parsed))
    except Exception:
        return None

# --------------------------------------------------------------------------- #
# Evaluation core (reuse llm and tokenizer)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Evaluation core (reuse llm and tokenizer), now using args.max_tokens
# --------------------------------------------------------------------------- #
def run_evaluation(
    llm: LLM,
    tokenizer: transformers.PreTrainedTokenizer,
    sampling: SamplingParams,
    prompt_type: Optional[str],
    problems: List[Dict]
) -> List[Dict]:
    # Use the --max_tokens setting as our length threshold
    max_allowed = sampling.max_tokens

    valid_prompts: List[str] = []
    valid_samples: List[Dict] = []

    # Build & filter prompts
    for sample in problems:
        prompt = build_chat_prompt(sample["problem"], tokenizer, prompt_type)
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(token_ids) > max_allowed:
            LOG.warning(
                "Skipping problem: prompt length %d > max_tokens %d. Sample: %.80s",
                len(token_ids), max_allowed, sample["problem"]
            )
            continue

        valid_prompts.append(prompt)
        valid_samples.append(sample)

    if not valid_prompts:
        LOG.warning("No prompts left after length filtering.")
        return []

    LOG.info(
        "Generating %d completions for %d valid problems…",
        sampling.n, len(valid_prompts)
    )
    raw_out = llm.generate(valid_prompts, sampling)

    # Collect results only for the valid set
    results: List[Dict] = []
    for sample, out in zip(valid_samples, raw_out):
        outputs = [o.text.strip() for o in out.outputs]
        results.append({
            "problem": sample["problem"],
            "answer": sample["answer"],
            "prompt_type": prompt_type,
            "outputs": outputs
        })

    return results

# --------------------------------------------------------------------------- #
# Metrics calculation
# --------------------------------------------------------------------------- #
def compute_metrics(
    results: List[Dict],
    num_samples: int
) -> Dict[str, float]:
    total = len(results)
    pass_counters: Dict[int, int] = {k: 0 for k in PASS_K if k <= num_samples}
    avg_sums: Dict[int, float] = {k: 0.0 for k in PASS_K if k <= num_samples}

    for r in results:
        flags = [is_correct(o, r["answer"]) or 0.0 for o in r.get("outputs", [])]
        for k in pass_counters:
            top_k = flags[:k]
            pass_counters[k] += int(any(top_k))
            avg_sums[k] += (sum(top_k) / k) if k > 0 else 0.0

    metrics: Dict[str, float] = {"total": float(total)}
    for k in sorted(pass_counters):
        metrics[f"pass@{k}"] = pass_counters[k] / total if total else 0.0
        metrics[f"avg@{k}"] = avg_sums[k] / total if total else 0.0
    return metrics

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )
    llm = LLM(
        args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    all_data = load_composed_benchmarks(args.benchmark_path)
    root = Path(args.output_path)
    root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    combined_inferences: List[Dict] = []

    for alias, problems in all_data.items():
        ns = 32 if alias in FIXED_32_DATASETS else args.num_samples
        LOG.info("Evaluating dataset '%s' (%d problems) with %d samples…", alias, len(problems), ns)

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            n=ns
        )
        results = run_evaluation(
            llm, tokenizer, sampling, args.prompt_type, problems
        )

        summary[alias] = compute_metrics(results, ns)
        # Accumulate inference results with dataset labels
        for r in results:
            combined_inferences.append({"dataset": alias, **r})

    # Save combined metrics only
    metrics_file = root / "metrics_summary.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save combined inference results
    inference_file = root / "inference_summary.jsonl"
    with inference_file.open("w", encoding="utf-8") as f:
        for record in combined_inferences:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOG.info("All done! Metrics summary at %s and inference summary at %s", metrics_file, inference_file)

if __name__ == "__main__":
    main()

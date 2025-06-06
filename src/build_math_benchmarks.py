import os
import re
import csv
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd
from textwrap import dedent
from pathlib import Path
from typing import List, Dict, Optional

import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def build_math_benchmarks(
    out_dir: str = "./data/math_benchmarks",
    num_proc: int = 8
) -> None:
    """
    Download, preprocess, and combine multiple math benchmark test sets, then save to disk.
    Each dataset is standardized to have columns ["dataset", "problem", "answer"].

    Special cases:
      - GSM8K: extract only the final answer after the '#### ' marker.
      - OlympiadBench (simplerl): join the `solution` list into one problem string, and take
        the first element of `final_answer` as the answer.
    """
    specs: List[Dict[str, Optional[str]]] = [
        {"repo": "HuggingFaceH4/MATH-500",        "split": "test", "subset": None,   "alias": "math500"},
        {"repo": "math-ai/amc23",                 "split": "test", "subset": None,   "alias": "amc23"},
        {"repo": "zwhe99/simplerl-OlympiadBench", "split": "test", "subset": None,   "alias": "olympiadbench"},
        {"repo": "zwhe99/simplerl-minerva-math",  "split": "test", "subset": None,   "alias": "minervabench"},
        {"repo": "HuggingFaceH4/aime_2024",       "split": "train","subset": None,   "alias": "aime24"},
        {"repo": "math-ai/aime25",                "split": "test", "subset": None,   "alias": "aime25"},
        {"repo": "openai/gsm8k",                  "split": "test", "subset": "main", "alias": "gsm8k"},
    ]

    datasets_list = []
    for spec in specs:
        # 1) Load dataset (with or without a subset)
        if spec["subset"]:
            ds = datasets.load_dataset(
                spec["repo"], spec["subset"],
                split=spec["split"], trust_remote_code=True
            )
        else:
            ds = datasets.load_dataset(
                spec["repo"],
                split=spec["split"], trust_remote_code=True
            )

        # 2) Normalize each example to {"dataset","problem","answer"}
        def _normalize(example, alias=spec["alias"]):
            if alias == "gsm8k":
                # GSM8K: question stays as-is; answer is text after '#### '
                problem = example["question"]
                raw = example["answer"]
                m = re.search(r"####\s*(.+)", raw)
                answer = m.group(1).strip() if m else raw.strip()

            elif alias == "olympiadbench":
                # Simplerl-OlympiadBench: join solution list → problem;
                # take first element of final_answer list → answer.
                problem = " ".join(example["solution"])
                fas = example.get("final_answer", [])
                answer = fas[0].strip() if isinstance(fas, list) and fas else ""

            else:
                # Default case: use 'problem' or fallback to 'question'; answer from 'answer'
                problem = example.get("problem", example.get("question", ""))
                answer = example.get("answer", "").strip()

            return {"dataset": alias, "problem": problem, "answer": answer}

        # 3) Apply mapping and drop any other columns
        keep = {"dataset", "problem", "answer"}
        ds = ds.map(
            _normalize,
            num_proc=num_proc,
            remove_columns=[c for c in ds.column_names if c not in keep]
        )

        datasets_list.append(ds)

    # 4) Concatenate all benchmarks
    combined = datasets.concatenate_datasets(datasets_list)

    # 5) Save to disk
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(out_path))
    print(f"Saved combined math benchmarks to {out_path.resolve()}")

if __name__ == "__main__":
    
    build_math_benchmarks(
        out_dir="./data/math_benchmarks",
        num_proc=8
    )
    
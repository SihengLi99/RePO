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
        
def process_deepmath(
    sample_size: Optional[int] = None,
    max_length: Optional[int] = 4096,
    num_proc: int = 8
) -> DatasetDict:
    """
    Load, preprocess, optionally filter by length, sample, and save DeepMath.

    Args:
        sample_size (int, optional): Number of examples to keep after filtering. 
                                     If None, keep all.
        max_length (int, optional): Maximum allowed token count per example.
                                    If None, skip length filtering.
        num_proc (int): Number of processes for mapping and filtering.

    Returns:
        DatasetDict: The final processed dataset.
    """
    # 1. Load the raw DeepMath dataset
    dataset = load_dataset("zwhe99/DeepMath-103K")

    # 2. Define preprocessing function
    def _process(item):
        item["prompt"] = [{"role": "user", "content": item["question"]}]
        item["messages"] = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["r1_solution_1"]},
        ]
        item["solution"] = item["r1_solution_1"]
        item["ground_truth"] = item["final_answer"]
        return item

    # 3. Apply mapping and drop unused columns
    keep_cols = ["prompt", "messages", "solution", "ground_truth"]
    drop_cols = [c for c in dataset["train"].column_names if c not in keep_cols]
    dataset = dataset.map(_process, num_proc=num_proc, remove_columns=drop_cols)

    # 4. Initialize tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")

    # 5. Filter by token length if max_length is specified
    if max_length is not None:
        def _filter_by_length(item):
            # Tokenize using the chat template without adding generation prompt
            tokens = tokenizer.apply_chat_template(
                item["messages"], add_generation_prompt=False, tokenize=True
            )
            return len(tokens) < max_length

        dataset = dataset.filter(_filter_by_length, num_proc=num_proc)

    # 6. Optionally sample the training split
    if sample_size:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(sample_size))

    # 7. Print final split sizes
    print("Final dataset splits and sizes:")
    for split, ds in dataset.items():
        print(f"- {split}: {len(ds)} examples")

    # 8. Save to disk with descriptive folder name
    size_desc = f"{sample_size}" if sample_size else "all"
    len_desc = f"-maxlen{max_length}" if max_length else ""
    folder_name = f"DeepMath-{size_desc}samples{len_desc}"
    save_path = os.path.join("data", folder_name)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"Saved processed dataset to: {save_path}")

    return dataset

if __name__ == "__main__":
    
    process_deepmath(sample_size=1024, max_length=None, num_proc=8)

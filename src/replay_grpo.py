# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from functools import partial
from typing import Optional, List
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    GRPOTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from rewards import REWARD_FUNCS_REGISTRY
from utils import get_tokenizer, PROCESS_FUNC_MAP

from replay_grpo_trainer import ReplayGRPOTrainer


logger = logging.getLogger(__name__)

@dataclass
class ReplayGRPOConfig(trl.GRPOConfig):
    """
    Extension of `trl.GRPOConfig` that adds the parameters required by the
    prompt-level reply cache implemented in `ReplayGRPOTrainer`.
    """

    # ────────────────────────────── templating ───────────────────────────── #
    chat_template:   Optional[str] = field(
        default=None,
        metadata={"help": "Chat template name (overrides the model default)."},
    )
    prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt template name (overrides the model default)."},
    )

    # ─────────────────────────── reward functions ────────────────────────── #
    reward_funcs: str = field(
        default="format_correctness",
        metadata={"help": "Underscore-separated list of reward function names."},
    )

    # convenience accessor
    @property
    def reward_funcs_list(self) -> list[str]:
        return [s.strip() for s in self.reward_funcs.split("_") if s.strip()]

    # ──────────────────────────── replay settings ─────────────────────────── #
    replay_strategy: str = field(
        default="random",
        metadata={
            "help": (
                "Replay strategy to use for the replay cache. Options: `topk` (default), "
                "`replay`, `none`."
            )
        },
    )
    
    num_replays: int = field(
        default=0,
        metadata={
            "help": (
                "Probability that a reply slot for a given prompt is served from the "
                "in-memory cache instead of being freshly generated."
            )
        },
    )
    
    min_cache_to_replay: int = field(
        default=16,
        metadata={
            "help": (
                "Minimum number of cached completions required before enabling the replay strategy. "
                "Replay will be skipped until this threshold is reached."
            )
        },
    )

    max_cache_size: int = field(
        default=8,
        metadata={
            "help": (
                "Maximum number of completions stored per prompt.  Implemented with "
                "a `collections.deque(maxlen=max_cache_size)`, so the oldest entry "
                "is automatically evicted on overflow."
            )
        },
    )
    
    adv_norm_mode: str = field(
        default="split",
        metadata={
            "help": (
                "Norm mode for the adversarial loss. Options: `split` (default), `mixed`."
            )
        },
    )
    
    off_policy_loss_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "Coefficient for the off-policy loss term. 0.0 disables the off-policy loss."
            )
        },
    )

    # ───────────────────────────── dataset / I/O ─────────────────────────── #
    dataset_num_proc: int = field(
        default=8,
        metadata={"help": "Number of workers for dataset preprocessing."},
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Use only the first N examples for training; `None` = use all."
        },
    )

    # ─────────────────────────────── saving  ─────────────────────────────── #
    fsdp_save_full_state_dict: bool = field(
        default=False,
        metadata={"help": "Save full state dict when using FSDP."},
    )
    
def load_grpo_dataset(dataset_name, dataset_config, num_examples, prompt_template, tokenizer, num_proc):

    dataset = datasets.load_from_disk(dataset_name)        
    dataset = dataset.remove_columns(["messages"])
    
    return dataset

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args.chat_template)
    
    ################
    # Load datasets
    ################
    dataset = load_grpo_dataset(
        dataset_name=script_args.dataset_name,
        dataset_config=script_args.dataset_config,
        num_examples=training_args.num_examples,
        prompt_template=training_args.prompt_template,
        tokenizer=tokenizer,
        num_proc=training_args.dataset_num_proc,
    )

    # Get reward functions
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in training_args.reward_funcs_list]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = ReplayGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled and training_args.fsdp_save_full_state_dict:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["grpo"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ReplayGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

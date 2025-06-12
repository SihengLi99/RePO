#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH --job-name=EvalMath
#SBATCH --output=log/eval_math_%j.out

# ----------------------------- cluster setup ---------------------------------
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=27700

echo "Nodes allocated:"
for n in "${NODELIST[@]}"; do
  echo "  - $n"
done

export NCCL_ASYNC_ERROR_HANDLING=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export ACCELERATE_LOG_LEVEL=info

# ----------------------------- paths & model ---------------------------------
PROJECT_HOME=/mnt/petrelfs/lisiheng/RePO
OUTPUT_HOME=/mnt/lustrenew/mllm_safety-shared/lisiheng

MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Siheng99

# MODEL_NAME=Qwen2.5-Math-1.5B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-GRPO

# MODEL_NAME=Qwen2.5-Math-1.5B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-recent_k-8-16-16-split-1.0-ReplayGRPO

# MODEL_NAME=Qwen2.5-Math-7B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-GRPO

# MODEL_NAME=Qwen2.5-Math-7B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-recent_k-8-16-16-split-1.0-ReplayGRPO

# MODEL_NAME=Qwen3-1.7B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-GRPO

# MODEL_NAME=Qwen3-1.7B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-highest_k-8-16-16-split-1.0-ReplayGRPO

MODEL_NAME=Qwen3-1.7B-DeepMath-1024samples-RePO

MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

SCRIPT=$PROJECT_HOME/src/evaluate_math.py
BENCHMARK_PATH=$PROJECT_HOME/data/math_benchmarks

PROMPT_TYPE=None
NUM_SAMPLES=1
TEMPERATURE=0.2
TOP_P=0.95
MAX_TOKENS=2048

TENSOR_PARALLEL_SIZE=$GPUS_PER_NODE

OUTPUT_DIR=$PROJECT_HOME/results/${MODEL_NAME}/math-${PROMPT_TYPE}-${NUM_SAMPLES}-${TEMPERATURE}-${TOP_P}-${MAX_TOKENS}

# ----------------------------- run evaluation --------------------------------
python $SCRIPT \
   --model_name_or_path $MODEL_NAME_OR_PATH \
   --num_samples $NUM_SAMPLES \
   --temperature $TEMPERATURE \
   --top_p $TOP_P \
   --max_tokens $MAX_TOKENS \
   --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
   --benchmark_path $BENCHMARK_PATH \
   --output_path $OUTPUT_DIR 
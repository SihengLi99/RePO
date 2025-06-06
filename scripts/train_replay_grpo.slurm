#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH --job-name=ReplayGRPO
#SBATCH --output=log/replay_grpo_%j.out
#SBATCH --exclude=SH-IDCA1404-10-140-54-15,SH-IDCA1404-10-140-54-47,SH-IDCA1404-10-140-54-76

#--------------------------------------------------
# Node / network setup
#--------------------------------------------------
HOSTNAME=$(hostname)
MASTER_ADDR=$HOSTNAME       # single-node launch
MASTER_PORT=27500

echo "Running on $HOSTNAME"
echo "MASTER_ADDR = $MASTER_ADDR, MASTER_PORT = $MASTER_PORT"

#--------------------------------------------------
# Cleanup: free pg_port 51216 and kill zombie vLLM servers
#--------------------------------------------------
PG_PORT=51216        # vLLM’s default process-group port
# If the port is busy, force-kill whatever owns it
if lsof -i :$PG_PORT &>/dev/null; then
  echo "[Cleanup] Port $PG_PORT is busy – terminating owner(s)."
  fuser -k ${PG_PORT}/tcp
fi
# Terminate any previous vllm-serve instance that used the same model path
pkill -f "trl vllm-serve.*$MODEL_NAME_OR_PATH" 2>/dev/null || true

#--------------------------------------------------
# Environment
#--------------------------------------------------
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export ACCELERATE_LOG_LEVEL=info
export WANDB_API_KEY=85a3c5af1814c40a13d5d9e64783857cf260b506
export WANDB_PROJECT=ReplayGRPO_Math
export no_proxy="$HOSTNAME"
export NO_PROXY="$HOSTNAME"

#--------------------------------------------------
# Paths & model
#--------------------------------------------------
PROJECT_HOME=/mnt/petrelfs/lisiheng/RePO
OUTPUT_HOME=/mnt/lustrenew/mllm_safety-shared/lisiheng

MODEL_DIR=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen
MODEL_NAME=Qwen3-1.7B
MAX_COMPLETION_LENGTH=1024

MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

DATASET_DIR=data
DATASET_NAME=DeepMath-1024samples
DATASET_NAME_OR_PATH=$DATASET_DIR/$DATASET_NAME

#--------------------------------------------------
# Training hyperparams
#--------------------------------------------------
NUM_TRAIN_EPOCHS=3
REWARD_FUNCS=accuracy
MAX_PROMPT_LENGTH=512
TEMPERATURE=1.0
TOP_K=50
LOSS_TYPE=grpo
BETA=0.0
EPSILON=0.20
EPSILON_HIGH=0.20
NUM_ITERATIONS=1
SCALE_REWARDS=true
ADV_NORM_MODE=split
OFF_POLICY_LOSS_COEF=1.0

# none all random_k highest_k max_variance_k recent_k
REPLAY_STRATEGY=highest_k
NUM_REPLAYS=8
NUM_GENERATIONS=8
MIN_CACHE_TO_REPLAY=16
MAX_CACHE_SIZE=16
PER_DEVICE_TRAIN_BATCH_SIZE=8

#--------------------------------------------------
# GPU partitioning
#--------------------------------------------------
#   4,5,6,7 → accelerate training
#   0,1,2,3 → vLLM server
ACCEL_GPUS="4,5,6,7"
VLLM_GPUS="0,1,2,3"
VLLM_PORT=27800

#--------------------------------------------------
# Start vLLM server on GPUs 0–3
#--------------------------------------------------
export CUDA_VISIBLE_DEVICES=$VLLM_GPUS
trl vllm-serve \
  --model $MODEL_NAME_OR_PATH \
  --tensor_parallel_size 4 \
  --port $VLLM_PORT &

# give vLLM a moment to spin up
sleep 5

#--------------------------------------------------
# Launch Accelerate on GPUs 4–7
#--------------------------------------------------
export CUDA_VISIBLE_DEVICES=$ACCEL_GPUS

WORLD_SIZE=4
LEARNING_RATE=1e-6
TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE))

accelerate launch \
  --config_file configs/accelerate_ds.yaml \
  --num_machines 1 \
  --num_processes ${WORLD_SIZE} \
  src/replay_grpo.py \
    --config configs/grpo.yaml \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --dataset_name=$DATASET_NAME_OR_PATH \
    --output_dir=$OUTPUT_HOME/checkpoints/${MODEL_NAME}-${DATASET_NAME}-${NUM_TRAIN_EPOCHS}-${REWARD_FUNCS}-${MAX_PROMPT_LENGTH}-${MAX_COMPLETION_LENGTH}-${NUM_GENERATIONS}-${TEMPERATURE}-${TOP_K}-${LOSS_TYPE}-${BETA}-${EPSILON}-${EPSILON_HIGH}-${NUM_ITERATIONS}-${SCALE_REWARDS}-${LEARNING_RATE}-${TOTAL_BATCH_SIZE}-${REPLAY_STRATEGY}-${NUM_REPLAYS}-${MIN_CACHE_TO_REPLAY}-${MAX_CACHE_SIZE}-${ADV_NORM_MODE}-${OFF_POLICY_LOSS_COEF}-ReplayGRPO \
    --fsdp_save_full_state_dict=true \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --reward_funcs=$REWARD_FUNCS \
    --max_prompt_length=$MAX_PROMPT_LENGTH \
    --max_completion_length=$MAX_COMPLETION_LENGTH \
    --num_generations=$NUM_GENERATIONS \
    --temperature=$TEMPERATURE \
    --top_k=$TOP_K \
    --loss_type=$LOSS_TYPE \
    --beta=$BETA \
    --epsilon=$EPSILON \
    --epsilon_high=$EPSILON_HIGH \
    --num_iterations=$NUM_ITERATIONS \
    --scale_rewards=$SCALE_REWARDS \
    --use_vllm=true \
    --vllm_server_host=$MASTER_ADDR \
    --vllm_server_port=$VLLM_PORT \
    --vllm_server_timeout=360.0 \
    --replay_strategy=$REPLAY_STRATEGY \
    --min_cache_to_replay=$MIN_CACHE_TO_REPLAY \
    --num_replays=$NUM_REPLAYS \
    --max_cache_size=$MAX_CACHE_SIZE \
    --adv_norm_mode=$ADV_NORM_MODE \
    --off_policy_loss_coef=$OFF_POLICY_LOSS_COEF \
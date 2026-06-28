#!/usr/bin/env bash
# Stage 4 电商垂域 GRPO 启动脚本。
#
# 默认从 Stage4 100k balanced SFT best 初始化：
#   - policy: 继续训练 LoRA
#   - reference: 冻结的同一份 SFT checkpoint，用于 KL 约束
#   - SigLIP2 / Qwen backbone / projector 默认冻结
#
# 常用快速实验：
#   MAX_SAMPLES=256 MAX_STEPS=20 bash scripts/train_stage4_grpo.sh
#
# 稍正式一点的实验：
#   MAX_SAMPLES=100000 MAX_STEPS=3000 OUTPUT_DIR=/root/autodl-tmp/checkpoints/stage4_abo_grpo_short_reward_v2 \
#     bash scripts/train_stage4_grpo.sh

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_models/Qwen3-1.7B}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384}"

# 默认使用 SFT balanced 数据作为 GRPO prompt 源。
# 原来的 grpo/train.json 只有 brand/type/color/style 四类，会漏掉 title/summary；
# Dataset 会把两轮 SFT 样本动态转成一轮 prompt + reward.references。
ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train_100k_balanced.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/val.json}"

INIT_PROJECTOR_PATH="${INIT_PROJECTOR_PATH:-/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/projector.pt}"
INIT_LORA_PATH="${INIT_LORA_PATH:-/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/lora_adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage4_abo_grpo_short_reward_v2}"

MAX_SAMPLES="${MAX_SAMPLES:-100000}"
MAX_STEPS="${MAX_STEPS:-3000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CLIP_RANGE="${CLIP_RANGE:-0.2}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-5.0}"
KL_BETA="${KL_BETA:-0.05}"
TOKEN_F1_REWARD_WEIGHT="${TOKEN_F1_REWARD_WEIGHT:-0.1}"
GENERATIVE_F1_REWARD_WEIGHT="${GENERATIVE_F1_REWARD_WEIGHT:-1.0}"
GENERATIVE_EXACT_BONUS="${GENERATIVE_EXACT_BONUS:-0.1}"
STYLE_F1_REWARD_WEIGHT="${STYLE_F1_REWARD_WEIGHT:-0.25}"
LENGTH_PENALTY_WEIGHT="${LENGTH_PENALTY_WEIGHT:-0.02}"
MAX_REWARD_TOKENS="${MAX_REWARD_TOKENS:-8}"
GENERATIVE_MAX_REWARD_TOKENS="${GENERATIVE_MAX_REWARD_TOKENS:-48}"
TASK_SAMPLING_WEIGHTS="${TASK_SAMPLING_WEIGHTS:-{\"product_style_qa\":2.5,\"product_title_generation\":2.0,\"product_attribute_summary\":1.5}}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.002}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-750}"
TRAIN_PROJECTOR="${TRAIN_PROJECTOR:-0}"

NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_SAMPLES="${EVAL_SAMPLES:-300}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

echo "========== Stage 4 电商垂域 GRPO =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
echo "VAL_ANNOTATION_PATH=${VAL_ANNOTATION_PATH}"
echo "INIT_PROJECTOR_PATH=${INIT_PROJECTOR_PATH}"
echo "INIT_LORA_PATH=${INIT_LORA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "LR=${LR}"
echo "KL_BETA=${KL_BETA}"
echo "TASK_SAMPLING_WEIGHTS=${TASK_SAMPLING_WEIGHTS}"
echo "EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE}"
echo "TRAIN_PROJECTOR=${TRAIN_PROJECTOR}"
echo "==========================================="

CMD=(
  python -m vlm.training.train_stage4_grpo
  --qwen-path "${QWEN_PATH}"
  --siglip-path "${SIGLIP_PATH}"
  --annotation-path "${ANNOTATION_PATH}"
  --init-projector-path "${INIT_PROJECTOR_PATH}"
  --init-lora-path "${INIT_LORA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-samples "${MAX_SAMPLES}"
  --max-steps "${MAX_STEPS}"
  --num-epochs "${NUM_EPOCHS}"
  --prompt-batch-size "${PROMPT_BATCH_SIZE}"
  --num-generations "${NUM_GENERATIONS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --learning-rate "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --clip-range "${CLIP_RANGE}"
  --advantage-clip "${ADVANTAGE_CLIP}"
  --kl-beta "${KL_BETA}"
  --token-f1-reward-weight "${TOKEN_F1_REWARD_WEIGHT}"
  --generative-f1-reward-weight "${GENERATIVE_F1_REWARD_WEIGHT}"
  --generative-exact-bonus "${GENERATIVE_EXACT_BONUS}"
  --style-f1-reward-weight "${STYLE_F1_REWARD_WEIGHT}"
  --length-penalty-weight "${LENGTH_PENALTY_WEIGHT}"
  --max-reward-tokens "${MAX_REWARD_TOKENS}"
  --generative-max-reward-tokens "${GENERATIVE_MAX_REWARD_TOKENS}"
  --task-sampling-weights "${TASK_SAMPLING_WEIGHTS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}"
  --early-stop-min-steps "${EARLY_STOP_MIN_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-samples "${EVAL_SAMPLES}"
  --torch-dtype "${TORCH_DTYPE}"
  --device "${DEVICE}"
)

if [[ -n "${VAL_ANNOTATION_PATH}" ]]; then
  CMD+=(--val-annotation-path "${VAL_ANNOTATION_PATH}")
fi

if [[ "${TRAIN_PROJECTOR}" == "1" ]]; then
  CMD+=(--train-projector)
fi

"${CMD[@]}"

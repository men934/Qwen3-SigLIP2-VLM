#!/usr/bin/env bash
# Stage 3 文档/OCR/图表垂域微调启动脚本。
#
# 默认从固定分辨率 Stage 2 LoRA 150k r32 checkpoint 继续训练：
#   - projector 从 Stage 2 加载并继续训练
#   - Qwen3 主干冻结
#   - Stage 2 LoRA adapter 加载后继续训练
#   - SigLIP2 vision encoder 冻结

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 模型路径
QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_models/Qwen3-1.7B}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384}"

# Stage 3 统一数据路径
ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/train.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/val.json}"

# 固定分辨率 Stage 2 起始权重
STAGE2_PROJECTOR_PATH="${STAGE2_PROJECTOR_PATH:-/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/projector.pt}"
STAGE2_LORA_PATH="${STAGE2_LORA_PATH:-/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/lora_adapter}"

# 输出路径
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix}"

# 训练规模。MAX_STEPS=none 表示完整跑完 NUM_EPOCHS。
MAX_SAMPLES="${MAX_SAMPLES:-none}"
MAX_STEPS="${MAX_STEPS:-none}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 显存相关参数。
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

# Stage 3 主线沿用固定分辨率，方便直接继承 Stage 2 固定 baseline。
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MIN_PIXELS="${MIN_PIXELS:-147456}"   # 384 * 384
MAX_PIXELS="${MAX_PIXELS:-451584}"   # 672 * 672
USE_SIGLIP_ABS_POS="${USE_SIGLIP_ABS_POS:-1}"
USE_SIGLIP_QK_2D_ROPE="${USE_SIGLIP_QK_2D_ROPE:-0}"
SIGLIP_ROPE_BASE="${SIGLIP_ROPE_BASE:-10000.0}"
SIGLIP_ROPE_DIM="${SIGLIP_ROPE_DIM:-none}"

# Stage 3 是继续微调已有 LoRA，学习率比 Stage 2 略小。
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 运行参数。
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
VERIFY_IMAGES="${VERIFY_IMAGES:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "========== Stage 3 文档/OCR/图表垂域微调 =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "SIGLIP_PATH=${SIGLIP_PATH}"
echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
echo "VAL_ANNOTATION_PATH=${VAL_ANNOTATION_PATH}"
echo "STAGE2_PROJECTOR_PATH=${STAGE2_PROJECTOR_PATH}"
echo "STAGE2_LORA_PATH=${STAGE2_LORA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "DYNAMIC_RESOLUTION=${DYNAMIC_RESOLUTION}"
echo "LR=${LR}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_BATCHES=${EVAL_BATCHES}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "DEVICE=${DEVICE}"
echo "=================================================="

CMD=(
  python -m vlm.training.train_stage3
  --qwen-path "${QWEN_PATH}"
  --siglip-path "${SIGLIP_PATH}"
  --annotation-path "${ANNOTATION_PATH}"
  --stage2-projector-path "${STAGE2_PROJECTOR_PATH}"
  --stage2-lora-path "${STAGE2_LORA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-samples "${MAX_SAMPLES}"
  --max-steps "${MAX_STEPS}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --image-size "${IMAGE_SIZE}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --siglip-rope-base "${SIGLIP_ROPE_BASE}"
  --siglip-rope-dim "${SIGLIP_ROPE_DIM}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --torch-dtype "${TORCH_DTYPE}"
  --device "${DEVICE}"
)

if [[ -n "${VAL_ANNOTATION_PATH}" ]]; then
  CMD+=(--val-annotation-path "${VAL_ANNOTATION_PATH}")
fi

if [[ "${DYNAMIC_RESOLUTION}" == "1" ]]; then
  CMD+=(--dynamic-resolution)
fi

if [[ "${USE_SIGLIP_ABS_POS}" == "0" ]]; then
  CMD+=(--no-siglip-abs-pos-embedding)
fi

if [[ "${USE_SIGLIP_QK_2D_ROPE}" == "1" ]]; then
  CMD+=(--use-siglip-qk-2d-rope)
fi

if [[ "${VERIFY_IMAGES}" == "1" ]]; then
  CMD+=(--verify-images)
fi

"${CMD[@]}"

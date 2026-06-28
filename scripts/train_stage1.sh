#!/usr/bin/env bash
# Stage 1 视觉-语言对齐训练启动脚本。
#
# 这个脚本用于训练：
#   SigLIP2 -> PatchMerger -> MLPProjector -> Qwen3
#
# Stage 1 的冻结策略：
#   - SigLIP2 vision encoder 冻结
#   - Qwen3 language model 冻结
#   - 只训练 MLPProjector
#
# 默认值偏向“小规模 sanity training”，避免第一次运行时直接扫完整 558K 数据。
# 你可以通过环境变量覆盖这些默认值，例如：
#
#   MAX_SAMPLES=1024 MAX_STEPS=100 BATCH_SIZE=2 GRAD_ACCUM=8 \
#   bash scripts/train_stage1.sh
#
# 如果要跑完整数据，可以使用：
#
#   MAX_SAMPLES=none MAX_STEPS=none bash scripts/train_stage1.sh

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 模型路径
QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_models/Qwen3-1.7B}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384}"

# 数据路径
ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/hf_datasets/LLaVA-Pretrain}"

# 输出路径
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage1_align}"

# 训练规模。默认先跑一个小规模版本，确认 loss/backward/checkpoint 正常。
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 显存相关参数。
# 当前模型在 RTX PRO 6000 上 batch_size=2 通常没问题；如果 OOM，先降到 1。
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

# 动态分辨率与 SigLIP2 内部 2D RoPE。
# 新路线推荐：
#   DYNAMIC_RESOLUTION=1 USE_SIGLIP_ABS_POS=0 USE_SIGLIP_QK_2D_ROPE=1
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MIN_PIXELS="${MIN_PIXELS:-147456}"   # 384 * 384
MAX_PIXELS="${MAX_PIXELS:-451584}"   # 672 * 672
USE_SIGLIP_ABS_POS="${USE_SIGLIP_ABS_POS:-1}"
USE_SIGLIP_QK_2D_ROPE="${USE_SIGLIP_QK_2D_ROPE:-0}"
SIGLIP_ROPE_BASE="${SIGLIP_ROPE_BASE:-10000.0}"
SIGLIP_ROPE_DIM="${SIGLIP_ROPE_DIM:-none}"

# 优化器参数。Stage 1 只训 projector，学习率可以比 SFT 阶段大一些。
LR="${LR:-1e-3}"
VISION_LR="${VISION_LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 两阶段 Stage1 策略：
#   前 UNFREEZE_VISION_AFTER_STEPS 个 optimizer step 只训练 projector；
#   之后解冻 SigLIP2 最后 UNFREEZE_VISION_LAST_LAYERS 层，用 VISION_LR 小学习率训练。
UNFREEZE_VISION_AFTER_STEPS="${UNFREEZE_VISION_AFTER_STEPS:-none}"
UNFREEZE_VISION_LAST_LAYERS="${UNFREEZE_VISION_LAST_LAYERS:-0}"

# 运行参数。
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

echo "========== Stage 1 视觉-语言对齐训练 =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "SIGLIP_PATH=${SIGLIP_PATH}"
echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
echo "VAL_ANNOTATION_PATH=${VAL_ANNOTATION_PATH}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "DYNAMIC_RESOLUTION=${DYNAMIC_RESOLUTION}"
echo "MIN_PIXELS=${MIN_PIXELS}"
echo "MAX_PIXELS=${MAX_PIXELS}"
echo "USE_SIGLIP_ABS_POS=${USE_SIGLIP_ABS_POS}"
echo "USE_SIGLIP_QK_2D_ROPE=${USE_SIGLIP_QK_2D_ROPE}"
echo "LR=${LR}"
echo "VISION_LR=${VISION_LR}"
echo "UNFREEZE_VISION_AFTER_STEPS=${UNFREEZE_VISION_AFTER_STEPS}"
echo "UNFREEZE_VISION_LAST_LAYERS=${UNFREEZE_VISION_LAST_LAYERS}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_BATCHES=${EVAL_BATCHES}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "DEVICE=${DEVICE}"
echo "================================================"

CMD=(
  python -m vlm.training.train_stage1
  --qwen-path "${QWEN_PATH}"
  --siglip-path "${SIGLIP_PATH}"
  --annotation-path "${ANNOTATION_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --max-samples "${MAX_SAMPLES}"
  --max-steps "${MAX_STEPS}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --image-size "${IMAGE_SIZE}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LR}"
  --vision-learning-rate "${VISION_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --siglip-rope-base "${SIGLIP_ROPE_BASE}"
  --siglip-rope-dim "${SIGLIP_ROPE_DIM}"
  --unfreeze-vision-after-steps "${UNFREEZE_VISION_AFTER_STEPS}"
  --unfreeze-vision-last-layers "${UNFREEZE_VISION_LAST_LAYERS}"
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

"${CMD[@]}"

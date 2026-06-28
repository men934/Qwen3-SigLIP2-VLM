#!/usr/bin/env bash
# Stage 4 电商垂域 SFT 启动脚本。
#
# 这个脚本默认做一个 5k 样本的 ABO 电商小规模实验：
#   - 从 Stage 3 最优 checkpoint 继续训练
#   - 训练 projector + Qwen LoRA
#   - 冻结 SigLIP2 和 Qwen3 主干
#   - 固定 384 分辨率，保持和 Stage 3 checkpoint 一致
#
# 你可以通过环境变量覆盖任何路径或训练超参，例如：
#
#   MAX_SAMPLES=20000 OUTPUT_DIR=/root/autodl-tmp/checkpoints/stage4_abo_sft_20k \
#     bash scripts/train_stage4_sft.sh

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 模型路径。
QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_models/Qwen3-1.7B}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384}"

# Stage 4 ABO SFT 数据路径。
ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/val.json}"

# 从 Stage 3 checkpoint 初始化。
INIT_PROJECTOR_PATH="${INIT_PROJECTOR_PATH:-/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix/step_006000/projector.pt}"
INIT_LORA_PATH="${INIT_LORA_PATH:-/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix/step_006000/lora_adapter}"

# 输出目录。
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage4_abo_sft_5k}"

# 训练规模。5k 样本 / grad_accum=8 大约是 625 个 optimizer step。
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
MAX_STEPS="${MAX_STEPS:-none}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 显存相关参数。
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

# Stage 4 主线沿用固定分辨率和 SigLIP2 absolute position embedding。
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MIN_PIXELS="${MIN_PIXELS:-147456}"   # 384 * 384
MAX_PIXELS="${MAX_PIXELS:-451584}"   # 672 * 672
USE_SIGLIP_ABS_POS="${USE_SIGLIP_ABS_POS:-1}"
USE_SIGLIP_QK_2D_ROPE="${USE_SIGLIP_QK_2D_ROPE:-0}"
SIGLIP_ROPE_BASE="${SIGLIP_ROPE_BASE:-10000.0}"
SIGLIP_ROPE_DIM="${SIGLIP_ROPE_DIM:-none}"

# Stage 4 继续训练已有 LoRA，学习率比 Stage 3 再保守一些。
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 日志、保存、验证。
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-250}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
VERIFY_IMAGES="${VERIFY_IMAGES:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "========== Stage 4 电商垂域 SFT =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "SIGLIP_PATH=${SIGLIP_PATH}"
echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
echo "VAL_ANNOTATION_PATH=${VAL_ANNOTATION_PATH}"
echo "INIT_PROJECTOR_PATH=${INIT_PROJECTOR_PATH}"
echo "INIT_LORA_PATH=${INIT_LORA_PATH}"
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
echo "=========================================="

CMD=(
  python -m vlm.training.train_stage4_sft
  --qwen-path "${QWEN_PATH}"
  --siglip-path "${SIGLIP_PATH}"
  --annotation-path "${ANNOTATION_PATH}"
  --init-projector-path "${INIT_PROJECTOR_PATH}"
  --init-lora-path "${INIT_LORA_PATH}"
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

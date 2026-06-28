#!/usr/bin/env bash
# Stage 2 多模态指令微调启动脚本。
#
# 这个脚本用于训练：
#   SigLIP2 -> PatchMerger -> Projector -> Qwen3
#
# 默认策略：
#   - SigLIP2 vision encoder 冻结
#   - Qwen3 language model 冻结
#   - 加载 Stage 1 projector 继续训练
#
# 如果安装了 peft，可以设置 USE_LORA=1，训练 projector + Qwen LoRA。

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 模型路径
QWEN_PATH="${QWEN_PATH:-/root/autodl-tmp/hf_models/Qwen3-1.7B}"
SIGLIP_PATH="${SIGLIP_PATH:-/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384}"

# 数据路径
ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/llava_instruct_150k.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/hf_datasets/coco/train2014}"

# Stage 1 初始化权重
STAGE1_PROJECTOR_PATH="${STAGE1_PROJECTOR_PATH:-/root/autodl-tmp/checkpoints/stage1_align_50k/step_003000/projector.pt}"
STAGE1_VISION_PATH="${STAGE1_VISION_PATH:-}"

# 输出路径
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage2_llava_instruct}"

# 训练规模。默认是小规模 sanity training。
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 显存相关参数。
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-768}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

# 动态分辨率与 SigLIP2 内部 2D RoPE。需要和 Stage1 checkpoint 的结构保持一致。
DYNAMIC_RESOLUTION="${DYNAMIC_RESOLUTION:-0}"
MIN_PIXELS="${MIN_PIXELS:-147456}"   # 384 * 384
MAX_PIXELS="${MAX_PIXELS:-451584}"   # 672 * 672
USE_SIGLIP_ABS_POS="${USE_SIGLIP_ABS_POS:-1}"
USE_SIGLIP_QK_2D_ROPE="${USE_SIGLIP_QK_2D_ROPE:-0}"
SIGLIP_ROPE_BASE="${SIGLIP_ROPE_BASE:-10000.0}"
SIGLIP_ROPE_DIM="${SIGLIP_ROPE_DIM:-none}"

# 优化器参数。Stage 2 默认 projector-only，学习率比 Stage 1 小一些。
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# LoRA 参数。当前环境未安装 peft 时不要打开 USE_LORA。
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

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

echo "========== Stage 2 多模态指令微调 =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "SIGLIP_PATH=${SIGLIP_PATH}"
echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
echo "VAL_ANNOTATION_PATH=${VAL_ANNOTATION_PATH}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "STAGE1_PROJECTOR_PATH=${STAGE1_PROJECTOR_PATH}"
echo "STAGE1_VISION_PATH=${STAGE1_VISION_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "DYNAMIC_RESOLUTION=${DYNAMIC_RESOLUTION}"
echo "MIN_PIXELS=${MIN_PIXELS}"
echo "MAX_PIXELS=${MAX_PIXELS}"
echo "USE_SIGLIP_ABS_POS=${USE_SIGLIP_ABS_POS}"
echo "USE_SIGLIP_QK_2D_ROPE=${USE_SIGLIP_QK_2D_ROPE}"
echo "LR=${LR}"
echo "USE_LORA=${USE_LORA}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_BATCHES=${EVAL_BATCHES}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "DEVICE=${DEVICE}"
echo "============================================"

CMD=(
  python -m vlm.training.train_stage2
  --qwen-path "${QWEN_PATH}"
  --siglip-path "${SIGLIP_PATH}"
  --annotation-path "${ANNOTATION_PATH}"
  --image-root "${IMAGE_ROOT}"
  --stage1-projector-path "${STAGE1_PROJECTOR_PATH}"
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
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
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

if [[ -n "${STAGE1_VISION_PATH}" ]]; then
  CMD+=(--stage1-vision-path "${STAGE1_VISION_PATH}")
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

if [[ "${USE_LORA}" == "1" ]]; then
  CMD+=(--use-lora)
fi

"${CMD[@]}"

#!/usr/bin/env bash
# Build the Stage4 visual_v3 GRPO prompt set from existing SFT splits.

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

TRAIN_SFT_PATH="${TRAIN_SFT_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train_100k_balanced.json}"
VAL_SFT_PATH="${VAL_SFT_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/val.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo_visual_v3}"
SEED="${SEED:-3407}"
TOP_TYPE_COUNT="${TOP_TYPE_COUNT:-96}"
TOP_BRAND_COUNT="${TOP_BRAND_COUNT:-48}"
VAL_PER_TASK="${VAL_PER_TASK:-300}"
TASK_QUOTAS="${TASK_QUOTAS:-{\"product_type_qa\":22000,\"product_color_qa\":22000,\"product_brand_qa\":14000,\"product_attribute_summary\":9000,\"product_title_generation\":9000,\"product_style_qa\":4000}}"

python -m vlm.data.build_stage4_grpo_clean \
  --train-sft-path "${TRAIN_SFT_PATH}" \
  --val-sft-path "${VAL_SFT_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  --top-type-count "${TOP_TYPE_COUNT}" \
  --top-brand-count "${TOP_BRAND_COUNT}" \
  --val-per-task "${VAL_PER_TASK}" \
  --task-quotas "${TASK_QUOTAS}"

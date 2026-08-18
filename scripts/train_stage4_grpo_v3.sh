#!/usr/bin/env bash
# Stage4 GRPO visual_v3 run on the cleaned prompt set.

set -euo pipefail

PROJECT_ROOT="/root/qwen3_siglip2_vlm"
cd "${PROJECT_ROOT}"

ANNOTATION_PATH="${ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo_visual_v3/train.json}"
VAL_ANNOTATION_PATH="${VAL_ANNOTATION_PATH:-/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo_visual_v3/val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/stage4_abo_grpo_visual_v3}"
MAX_SAMPLES="${MAX_SAMPLES:-80000}"
MAX_STEPS="${MAX_STEPS:-20000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40}"
LR="${LR:-3e-6}"
KL_BETA="${KL_BETA:-0.08}"
REWARD_PROFILE="${REWARD_PROFILE:-visual_v3}"
TASK_SAMPLING_WEIGHTS="${TASK_SAMPLING_WEIGHTS:-}"
if [[ -z "${TASK_SAMPLING_WEIGHTS}" ]]; then
  TASK_SAMPLING_WEIGHTS='{"product_color_qa":1.6,"product_type_qa":1.4,"product_attribute_summary":1.1,"product_brand_qa":0.7,"product_title_generation":0.8,"product_style_qa":0.6}'
fi
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0015}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_SAMPLES="${EVAL_SAMPLES:-600}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-25}"

export ANNOTATION_PATH
export VAL_ANNOTATION_PATH
export OUTPUT_DIR
export MAX_SAMPLES
export MAX_STEPS
export NUM_GENERATIONS
export MAX_NEW_TOKENS
export LR
export KL_BETA
export REWARD_PROFILE
export TASK_SAMPLING_WEIGHTS
export EARLY_STOP_PATIENCE
export EARLY_STOP_MIN_DELTA
export EARLY_STOP_MIN_STEPS
export EVAL_EVERY
export EVAL_SAMPLES
export SAVE_EVERY
export LOG_EVERY

bash scripts/train_stage4_grpo.sh

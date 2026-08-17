# Experiment Configs

These files contain environment variables for the main experiments referenced in the README.

They are optional. Existing scripts still work without them:

```bash
bash scripts/train_stage1.sh
```

To run with one config:

```bash
set -a
source configs/stage4_grpo_short_v2.env
set +a
bash scripts/train_stage4_grpo.sh
```

For the cleaned-label GRPO run:

```bash
bash scripts/build_stage4_grpo_clean.sh
set -a
source configs/stage4_grpo_visual_v3.env
set +a
bash scripts/train_stage4_grpo.sh
```

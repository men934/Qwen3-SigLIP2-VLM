# Qwen3-SigLIP2-VLM

基于 **Qwen3-1.7B** 和 **SigLIP2 SO400M** 的轻量 VLM 训练项目。模型使用 SigLIP2 提取视觉特征，通过 2x2 Patch Merger 和两层 MLP Projector 将视觉 token 映射到 Qwen hidden space，再把视觉 token 插入文本序列后送入 Qwen 做自回归训练和生成。

仓库覆盖四个阶段：Stage1 视觉语言对齐、Stage2 通用多模态指令微调、Stage3 文档/OCR/图表垂域 SFT、Stage4 电商垂域 SFT 与 GRPO-style reward optimization。

![Qwen3-SigLIP2-VLM project overview](docs/assets/project_hero.png)

## 主要实现

- **VLM 结构**：SigLIP2 vision encoder + 2x2 Patch Merger + MLP Projector + Qwen3 CausalLM。
- **视觉 token 插入**：collator 保留单个 `<image>` token，模型 forward 中将其替换为多枚 projected visual tokens，并同步扩展 attention mask 与 labels。
- **动态分辨率分支**：图像尺寸按 `patch_size * merge_size` 对齐，batch 内 padding 后再根据真实 patch grid 裁掉 padding token。
- **SigLIP2 Q/K 2D RoPE 实验**：在 SigLIP attention 的 Q/K 上加入二维 RoPE，用于和固定分辨率 + absolute position embedding 分支对比。
- **LoRA 微调链路**：Stage2/3/4 复用 projector checkpoint，并在 Qwen 上加载、继续训练 LoRA adapter。
- **电商 GRPO-style 实验**：基于 ABO 商品图任务实现在线采样、组内 advantage 标准化、KL penalty、early stopping 和任务自适应 reward。

## Model Design

```text
image
  -> SigLIP2 Vision Encoder
  -> optional 2D Q/K RoPE inside SigLIP attention
  -> 2x2 Patch Merger
  -> 2-layer MLP Projector
  -> visual tokens

text prompt
  -> Qwen tokenizer
  -> text tokens

visual tokens + text tokens
  -> Qwen3-1.7B Causal LM
  -> answer
```

关键模块：

| Module | File | Description |
|---|---|---|
| Patch Merger | `src/vlm/models/patch_merger.py` | 将相邻 2x2 patch token concat，降低视觉 token 数量 |
| Projector | `src/vlm/models/projector.py` | 两层 MLP，将 SigLIP hidden size 映射到 Qwen hidden size |
| VLM Model | `src/vlm/models/vlm_model.py` | 视觉编码、视觉 token 插入、语言模型 forward/generation |
| Image Processor | `src/vlm/data/image_processing.py` | 固定/动态分辨率图像预处理、patch grid 记录、batch padding |
| Collator | `src/vlm/data/collator.py` | 构造 `input_ids`、`labels`、`pixel_values` 和图像元信息 |

## Training Pipeline

![Qwen3-SigLIP2-VLM multi-stage training pipeline](docs/assets/training_pipeline_overview.png)

### Stage 1: Visual-Language Alignment

冻结 SigLIP2 和 Qwen3，训练 Patch Merger / Projector，使视觉 token 能接入 Qwen 的 hidden space。

数据：

- LLaVA-Pretrain 558K / LLaVA-CC3M-Pretrain
- COCO train2014 images

入口：

```bash
bash scripts/train_stage1.sh
```

### Stage 2: General Multimodal Instruction Tuning

从 Stage1 projector 初始化，加入通用视觉指令数据。主线实验使用 Qwen LoRA，并继续训练 projector。

数据：

- LLaVA-1.5 instruction data
- ShareGPT4V subset
- MMInstruct subset

入口：

```bash
bash scripts/train_stage2.sh
```

### Stage 3: Document / OCR / Chart Domain SFT

从 Stage2 projector + LoRA checkpoint 初始化，继续训练文档理解、OCR 和图表问答数据。

数据混合：

- DocVQA
- TextVQA
- ChartQA
- CORD
- FUNSD / SROIE-style structured extraction data

入口：

```bash
bash scripts/train_stage3.sh
```

300 样本评估结果：

| Checkpoint | EM | F1 | Chart Relaxed |
|---|---:|---:|---:|
| Stage1 fixed 50k | 0.0000 | 0.0226 | 0.0100 |
| Stage2 fixed 150k r32 | 0.0000 | 0.0602 | 0.0167 |
| Stage3 doc/ocr mix | **0.1833** | **0.2324** | **0.0600** |

![Stage3 overall metrics](docs/assets/stage3_overall_metrics.png)

![Stage3 by-source F1](docs/assets/stage3_by_source_f1.png)

### Stage 4: E-commerce Domain SFT

Stage4 使用 Amazon Berkeley Objects 构造商品图问答任务，包括品牌、颜色、类型、风格、标题生成和属性总结。

任务类型：

- `product_brand_qa`
- `product_color_qa`
- `product_type_qa`
- `product_style_qa`
- `product_title_generation`
- `product_attribute_summary`

入口：

```bash
bash scripts/train_stage4_sft.sh
```

100k balanced SFT 的验证 loss：

![Stage4 SFT loss](docs/assets/stage4_sft_loss_curve.png)

## GRPO-style Reward Optimization

Stage4 SFT 之后，仓库实现了一个电商垂域的在线 GRPO-style 训练脚本：

```bash
bash scripts/train_stage4_grpo.sh
```

训练流程：

```text
1. 对每个 prompt 采样 G 个回复
2. 使用任务自适应规则 reward 计算奖励 R_i
3. 在同一 prompt 的 G 个回复内部做 advantage 标准化
   A_i = (R_i - mean(R)) / (std(R) + eps)
4. 裁剪 advantage
5. 计算 policy logprob 与 reference logprob
6. 使用 clipped surrogate + KL penalty 更新 LoRA
```

实现边界：

- 这是 **online single-update GRPO-style trainer**，不是完整 PPO/GRPO replay trainer。
- 每组回复只更新一次；`old_logps` 使用 `current_logps.detach()` 作为采样时 policy 的快照。
- loss 使用 sequence-level average logprob，不是逐 token 的多轮 RLHF objective。
- 默认 `prompt_batch_size=1`，每个 optimizer step 处理一个 prompt 和 `NUM_GENERATIONS` 个回复。
- 默认只训练 Qwen LoRA，冻结 SigLIP2、Qwen backbone 和 projector。
- reward 是任务规则函数，适合 brand/type/color/style 等短答案任务，以及 title/summary 的 token F1 shaping。

GRPO 对比实验：

| Run | Main Setting | Observation |
|---|---|---|
| GRPO 300-step | short-answer reward | generation F1 提升，但整体 EM/F1 下降 |
| GRPO 20k | long run, `NUM_GENERATIONS=4` | validation reward 变高，但真实评估整体退化，出现 reward hacking |
| GRPO short reward v2 | short run + stronger KL + early stopping + task-adaptive reward | 整体 F1 小幅超过 SFT，generation F1 高于 SFT |

GRPO short reward v2 使用：

- `KL_BETA=0.05`
- `LR=5e-6`
- `MAX_STEPS=3000`
- `NUM_GENERATIONS=4`
- early stopping
- title/summary 使用 token F1 主 reward
- style/title/summary 任务重采样

训练在 step 2500 early stopped，best 出现在 step 1500：

| Step | Val Reward |
|---:|---:|
| 250 | 0.7080 |
| 500 | 0.7174 |
| 1500 | **0.7196** |
| 2500 | 0.6988 |

![Stage4 GRPO metrics](docs/assets/stage4_grpo_short_v2_metrics.png)

## Evaluation

Stage4 电商 300 样本评估结果：

| Checkpoint | EM | F1 | Short EM | Generation F1 |
|---|---:|---:|---:|---:|
| Stage4 SFT 100k balanced | **0.5000** | 0.6593 | **0.6931** | 0.5422 |
| GRPO 300-step | 0.4867 | 0.6498 | 0.6683 | **0.5589** |
| GRPO 20k | 0.4767 | 0.6409 | 0.6683 | 0.5358 |
| GRPO short reward v2 | 0.4933 | **0.6609** | 0.6881 | 0.5528 |

![Stage4 overall metrics](docs/assets/stage4_overall_metrics.png)

按任务 F1：

| Task | SFT100k F1 | GRPO short v2 F1 | Change |
|---|---:|---:|---:|
| attribute summary | 0.4943 | **0.5216** | +0.0273 |
| brand QA | **0.8256** | **0.8256** | +0.0000 |
| color QA | 0.6091 | **0.6352** | +0.0261 |
| style QA | **0.3800** | 0.3400 | -0.0400 |
| title generation | **0.5830** | 0.5792 | -0.0038 |
| type QA | **0.8421** | 0.8246 | -0.0175 |

![Stage4 by-task F1](docs/assets/stage4_by_task_f1.png)

结论：

- Stage4 SFT 100k balanced 是整体最稳的 checkpoint。
- GRPO short reward v2 将整体 F1 从 0.6593 提升到 0.6609，generation F1 从 0.5422 提升到 0.5528。
- `product_style_qa` 和 `product_title_generation` 仍受 reward 设计影响较大。

## Repository Structure

```text
qwen3_siglip2_vlm/
├── configs/
│   ├── stage1_alignment.env
│   ├── stage2_lora_150k.env
│   ├── stage3_doc_ocr_mix.env
│   ├── stage4_sft_100k_balanced.env
│   └── stage4_grpo_short_v2.env
├── scripts/
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   ├── train_stage3.sh
│   ├── train_stage4_sft.sh
│   └── train_stage4_grpo.sh
├── src/vlm/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── eval/
│   └── inference/
├── tools/
├── docs/assets/
├── requirements.txt
└── README.md
```

## 安装与环境

```bash
git clone https://github.com/men934/Qwen3-SigLIP2-VLM.git
cd Qwen3-SigLIP2-VLM

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

设置 `PYTHONPATH`：

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

训练脚本默认使用 `/root/autodl-tmp` 下的模型、数据集和 checkpoint。路径可通过环境变量覆盖：

```bash
QWEN_PATH=/path/to/Qwen3-1.7B \
SIGLIP_PATH=/path/to/siglip2-so400m-patch14-384 \
OUTPUT_DIR=/path/to/checkpoints/stage1 \
bash scripts/train_stage1.sh
```

## Configs

`configs/` 下保存了几组实验环境变量。它们不会改变现有脚本的默认行为，只是把 README 中的主线实验配置固定下来，便于复查和复跑。

使用方式：

```bash
set -a
source configs/stage4_grpo_short_v2.env
set +a
bash scripts/train_stage4_grpo.sh
```

也可以继续直接使用脚本默认参数或在命令前传入环境变量：

```bash
MAX_SAMPLES=100000 MAX_STEPS=3000 bash scripts/train_stage4_grpo.sh
```

## 数据说明

本文实验使用的数据如下：

| 阶段 | 数据 |
|---|---|
| Stage1 | LLaVA-Pretrain / LLaVA-CC3M-Pretrain + COCO train2014 |
| Stage2 | LLaVA-1.5 instruction data + ShareGPT4V/MMInstruct subsets |
| Stage3 | DocVQA, TextVQA, ChartQA, CORD/FUNSD/SROIE-style document data |
| Stage4 | Amazon Berkeley Objects derived e-commerce image-text tasks |

## 常用命令

```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
bash scripts/train_stage3.sh
bash scripts/train_stage4_sft.sh
bash scripts/train_stage4_grpo.sh
```

Stage4 电商垂域评估：

```bash
PYTHONPATH=src python -m vlm.eval.eval_stage4_ecommerce \
  --max-samples 300 \
  --checkpoints stage4_100k_balanced,stage4_grpo_short_v2 \
  --output-dir outputs/stage4_eval_300 \
  --max-new-tokens 64
```

## 备注

- 当前代码以单机实验脚本为主，没有接入 DeepSpeed/FSDP。
- 动态分辨率 + SigLIP2 Q/K 2D RoPE 是架构实验分支；当前结果没有稳定超过固定分辨率分支。
- Stage4 GRPO short reward v2 改善了整体 F1、generation F1、属性总结和颜色问答，但 style/title 任务仍需要更细的 reward 设计。

"""Stage 4 e-commerce GRPO-style reward optimization.

Default setup:
    - 冻结 SigLIP2 vision encoder。
    - 冻结 Qwen3 backbone。
    - 冻结 projector，避免稀疏 reward 把视觉-语言对齐层拉偏。
    - 只训练 Qwen LoRA adapter。
    - reference model 固定为 GRPO 开始前的 Stage4 SFT policy，用于 KL 约束。

Training loop:
    1. 对每个 prompt 采样 G 个回复。
    2. 用规则 reward 计算每个回复的奖励 R_i。
    3. 在同一个 prompt 的 G 个回复内部做标准化：

           A_i = (R_i - mean(R)) / (std(R) + eps)

    4. 对 advantage 做裁剪，避免少量 reward 噪声造成过大梯度。
    5. 计算 policy 当前 logprob、old logprob 和 reference logprob。
    6. 使用 clipped policy objective + KL penalty 更新 LoRA。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.conversation import DEFAULT_IMAGE_TOKEN, IM_END, IM_START
from vlm.data import ecommerce_reward
from vlm.data.grpo_dataset import GRPODataset
from vlm.models.vlm_model import IGNORE_INDEX, QwenSiglipVLM, VLMModelConfig


@dataclass
class Stage4GRPOConfig:
    """Stage4 GRPO 训练配置。"""

    qwen_path: str = "/root/autodl-tmp/hf_models/Qwen3-1.7B"
    siglip_path: str = "/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384"
    annotation_path: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo/train.json"
    val_annotation_path: str | None = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo/val.json"
    init_projector_path: str = "/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/projector.pt"
    init_lora_path: str = "/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/lora_adapter"
    resume_from_checkpoint: str | None = None
    output_dir: str = "/root/autodl-tmp/checkpoints/stage4_abo_grpo"

    image_size: int = 384
    max_prompt_length: int = 256
    max_new_tokens: int = 24
    max_samples: int | None = 2000
    max_steps: int | None = 500
    num_epochs: int = 1

    # Each optimizer step handles one prompt and G sampled responses.
    prompt_batch_size: int = 1
    num_generations: int = 4
    temperature: float = 0.7
    top_p: float = 0.9

    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # GRPO/PPO 风格裁剪与 KL 约束。
    clip_range: float = 0.2
    advantage_clip: float = 5.0
    kl_beta: float = 0.02
    advantage_eps: float = 1.0e-6

    # reward shaping。短答案仍以 exact/contains 为主；标题/摘要改用 token F1 作为主信号。
    token_f1_reward_weight: float = 0.1
    generative_f1_reward_weight: float = 1.0
    generative_exact_bonus: float = 0.1
    style_f1_reward_weight: float = 0.25
    length_penalty_weight: float = 0.02
    max_reward_tokens: int = 8
    generative_max_reward_tokens: int = 48
    reward_profile: str = "legacy"

    # 按任务过采样。20k 实验显示 style/title 被 GRPO 伤害较大，所以默认提高这两类出现频率。
    # JSON string so shell environment variables can override it directly.
    task_sampling_weights: str = (
        '{"product_style_qa": 2.5, "product_title_generation": 2.0, '
        '"product_attribute_summary": 1.5}'
    )

    # Early stopping for reward over-optimization.
    early_stop_patience: int = 4
    early_stop_min_delta: float = 0.002
    early_stop_min_steps: int = 750

    freeze_projector: bool = True
    num_workers: int = 2
    seed: int = 42
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 100
    eval_samples: int = 100
    torch_dtype: str = "bfloat16"
    device: str = "cuda"


def optional_int(value: str) -> int | None:
    """argparse 小工具：允许 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def parse_args() -> Stage4GRPOConfig:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Stage 4 电商垂域 GRPO 训练")
    parser.add_argument("--qwen-path", default=Stage4GRPOConfig.qwen_path)
    parser.add_argument("--siglip-path", default=Stage4GRPOConfig.siglip_path)
    parser.add_argument("--annotation-path", default=Stage4GRPOConfig.annotation_path)
    parser.add_argument("--val-annotation-path", default=Stage4GRPOConfig.val_annotation_path)
    parser.add_argument("--init-projector-path", default=Stage4GRPOConfig.init_projector_path)
    parser.add_argument("--init-lora-path", default=Stage4GRPOConfig.init_lora_path)
    parser.add_argument("--resume-from-checkpoint", default=Stage4GRPOConfig.resume_from_checkpoint)
    parser.add_argument("--output-dir", default=Stage4GRPOConfig.output_dir)

    parser.add_argument("--image-size", type=int, default=Stage4GRPOConfig.image_size)
    parser.add_argument("--max-prompt-length", type=int, default=Stage4GRPOConfig.max_prompt_length)
    parser.add_argument("--max-new-tokens", type=int, default=Stage4GRPOConfig.max_new_tokens)
    parser.add_argument("--max-samples", type=optional_int, default=Stage4GRPOConfig.max_samples)
    parser.add_argument("--max-steps", type=optional_int, default=Stage4GRPOConfig.max_steps)
    parser.add_argument("--num-epochs", type=int, default=Stage4GRPOConfig.num_epochs)
    parser.add_argument("--prompt-batch-size", type=int, default=Stage4GRPOConfig.prompt_batch_size)
    parser.add_argument("--num-generations", type=int, default=Stage4GRPOConfig.num_generations)
    parser.add_argument("--temperature", type=float, default=Stage4GRPOConfig.temperature)
    parser.add_argument("--top-p", type=float, default=Stage4GRPOConfig.top_p)

    parser.add_argument("--learning-rate", type=float, default=Stage4GRPOConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage4GRPOConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Stage4GRPOConfig.max_grad_norm)
    parser.add_argument("--clip-range", type=float, default=Stage4GRPOConfig.clip_range)
    parser.add_argument("--advantage-clip", type=float, default=Stage4GRPOConfig.advantage_clip)
    parser.add_argument("--kl-beta", type=float, default=Stage4GRPOConfig.kl_beta)
    parser.add_argument("--token-f1-reward-weight", type=float, default=Stage4GRPOConfig.token_f1_reward_weight)
    parser.add_argument(
        "--generative-f1-reward-weight",
        type=float,
        default=Stage4GRPOConfig.generative_f1_reward_weight,
    )
    parser.add_argument(
        "--generative-exact-bonus",
        type=float,
        default=Stage4GRPOConfig.generative_exact_bonus,
    )
    parser.add_argument("--style-f1-reward-weight", type=float, default=Stage4GRPOConfig.style_f1_reward_weight)
    parser.add_argument("--length-penalty-weight", type=float, default=Stage4GRPOConfig.length_penalty_weight)
    parser.add_argument("--max-reward-tokens", type=int, default=Stage4GRPOConfig.max_reward_tokens)
    parser.add_argument(
        "--generative-max-reward-tokens",
        type=int,
        default=Stage4GRPOConfig.generative_max_reward_tokens,
    )
    parser.add_argument("--reward-profile", default=Stage4GRPOConfig.reward_profile)
    parser.add_argument("--task-sampling-weights", default=Stage4GRPOConfig.task_sampling_weights)
    parser.add_argument("--early-stop-patience", type=int, default=Stage4GRPOConfig.early_stop_patience)
    parser.add_argument("--early-stop-min-delta", type=float, default=Stage4GRPOConfig.early_stop_min_delta)
    parser.add_argument("--early-stop-min-steps", type=int, default=Stage4GRPOConfig.early_stop_min_steps)
    parser.add_argument("--train-projector", action="store_true")

    parser.add_argument("--num-workers", type=int, default=Stage4GRPOConfig.num_workers)
    parser.add_argument("--seed", type=int, default=Stage4GRPOConfig.seed)
    parser.add_argument("--log-every", type=int, default=Stage4GRPOConfig.log_every)
    parser.add_argument("--save-every", type=int, default=Stage4GRPOConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=Stage4GRPOConfig.eval_every)
    parser.add_argument("--eval-samples", type=int, default=Stage4GRPOConfig.eval_samples)
    parser.add_argument("--torch-dtype", default=Stage4GRPOConfig.torch_dtype)
    parser.add_argument("--device", default=Stage4GRPOConfig.device)

    args = parser.parse_args()
    return Stage4GRPOConfig(
        qwen_path=args.qwen_path,
        siglip_path=args.siglip_path,
        annotation_path=args.annotation_path,
        val_annotation_path=args.val_annotation_path,
        init_projector_path=args.init_projector_path,
        init_lora_path=args.init_lora_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        prompt_batch_size=args.prompt_batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        clip_range=args.clip_range,
        advantage_clip=args.advantage_clip,
        kl_beta=args.kl_beta,
        token_f1_reward_weight=args.token_f1_reward_weight,
        generative_f1_reward_weight=args.generative_f1_reward_weight,
        generative_exact_bonus=args.generative_exact_bonus,
        style_f1_reward_weight=args.style_f1_reward_weight,
        length_penalty_weight=args.length_penalty_weight,
        max_reward_tokens=args.max_reward_tokens,
        generative_max_reward_tokens=args.generative_max_reward_tokens,
        reward_profile=args.reward_profile,
        task_sampling_weights=args.task_sampling_weights,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_steps=args.early_stop_min_steps,
        freeze_projector=not args.train_projector,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    """Set random seeds."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_identity(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return samples unchanged."""

    return examples


def parse_task_weights(raw: str) -> dict[str, float]:
    """Parse task-sampling weights from a JSON dict."""

    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("--task-sampling-weights 必须是 JSON object。")
    output: dict[str, float] = {}
    for key, value in data.items():
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"任务 {key} 的采样权重必须为正数，当前为 {weight}。")
        output[str(key)] = weight
    return output


def build_weighted_sampler(dataset: GRPODataset, config: Stage4GRPOConfig) -> WeightedRandomSampler | None:
    """Build a weighted sampler over tasks."""

    task_weights = parse_task_weights(config.task_sampling_weights)
    if not task_weights:
        return None

    weights = []
    task_counter: Counter[str] = Counter()
    weighted_counter: Counter[str] = Counter()
    for raw_sample in dataset.samples:
        task = str(raw_sample.get("task", "unknown"))
        weight = task_weights.get(task, 1.0)
        weights.append(weight)
        task_counter[task] += 1
        weighted_counter[task] += int(round(weight * 100))

    print("[data] task counts:", dict(sorted(task_counter.items())))
    print("[data] task sampling weights:", task_weights)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(config: Stage4GRPOConfig) -> tuple[DataLoader, GRPODataset | None, VLMDataCollator]:
    """Build training and validation loaders."""

    train_dataset = GRPODataset(
        annotation_path=config.annotation_path,
        verify_images=False,
        max_samples=config.max_samples,
    )
    sampler = build_weighted_sampler(train_dataset, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.prompt_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_identity,
        drop_last=False,
    )
    val_dataset = None
    if config.val_annotation_path and config.eval_samples > 0:
        val_dataset = GRPODataset(
            annotation_path=config.val_annotation_path,
            verify_images=False,
            max_samples=config.eval_samples,
        )

    helper = VLMDataCollator(
        CollatorConfig(
            tokenizer_path=config.qwen_path,
            image_processor_path=config.siglip_path,
            image_size=config.image_size,
            max_length=config.max_prompt_length,
        )
    )
    return train_loader, val_dataset, helper


def build_model(
    config: Stage4GRPOConfig,
    helper: VLMDataCollator,
    *,
    trainable: bool,
    projector_path: str | Path | None = None,
    lora_path: str | Path | None = None,
) -> QwenSiglipVLM:
    """加载 Stage4 SFT 模型。

    trainable=True:
        policy model。LoRA 可训练，projector 默认冻结。

    trainable=False:
        reference model。所有参数冻结，只用于 KL 约束。
    """

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path=config.qwen_path,
            siglip_path=config.siglip_path,
            image_token_id=helper.image_token_id,
            tokenizer_length=len(helper.tokenizer),
            freeze_vision_encoder=True,
            freeze_language_model=True,
            torch_dtype=config.torch_dtype,
        )
    )

    projector_path = Path(projector_path or config.init_projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"初始化 projector 不存在：{projector_path}")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    from peft import PeftModel

    lora_path = Path(lora_path or config.init_lora_path)
    if not lora_path.is_dir():
        raise FileNotFoundError(f"初始化 LoRA adapter 不存在：{lora_path}")
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        str(lora_path),
        is_trainable=trainable,
    )

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = trainable and not config.freeze_projector
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    return model


def checkpoint_paths(checkpoint_dir: str | Path) -> tuple[Path, Path, Path]:
    ckpt_dir = Path(checkpoint_dir)
    projector_path = ckpt_dir / "projector.pt"
    lora_path = ckpt_dir / "lora_adapter"
    trainer_state_path = ckpt_dir / "trainer_state.pt"
    if not projector_path.is_file():
        raise FileNotFoundError(f"resume checkpoint 缺少 projector.pt：{projector_path}")
    if not lora_path.is_dir():
        raise FileNotFoundError(f"resume checkpoint 缺少 lora_adapter：{lora_path}")
    if not trainer_state_path.is_file():
        raise FileNotFoundError(f"resume checkpoint 缺少 trainer_state.pt：{trainer_state_path}")
    return projector_path, lora_path, trainer_state_path


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def best_val_from_metrics(metrics_path: Path) -> float:
    if not metrics_path.is_file():
        return float("-inf")
    best = float("-inf")
    for row in csv.DictReader(metrics_path.open("r", encoding="utf-8")):
        raw = row.get("val_reward")
        if raw:
            best = max(best, float(raw))
    return best


def build_prompt(sample: dict[str, Any], image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """把 GRPO user message 格式化成 Qwen ChatML prompt。"""

    user_content = sample["messages"][0]["content"]
    if image_token not in user_content:
        user_content = f"{image_token}\n{user_content}"
    return f"{IM_START}user\n{user_content}{IM_END}\n{IM_START}assistant\n"


def normalize_answer(text: str) -> str:
    """短答案归一化：小写、去标点、合并空格。"""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, references: list[str]) -> float:
    """轻量 token F1，用作稀疏 exact reward 的辅助 shaping。"""

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


GENERATION_TASKS = {
    "product_title_generation",
    "product_attribute_summary",
}


def legacy_reward_one(
    prediction: str,
    references: list[str],
    config: Stage4GRPOConfig,
    *,
    task: str = "",
) -> tuple[float, float]:
    normalized_pred = normalize_answer(prediction)
    normalized_refs = [normalize_answer(item) for item in references]
    exact = float(any(normalized_pred == ref for ref in normalized_refs))
    contains = float(
        any(
            normalized_pred
            and ref
            and (normalized_pred in ref or ref in normalized_pred)
            for ref in normalized_refs
        )
    )
    f1 = token_f1(prediction, references)
    length = len(normalized_pred.split())

    if task in GENERATION_TASKS:
        max_tokens = config.generative_max_reward_tokens
        length_penalty = max(0, length - max_tokens) * config.length_penalty_weight
        reward = config.generative_f1_reward_weight * f1
        reward += config.generative_exact_bonus * exact
        reward += 0.1 * (1.0 - exact) * contains
        reward -= length_penalty
        return reward, exact

    length_penalty = max(0, length - config.max_reward_tokens) * config.length_penalty_weight
    reward = exact + 0.5 * (1.0 - exact) * contains
    f1_weight = config.style_f1_reward_weight if task == "product_style_qa" else config.token_f1_reward_weight
    reward += f1_weight * f1 - length_penalty
    return reward, exact


def visual_v3_reward_one(
    prediction: str,
    references: list[str],
    config: Stage4GRPOConfig,
    *,
    task: str = "",
) -> tuple[float, float]:
    normalized_pred = ecommerce_reward.canonicalize(prediction, task)
    normalized_refs = [ecommerce_reward.canonicalize(item, task) for item in references]
    exact = float(any(normalized_pred and normalized_pred == ref for ref in normalized_refs))
    contains = float(
        any(
            normalized_pred
            and ref
            and (normalized_pred in ref or ref in normalized_pred)
            for ref in normalized_refs
        )
    )
    f1 = ecommerce_reward.token_f1(prediction, references, task)
    token_count = len(normalized_pred.split())

    if task in GENERATION_TASKS:
        length_penalty = max(0, token_count - config.generative_max_reward_tokens)
        reward = config.generative_f1_reward_weight * f1
        reward += config.generative_exact_bonus * exact
        reward += 0.05 * contains
        reward -= config.length_penalty_weight * length_penalty
        if task == "product_attribute_summary":
            lowered = prediction.lower()
            field_bonus = 0.0
            for marker in ("title", "brand", "product type", "color"):
                if marker in lowered:
                    field_bonus += 0.025
            reward += min(field_bonus, 0.1)
        return reward, exact

    if token_count == 0:
        return -0.2, 0.0

    max_tokens = config.max_reward_tokens
    if task == "product_type_qa":
        max_tokens = 6
    elif task == "product_brand_qa":
        max_tokens = 8
    elif task == "product_style_qa":
        max_tokens = 5
    length_penalty = max(0, token_count - max_tokens) * config.length_penalty_weight

    if task == "product_type_qa":
        reward = exact + 0.35 * (1.0 - exact) * f1 + 0.1 * contains
    elif task == "product_color_qa":
        reward = exact + 0.25 * (1.0 - exact) * f1 + 0.15 * contains
    elif task == "product_brand_qa":
        reward = exact + 0.2 * (1.0 - exact) * f1 + 0.1 * contains
    elif task == "product_style_qa":
        reward = exact + 0.5 * (1.0 - exact) * f1 + 0.1 * contains
    else:
        reward = exact + config.token_f1_reward_weight * f1 + 0.1 * contains

    return reward - length_penalty, exact


def reward_one(
    prediction: str,
    references: list[str],
    config: Stage4GRPOConfig,
    *,
    task: str = "",
) -> tuple[float, float]:
    if config.reward_profile == "legacy":
        return legacy_reward_one(prediction, references, config, task=task)
    if config.reward_profile == "visual_v3":
        return visual_v3_reward_one(prediction, references, config, task=task)
    raise ValueError(f"未知 reward_profile：{config.reward_profile}")


def trim_generated_ids(token_ids: Tensor, tokenizer) -> list[int]:
    """清理 generate 输出，得到用于 logprob 计算的 response token ids。"""

    ids = [int(item) for item in token_ids.detach().cpu().tolist()]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    cleaned: list[int] = []
    for token_id in ids:
        if pad_id is not None and token_id == pad_id:
            break
        cleaned.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
    if not cleaned:
        cleaned = [eos_id if eos_id is not None else tokenizer.pad_token_id]
    return cleaned


@torch.no_grad()
def generate_responses(
    model: QwenSiglipVLM,
    helper: VLMDataCollator,
    sample: dict[str, Any],
    config: Stage4GRPOConfig,
    device: torch.device,
) -> tuple[str, list[str], list[list[int]]]:
    """对一个 prompt 采样 G 个回复。"""

    prompt = build_prompt(sample, image_token=helper.image_token)
    encoded = helper.tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=config.max_prompt_length,
        return_tensors="pt",
    )
    image_batch = helper.image_processor.process_batch([sample["image_path"]])
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = image_batch.pixel_values.to(device)

    visual_embeds, _ = model.encode_images(pixel_values, image_infos=image_batch.infos)
    multimodal_inputs = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        visual_embeds=visual_embeds,
    )

    do_sample = config.temperature > 0
    generation_kwargs = {
        "inputs_embeds": multimodal_inputs["inputs_embeds"],
        "attention_mask": multimodal_inputs["attention_mask"],
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": config.num_generations,
        "eos_token_id": helper.tokenizer.eos_token_id,
        "pad_token_id": helper.tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
        )
    output_ids = model.language_model.generate(**generation_kwargs)

    response_ids = [trim_generated_ids(row, helper.tokenizer) for row in output_ids]
    response_texts = []
    for ids in response_ids:
        text = helper.tokenizer.decode(ids, skip_special_tokens=False)
        if IM_END in text:
            text = text.split(IM_END, 1)[0]
        response_texts.append(text.strip())
    return prompt, response_texts, response_ids


def pad_1d(sequences: list[Tensor], pad_value: int, device: torch.device) -> Tensor:
    """把一组 1D tensor padding 成 batch。"""

    max_len = max(item.numel() for item in sequences)
    output = torch.full(
        (len(sequences), max_len),
        pad_value,
        dtype=sequences[0].dtype,
        device=device,
    )
    for index, item in enumerate(sequences):
        output[index, : item.numel()] = item
    return output


def sequence_logprobs(
    model: QwenSiglipVLM,
    helper: VLMDataCollator,
    sample: dict[str, Any],
    prompt: str,
    response_ids: list[list[int]],
    device: torch.device,
) -> Tensor:
    """Compute average token logprob for each response."""

    tokenizer = helper.tokenizer
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(device)

    input_sequences = []
    label_sequences = []
    attention_sequences = []
    for ids in response_ids:
        resp = torch.tensor(ids, dtype=torch.long, device=device)
        full_ids = torch.cat([prompt_ids, resp], dim=0)
        labels = torch.full_like(full_ids, IGNORE_INDEX)
        labels[prompt_ids.numel() :] = resp
        input_sequences.append(full_ids)
        label_sequences.append(labels)
        attention_sequences.append(torch.ones_like(full_ids))

    input_ids = pad_1d(input_sequences, tokenizer.pad_token_id, device)
    labels = pad_1d(label_sequences, IGNORE_INDEX, device)
    attention_mask = pad_1d(attention_sequences, 0, device)

    image_batch = helper.image_processor.process_batch([sample["image_path"]])
    pixel_values = image_batch.pixel_values.to(device)
    visual_embeds, _ = model.encode_images(pixel_values, image_infos=image_batch.infos)
    if isinstance(visual_embeds, Tensor):
        visual_embeds = visual_embeds.repeat(len(response_ids), 1, 1)
    else:
        visual_embeds = visual_embeds * len(response_ids)

    multimodal_inputs = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        visual_embeds=visual_embeds,
    )
    outputs = model.language_model(
        inputs_embeds=multimodal_inputs["inputs_embeds"],
        attention_mask=multimodal_inputs["attention_mask"],
        use_cache=False,
    )

    expanded_labels = multimodal_inputs["labels"]
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = expanded_labels[:, 1:]
    valid_mask = shift_labels.ne(IGNORE_INDEX)

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logprobs = token_logprobs * valid_mask
    token_counts = valid_mask.sum(dim=1).clamp_min(1)
    return token_logprobs.sum(dim=1) / token_counts


def grpo_loss(
    current_logps: Tensor,
    ref_logps: Tensor,
    rewards: Tensor,
    config: Stage4GRPOConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Compute clipped GRPO loss with KL penalty."""

    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if float(reward_std.detach().cpu()) < config.advantage_eps:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - reward_mean) / (reward_std + config.advantage_eps)
    advantages = advantages.clamp(-config.advantage_clip, config.advantage_clip)

    old_logps = current_logps.detach()
    ratio = torch.exp(current_logps - old_logps)
    clipped_ratio = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    # 这是常用的非负 KL 估计形式。ref_logps 越接近 current_logps，KL 越接近 0。
    ref_delta = ref_logps.detach() - current_logps
    kl = torch.exp(ref_delta) - ref_delta - 1.0

    policy_loss = -surrogate.mean()
    kl_loss = kl.mean()
    loss = policy_loss + config.kl_beta * kl_loss
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "reward_mean": float(reward_mean.detach().cpu()),
        "reward_std": float(reward_std.detach().cpu()),
        "adv_mean": float(advantages.mean().detach().cpu()),
        "adv_abs_mean": float(advantages.abs().mean().detach().cpu()),
    }
    return loss, stats


def save_checkpoint(
    model: QwenSiglipVLM,
    optimizer: torch.optim.Optimizer,
    config: Stage4GRPOConfig,
    step: int,
    output_dir: Path,
    *,
    name: str | None = None,
    val_reward: float | None = None,
) -> None:
    """保存 GRPO checkpoint。"""

    ckpt_dir = output_dir / (name if name is not None else f"step_{step:06d}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), ckpt_dir / "projector.pt")
    model.language_model.save_pretrained(ckpt_dir / "lora_adapter")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "val_reward": val_reward,
        },
        ckpt_dir / "trainer_state.pt",
    )
    payload = asdict(config)
    payload["step"] = step
    payload["val_reward"] = val_reward
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] checkpoint -> {ckpt_dir}")


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    """追加写入训练指标。"""

    fieldnames = [
        "step",
        "loss",
        "policy_loss",
        "kl",
        "reward_mean",
        "reward_std",
        "exact_mean",
        "val_reward",
        "elapsed",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_metrics(metrics_path: Path, output_path: Path) -> None:
    """绘制 GRPO reward/loss/KL 曲线。"""

    if not metrics_path.is_file():
        return
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    reward = [float(row["reward_mean"]) for row in rows]
    exact = [float(row["exact_mean"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    kl = [float(row["kl"]) for row in rows]
    val_steps = [int(row["step"]) for row in rows if row.get("val_reward")]
    val_rewards = [float(row["val_reward"]) for row in rows if row.get("val_reward")]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(steps, reward, label="train reward", linewidth=1.4)
    axes[0].plot(steps, exact, label="train exact", linewidth=1.4)
    if val_steps:
        axes[0].plot(val_steps, val_rewards, marker="o", label="val greedy reward")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, loss, label="loss", color="tab:red")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(steps, kl, label="KL", color="tab:purple")
    axes[2].set_xlabel("optimizer step")
    axes[2].set_ylabel("KL")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle("Stage 4 GRPO Training Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_reward(
    model: QwenSiglipVLM,
    helper: VLMDataCollator,
    dataset: GRPODataset | None,
    config: Stage4GRPOConfig,
    device: torch.device,
) -> float | None:
    """在验证集上用 greedy generation 估计平均 reward。"""

    if dataset is None:
        return None
    was_training = model.training
    model.eval()
    rewards = []
    eval_config = replace(config, temperature=0.0, top_p=1.0, num_generations=1)
    for index in range(min(config.eval_samples, len(dataset))):
        sample = dataset[index]
        _, texts, _ = generate_responses(model, helper, sample, eval_config, device)
        refs = sample["reward"]["answers"]
        reward, _ = reward_one(texts[0], refs, config, task=sample.get("task", ""))
        rewards.append(reward)
    if was_training:
        model.train()
    return sum(rewards) / max(1, len(rewards))


def train(config: Stage4GRPOConfig) -> None:
    """执行 Stage4 GRPO 训练。"""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    plot_path = output_dir / "grpo_metrics.png"

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")
    print(f"[init] output_dir={output_dir}")
    print(f"[init] init_lora_path={config.init_lora_path}")
    if config.resume_from_checkpoint:
        print(f"[init] resume_from_checkpoint={config.resume_from_checkpoint}")
    print(f"[init] train_projector={not config.freeze_projector}")

    train_loader, val_dataset, helper = build_dataloaders(config)
    print(f"[data] train prompts={len(train_loader.dataset)} batches={len(train_loader)}")
    if val_dataset is not None:
        print(f"[data] val prompts={len(val_dataset)}")

    resume_state: dict[str, Any] | None = None
    resume_projector_path: Path | None = None
    resume_lora_path: Path | None = None
    if config.resume_from_checkpoint:
        resume_projector_path, resume_lora_path, resume_state_path = checkpoint_paths(
            config.resume_from_checkpoint
        )
        resume_state = torch.load(resume_state_path, map_location="cpu")

    policy = build_model(
        config,
        helper,
        trainable=True,
        projector_path=resume_projector_path,
        lora_path=resume_lora_path,
    ).to(device)
    reference = build_model(config, helper, trainable=False).to(device)
    policy.train()
    reference.eval()
    policy.print_trainable_parameters()

    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有可训练参数，请检查 LoRA 是否以 is_trainable=True 加载。")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    global_step = int(resume_state.get("step", 0)) if resume_state is not None else 0
    best_val_reward = best_val_from_metrics(metrics_path)
    if resume_state is not None:
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            move_optimizer_state(optimizer, device)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group["weight_decay"] = config.weight_decay
        if resume_state.get("val_reward") is not None:
            best_val_reward = max(best_val_reward, float(resume_state["val_reward"]))
        print(
            "[resume] "
            f"global_step={global_step} "
            f"best_val_reward={best_val_reward:.4f} "
            f"lr={config.learning_rate}"
        )
    bad_eval_count = 0
    start = time.time()
    running: dict[str, float] = {}
    running_count = 0

    for epoch in range(config.num_epochs):
        print(f"[train] epoch {epoch + 1}/{config.num_epochs}")
        for batch in train_loader:
            if len(batch) != 1:
                raise ValueError("GRPO 当前只支持 prompt_batch_size=1。")
            sample = batch[0]

            policy.eval()
            prompt, response_texts, response_ids = generate_responses(
                policy,
                helper,
                sample,
                config,
                device,
            )

            refs = sample["reward"]["answers"]
            task = sample.get("task", "")
            reward_items = [
                reward_one(text, refs, config, task=task)
                for text in response_texts
            ]
            rewards = torch.tensor(
                [item[0] for item in reward_items],
                dtype=torch.float32,
                device=device,
            )
            exact_mean = sum(item[1] for item in reward_items) / len(reward_items)

            policy.train()
            current_logps = sequence_logprobs(
                policy,
                helper,
                sample,
                prompt,
                response_ids,
                device,
            )
            with torch.no_grad():
                ref_logps = sequence_logprobs(
                    reference,
                    helper,
                    sample,
                    prompt,
                    response_ids,
                    device,
                )

            loss, stats = grpo_loss(current_logps, ref_logps, rewards, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            global_step += 1

            stats["exact_mean"] = exact_mean
            stats["grad_norm"] = float(grad_norm.detach().cpu())
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + value
            running_count += 1

            val_reward = None
            if config.eval_every and global_step % config.eval_every == 0:
                val_reward = evaluate_reward(policy, helper, val_dataset, config, device)
                if val_reward is not None:
                    improved = val_reward > best_val_reward + config.early_stop_min_delta
                    if improved:
                        best_val_reward = val_reward
                        bad_eval_count = 0
                        save_checkpoint(
                            policy,
                            optimizer,
                            config,
                            global_step,
                            output_dir,
                            name="best",
                            val_reward=val_reward,
                        )
                    else:
                        bad_eval_count += 1

            if global_step % config.log_every == 0:
                averaged = {key: value / running_count for key, value in running.items()}
                elapsed = time.time() - start
                print(
                    "[log] "
                    f"step={global_step} "
                    f"loss={averaged['loss']:.4f} "
                    f"reward={averaged['reward_mean']:.4f} "
                    f"exact={averaged['exact_mean']:.4f} "
                    f"kl={averaged['kl']:.5f} "
                    + (f"val_reward={val_reward:.4f} " if val_reward is not None else "")
                    + f"elapsed={elapsed:.1f}s"
                )
                append_metrics(
                    metrics_path,
                    {
                        "step": global_step,
                        **averaged,
                        "val_reward": val_reward,
                        "elapsed": elapsed,
                    },
                )
                plot_metrics(metrics_path, plot_path)
                running.clear()
                running_count = 0

            if config.save_every and global_step % config.save_every == 0:
                save_checkpoint(policy, optimizer, config, global_step, output_dir)

            if (
                config.early_stop_patience > 0
                and global_step >= config.early_stop_min_steps
                and bad_eval_count >= config.early_stop_patience
            ):
                plot_metrics(metrics_path, plot_path)
                print(
                    "[early_stop] "
                    f"连续 {bad_eval_count} 次验证没有超过 best_val_reward="
                    f"{best_val_reward:.4f} + min_delta={config.early_stop_min_delta}，"
                    "提前停止 GRPO。"
                )
                return

            if config.max_steps is not None and global_step >= config.max_steps:
                if not config.save_every or global_step % config.save_every != 0:
                    save_checkpoint(policy, optimizer, config, global_step, output_dir)
                plot_metrics(metrics_path, plot_path)
                print("[done] 达到 max_steps，GRPO 训练结束。")
                return

    save_checkpoint(policy, optimizer, config, global_step, output_dir)
    plot_metrics(metrics_path, plot_path)
    print("[done] Stage4 GRPO 训练完成。")


if __name__ == "__main__":
    train(parse_args())

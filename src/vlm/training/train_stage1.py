"""Stage 1：视觉-语言对齐训练脚本。

Stage 1 的目标不是让模型学会复杂问答，而是先让 Qwen3 能够“读懂”
SigLIP2 + PatchMerger + Projector 产生的视觉 token。

训练策略：

    SigLIP2 vision encoder: 冻结
    Qwen3 language model:   冻结
    PatchMerger:            无参数
    MLPProjector:           训练

也就是说，Stage 1 只训练一个很小的视觉到语言空间的映射层：

    visual tokens -> Qwen hidden space

这个脚本故意写成简单可读的 PyTorch training loop，而不是一上来使用
Trainer/Accelerate/DeepSpeed。这样更适合学习每一步发生了什么。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.llava_pretrain_dataset import LlavaPretrainDataset
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


@dataclass
class Stage1TrainConfig:
    """Stage 1 训练配置。

    默认值偏向“先跑通 sanity training”，而不是直接跑完整训练。
    如果要跑完整 558K，可以显式把 ``max_samples`` 和 ``max_steps`` 设为 None。
    """

    qwen_path: str = "/root/autodl-tmp/hf_models/Qwen3-1.7B"
    siglip_path: str = "/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384"
    annotation_path: str = (
        "/root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    )
    val_annotation_path: str | None = None
    image_root: str = "/root/autodl-tmp/hf_datasets/LLaVA-Pretrain"
    output_dir: str = "/root/autodl-tmp/checkpoints/stage1_align"

    image_size: int = 384
    dynamic_resolution: bool = False
    min_pixels: int = 384 * 384
    max_pixels: int = 672 * 672
    max_length: int = 512
    max_samples: int | None = 128
    max_steps: int | None = 20
    num_epochs: int = 1

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-3
    vision_learning_rate: float = 2.0e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False
    siglip_rope_base: float = 10000.0
    siglip_rope_dim: int | None = None
    unfreeze_vision_after_steps: int | None = None
    unfreeze_vision_last_layers: int = 0

    num_workers: int = 2
    seed: int = 42
    log_every: int = 1
    save_every: int = 20
    eval_every: int = 50
    eval_batches: int = 20

    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    verify_images: bool = False


def parse_args() -> Stage1TrainConfig:
    parser = argparse.ArgumentParser(description="Stage 1 视觉-语言对齐训练")

    parser.add_argument("--qwen-path", default=Stage1TrainConfig.qwen_path)
    parser.add_argument("--siglip-path", default=Stage1TrainConfig.siglip_path)
    parser.add_argument("--annotation-path", default=Stage1TrainConfig.annotation_path)
    parser.add_argument("--val-annotation-path", default=Stage1TrainConfig.val_annotation_path)
    parser.add_argument("--image-root", default=Stage1TrainConfig.image_root)
    parser.add_argument("--output-dir", default=Stage1TrainConfig.output_dir)

    parser.add_argument("--image-size", type=int, default=Stage1TrainConfig.image_size)
    parser.add_argument("--dynamic-resolution", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=Stage1TrainConfig.min_pixels)
    parser.add_argument("--max-pixels", type=int, default=Stage1TrainConfig.max_pixels)
    parser.add_argument("--max-length", type=int, default=Stage1TrainConfig.max_length)
    parser.add_argument("--max-samples", type=optional_int, default=Stage1TrainConfig.max_samples)
    parser.add_argument("--max-steps", type=optional_int, default=Stage1TrainConfig.max_steps)
    parser.add_argument("--num-epochs", type=int, default=Stage1TrainConfig.num_epochs)

    parser.add_argument("--batch-size", type=int, default=Stage1TrainConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=Stage1TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=Stage1TrainConfig.learning_rate)
    parser.add_argument(
        "--vision-learning-rate",
        type=float,
        default=Stage1TrainConfig.vision_learning_rate,
    )
    parser.add_argument("--weight-decay", type=float, default=Stage1TrainConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Stage1TrainConfig.max_grad_norm)
    parser.add_argument(
        "--no-siglip-abs-pos-embedding",
        action="store_true",
        help="关闭 SigLIP2 原生 absolute position embedding。通常要和 --use-siglip-qk-2d-rope 一起使用。",
    )
    parser.add_argument("--use-siglip-qk-2d-rope", action="store_true")
    parser.add_argument("--siglip-rope-base", type=float, default=Stage1TrainConfig.siglip_rope_base)
    parser.add_argument("--siglip-rope-dim", type=optional_int, default=Stage1TrainConfig.siglip_rope_dim)
    parser.add_argument(
        "--unfreeze-vision-after-steps",
        type=optional_int,
        default=Stage1TrainConfig.unfreeze_vision_after_steps,
        help="达到该 optimizer step 后解冻 SigLIP2 最后 N 层；none 表示不解冻。",
    )
    parser.add_argument(
        "--unfreeze-vision-last-layers",
        type=int,
        default=Stage1TrainConfig.unfreeze_vision_last_layers,
    )

    parser.add_argument("--num-workers", type=int, default=Stage1TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=Stage1TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=Stage1TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=Stage1TrainConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=Stage1TrainConfig.eval_every)
    parser.add_argument("--eval-batches", type=int, default=Stage1TrainConfig.eval_batches)

    parser.add_argument("--torch-dtype", default=Stage1TrainConfig.torch_dtype)
    parser.add_argument("--device", default=Stage1TrainConfig.device)
    parser.add_argument("--verify-images", action="store_true")

    args = parser.parse_args()
    return Stage1TrainConfig(
        qwen_path=args.qwen_path,
        siglip_path=args.siglip_path,
        annotation_path=args.annotation_path,
        val_annotation_path=args.val_annotation_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        dynamic_resolution=args.dynamic_resolution,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_length=args.max_length,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        vision_learning_rate=args.vision_learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_siglip_abs_pos_embedding=not args.no_siglip_abs_pos_embedding,
        use_siglip_qk_2d_rope=args.use_siglip_qk_2d_rope,
        siglip_rope_base=args.siglip_rope_base,
        siglip_rope_dim=args.siglip_rope_dim,
        unfreeze_vision_after_steps=args.unfreeze_vision_after_steps,
        unfreeze_vision_last_layers=args.unfreeze_vision_last_layers,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        torch_dtype=args.torch_dtype,
        device=args.device,
        verify_images=args.verify_images,
    )


def optional_int(value: str) -> int | None:
    """argparse 用的小工具：允许传入 none/null 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def set_seed(seed: int) -> None:
    """固定随机种子，让小规模 sanity run 更容易复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 里的 Tensor 移到训练设备上，非 Tensor 元信息保持不动。"""

    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def build_dataloader(config: Stage1TrainConfig) -> tuple[DataLoader, DataLoader | None, VLMDataCollator]:
    """构造训练/验证 Dataset、Collator 和 DataLoader。"""

    train_dataset = LlavaPretrainDataset(
        annotation_path=config.annotation_path,
        image_root=config.image_root,
        verify_images=config.verify_images,
        max_samples=config.max_samples,
    )

    collator = VLMDataCollator(
        CollatorConfig(
            tokenizer_path=config.qwen_path,
            image_processor_path=config.siglip_path,
            image_size=config.image_size,
            dynamic_resolution=config.dynamic_resolution,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_length=config.max_length,
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_dataloader = None
    if config.val_annotation_path:
        val_dataset = LlavaPretrainDataset(
            annotation_path=config.val_annotation_path,
            image_root=config.image_root,
            verify_images=config.verify_images,
            max_samples=None,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return train_dataloader, val_dataloader, collator


def build_model(config: Stage1TrainConfig, collator: VLMDataCollator) -> QwenSiglipVLM:
    """构造 Stage 1 模型，并按策略冻结 vision tower 和 language model。"""

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path=config.qwen_path,
            siglip_path=config.siglip_path,
            image_token_id=collator.image_token_id,
            tokenizer_length=len(collator.tokenizer),
            freeze_vision_encoder=True,
            freeze_language_model=True,
            use_siglip_abs_pos_embedding=config.use_siglip_abs_pos_embedding,
            use_siglip_qk_2d_rope=config.use_siglip_qk_2d_rope,
            siglip_rope_base=config.siglip_rope_base,
            siglip_rope_dim=config.siglip_rope_dim,
            torch_dtype=config.torch_dtype,
        )
    )
    return model


def build_optimizer(
    model: QwenSiglipVLM,
    config: Stage1TrainConfig,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    """按当前 requires_grad 状态构造 optimizer。

    Stage 1 初始通常只训练 projector。等到需要解冻 SigLIP 后几层时，我们会重新
    调用这个函数，让 optimizer 增加 vision tower 的小学习率参数组。
    """

    projector_params = [
        param for param in model.projector.parameters() if param.requires_grad
    ]
    vision_params = [
        param for param in model.vision_encoder.parameters() if param.requires_grad
    ]

    param_groups: list[dict[str, Any]] = []
    if projector_params:
        param_groups.append(
            {
                "params": projector_params,
                "lr": config.learning_rate,
                "name": "projector",
            }
        )
    if vision_params:
        param_groups.append(
            {
                "params": vision_params,
                "lr": config.vision_learning_rate,
                "name": "vision_encoder",
            }
        )

    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查冻结策略。")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
    )
    trainable_params = [
        param for group in param_groups for param in group["params"]
    ]
    return optimizer, trainable_params


def unfreeze_siglip_last_layers(model: QwenSiglipVLM, num_layers: int) -> int:
    """解冻 SigLIP2 vision encoder 最后 num_layers 个 Transformer block。

    只解冻后几层的原因：
        动态分辨率 + Q/K 2D RoPE 改变了视觉塔内部的位置机制，但完全微调整个
        SO400M 视觉塔显存和过拟合风险都更高。先让最后几层适配新位置机制，是更
        保守的折中。
    """

    if num_layers <= 0:
        return 0

    layers = model.vision_encoder.encoder.layers
    num_layers = min(num_layers, len(layers))

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # post_layernorm 紧跟 encoder 输出，解冻它能让最后几层特征尺度有一点适配空间。
    for param in model.vision_encoder.post_layernorm.parameters():
        param.requires_grad = True

    return num_layers


def vision_trainable_state_dict(model: QwenSiglipVLM) -> dict[str, torch.Tensor]:
    """只收集 SigLIP2 中 requires_grad=True 的参数，用于轻量保存。"""

    state = {}
    for name, param in model.vision_encoder.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu()
    return state


def add_new_trainable_params_to_optimizer(
    optimizer: torch.optim.Optimizer,
    trainable_params: list[torch.nn.Parameter],
    model: QwenSiglipVLM,
    config: Stage1TrainConfig,
) -> list[torch.nn.Parameter]:
    """解冻 vision 后，把新增参数追加进 optimizer，保留 projector 的 Adam 状态。"""

    known_param_ids = {id(param) for param in trainable_params}
    new_vision_params = [
        param
        for param in model.vision_encoder.parameters()
        if param.requires_grad and id(param) not in known_param_ids
    ]
    if new_vision_params:
        optimizer.add_param_group(
            {
                "params": new_vision_params,
                "lr": config.vision_learning_rate,
                "name": "vision_encoder",
            }
        )
        trainable_params = trainable_params + new_vision_params
    return trainable_params


def save_checkpoint(
    model: QwenSiglipVLM,
    optimizer: torch.optim.Optimizer,
    config: Stage1TrainConfig,
    step: int,
    output_dir: Path,
) -> None:
    """保存 Stage 1 checkpoint。

    Stage 1 只训练 projector，所以主要保存 projector 权重即可。
    optimizer state 也保存，方便中断后继续训练。
    """

    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.projector.state_dict(), ckpt_dir / "projector.pt")
    vision_state = vision_trainable_state_dict(model)
    if vision_state:
        torch.save(vision_state, ckpt_dir / "vision_encoder_trainable.pt")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        ckpt_dir / "trainer_state.pt",
    )

    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    print(f"[save] checkpoint 已保存到 {ckpt_dir}")


def append_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    """把训练/验证指标追加写入 CSV。"""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "train_loss", "val_loss", "grad_norm", "elapsed"]
    file_exists = metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_loss_curve(metrics_path: Path, output_path: Path) -> None:
    """根据 metrics.csv 画 loss 曲线。"""

    if not metrics_path.exists():
        return

    steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            if row.get("train_loss"):
                steps.append(step)
                train_losses.append(float(row["train_loss"]))
            if row.get("val_loss"):
                val_steps.append(step)
                val_losses.append(float(row["val_loss"]))

    if not steps and not val_steps:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    if steps:
        plt.plot(steps, train_losses, label="train loss", linewidth=1.5)
    if val_steps:
        plt.plot(val_steps, val_losses, label="val loss", marker="o", linewidth=1.5)
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("Stage 1 Alignment Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


@torch.no_grad()
def evaluate(
    model: QwenSiglipVLM,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """在验证集上计算平均 loss。"""

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pixel_values=batch["pixel_values"],
            image_infos=batch["image_infos"],
        )
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("验证时模型没有返回 loss。")
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    if was_training:
        model.train()

    if total_batches == 0:
        raise ValueError("验证 dataloader 没有产生 batch。")
    return total_loss / total_batches


def train(config: Stage1TrainConfig) -> None:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}")
    print(f"[init] output_dir: {output_dir}")

    dataloader, val_dataloader, collator = build_dataloader(config)
    print(f"[data] batches per epoch: {len(dataloader)}")
    if val_dataloader is not None:
        print(f"[data] val batches: {len(val_dataloader)}")

    model = build_model(config, collator)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    optimizer, trainable_params = build_optimizer(model, config)
    vision_unfrozen = False
    if (
        config.unfreeze_vision_after_steps == 0
        and config.unfreeze_vision_last_layers > 0
    ):
        actual_layers = unfreeze_siglip_last_layers(
            model,
            config.unfreeze_vision_last_layers,
        )
        optimizer, trainable_params = build_optimizer(model, config)
        vision_unfrozen = True
        print(
            "[init] 已在训练开始前解冻 "
            f"SigLIP2 最后 {actual_layers} 层，vision_lr={config.vision_learning_rate}"
        )
        model.print_trainable_parameters()

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / "metrics.csv"
    loss_curve_path = output_dir / "loss_curve.png"

    for epoch in range(config.num_epochs):
        print(f"[train] epoch {epoch + 1}/{config.num_epochs}")

        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_infos=batch["image_infos"],
            )

            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("模型没有返回 loss，请检查 labels。")

            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            running_loss += float(loss.detach().cpu())
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        config.max_grad_norm,
                    )
                else:
                    grad_norm = torch.tensor(0.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if (
                    not vision_unfrozen
                    and config.unfreeze_vision_after_steps is not None
                    and config.unfreeze_vision_last_layers > 0
                    and global_step >= config.unfreeze_vision_after_steps
                ):
                    actual_layers = unfreeze_siglip_last_layers(
                        model,
                        config.unfreeze_vision_last_layers,
                    )
                    trainable_params = add_new_trainable_params_to_optimizer(
                        optimizer,
                        trainable_params,
                        model,
                        config,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    vision_unfrozen = True
                    print(
                        "[train] 已解冻 "
                        f"SigLIP2 最后 {actual_layers} 层，"
                        f"vision_lr={config.vision_learning_rate}"
                    )
                    model.print_trainable_parameters()

                if global_step % config.log_every == 0:
                    avg_loss = running_loss / (config.log_every * config.gradient_accumulation_steps)
                    elapsed = time.time() - start_time
                    val_loss = None
                    if (
                        val_dataloader is not None
                        and config.eval_every
                        and global_step % config.eval_every == 0
                    ):
                        val_loss = evaluate(
                            model=model,
                            dataloader=val_dataloader,
                            device=device,
                            max_batches=config.eval_batches,
                        )
                    print(
                        "[log] "
                        f"step={global_step} "
                        f"loss={avg_loss:.4f} "
                        + (f"val_loss={val_loss:.4f} " if val_loss is not None else "")
                        + (
                            f"grad_norm={float(grad_norm):.4f} "
                            f"visual_tokens={outputs['visual_token_count']} "
                            f"seq_len={outputs['expanded_attention_mask'].shape[1]} "
                            f"elapsed={elapsed:.1f}s"
                        )
                    )
                    append_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "train_loss": avg_loss,
                            "val_loss": val_loss,
                            "grad_norm": float(grad_norm),
                            "elapsed": elapsed,
                        },
                    )
                    plot_loss_curve(metrics_path, loss_curve_path)
                    running_loss = 0.0

                if config.save_every and global_step % config.save_every == 0:
                    save_checkpoint(model, optimizer, config, global_step, output_dir)

                if config.max_steps is not None and global_step >= config.max_steps:
                    if not config.save_every or global_step % config.save_every != 0:
                        save_checkpoint(model, optimizer, config, global_step, output_dir)
                    plot_loss_curve(metrics_path, loss_curve_path)
                    print("[done] 达到 max_steps，训练结束。")
                    return

        # 如果一个 epoch 结束时还有未 step 的梯度，也执行一次 optimizer.step。
        if micro_step % config.gradient_accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                not vision_unfrozen
                and config.unfreeze_vision_after_steps is not None
                and config.unfreeze_vision_last_layers > 0
                and global_step >= config.unfreeze_vision_after_steps
            ):
                actual_layers = unfreeze_siglip_last_layers(
                    model,
                    config.unfreeze_vision_last_layers,
                )
                trainable_params = add_new_trainable_params_to_optimizer(
                    optimizer,
                    trainable_params,
                    model,
                    config,
                )
                optimizer.zero_grad(set_to_none=True)
                vision_unfrozen = True
                print(
                    "[train] 已解冻 "
                    f"SigLIP2 最后 {actual_layers} 层，"
                    f"vision_lr={config.vision_learning_rate}"
                )
                model.print_trainable_parameters()
            print(
                "[log] "
                f"step={global_step} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"visual_tokens={outputs['visual_token_count']}"
            )
            append_metrics(
                output_dir / "metrics.csv",
                {
                    "step": global_step,
                    "train_loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm),
                    "elapsed": time.time() - start_time,
                },
            )
            plot_loss_curve(output_dir / "metrics.csv", output_dir / "loss_curve.png")

    save_checkpoint(model, optimizer, config, global_step, output_dir)
    plot_loss_curve(output_dir / "metrics.csv", output_dir / "loss_curve.png")
    print("[done] 训练完成。")


if __name__ == "__main__":
    # 示例：
    #
    # 先跑一个很小的 sanity training：
    #
    #   PYTHONPATH=/root/qwen3_siglip2_vlm/src \\
    #   python -m vlm.training.train_stage1 \\
    #     --max-samples 8 \\
    #     --max-steps 1 \\
    #     --batch-size 1 \\
    #     --gradient-accumulation-steps 1
    #
    # 确认没问题后，再逐步扩大 max_samples/max_steps。
    train(parse_args())

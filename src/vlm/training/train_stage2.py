"""Stage 2 multimodal instruction tuning.

Supported strategies:
    1. projector-only
    2. projector + Qwen LoRA
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
from vlm.data.llava_instruct_dataset import LlavaInstructDataset
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


@dataclass
class Stage2TrainConfig:
    """Stage 2 训练配置。"""

    qwen_path: str = "/root/autodl-tmp/hf_models/Qwen3-1.7B"
    siglip_path: str = "/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384"
    annotation_path: str = (
        "/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/llava_instruct_150k.json"
    )
    val_annotation_path: str | None = None
    image_root: str = "/root/autodl-tmp/hf_datasets/coco/train2014"
    stage1_projector_path: str = (
        "/root/autodl-tmp/checkpoints/stage1_align_50k/step_003000/projector.pt"
    )
    stage1_vision_path: str | None = None
    output_dir: str = "/root/autodl-tmp/checkpoints/stage2_llava_instruct"

    image_size: int = 384
    dynamic_resolution: bool = False
    min_pixels: int = 384 * 384
    max_pixels: int = 672 * 672
    max_length: int = 768
    max_samples: int | None = 128
    max_steps: int | None = 20
    num_epochs: int = 1

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False
    siglip_rope_base: float = 10000.0
    siglip_rope_dim: int | None = None

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    num_workers: int = 2
    seed: int = 42
    log_every: int = 1
    save_every: int = 20
    eval_every: int = 50
    eval_batches: int = 20

    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    verify_images: bool = False


def parse_args() -> Stage2TrainConfig:
    parser = argparse.ArgumentParser(description="Stage 2 多模态指令微调")

    parser.add_argument("--qwen-path", default=Stage2TrainConfig.qwen_path)
    parser.add_argument("--siglip-path", default=Stage2TrainConfig.siglip_path)
    parser.add_argument("--annotation-path", default=Stage2TrainConfig.annotation_path)
    parser.add_argument("--val-annotation-path", default=Stage2TrainConfig.val_annotation_path)
    parser.add_argument("--image-root", default=Stage2TrainConfig.image_root)
    parser.add_argument(
        "--stage1-projector-path",
        default=Stage2TrainConfig.stage1_projector_path,
    )
    parser.add_argument("--stage1-vision-path", default=Stage2TrainConfig.stage1_vision_path)
    parser.add_argument("--output-dir", default=Stage2TrainConfig.output_dir)

    parser.add_argument("--image-size", type=int, default=Stage2TrainConfig.image_size)
    parser.add_argument("--dynamic-resolution", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=Stage2TrainConfig.min_pixels)
    parser.add_argument("--max-pixels", type=int, default=Stage2TrainConfig.max_pixels)
    parser.add_argument("--max-length", type=int, default=Stage2TrainConfig.max_length)
    parser.add_argument("--max-samples", type=optional_int, default=Stage2TrainConfig.max_samples)
    parser.add_argument("--max-steps", type=optional_int, default=Stage2TrainConfig.max_steps)
    parser.add_argument("--num-epochs", type=int, default=Stage2TrainConfig.num_epochs)

    parser.add_argument("--batch-size", type=int, default=Stage2TrainConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=Stage2TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=Stage2TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage2TrainConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Stage2TrainConfig.max_grad_norm)
    parser.add_argument(
        "--no-siglip-abs-pos-embedding",
        action="store_true",
        help="关闭 SigLIP2 原生 absolute position embedding。通常要和 --use-siglip-qk-2d-rope 一起使用。",
    )
    parser.add_argument("--use-siglip-qk-2d-rope", action="store_true")
    parser.add_argument("--siglip-rope-base", type=float, default=Stage2TrainConfig.siglip_rope_base)
    parser.add_argument("--siglip-rope-dim", type=optional_int, default=Stage2TrainConfig.siglip_rope_dim)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=Stage2TrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=Stage2TrainConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=Stage2TrainConfig.lora_dropout)
    parser.add_argument(
        "--lora-target-modules",
        default=Stage2TrainConfig.lora_target_modules,
        help="逗号分隔的 LoRA target module 名称。",
    )

    parser.add_argument("--num-workers", type=int, default=Stage2TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=Stage2TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=Stage2TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=Stage2TrainConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=Stage2TrainConfig.eval_every)
    parser.add_argument("--eval-batches", type=int, default=Stage2TrainConfig.eval_batches)

    parser.add_argument("--torch-dtype", default=Stage2TrainConfig.torch_dtype)
    parser.add_argument("--device", default=Stage2TrainConfig.device)
    parser.add_argument("--verify-images", action="store_true")

    args = parser.parse_args()
    return Stage2TrainConfig(
        qwen_path=args.qwen_path,
        siglip_path=args.siglip_path,
        annotation_path=args.annotation_path,
        val_annotation_path=args.val_annotation_path,
        image_root=args.image_root,
        stage1_projector_path=args.stage1_projector_path,
        stage1_vision_path=args.stage1_vision_path,
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
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_siglip_abs_pos_embedding=not args.no_siglip_abs_pos_embedding,
        use_siglip_qk_2d_rope=args.use_siglip_qk_2d_rope,
        siglip_rope_base=args.siglip_rope_base,
        siglip_rope_dim=args.siglip_rope_dim,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
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
    """固定随机种子。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 里的 Tensor 移到训练设备上。"""

    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output


def build_dataloader(
    config: Stage2TrainConfig,
) -> tuple[DataLoader, DataLoader | None, VLMDataCollator]:
    """构造 Stage 2 的 Dataset、Collator 和 DataLoader。"""

    train_dataset = LlavaInstructDataset(
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
        val_dataset = LlavaInstructDataset(
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


def build_model(config: Stage2TrainConfig, collator: VLMDataCollator) -> QwenSiglipVLM:
    """构造 Stage 2 模型，并加载 Stage 1 projector。"""

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

    projector_path = Path(config.stage1_projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"Stage 1 projector 不存在：{projector_path}")

    state_dict = torch.load(projector_path, map_location="cpu")
    model.projector.load_state_dict(state_dict)
    print(f"[init] 已加载 Stage 1 projector: {projector_path}")

    if config.stage1_vision_path:
        vision_path = Path(config.stage1_vision_path)
        if not vision_path.is_file():
            raise FileNotFoundError(f"Stage 1 vision 权重不存在：{vision_path}")
        vision_state = torch.load(vision_path, map_location="cpu")
        missing, unexpected = model.vision_encoder.load_state_dict(
            vision_state,
            strict=False,
        )
        if unexpected:
            raise RuntimeError(f"加载 Stage 1 vision 权重时出现 unexpected keys: {unexpected}")
        print(
            f"[init] 已加载 Stage 1 vision trainable 权重: {vision_path} "
            f"(missing={len(missing)})"
        )

    if config.use_lora:
        apply_lora(model, config)

    return model


def apply_lora(model: QwenSiglipVLM, config: Stage2TrainConfig) -> None:
    """给 Qwen language model 添加 LoRA adapter。"""

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "你开启了 --use-lora，但当前环境没有安装 peft。"
            "可以先安装 peft，或去掉 --use-lora 跑 projector-only debug run。"
        ) from exc

    target_modules = [
        item.strip() for item in config.lora_target_modules.split(",") if item.strip()
    ]
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    print("[init] Qwen language model 已添加 LoRA。")


def save_checkpoint(
    model: QwenSiglipVLM,
    optimizer: torch.optim.Optimizer,
    config: Stage2TrainConfig,
    step: int,
    output_dir: Path,
) -> None:
    """保存 Stage 2 checkpoint。"""

    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.projector.state_dict(), ckpt_dir / "projector.pt")
    if config.use_lora:
        model.language_model.save_pretrained(ckpt_dir / "lora_adapter")

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

    fieldnames = ["step", "train_loss", "val_loss", "grad_norm", "elapsed"]
    file_exists = metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_loss_curve(metrics_path: Path, output_path: Path) -> None:
    """根据 metrics.csv 画 Stage 2 loss 曲线。"""

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
    plt.title("Stage 2 Instruction Tuning Loss")
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


def train(config: Stage2TrainConfig) -> None:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}")
    print(f"[init] output_dir: {output_dir}")
    print(f"[init] use_lora: {config.use_lora}")

    dataloader, val_dataloader, collator = build_dataloader(config)
    print(f"[data] batches per epoch: {len(dataloader)}")
    if val_dataloader is not None:
        print(f"[data] val batches: {len(val_dataloader)}")

    model = build_model(config, collator)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有可训练参数，请检查冻结策略或 LoRA 配置。")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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

            (loss / config.gradient_accumulation_steps).backward()
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

                if global_step % config.log_every == 0:
                    avg_loss = running_loss / (
                        config.log_every * config.gradient_accumulation_steps
                    )
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

        if micro_step % config.gradient_accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(
                "[log] "
                f"step={global_step} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"visual_tokens={outputs['visual_token_count']}"
            )
            append_metrics(
                metrics_path,
                {
                    "step": global_step,
                    "train_loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm),
                    "elapsed": time.time() - start_time,
                },
            )
            plot_loss_curve(metrics_path, loss_curve_path)

    save_checkpoint(model, optimizer, config, global_step, output_dir)
    plot_loss_curve(metrics_path, loss_curve_path)
    print("[done] 训练完成。")


if __name__ == "__main__":
    train(parse_args())

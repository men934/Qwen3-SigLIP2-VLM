"""Stage 3 document/OCR/chart domain SFT.

Default initialization:
    - 加载 Stage 2 projector。
    - 加载 Stage 2 LoRA adapter，并保持 LoRA 可训练。
    - SigLIP2 继续冻结。
    - Qwen3 主干继续冻结。
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
from vlm.data.domain_mix_dataset import DomainMixDataset
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


@dataclass
class Stage3TrainConfig:
    """Stage 3 训练配置。"""

    qwen_path: str = "/root/autodl-tmp/hf_models/Qwen3-1.7B"
    siglip_path: str = "/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384"
    annotation_path: str = "/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/train.json"
    val_annotation_path: str | None = "/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/val.json"
    stage2_projector_path: str = (
        "/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/projector.pt"
    )
    stage2_lora_path: str = (
        "/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/lora_adapter"
    )
    output_dir: str = "/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix"

    image_size: int = 384
    dynamic_resolution: bool = False
    min_pixels: int = 384 * 384
    max_pixels: int = 672 * 672
    max_length: int = 1024
    max_samples: int | None = None
    max_steps: int | None = None
    num_epochs: int = 1

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 8.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False
    siglip_rope_base: float = 10000.0
    siglip_rope_dim: int | None = None

    num_workers: int = 4
    seed: int = 42
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500
    eval_batches: int = 100

    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    verify_images: bool = False


def optional_int(value: str) -> int | None:
    """argparse 用的小工具：允许 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def parse_args() -> Stage3TrainConfig:
    parser = argparse.ArgumentParser(description="Stage 3 文档/OCR/图表垂域微调")
    parser.add_argument("--qwen-path", default=Stage3TrainConfig.qwen_path)
    parser.add_argument("--siglip-path", default=Stage3TrainConfig.siglip_path)
    parser.add_argument("--annotation-path", default=Stage3TrainConfig.annotation_path)
    parser.add_argument("--val-annotation-path", default=Stage3TrainConfig.val_annotation_path)
    parser.add_argument("--stage2-projector-path", default=Stage3TrainConfig.stage2_projector_path)
    parser.add_argument("--stage2-lora-path", default=Stage3TrainConfig.stage2_lora_path)
    parser.add_argument("--output-dir", default=Stage3TrainConfig.output_dir)

    parser.add_argument("--image-size", type=int, default=Stage3TrainConfig.image_size)
    parser.add_argument("--dynamic-resolution", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=Stage3TrainConfig.min_pixels)
    parser.add_argument("--max-pixels", type=int, default=Stage3TrainConfig.max_pixels)
    parser.add_argument("--max-length", type=int, default=Stage3TrainConfig.max_length)
    parser.add_argument("--max-samples", type=optional_int, default=Stage3TrainConfig.max_samples)
    parser.add_argument("--max-steps", type=optional_int, default=Stage3TrainConfig.max_steps)
    parser.add_argument("--num-epochs", type=int, default=Stage3TrainConfig.num_epochs)

    parser.add_argument("--batch-size", type=int, default=Stage3TrainConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=Stage3TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=Stage3TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage3TrainConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Stage3TrainConfig.max_grad_norm)
    parser.add_argument("--no-siglip-abs-pos-embedding", action="store_true")
    parser.add_argument("--use-siglip-qk-2d-rope", action="store_true")
    parser.add_argument("--siglip-rope-base", type=float, default=Stage3TrainConfig.siglip_rope_base)
    parser.add_argument("--siglip-rope-dim", type=optional_int, default=Stage3TrainConfig.siglip_rope_dim)

    parser.add_argument("--num-workers", type=int, default=Stage3TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=Stage3TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=Stage3TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=Stage3TrainConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=Stage3TrainConfig.eval_every)
    parser.add_argument("--eval-batches", type=int, default=Stage3TrainConfig.eval_batches)
    parser.add_argument("--torch-dtype", default=Stage3TrainConfig.torch_dtype)
    parser.add_argument("--device", default=Stage3TrainConfig.device)
    parser.add_argument("--verify-images", action="store_true")

    args = parser.parse_args()
    return Stage3TrainConfig(
        qwen_path=args.qwen_path,
        siglip_path=args.siglip_path,
        annotation_path=args.annotation_path,
        val_annotation_path=args.val_annotation_path,
        stage2_projector_path=args.stage2_projector_path,
        stage2_lora_path=args.stage2_lora_path,
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
    config: Stage3TrainConfig,
) -> tuple[DataLoader, DataLoader | None, VLMDataCollator]:
    """构造 Stage 3 Dataset、Collator 和 DataLoader。"""

    train_dataset = DomainMixDataset(
        annotation_path=config.annotation_path,
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
        val_dataset = DomainMixDataset(
            annotation_path=config.val_annotation_path,
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


def build_model(config: Stage3TrainConfig, collator: VLMDataCollator) -> QwenSiglipVLM:
    """构造 Stage 3 模型，并加载 Stage 2 projector + LoRA。"""

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

    projector_path = Path(config.stage2_projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"Stage 2 projector 不存在：{projector_path}")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    print(f"[init] 已加载 Stage 2 projector: {projector_path}")

    lora_path = Path(config.stage2_lora_path)
    if not lora_path.is_dir():
        raise FileNotFoundError(f"Stage 2 LoRA adapter 不存在：{lora_path}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Stage 3 需要 peft 来加载 Stage 2 LoRA adapter。") from exc

    # is_trainable=True 表示继续训练这个 adapter，而不是只用于推理。
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        str(lora_path),
        is_trainable=True,
    )
    print(f"[init] 已加载并解冻 Stage 2 LoRA adapter: {lora_path}")
    return model


def save_checkpoint(
    model: QwenSiglipVLM,
    optimizer: torch.optim.Optimizer,
    config: Stage3TrainConfig,
    step: int,
    output_dir: Path,
) -> None:
    """保存 Stage 3 checkpoint。"""

    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), ckpt_dir / "projector.pt")
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


def save_best_checkpoint(
    model: QwenSiglipVLM,
    optimizer: torch.optim.Optimizer,
    config: Stage3TrainConfig,
    step: int,
    val_loss: float,
    output_dir: Path,
) -> None:
    """Save the best validation checkpoint to ``best``."""

    ckpt_dir = output_dir / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), ckpt_dir / "projector.pt")
    model.language_model.save_pretrained(ckpt_dir / "lora_adapter")
    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        ckpt_dir / "trainer_state.pt",
    )
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = asdict(config)
        payload["best_step"] = step
        payload["best_val_loss"] = val_loss
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] best checkpoint 已更新到 {ckpt_dir} (step={step}, val_loss={val_loss:.4f})")


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
    """根据 metrics.csv 绘制 Stage 3 loss 曲线。"""

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
    plt.title("Stage 3 Domain Tuning Loss")
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
        if outputs["loss"] is None:
            raise RuntimeError("验证时模型没有返回 loss。")
        total_loss += float(outputs["loss"].detach().cpu())
        total_batches += 1
    if was_training:
        model.train()
    if total_batches == 0:
        raise ValueError("验证 dataloader 没有产生 batch。")
    return total_loss / total_batches


def train(config: Stage3TrainConfig) -> None:
    """执行 Stage 3 训练。"""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}")
    print(f"[init] output_dir: {output_dir}")
    print(f"[init] stage2_projector_path: {config.stage2_projector_path}")
    print(f"[init] stage2_lora_path: {config.stage2_lora_path}")

    dataloader, val_dataloader, collator = build_dataloader(config)
    print(f"[data] train batches per epoch: {len(dataloader)}")
    if val_dataloader is not None:
        print(f"[data] val batches: {len(val_dataloader)}")

    model = build_model(config, collator)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有可训练参数，请检查 LoRA/projector 是否正确加载。")

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
    best_val_loss = float("inf")

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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    config.max_grad_norm,
                )
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
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_best_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                config=config,
                                step=global_step,
                                val_loss=val_loss,
                                output_dir=output_dir,
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
    print("[done] Stage 3 训练完成。")


if __name__ == "__main__":
    train(parse_args())

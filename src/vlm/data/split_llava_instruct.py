"""Create a fixed train/val split for LLaVA-Instruct-150K.

    - train.json：真正参与训练的样本；
    - val.json：不参与训练，只用于周期性计算 val loss。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="切分 LLaVA-Instruct-150K train/val")
    parser.add_argument(
        "--annotation-path",
        default="/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/llava_instruct_150k.json",
        help="原始 LLaVA-Instruct JSON 路径。",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/splits",
        help="切分后 JSON 的保存目录。",
    )
    parser.add_argument("--val-size", type=int, default=5000, help="验证集样本数。")
    parser.add_argument("--seed", type=int, default=42, help="随机切分种子。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 train.json / val.json。",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    """读取原始标注文件，并检查顶层结构。"""

    if not path.is_file():
        raise FileNotFoundError(f"标注文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"标注文件顶层应该是 list，当前是 {type(data).__name__}。")
    if not data:
        raise ValueError("标注文件为空。")
    return data


def save_json(path: Path, data: list[dict[str, Any]], overwrite: bool) -> None:
    """保存 JSON，并避免无意覆盖已有切分。"""

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} 已存在；如需覆盖，请加 --overwrite。")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main() -> None:
    """执行固定随机切分。"""

    args = parse_args()
    annotation_path = Path(args.annotation_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_json(annotation_path)
    if args.val_size <= 0:
        raise ValueError(f"val-size 必须为正数，当前为 {args.val_size}。")
    if args.val_size >= len(samples):
        raise ValueError(
            f"val-size 必须小于总样本数，总样本数为 {len(samples)}，"
            f"当前 val-size 为 {args.val_size}。"
        )

    indices = list(range(len(samples)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    val_indices = set(indices[: args.val_size])
    train_samples = [sample for i, sample in enumerate(samples) if i not in val_indices]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_indices]

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    meta_path = output_dir / "split_meta.json"

    save_json(train_path, train_samples, overwrite=args.overwrite)
    save_json(val_path, val_samples, overwrite=args.overwrite)

    meta = {
        "source": str(annotation_path),
        "seed": args.seed,
        "total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[split] 完成 LLaVA-Instruct 切分")
    print(f"[split] source: {annotation_path}")
    print(f"[split] train:  {train_path} ({len(train_samples)} samples)")
    print(f"[split] val:    {val_path} ({len(val_samples)} samples)")
    print(f"[split] meta:   {meta_path}")


if __name__ == "__main__":
    main()

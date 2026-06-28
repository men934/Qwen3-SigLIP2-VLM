"""为 LLaVA-Pretrain 生成固定 train/val 划分。

Stage 1 的 LLaVA-Pretrain 原始 JSON 没有自带 train/val/test 划分。
为了观察 projector 是否真的泛化，而不是只看 train loss，我们需要固定随机划分出
一个验证集。

推荐用法：

    python tools/split_llava_pretrain.py \
      --input /root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
      --output-dir /root/autodl-tmp/hf_datasets/LLaVA-Pretrain/splits \
      --val-size 5000 \
      --seed 42

输出：

    train.json
    val.json
    split_info.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分 LLaVA-Pretrain train/val")
    parser.add_argument(
        "--input",
        required=True,
        help="原始 blip_laion_cc_sbu_558k.json 路径。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，会写入 train.json / val.json / split_info.json。",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="验证集样本数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。固定后每次划分一致。",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"输入 JSON 顶层必须是 list，当前为 {type(data).__name__}。")
    return data


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.val_size <= 0:
        raise ValueError(f"val-size 必须为正数，当前为 {args.val_size}。")

    data = load_json(input_path)
    total = len(data)
    if args.val_size >= total:
        raise ValueError(f"val-size={args.val_size} 不能大于等于总样本数 {total}。")

    indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    val_indices = set(indices[: args.val_size])
    train = []
    val = []
    for i, sample in enumerate(data):
        if i in val_indices:
            val.append(sample)
        else:
            train.append(sample)

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    info_path = output_dir / "split_info.json"

    dump_json(train_path, train)
    dump_json(val_path, val)
    dump_json(
        info_path,
        {
            "input": str(input_path),
            "seed": args.seed,
            "total": total,
            "train_size": len(train),
            "val_size": len(val),
            "train_path": str(train_path),
            "val_path": str(val_path),
        },
    )

    print("划分完成：")
    print(f"  total: {total}")
    print(f"  train: {len(train)} -> {train_path}")
    print(f"  val:   {len(val)} -> {val_path}")
    print(f"  info:  {info_path}")


if __name__ == "__main__":
    main()

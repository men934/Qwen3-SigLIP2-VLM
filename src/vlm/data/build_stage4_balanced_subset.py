"""构建 Stage 4 电商 SFT 的均衡子集。

为什么需要这个脚本？
    ABO metadata 里不同任务的样本数不完全均衡，例如 style 任务明显少于
    brand/type/title/summary。如果直接取 train.json 前 100k，模型会更多看到大类
    任务，小类任务提升可能不明显。

这个脚本按 ``task`` 分桶，然后从每个任务里随机抽取接近相同数量的样本，生成一个
新的 JSON。这样 100k 训练更适合做阶段性实验：

    - 训练时间比 717k 全量短很多。
    - 每个任务都有足够曝光。
    - 评估结果更容易解释。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="构建 Stage 4 电商均衡 SFT 子集")
    parser.add_argument(
        "--input-path",
        default="/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train.json",
        help="原始 Stage 4 SFT train.json。",
    )
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train_100k_balanced.json",
        help="输出的均衡子集 JSON。",
    )
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    """读取 JSON list。"""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层应该是 list，实际是 {type(data).__name__}。")
    return data


def build_balanced_subset(
    samples: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """按 task 均衡抽样。

    如果某个任务样本不足，就先全部取出，再把剩余额度分配给其它任务。本项目的 ABO
    数据里每个任务都超过 16k，所以 100k 六任务均衡抽样不会触发不足逻辑。
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples 必须为正数，当前为 {num_samples}。")

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[sample.get("task", "")].append(sample)

    if not buckets:
        raise ValueError("没有可抽样的 task bucket。")

    tasks = sorted(buckets)
    base_quota = num_samples // len(tasks)
    remainder = num_samples % len(tasks)

    selected: list[dict[str, Any]] = []
    spare_capacity: list[str] = []
    deficit = 0
    for index, task in enumerate(tasks):
        quota = base_quota + (1 if index < remainder else 0)
        bucket = list(buckets[task])
        rng.shuffle(bucket)
        if len(bucket) >= quota:
            selected.extend(bucket[:quota])
            if len(bucket) > quota:
                spare_capacity.append(task)
        else:
            selected.extend(bucket)
            deficit += quota - len(bucket)

    # 如果有任务样本不足，把缺口从仍有剩余样本的任务中补齐。
    if deficit > 0:
        already_selected_ids = {id(sample) for sample in selected}
        leftovers = []
        for task in spare_capacity:
            for sample in buckets[task]:
                if id(sample) not in already_selected_ids:
                    leftovers.append(sample)
        rng.shuffle(leftovers)
        selected.extend(leftovers[:deficit])

    if len(selected) > num_samples:
        selected = selected[:num_samples]
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_json(input_path)
    subset = build_balanced_subset(
        samples=samples,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"[done] input: {input_path}")
    print(f"[done] output: {output_path}")
    print(f"[done] total: {len(subset)}")
    print("[done] task distribution:")
    for task, count in Counter(sample.get("task", "") for sample in subset).most_common():
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()

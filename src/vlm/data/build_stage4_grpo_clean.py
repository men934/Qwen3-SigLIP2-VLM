"""Build a cleaner Stage 4 GRPO dataset from the SFT splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlm.data.ecommerce_reward import answer_aliases, reliable_for_grpo


DEFAULT_TASK_QUOTAS = {
    "product_type_qa": 22000,
    "product_color_qa": 22000,
    "product_brand_qa": 14000,
    "product_attribute_summary": 9000,
    "product_title_generation": 9000,
    "product_style_qa": 4000,
}


@dataclass(frozen=True)
class CleanGRPOConfig:
    train_sft_path: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/train_100k_balanced.json"
    val_sft_path: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/val.json"
    output_root: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo_visual_v3"
    seed: int = 3407
    top_type_count: int = 96
    top_brand_count: int = 48
    task_quotas: str = json.dumps(DEFAULT_TASK_QUOTAS, sort_keys=True)
    val_per_task: int = 300


def parse_args() -> CleanGRPOConfig:
    parser = argparse.ArgumentParser(description="Build clean Stage4 GRPO data")
    parser.add_argument("--train-sft-path", default=CleanGRPOConfig.train_sft_path)
    parser.add_argument("--val-sft-path", default=CleanGRPOConfig.val_sft_path)
    parser.add_argument("--output-root", default=CleanGRPOConfig.output_root)
    parser.add_argument("--seed", type=int, default=CleanGRPOConfig.seed)
    parser.add_argument("--top-type-count", type=int, default=CleanGRPOConfig.top_type_count)
    parser.add_argument("--top-brand-count", type=int, default=CleanGRPOConfig.top_brand_count)
    parser.add_argument("--task-quotas", default=CleanGRPOConfig.task_quotas)
    parser.add_argument("--val-per-task", type=int, default=CleanGRPOConfig.val_per_task)
    args = parser.parse_args()
    return CleanGRPOConfig(
        train_sft_path=args.train_sft_path,
        val_sft_path=args.val_sft_path,
        output_root=args.output_root,
        seed=args.seed,
        top_type_count=args.top_type_count,
        top_brand_count=args.top_brand_count,
        task_quotas=args.task_quotas,
        val_per_task=args.val_per_task,
    )


def load_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层必须是 JSON list。")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_task_quotas(raw: str) -> dict[str, int]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("--task-quotas 必须是 JSON object。")
    quotas: dict[str, int] = {}
    for key, value in data.items():
        quota = int(value)
        if quota <= 0:
            raise ValueError(f"任务 {key} 的 quota 必须为正数。")
        quotas[str(key)] = quota
    return quotas


def answer_of(sample: dict[str, Any]) -> str:
    answers = sample.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0]).strip()
    messages = sample.get("messages") or []
    if len(messages) >= 2:
        return str(messages[-1].get("content", "")).strip()
    return ""


def prompt_of(sample: dict[str, Any]) -> dict[str, str]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"样本 {sample.get('id', '<unknown>')} 缺少 messages。")
    user = messages[0]
    if user.get("role") != "user":
        raise ValueError(f"样本 {sample.get('id', '<unknown>')} 第一条消息必须是 user。")
    return {"role": "user", "content": str(user.get("content", ""))}


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    task = str(sample.get("task", ""))
    answer = answer_of(sample)
    aliases = answer_aliases(answer, task)
    return {
        "id": f"{sample.get('id', 'sample')}_clean_grpo",
        "image_path": sample["image_path"],
        "messages": [prompt_of(sample)],
        "source": sample.get("source", "abo"),
        "task": task,
        "answer": answer,
        "reward": {
            "type": "visual_v3",
            "answers": aliases,
            "target_field": task,
            "normalization": "canonical_alias_token_f1",
        },
    }


def top_values(samples: list[dict[str, Any]], task: str, limit: int) -> set[str]:
    counter = Counter(answer_of(sample) for sample in samples if sample.get("task") == task)
    return {value for value, _ in counter.most_common(limit) if value}


def filter_samples(
    samples: list[dict[str, Any]],
    *,
    top_types: set[str],
    top_brands: set[str],
) -> list[dict[str, Any]]:
    kept = []
    for sample in samples:
        if reliable_for_grpo(sample, top_types=top_types, top_brands=top_brands):
            kept.append(sample)
    return kept


def balanced_cap(samples: list[dict[str, Any]], quotas: dict[str, int], seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_task[str(sample.get("task", ""))].append(sample)

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for task, task_samples in sorted(by_task.items()):
        rng.shuffle(task_samples)
        quota = quotas.get(task, len(task_samples))
        output.extend(task_samples[:quota])
    rng.shuffle(output)
    return output


def cap_val(samples: list[dict[str, Any]], per_task: int, seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_task[str(sample.get("task", ""))].append(sample)

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for _, task_samples in sorted(by_task.items()):
        rng.shuffle(task_samples)
        output.extend(task_samples[:per_task])
    rng.shuffle(output)
    return output


def task_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(sample.get("task", "")) for sample in samples).items()))


def main() -> None:
    config = parse_args()
    train_sft = load_json(config.train_sft_path)
    val_sft = load_json(config.val_sft_path)

    top_types = top_values(train_sft, "product_type_qa", config.top_type_count)
    top_brands = top_values(train_sft, "product_brand_qa", config.top_brand_count)
    quotas = parse_task_quotas(config.task_quotas)

    train_filtered = filter_samples(train_sft, top_types=top_types, top_brands=top_brands)
    val_filtered = filter_samples(val_sft, top_types=top_types, top_brands=top_brands)

    train_balanced = balanced_cap(train_filtered, quotas, seed=config.seed)
    val_balanced = cap_val(val_filtered, per_task=config.val_per_task, seed=config.seed + 1)

    train_grpo = [convert_sample(sample) for sample in train_balanced]
    val_grpo = [convert_sample(sample) for sample in val_balanced]

    output_root = Path(config.output_root)
    write_json(output_root / "train.json", train_grpo)
    write_json(output_root / "val.json", val_grpo)
    write_json(
        output_root / "manifest.json",
        {
            "config": asdict(config),
            "top_types": sorted(top_types),
            "top_brands": sorted(top_brands),
            "raw_counts": {
                "train": task_counts(train_sft),
                "val": task_counts(val_sft),
            },
            "filtered_counts": {
                "train": task_counts(train_filtered),
                "val": task_counts(val_filtered),
            },
            "output_counts": {
                "train": task_counts(train_grpo),
                "val": task_counts(val_grpo),
            },
        },
    )

    print(f"[done] clean GRPO train={len(train_grpo)} val={len(val_grpo)}")
    print(f"[done] train counts={task_counts(train_grpo)}")
    print(f"[done] val counts={task_counts(val_grpo)}")
    print(f"[done] output_root={output_root}")


if __name__ == "__main__":
    main()

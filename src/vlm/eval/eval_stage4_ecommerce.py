"""Stage 4 e-commerce quantitative evaluation.

指标说明：
    - normalized exact match:
        先小写、去标点、合并空格，再判断预测和标准答案是否完全一致。
        适合 brand/type/color/style 这种短答案任务。

    - token F1:
        按 token overlap 算 F1。适合标题生成、属性摘要这类“没有唯一标准表述”的任务。

Use ``--max-samples none`` to evaluate the full test split.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vlm.data.domain_mix_dataset import DomainMixDataset
from vlm.eval.eval_stage3_domain import (
    CheckpointSpec,
    build_helper,
    checkpoint_exists,
    exact_match,
    generate_one,
    load_model,
    token_f1,
)


SHORT_ANSWER_TASKS = {
    "product_brand_qa",
    "product_type_qa",
    "product_color_qa",
    "product_style_qa",
}

GENERATION_TASKS = {
    "product_title_generation",
    "product_attribute_summary",
}


def optional_int(value: str) -> int | None:
    """argparse 小工具：允许 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Stage 4 电商垂域定量评估")
    parser.add_argument("--qwen-path", default="/root/autodl-tmp/hf_models/Qwen3-1.7B")
    parser.add_argument(
        "--siglip-path",
        default="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--test-annotation-path",
        default="/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/sft/test.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/checkpoints/stage4_eval_300",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--min-pixels", type=int, default=384 * 384)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=optional_int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoints",
        default="stage3,stage4_5k",
        help=(
            "逗号分隔的 checkpoint 名称。可选："
            "stage1,stage1_dynamic,stage2,stage2_dynamic,stage3,"
            "stage4_5k,stage4_100k_balanced,stage4_grpo,stage4_grpo_20k,"
            "stage4_grpo_short_v2。"
            "默认先对比 Stage3 和 Stage4-5k。"
        ),
    )
    return parser.parse_args()


def default_checkpoint_specs() -> dict[str, CheckpointSpec]:
    """Return registered e-commerce evaluation checkpoints."""

    return {
        "stage1": CheckpointSpec(
            name="stage1_fixed_50k",
            projector_path="/root/autodl-tmp/checkpoints/stage1_align_50k/step_003000/projector.pt",
        ),
        "stage1_dynamic": CheckpointSpec(
            name="stage1_dynamic_qk_rope_50k",
            projector_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/projector.pt",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
        "stage2": CheckpointSpec(
            name="stage2_fixed_150k_r32",
            projector_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/lora_adapter",
        ),
        "stage2_dynamic": CheckpointSpec(
            name="stage2_dynamic_qk_rope_150k_r32",
            projector_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/lora_adapter",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
        "stage3": CheckpointSpec(
            name="stage3_doc_ocr_mix",
            projector_path="/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix/step_006000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix/step_006000/lora_adapter",
        ),
        "stage4_5k": CheckpointSpec(
            name="stage4_abo_sft_5k_best",
            projector_path="/root/autodl-tmp/checkpoints/stage4_abo_sft_5k/best/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage4_abo_sft_5k/best/lora_adapter",
        ),
        "stage4_100k_balanced": CheckpointSpec(
            name="stage4_abo_sft_100k_balanced_best",
            projector_path="/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage4_abo_sft_100k_balanced/best/lora_adapter",
        ),
        "stage4_grpo": CheckpointSpec(
            name="stage4_abo_grpo_best",
            projector_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo/best/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo/best/lora_adapter",
        ),
        "stage4_grpo_20k": CheckpointSpec(
            name="stage4_abo_grpo_20k_best",
            projector_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo_20k/best/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo_20k/best/lora_adapter",
        ),
        "stage4_grpo_short_v2": CheckpointSpec(
            name="stage4_abo_grpo_short_reward_v2_best",
            projector_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo_short_reward_v2/best/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage4_abo_grpo_short_reward_v2/best/lora_adapter",
        ),
    }


def select_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    """根据 --checkpoints 选择要评估的 checkpoint，并跳过缺失项。"""

    registry = default_checkpoint_specs()
    names = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    specs = []
    for name in names:
        if name not in registry:
            raise KeyError(f"未知 checkpoint 名称：{name}，可选：{sorted(registry)}")
        spec = registry[name]
        if checkpoint_exists(spec):
            specs.append(spec)
        else:
            print(f"[skip] checkpoint 缺文件，跳过：{spec.name}")
    if not specs:
        raise FileNotFoundError("没有找到任何可评估 checkpoint。")
    return specs


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """读取测试集，并用固定 seed 抽样，保证多次评估可复现。"""

    dataset = DomainMixDataset(
        annotation_path=args.test_annotation_path,
        verify_images=False,
        max_samples=None,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    if args.max_samples is not None and args.max_samples < len(samples):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(samples)), args.max_samples))
        samples = [samples[index] for index in indices]
    return samples


def score_prediction(prediction: str, sample: dict[str, Any]) -> dict[str, float]:
    """按电商任务类型计算指标。

    对短答案任务，EM 是最重要指标；对标题/摘要任务，F1 更有参考价值。
    这里仍统一输出两个指标，便于表格和图表直接对比。
    """

    references = sample.get("answers") or [sample["messages"][-1]["content"]]
    task = sample.get("task", "")
    em = exact_match(prediction, references)
    f1 = token_f1(prediction, references)
    short_answer_em = em if task in SHORT_ANSWER_TASKS else 0.0
    generation_f1 = f1 if task in GENERATION_TASKS else 0.0
    return {
        "em": em,
        "f1": f1,
        "short_answer_em": short_answer_em,
        "generation_f1": generation_f1,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合整体、短答案、生成任务和每个 task 的指标。"""

    metric_keys = ["em", "f1", "short_answer_em", "generation_f1"]

    def summarize(subset: list[dict[str, Any]]) -> dict[str, float | int]:
        if not subset:
            return {"count": 0, **{key: 0.0 for key in metric_keys}}
        output: dict[str, float | int] = {"count": len(subset)}
        for key in metric_keys:
            output[key] = sum(row["scores"][key] for row in subset) / len(subset)
        return output

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    short_rows = []
    generation_rows = []
    for row in rows:
        task = row["task"]
        by_task[task].append(row)
        if task in SHORT_ANSWER_TASKS:
            short_rows.append(row)
        if task in GENERATION_TASKS:
            generation_rows.append(row)

    return {
        "overall": summarize(rows),
        "short_answer": summarize(short_rows),
        "generation": summarize(generation_rows),
        "by_task": {key: summarize(value) for key, value in sorted(by_task.items())},
    }


def evaluate_checkpoint(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """评估单个 checkpoint。"""

    print(f"[eval] 开始评估 {spec.name}")
    helper = build_helper(args, spec)
    model = load_model(args, spec, helper, device)

    rows = []
    for index, sample in enumerate(samples):
        prediction = generate_one(
            model=model,
            helper=helper,
            sample=sample,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        scores = score_prediction(prediction, sample)
        rows.append(
            {
                "checkpoint": spec.name,
                "index": index,
                "id": sample.get("id", ""),
                "source": sample.get("source", ""),
                "task": sample.get("task", ""),
                "prompt": sample["messages"][0]["content"],
                "prediction": prediction,
                "references": sample.get("answers", []),
                "scores": scores,
            }
        )
        if (index + 1) % 20 == 0:
            print(f"[eval] {spec.name}: {index + 1}/{len(samples)}")

    metrics = {
        "checkpoint": spec.name,
        "spec": asdict(spec),
        **aggregate(rows),
    }

    del model
    del helper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, rows


def write_report(path: Path, all_metrics: list[dict[str, Any]]) -> None:
    """Write a Markdown summary report."""

    lines = ["# Stage 4 E-commerce Evaluation", ""]
    lines.append("## Overall")
    lines.append("| checkpoint | count | EM | F1 | short EM | generation F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metrics in all_metrics:
        overall = metrics["overall"]
        short_answer = metrics["short_answer"]
        generation = metrics["generation"]
        lines.append(
            f"| {metrics['checkpoint']} | {overall['count']} | "
            f"{overall['em']:.4f} | {overall['f1']:.4f} | "
            f"{short_answer['em']:.4f} | {generation['f1']:.4f} |"
        )

    lines.append("")
    lines.append("## By Task")
    for metrics in all_metrics:
        lines.append("")
        lines.append(f"### {metrics['checkpoint']}")
        lines.append("| task | count | EM | F1 |")
        lines.append("|---|---:|---:|---:|")
        for task, row in metrics["by_task"].items():
            lines.append(
                f"| {task} | {row['count']} | {row['em']:.4f} | {row['f1']:.4f} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics(output_dir: Path, all_metrics: list[dict[str, Any]]) -> None:
    """绘制整体和按任务的对比图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [item["checkpoint"] for item in all_metrics]
    x = np.arange(len(labels))

    overall_items = [
        ("overall", "em", "Overall EM"),
        ("overall", "f1", "Overall F1"),
        ("short_answer", "em", "Short EM"),
        ("generation", "f1", "Generation F1"),
    ]
    width = 0.18
    plt.figure(figsize=(12, 5.8))
    for offset, (section, key, label) in enumerate(overall_items):
        values = [item[section][key] for item in all_metrics]
        bars = plt.bar(x + (offset - 1.5) * width, values, width=width, label=label)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.xticks(x, labels, rotation=15, ha="right")
    max_value = max(item["overall"]["f1"] for item in all_metrics)
    plt.ylim(0, max(0.25, max_value * 1.35))
    plt.ylabel("Score")
    plt.title("Stage 4 E-commerce Evaluation")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overall_metrics_comparison.png", dpi=180)
    plt.close()

    tasks = sorted({task for item in all_metrics for task in item["by_task"]})
    if not tasks:
        return
    width = min(0.22, 0.8 / max(1, len(all_metrics)))
    start = -width * (len(all_metrics) - 1) / 2
    x = np.arange(len(tasks))
    plt.figure(figsize=(13, 6.2))
    for index, item in enumerate(all_metrics):
        values = [item["by_task"].get(task, {}).get("f1", 0.0) for task in tasks]
        bars = plt.bar(
            x + start + index * width,
            values,
            width=width,
            label=item["checkpoint"],
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    plt.xticks(x, tasks, rotation=20, ha="right")
    max_value = max(
        item["by_task"].get(task, {}).get("f1", 0.0)
        for item in all_metrics
        for task in tasks
    )
    plt.ylim(0, max(0.25, max_value * 1.3))
    plt.ylabel("Token F1")
    plt.title("Stage 4 E-commerce Evaluation: F1 by Task")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "by_task_f1_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args)
    specs = select_checkpoints(args)
    print(f"[data] test samples: {len(samples)}")
    print("[eval] checkpoints:", ", ".join(spec.name for spec in specs))

    all_metrics = []
    all_predictions = []
    for spec in specs:
        metrics, rows = evaluate_checkpoint(args, spec, samples, device)
        all_metrics.append(metrics)
        all_predictions.extend(rows)

        # 每评估完一个模型就落盘，避免长评估被打断后结果全丢。
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        write_report(output_dir / "report.md", all_metrics)
        plot_metrics(output_dir, all_metrics)

    print(f"[done] metrics -> {output_dir / 'metrics.json'}")
    print(f"[done] predictions -> {output_dir / 'predictions.json'}")
    print(f"[done] report -> {output_dir / 'report.md'}")
    print(f"[done] overall plot -> {output_dir / 'overall_metrics_comparison.png'}")
    print(f"[done] by-task plot -> {output_dir / 'by_task_f1_comparison.png'}")


if __name__ == "__main__":
    main()

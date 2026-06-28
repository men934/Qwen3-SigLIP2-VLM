"""绘制 Stage 3 定量评估对比图。

``eval_stage3_domain.py`` 会输出结构化的 ``metrics.json``，但只看 JSON/Markdown
不够直观。这个脚本负责把几个关键指标画成图片，方便放进 README、汇报 PPT 或面试
项目说明里。

默认生成两张图：
    1. overall_metrics_comparison.png
       对比不同 checkpoint 的整体 EM / F1 / Chart relaxed accuracy。

    2. by_source_f1_comparison.png
       对比不同 checkpoint 在各数据源上的 F1。这里默认排除 CORD，因为当前 300 条
       子集里 CORD 只有 3 条，而且 JSON 任务需要单独拉长生成长度后再评估。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHECKPOINT_LABELS = {
    "stage1_fixed_50k": "Stage1 Fixed",
    "stage1_dynamic_qk_rope_50k": "Stage1 Dynamic",
    "stage2_fixed_150k_r32": "Stage2 Fixed",
    "stage2_dynamic_qk_rope_150k_r32": "Stage2 Dynamic",
    "stage3_step_006000": "Stage3",
}

SOURCE_LABELS = {
    "chartqa": "ChartQA",
    "docvqa": "DocVQA",
    "infographic_vqa": "InfoVQA",
    "textvqa": "TextVQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 Stage 3 评估对比图")
    parser.add_argument(
        "--metrics-path",
        default="/root/autodl-tmp/checkpoints/stage3_eval_300/metrics.json",
        help="eval_stage3_domain.py 输出的 metrics.json。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="图片输出目录。默认和 metrics.json 在同一个目录。",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=["cord_v2"],
        help="不绘制的数据源。默认排除 cord_v2，可重复传入。",
    )
    return parser.parse_args()


def load_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    """读取 metrics.json。"""

    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"metrics.json 不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("metrics.json 顶层应该是 list。")
    return data


def checkpoint_label(name: str) -> str:
    """把 checkpoint 内部名转换成适合图表展示的短标签。"""

    if name in CHECKPOINT_LABELS:
        return CHECKPOINT_LABELS[name]
    if name.startswith("stage3_"):
        return "Stage3"
    return name


def source_label(name: str) -> str:
    """把数据源内部名转换成图表展示标签。"""

    return SOURCE_LABELS.get(name, name)


def plot_overall_metrics(metrics: list[dict[str, Any]], output_path: Path) -> None:
    """绘制整体 EM / F1 / Chart relaxed accuracy 对比。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoint_names = [item["checkpoint"] for item in metrics]
    labels = [checkpoint_label(name) for name in checkpoint_names]
    metric_items = [
        ("em", "EM"),
        ("f1", "F1"),
        ("relaxed_acc", "Chart Relaxed"),
    ]

    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(11, 5.8))
    for offset, (metric_key, metric_label) in enumerate(metric_items):
        values = [item["overall"][metric_key] for item in metrics]
        bars = plt.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=metric_label,
        )
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylim(0, max(0.3, max(item["overall"]["f1"] for item in metrics) * 1.25))
    plt.ylabel("Score")
    plt.title("Stage 3 Domain Evaluation: Overall Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_by_source_f1(
    metrics: list[dict[str, Any]],
    output_path: Path,
    exclude_sources: set[str],
) -> None:
    """绘制各数据源 F1 对比图，默认排除 CORD。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    all_sources = sorted(
        {
            source
            for item in metrics
            for source in item.get("by_source", {})
            if source not in exclude_sources
        }
    )
    if not all_sources:
        raise ValueError("没有可绘制的数据源，请检查 exclude_source 参数。")

    checkpoint_labels = [checkpoint_label(item["checkpoint"]) for item in metrics]
    x = np.arange(len(all_sources))
    width = min(0.16, 0.8 / max(1, len(metrics)))
    start = -width * (len(metrics) - 1) / 2

    plt.figure(figsize=(12, 6))
    for index, item in enumerate(metrics):
        values = [
            item.get("by_source", {}).get(source, {}).get("f1", 0.0)
            for source in all_sources
        ]
        bars = plt.bar(
            x + start + index * width,
            values,
            width=width,
            label=checkpoint_labels[index],
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

    plt.xticks(x, [source_label(source) for source in all_sources])
    max_value = max(
        item.get("by_source", {}).get(source, {}).get("f1", 0.0)
        for item in metrics
        for source in all_sources
    )
    plt.ylim(0, max(0.25, max_value * 1.25))
    plt.ylabel("Token F1")
    plt.title("Stage 3 Domain Evaluation: F1 by Source")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    overall_path = output_dir / "overall_metrics_comparison.png"
    by_source_path = output_dir / "by_source_f1_comparison.png"

    plot_overall_metrics(metrics, overall_path)
    plot_by_source_f1(metrics, by_source_path, set(args.exclude_source))

    print(f"[done] overall plot -> {overall_path}")
    print(f"[done] by-source plot -> {by_source_path}")


if __name__ == "__main__":
    main()

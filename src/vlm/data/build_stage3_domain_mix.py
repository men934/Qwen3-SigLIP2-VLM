"""Convert Stage 3 domain datasets into the unified JSON format.

Pipeline:
    1. 从 parquet 读取样本。
    2. 把图片 bytes 写成普通图片文件。
    3. 把各种字段统一成 ``messages`` 格式。
    4. 生成 train/val/test 三个 JSON。

输出样本格式：
    {
      "id": "docvqa_train_00000001",
      "image_path": "/root/autodl-tmp/.../images/docvqa/train/xxx.png",
      "messages": [
        {"role": "user", "content": "<image>\\nAnswer the question ..."},
        {"role": "assistant", "content": "answer"}
      ],
      "source": "docvqa",
      "task": "document_qa",
      "answers": ["answer", "..."],
      "eval": {"metric": "vqa"}
    }
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq


IMAGE_TOKEN = "<image>"


@dataclass(frozen=True)
class BuildConfig:
    """Stage 3 数据构建配置。"""

    domain_root: str = "/root/autodl-tmp/hf_datasets/domain_mix"
    output_root: str = "/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix"
    seed: int = 42
    batch_size: int = 512
    val_ratio_from_answered_validation: float = 0.5
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    max_per_source: int | None = None
    overwrite_images: bool = False


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="构建 Stage 3 domain_mix 统一数据")
    parser.add_argument("--domain-root", default=BuildConfig.domain_root)
    parser.add_argument("--output-root", default=BuildConfig.output_root)
    parser.add_argument("--seed", type=int, default=BuildConfig.seed)
    parser.add_argument("--batch-size", type=int, default=BuildConfig.batch_size)
    parser.add_argument(
        "--val-ratio-from-answered-validation",
        type=float,
        default=BuildConfig.val_ratio_from_answered_validation,
        help="DocVQA/TextVQA 等官方 test 无答案时，把 validation 按这个比例切成 val/test。",
    )
    parser.add_argument("--max-train-samples", type=optional_int, default=None)
    parser.add_argument("--max-val-samples", type=optional_int, default=None)
    parser.add_argument("--max-test-samples", type=optional_int, default=None)
    parser.add_argument("--max-per-source", type=optional_int, default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()
    return BuildConfig(
        domain_root=args.domain_root,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        val_ratio_from_answered_validation=args.val_ratio_from_answered_validation,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_per_source=args.max_per_source,
        overwrite_images=args.overwrite_images,
    )


def optional_int(value: str) -> int | None:
    """argparse 小工具：允许传入 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def sorted_parquets(pattern: str) -> list[Path]:
    """按文件名排序，保证构建结果可复现。"""

    return sorted(Path().glob(pattern))


def iter_parquet_rows(paths: Iterable[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    """流式读取 parquet，避免一次性把全量数据放进内存。"""

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def infer_image_suffix(image_obj: dict[str, Any], default: str = ".jpg") -> str:
    """根据原始 path 或 bytes 魔数推断图片后缀。"""

    raw_path = image_obj.get("path")
    if isinstance(raw_path, str) and Path(raw_path).suffix:
        suffix = Path(raw_path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return suffix

    data = image_obj.get("bytes") or b""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return ".webp"
    return default


def write_image(
    image_obj: dict[str, Any],
    image_path: Path,
    *,
    overwrite: bool,
) -> Path:
    """把 parquet 中的 image bytes 写成普通图片文件。"""

    data = image_obj.get("bytes")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("image.bytes 缺失，无法写出图片。")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not image_path.exists():
        image_path.write_bytes(data)
    return image_path


def clean_text(text: Any) -> str:
    """统一清理文本字段。"""

    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_answer_list(answers: Any) -> list[str]:
    """把不同数据集的 answer 字段统一成非空字符串列表。"""

    if answers is None:
        return []
    if isinstance(answers, str):
        candidates = [answers]
    elif isinstance(answers, list):
        candidates = answers
    else:
        candidates = [str(answers)]

    output = []
    for item in candidates:
        text = clean_text(item)
        if text:
            output.append(text)
    return output


def most_common_answer(answers: list[str]) -> str:
    """TextVQA 常有 10 个标注答案，这里用出现最多的答案作为训练目标。"""

    if not answers:
        return ""
    normalized_to_original: dict[str, str] = {}
    counter: Counter[str] = Counter()
    for answer in answers:
        key = normalize_for_count(answer)
        normalized_to_original.setdefault(key, answer)
        counter[key] += 1
    best_key, _ = counter.most_common(1)[0]
    return normalized_to_original[best_key]


def normalize_for_count(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    """生成项目统一的两轮对话。"""

    return [
        {"role": "user", "content": f"{IMAGE_TOKEN}\n{prompt}"},
        {"role": "assistant", "content": answer},
    ]


def stable_image_name(source: str, split: str, index: int, image_obj: dict[str, Any]) -> str:
    """生成稳定图片文件名，避免不同数据源 path=None 时互相覆盖。"""

    suffix = infer_image_suffix(image_obj)
    raw_path = image_obj.get("path")
    if isinstance(raw_path, str) and raw_path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(raw_path).stem)[:80]
        return f"{index:08d}_{stem}{suffix}"
    return f"{index:08d}{suffix}"


def make_sample(
    *,
    sample_id: str,
    image_path: Path,
    prompt: str,
    answer: str,
    answers: list[str],
    source: str,
    task: str,
    metric: str,
) -> dict[str, Any]:
    """构造统一样本。"""

    return {
        "id": sample_id,
        "image_path": str(image_path),
        "messages": make_messages(prompt, answer),
        "source": source,
        "task": task,
        "answers": answers,
        "eval": {"metric": metric},
    }


def build_vqa_rows(
    *,
    rows: Iterable[dict[str, Any]],
    source: str,
    split: str,
    output_root: Path,
    question_key: str,
    answer_key: str,
    task: str,
    metric: str,
    answer_selector: Callable[[list[str]], str],
    overwrite_images: bool,
    max_source_samples: int | None,
) -> list[dict[str, Any]]:
    """构建 DocVQA / InfographicVQA / TextVQA / ChartQA 这类问答样本。"""

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if max_source_samples is not None and len(samples) >= max_source_samples:
            break

        question = clean_text(row.get(question_key))
        answers = clean_answer_list(row.get(answer_key))
        answer = answer_selector(answers)
        image_obj = row.get("image")
        if not question or not answer or not isinstance(image_obj, dict):
            continue

        image_name = stable_image_name(source, split, index, image_obj)
        image_path = output_root / "images" / source / split / image_name
        write_image(image_obj, image_path, overwrite=overwrite_images)

        if task == "chart_qa":
            prompt = f"Answer the chart question with a concise answer.\nQuestion: {question}"
        elif task == "scene_text_qa":
            prompt = f"Answer the question by reading the visible text in the image.\nQuestion: {question}"
        else:
            prompt = f"Answer the document question with a concise answer.\nQuestion: {question}"

        raw_id = row.get("questionId", row.get("question_id", index))
        samples.append(
            make_sample(
                sample_id=f"{source}_{split}_{raw_id}",
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                answers=answers,
                source=source,
                task=task,
                metric=metric,
            )
        )
    return samples


def parse_cord_gt(row: dict[str, Any]) -> dict[str, Any] | None:
    """解析 CORD 的 ground_truth JSON，只保留 gt_parse。"""

    raw = row.get("ground_truth")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    gt_parse = obj.get("gt_parse")
    return gt_parse if isinstance(gt_parse, dict) else None


def build_cord_rows(
    *,
    rows: Iterable[dict[str, Any]],
    split: str,
    output_root: Path,
    overwrite_images: bool,
    max_source_samples: int | None,
) -> list[dict[str, Any]]:
    """构建 CORD 票据结构化抽取样本。"""

    source = "cord_v2"
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if max_source_samples is not None and len(samples) >= max_source_samples:
            break

        image_obj = row.get("image")
        gt_parse = parse_cord_gt(row)
        if not isinstance(image_obj, dict) or not gt_parse:
            continue

        image_name = stable_image_name(source, split, index, image_obj)
        image_path = output_root / "images" / source / split / image_name
        write_image(image_obj, image_path, overwrite=overwrite_images)
        answer = json.dumps(gt_parse, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        prompt = (
            "Extract the receipt information as compact JSON. "
            "Keep the keys and values visible in the image."
        )
        samples.append(
            make_sample(
                sample_id=f"{source}_{split}_{index:06d}",
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                answers=[answer],
                source=source,
                task="receipt_json_extraction",
                metric="json",
            )
        )
    return samples


def build_funsd_rows(
    *,
    rows: Iterable[dict[str, Any]],
    split: str,
    output_root: Path,
    overwrite_images: bool,
    max_source_samples: int | None,
) -> list[dict[str, Any]]:
    """构建 FUNSD 表单 OCR 转录样本。"""

    source = "funsd"
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if max_source_samples is not None and len(samples) >= max_source_samples:
            break

        image_obj = row.get("image")
        words = row.get("words")
        if not isinstance(image_obj, dict) or not isinstance(words, list):
            continue
        answer = clean_text(" ".join(str(word) for word in words if str(word).strip()))
        if not answer:
            continue

        image_name = stable_image_name(source, split, index, image_obj)
        image_path = output_root / "images" / source / split / image_name
        write_image(image_obj, image_path, overwrite=overwrite_images)
        prompt = "Read the text in this form and return the text in reading order."
        samples.append(
            make_sample(
                sample_id=f"{source}_{split}_{row.get('id', index)}",
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                answers=[answer],
                source=source,
                task="form_ocr",
                metric="ocr",
            )
        )
    return samples


def split_answered_validation(
    samples: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把带答案的 validation 确定性切成 val/test。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 (0, 1) 内，当前为 {val_ratio}。")
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    split_index = int(round(len(samples) * val_ratio))
    return samples[:split_index], samples[split_index:]


def cap_samples(samples: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """按全局上限截断样本。"""

    if limit is None:
        return samples
    return samples[:limit]


def write_json(path: Path, samples: list[dict[str, Any]]) -> None:
    """写 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def write_json_obj(path: Path, obj: dict[str, Any]) -> None:
    """写 JSON 对象。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_dataset(config: BuildConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """收集并转换所有支持的数据源。"""

    domain_root = Path(config.domain_root)
    output_root = Path(config.output_root)
    if not domain_root.exists():
        raise FileNotFoundError(f"domain_mix 根目录不存在：{domain_root}")

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"config": asdict(config), "sources": {}}

    def add_source(name: str, train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)
        manifest["sources"][name] = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }
        print(f"[build] {name}: train={len(train)} val={len(val)} test={len(test)}")

    # DocVQA 和 InfographicVQA 的 official test 没有答案，因此把 validation 切成 val/test。
    doc_train = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("DocVQA/DocVQA/train-*.parquet")), config.batch_size),
        source="docvqa",
        split="train",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="document_qa",
        metric="vqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    doc_eval = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("DocVQA/DocVQA/validation-*.parquet")), config.batch_size),
        source="docvqa",
        split="validation",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="document_qa",
        metric="vqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    doc_val, doc_test = split_answered_validation(
        doc_eval,
        val_ratio=config.val_ratio_from_answered_validation,
        seed=config.seed,
    )
    add_source("docvqa", doc_train, doc_val, doc_test)

    info_train = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("DocVQA/InfographicVQA/train-*.parquet")), config.batch_size),
        source="infographic_vqa",
        split="train",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="document_qa",
        metric="vqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    info_eval = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("DocVQA/InfographicVQA/validation-*.parquet")), config.batch_size),
        source="infographic_vqa",
        split="validation",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="document_qa",
        metric="vqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    info_val, info_test = split_answered_validation(
        info_eval,
        val_ratio=config.val_ratio_from_answered_validation,
        seed=config.seed + 1,
    )
    add_source("infographic_vqa", info_train, info_val, info_test)

    text_train = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("textvqa/data/train-*.parquet")), config.batch_size),
        source="textvqa",
        split="train",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="scene_text_qa",
        metric="vqa",
        answer_selector=most_common_answer,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    text_eval = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("textvqa/data/validation-*.parquet")), config.batch_size),
        source="textvqa",
        split="validation",
        output_root=output_root,
        question_key="question",
        answer_key="answers",
        task="scene_text_qa",
        metric="vqa",
        answer_selector=most_common_answer,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    text_val, text_test = split_answered_validation(
        text_eval,
        val_ratio=config.val_ratio_from_answered_validation,
        seed=config.seed + 2,
    )
    add_source("textvqa", text_train, text_val, text_test)

    chart_train = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("ChartQA/data/train-*.parquet")), config.batch_size),
        source="chartqa",
        split="train",
        output_root=output_root,
        question_key="query",
        answer_key="label",
        task="chart_qa",
        metric="chartqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    chart_val = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("ChartQA/data/val-*.parquet")), config.batch_size),
        source="chartqa",
        split="val",
        output_root=output_root,
        question_key="query",
        answer_key="label",
        task="chart_qa",
        metric="chartqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    chart_test = build_vqa_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("ChartQA/data/test-*.parquet")), config.batch_size),
        source="chartqa",
        split="test",
        output_root=output_root,
        question_key="query",
        answer_key="label",
        task="chart_qa",
        metric="chartqa",
        answer_selector=lambda answers: answers[0] if answers else "",
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    add_source("chartqa", chart_train, chart_val, chart_test)

    cord_train = build_cord_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("cord-v2/data/train-*.parquet")), config.batch_size),
        split="train",
        output_root=output_root,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    cord_val = build_cord_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("cord-v2/data/validation-*.parquet")), config.batch_size),
        split="validation",
        output_root=output_root,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    cord_test = build_cord_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("cord-v2/data/test-*.parquet")), config.batch_size),
        split="test",
        output_root=output_root,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    add_source("cord_v2", cord_train, cord_val, cord_test)

    funsd_train = build_funsd_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("funsd/data/train-*.parquet")), config.batch_size),
        split="train",
        output_root=output_root,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    funsd_test = build_funsd_rows(
        rows=iter_parquet_rows(sorted(domain_root.glob("funsd/data/test-*.parquet")), config.batch_size),
        split="test",
        output_root=output_root,
        overwrite_images=config.overwrite_images,
        max_source_samples=config.max_per_source,
    )
    add_source("funsd", funsd_train, [], funsd_test)

    manifest["sources"]["sroie_2019_text_recognition"] = {
        "train": 0,
        "val": 0,
        "test": 0,
        "note": "本地版本只看到 train.zip/test.zip 图片，未发现标签文件，已跳过。",
    }

    rng = random.Random(config.seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    train_samples = cap_samples(train_samples, config.max_train_samples)
    val_samples = cap_samples(val_samples, config.max_val_samples)
    test_samples = cap_samples(test_samples, config.max_test_samples)

    manifest["final_counts"] = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
    }
    return train_samples, val_samples, test_samples, manifest


def main() -> None:
    config = parse_args()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, test_samples, manifest = collect_dataset(config)
    write_json(output_root / "train.json", train_samples)
    write_json(output_root / "val.json", val_samples)
    write_json(output_root / "test.json", test_samples)
    write_json_obj(output_root / "manifest.json", manifest)

    print("[done] Stage 3 domain_mix 构建完成")
    print(f"[done] train={len(train_samples)} -> {output_root / 'train.json'}")
    print(f"[done] val={len(val_samples)} -> {output_root / 'val.json'}")
    print(f"[done] test={len(test_samples)} -> {output_root / 'test.json'}")
    print(f"[done] manifest -> {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

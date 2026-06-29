"""Build Stage 4 e-commerce SFT / GRPO data.

    /root/autodl-tmp/hf_datasets/stage4_ecommerce/abo/extracted

Outputs:

    output_root/
        sft/train.json
        sft/val.json
        sft/test.json
        grpo/train.json
        grpo/val.json
        grpo/test.json
        manifest.json

SFT sample:

    {
      "id": "abo_B000_color",
      "image_path": "/abs/path/to/image.jpg",
      "messages": [
        {"role": "user", "content": "<image>\\nWhat is the product color?"},
        {"role": "assistant", "content": "Black"}
      ],
      "source": "abo",
      "task": "product_color_qa",
      "answers": ["Black"],
      "eval": {"metric": "exact_match"}
    }

GRPO sample:

    {
      "id": "abo_B000_color_grpo",
      "image_path": "/abs/path/to/image.jpg",
      "messages": [
        {"role": "user", "content": "<image>\\nWhat is the product color? Answer with a short phrase."}
      ],
      "source": "abo",
      "task": "product_color_qa",
      "reward": {
        "type": "normalized_exact_match",
        "answers": ["Black"],
        "target_field": "color"
      },
      "answer": "Black"
    }

"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_TOKEN = "<image>"


@dataclass(frozen=True)
class Stage4BuildConfig:
    """Stage 4 电商数据构建配置。"""

    abo_root: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/abo/extracted"
    output_root: str = "/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo"
    seed: int = 42
    max_listings: int | None = None
    max_sft_samples: int | None = None
    max_grpo_samples: int | None = None
    val_ratio: float = 0.02
    test_ratio: float = 0.02
    min_image_size: int = 64
    require_english: bool = False


def parse_args() -> Stage4BuildConfig:
    parser = argparse.ArgumentParser(description="构建 Stage 4 电商 SFT/GRPO 数据")
    parser.add_argument("--abo-root", default=Stage4BuildConfig.abo_root)
    parser.add_argument("--output-root", default=Stage4BuildConfig.output_root)
    parser.add_argument("--seed", type=int, default=Stage4BuildConfig.seed)
    parser.add_argument("--max-listings", type=optional_int, default=None)
    parser.add_argument("--max-sft-samples", type=optional_int, default=None)
    parser.add_argument("--max-grpo-samples", type=optional_int, default=None)
    parser.add_argument("--val-ratio", type=float, default=Stage4BuildConfig.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=Stage4BuildConfig.test_ratio)
    parser.add_argument("--min-image-size", type=int, default=Stage4BuildConfig.min_image_size)
    parser.add_argument(
        "--require-english",
        action="store_true",
        help="只保留能取到英文文本字段的 listing。默认关闭，因为 product_type/color 等字段仍可用于可验证任务。",
    )
    args = parser.parse_args()
    return Stage4BuildConfig(
        abo_root=args.abo_root,
        output_root=args.output_root,
        seed=args.seed,
        max_listings=args.max_listings,
        max_sft_samples=args.max_sft_samples,
        max_grpo_samples=args.max_grpo_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_image_size=args.min_image_size,
        require_english=args.require_english,
    )


def optional_int(value: str) -> int | None:
    """argparse 小工具：允许 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def normalize_space(text: Any) -> str:
    """把任意字段清理成单行文本。"""

    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def value_from_list(
    values: Any,
    *,
    prefer_english: bool = True,
    allow_non_english: bool = True,
) -> str:
    """从 ABO 的多语言字段列表中取一个可用值。

    ABO 里很多字段是：

        [{"language_tag": "en_US", "value": "..."}, ...]

    这里优先取英文；如果没有英文且允许非英文，就退回第一条非空值。
    """

    if values is None:
        return ""
    if isinstance(values, str):
        return normalize_space(values)
    if isinstance(values, dict):
        return normalize_space(values.get("value"))
    if not isinstance(values, list):
        return normalize_space(values)

    candidates: list[tuple[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            text = normalize_space(item.get("value"))
            language = normalize_space(item.get("language_tag")).lower()
            if text:
                candidates.append((language, text))
        else:
            text = normalize_space(item)
            if text:
                candidates.append(("", text))

    if not candidates:
        return ""
    if prefer_english:
        for language, text in candidates:
            if language.startswith("en"):
                return text
    if allow_non_english:
        return candidates[0][1]
    return ""


def values_from_list(values: Any, limit: int = 5) -> list[str]:
    """从多值字段里取若干字符串，用于 bullet points / keywords。"""

    if not isinstance(values, list):
        text = value_from_list(values)
        return [text] if text else []

    output = []
    for item in values:
        text = value_from_list(item)
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def load_image_index(abo_root: Path, min_image_size: int) -> dict[str, Path]:
    """读取 images.csv.gz，建立 image_id -> small image path 映射。"""

    csv_path = abo_root / "images" / "metadata" / "images.csv.gz"
    image_root = abo_root / "images" / "small"
    if not csv_path.is_file():
        raise FileNotFoundError(f"ABO image metadata 不存在：{csv_path}")

    mapping: dict[str, Path] = {}
    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            rel_path = row.get("path")
            if not image_id or not rel_path:
                continue
            try:
                height = int(row.get("height") or 0)
                width = int(row.get("width") or 0)
            except ValueError:
                continue
            if height < min_image_size or width < min_image_size:
                continue
            image_path = image_root / rel_path
            if image_path.is_file():
                mapping[image_id] = image_path
    return mapping


def iter_listing_rows(abo_root: Path) -> Iterable[dict[str, Any]]:
    """遍历 ABO listings_*.json.gz。"""

    metadata_dir = abo_root / "listings" / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"ABO listing metadata 目录不存在：{metadata_dir}")

    for path in sorted(metadata_dir.glob("listings_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def extract_listing(record: dict[str, Any], image_index: dict[str, Path], require_english: bool) -> dict[str, Any] | None:
    """把 ABO 原始 listing 压成构建任务需要的字段。"""

    item_id = normalize_space(record.get("item_id"))
    main_image_id = normalize_space(record.get("main_image_id"))
    if not item_id or not main_image_id:
        return None
    image_path = image_index.get(main_image_id)
    if image_path is None:
        return None

    allow_non_english = not require_english
    title = value_from_list(record.get("item_name"), allow_non_english=allow_non_english)
    brand = value_from_list(record.get("brand"), allow_non_english=allow_non_english)
    color = value_from_list(record.get("color"), allow_non_english=allow_non_english)
    style = value_from_list(record.get("style"), allow_non_english=allow_non_english)
    product_type = value_from_list(record.get("product_type"), allow_non_english=True)
    bullets = values_from_list(record.get("bullet_point"), limit=4)

    node_names = []
    for node in record.get("node") or []:
        if isinstance(node, dict):
            node_name = normalize_space(node.get("node_name"))
            if node_name:
                node_names.append(node_name)
    category = node_names[0] if node_names else ""

    if require_english and not title:
        return None
    if not any([title, brand, color, style, product_type, category]):
        return None

    return {
        "item_id": item_id,
        "image_path": str(image_path),
        "title": title,
        "brand": brand,
        "color": color,
        "style": style,
        "product_type": product_type,
        "category": category,
        "bullet_points": bullets,
        "country": normalize_space(record.get("country")),
        "domain_name": normalize_space(record.get("domain_name")),
    }


def make_sft_sample(
    *,
    sample_id: str,
    image_path: str,
    prompt: str,
    answer: str,
    task: str,
    answers: list[str] | None = None,
    metric: str = "exact_match",
) -> dict[str, Any]:
    """生成 SFT 样本。"""

    answer = normalize_space(answer)
    return {
        "id": sample_id,
        "image_path": image_path,
        "messages": [
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        "source": "abo",
        "task": task,
        "answers": answers or [answer],
        "eval": {"metric": metric},
    }


def make_grpo_sample(
    *,
    sample_id: str,
    image_path: str,
    prompt: str,
    answer: str,
    task: str,
    target_field: str,
    reward_type: str = "normalized_exact_match",
) -> dict[str, Any]:
    """生成 GRPO 样本。

    GRPO 训练时不会把 answer 拼进 prompt。answer 只用于 reward 计算。
    """

    answer = normalize_space(answer)
    return {
        "id": f"{sample_id}_grpo",
        "image_path": image_path,
        "messages": [
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{prompt}"},
        ],
        "source": "abo",
        "task": task,
        "answer": answer,
        "reward": {
            "type": reward_type,
            "answers": [answer],
            "target_field": target_field,
            "normalization": "lower_strip_punct_space",
        },
    }


def build_samples_for_listing(listing: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从一条商品 listing 生成多条 SFT 和 GRPO 样本。"""

    item_id = listing["item_id"]
    image_path = listing["image_path"]
    sft_samples: list[dict[str, Any]] = []
    grpo_samples: list[dict[str, Any]] = []

    field_tasks = [
        (
            "product_type",
            listing.get("product_type"),
            "What is the product type shown in the image? Answer with the product type only.",
            "product_type_qa",
        ),
        (
            "color",
            listing.get("color"),
            "What is the main product color? Answer with a short color phrase.",
            "product_color_qa",
        ),
        (
            "brand",
            listing.get("brand"),
            "What is the product brand? Answer with the brand name only.",
            "product_brand_qa",
        ),
        (
            "style",
            listing.get("style"),
            "What product style is shown? Answer with a short phrase.",
            "product_style_qa",
        ),
    ]

    for field_name, answer, prompt, task in field_tasks:
        answer = normalize_space(answer)
        if not answer:
            continue
        sample_id = f"abo_{item_id}_{field_name}"
        sft_samples.append(
            make_sft_sample(
                sample_id=sample_id,
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                task=task,
            )
        )
        grpo_samples.append(
            make_grpo_sample(
                sample_id=sample_id,
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                task=task,
                target_field=field_name,
            )
        )

    title = normalize_space(listing.get("title"))
    if title:
        sft_samples.append(
            make_sft_sample(
                sample_id=f"abo_{item_id}_title",
                image_path=image_path,
                prompt="Write a concise e-commerce product title for this image.",
                answer=title,
                task="product_title_generation",
                metric="token_f1",
            )
        )

    summary_parts = []
    for key, label in [
        ("title", "Title"),
        ("brand", "Brand"),
        ("product_type", "Product type"),
        ("color", "Color"),
        ("style", "Style"),
    ]:
        value = normalize_space(listing.get(key))
        if value:
            summary_parts.append(f"{label}: {value}")
    bullets = listing.get("bullet_points") or []
    if bullets:
        summary_parts.append("Key features: " + "; ".join(bullets[:3]))
    if len(summary_parts) >= 2:
        sft_samples.append(
            make_sft_sample(
                sample_id=f"abo_{item_id}_summary",
                image_path=image_path,
                prompt="Summarize the product attributes visible or provided for this e-commerce item.",
                answer="\n".join(summary_parts),
                task="product_attribute_summary",
                metric="token_f1",
            )
        )

    return sft_samples, grpo_samples


def split_samples(
    samples: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """确定性切分 train/val/test。"""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio/test_ratio 必须非负，且二者之和小于 1。")
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    total = len(samples)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    return {
        "val": samples[:val_count],
        "test": samples[val_count : val_count + test_count],
        "train": samples[val_count + test_count :],
    }


def cap_samples(samples: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """按上限截断样本。"""

    if limit is None:
        return samples
    return samples[:limit]


def write_json(path: Path, data: Any) -> None:
    """写 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    config = parse_args()
    abo_root = Path(config.abo_root)
    output_root = Path(config.output_root)
    if not abo_root.exists():
        raise FileNotFoundError(f"ABO root 不存在：{abo_root}")

    image_index = load_image_index(abo_root, min_image_size=config.min_image_size)
    print(f"[data] image_index={len(image_index)}")

    sft_samples: list[dict[str, Any]] = []
    grpo_samples: list[dict[str, Any]] = []
    product_type_counter: Counter[str] = Counter()
    kept_listings = 0

    for raw in iter_listing_rows(abo_root):
        listing = extract_listing(
            raw,
            image_index=image_index,
            require_english=config.require_english,
        )
        if listing is None:
            continue
        kept_listings += 1
        if listing.get("product_type"):
            product_type_counter[listing["product_type"]] += 1
        one_sft, one_grpo = build_samples_for_listing(listing)
        sft_samples.extend(one_sft)
        grpo_samples.extend(one_grpo)
        if config.max_listings is not None and kept_listings >= config.max_listings:
            break

    sft_samples = cap_samples(sft_samples, config.max_sft_samples)
    grpo_samples = cap_samples(grpo_samples, config.max_grpo_samples)

    sft_splits = split_samples(
        sft_samples,
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    grpo_splits = split_samples(
        grpo_samples,
        seed=config.seed + 1,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    for split_name, split_samples_ in sft_splits.items():
        write_json(output_root / "sft" / f"{split_name}.json", split_samples_)
    for split_name, split_samples_ in grpo_splits.items():
        write_json(output_root / "grpo" / f"{split_name}.json", split_samples_)

    manifest = {
        "config": asdict(config),
        "kept_listings": kept_listings,
        "image_index": len(image_index),
        "sft_counts": {key: len(value) for key, value in sft_splits.items()},
        "grpo_counts": {key: len(value) for key, value in grpo_splits.items()},
        "top_product_types": product_type_counter.most_common(30),
        "format_notes": {
            "sft": "messages 包含 user + assistant，可直接复用 VLMDataCollator 训练。",
            "grpo": "messages 只包含 user；answer/reward 字段用于生成后计算奖励。",
        },
    }
    write_json(output_root / "manifest.json", manifest)

    print("[done] Stage 4 ecommerce 数据构建完成")
    print(f"[done] kept_listings={kept_listings}")
    print(f"[done] sft_counts={manifest['sft_counts']}")
    print(f"[done] grpo_counts={manifest['grpo_counts']}")
    print(f"[done] output_root={output_root}")


if __name__ == "__main__":
    main()

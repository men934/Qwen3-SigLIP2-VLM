"""Stage 3 垂域测试集定量评估脚本。

这个脚本评估的是“开放式生成回答”，不是分类准确率。不同任务使用不同指标：

    VQA / OCR:
        normalized exact match + token F1

    ChartQA:
        在 VQA 指标基础上，额外计算 relaxed accuracy。数字答案允许 5% 相对误差。

    CORD JSON:
        JSON parse rate + 扁平字段 token F1。这里不是严格结构化评测的最终版本，
        但足够作为 Stage 3 训练前后的工程对比。

默认会尝试比较：
    1. Stage1 固定分辨率
    2. Stage1 动态分辨率 + SigLIP Q/K 2D RoPE
    3. Stage2 固定分辨率 LoRA
    4. Stage2 动态分辨率 + SigLIP Q/K 2D RoPE LoRA
    5. Stage3 当前 checkpoint

如果某个 checkpoint 不存在，会自动跳过。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import string
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.conversation import DEFAULT_IMAGE_TOKEN, IM_END, IM_START
from vlm.data.domain_mix_dataset import DomainMixDataset
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


@dataclass(frozen=True)
class CheckpointSpec:
    """一个待评估 checkpoint 的加载信息。"""

    name: str
    projector_path: str
    lora_path: str | None = None
    vision_path: str | None = None
    dynamic_resolution: bool = False
    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False
    siglip_rope_base: float = 10000.0
    siglip_rope_dim: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 垂域测试集定量评估")
    parser.add_argument("--qwen-path", default="/root/autodl-tmp/hf_models/Qwen3-1.7B")
    parser.add_argument(
        "--siglip-path",
        default="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--test-annotation-path",
        default="/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/test.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/checkpoints/stage3_eval",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        default=None,
        help="Stage3 checkpoint 目录，例如 /root/autodl-tmp/checkpoints/stage3_doc_ocr_mix/step_010000。为空时自动找最新 step。",
    )
    parser.add_argument(
        "--stage3-root",
        default="/root/autodl-tmp/checkpoints/stage3_doc_ocr_mix",
        help="自动寻找 Stage3 最新 checkpoint 的根目录。",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--min-pixels", type=int, default=384 * 384)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=optional_int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def optional_int(value: str) -> int | None:
    """argparse 用的小工具：允许 none/null/-1 表示不限制。"""

    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def latest_checkpoint(root: str | Path) -> Path | None:
    """寻找形如 step_000000 的最新 checkpoint。"""

    root_path = Path(root)
    if not root_path.is_dir():
        return None
    candidates = [
        path
        for path in root_path.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def best_stage3_checkpoint(root: str | Path) -> Path | None:
    """优先选择 Stage3 验证集最优 checkpoint。

    选择顺序：
        1. 如果存在 ``best/projector.pt``，直接使用 best。
        2. 读取 ``metrics.csv``，找 val_loss 最低且已保存的 step_xxxxxx。
        3. 如果没有可用 val checkpoint，退回最新 step。
    """

    root_path = Path(root)
    if not root_path.is_dir():
        return None

    best_dir = root_path / "best"
    if (best_dir / "projector.pt").is_file() and (best_dir / "lora_adapter").is_dir():
        return best_dir

    metrics_path = root_path / "metrics.csv"
    best_step = None
    best_val = float("inf")
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_val = row.get("val_loss")
                raw_step = row.get("step")
                if not raw_val or not raw_step:
                    continue
                try:
                    val_loss = float(raw_val)
                    step = int(raw_step)
                except ValueError:
                    continue
                ckpt_dir = root_path / f"step_{step:06d}"
                if val_loss < best_val and (ckpt_dir / "projector.pt").is_file():
                    best_val = val_loss
                    best_step = step
    if best_step is not None:
        return root_path / f"step_{best_step:06d}"

    return latest_checkpoint(root_path)


def default_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    """构造默认对比列表，不存在的 checkpoint 后面会跳过。"""

    stage3_dir = (
        Path(args.stage3_checkpoint)
        if args.stage3_checkpoint
        else best_stage3_checkpoint(args.stage3_root)
    )
    specs = [
        CheckpointSpec(
            name="stage1_fixed_50k",
            projector_path="/root/autodl-tmp/checkpoints/stage1_align_50k/step_003000/projector.pt",
        ),
        CheckpointSpec(
            name="stage1_dynamic_qk_rope_50k",
            projector_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/projector.pt",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
        CheckpointSpec(
            name="stage2_fixed_150k_r32",
            projector_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/lora_adapter",
        ),
        CheckpointSpec(
            name="stage2_dynamic_qk_rope_150k_r32",
            projector_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/lora_adapter",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
    ]
    if stage3_dir is not None:
        specs.append(
            CheckpointSpec(
                name=f"stage3_{stage3_dir.name}",
                projector_path=str(stage3_dir / "projector.pt"),
                lora_path=str(stage3_dir / "lora_adapter"),
            )
        )
    return specs


def checkpoint_exists(spec: CheckpointSpec) -> bool:
    """检查 checkpoint 文件是否齐全。"""

    if not Path(spec.projector_path).is_file():
        return False
    if spec.lora_path is not None and not Path(spec.lora_path).is_dir():
        return False
    if spec.vision_path is not None and not Path(spec.vision_path).is_file():
        return False
    return True


def build_helper(args: argparse.Namespace, spec: CheckpointSpec) -> VLMDataCollator:
    """复用 collator 中的 tokenizer 和 image processor。"""

    return VLMDataCollator(
        CollatorConfig(
            tokenizer_path=args.qwen_path,
            image_processor_path=args.siglip_path,
            image_size=args.image_size,
            dynamic_resolution=spec.dynamic_resolution,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_length=args.max_length,
        )
    )


def load_model(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    helper: VLMDataCollator,
    device: torch.device,
) -> QwenSiglipVLM:
    """加载一个待评估模型。"""

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path=args.qwen_path,
            siglip_path=args.siglip_path,
            image_token_id=helper.image_token_id,
            tokenizer_length=len(helper.tokenizer),
            freeze_vision_encoder=True,
            freeze_language_model=True,
            use_siglip_abs_pos_embedding=spec.use_siglip_abs_pos_embedding,
            use_siglip_qk_2d_rope=spec.use_siglip_qk_2d_rope,
            siglip_rope_base=spec.siglip_rope_base,
            siglip_rope_dim=spec.siglip_rope_dim,
            torch_dtype=args.torch_dtype,
        )
    )

    model.projector.load_state_dict(torch.load(spec.projector_path, map_location="cpu"))
    if spec.vision_path:
        vision_state = torch.load(spec.vision_path, map_location="cpu")
        missing, unexpected = model.vision_encoder.load_state_dict(vision_state, strict=False)
        if unexpected:
            raise RuntimeError(f"{spec.name} 加载 vision 权重出现 unexpected keys: {unexpected}")
        print(f"[init] {spec.name} 已加载 vision 权重，missing={len(missing)}")

    if spec.lora_path:
        from peft import PeftModel

        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            spec.lora_path,
        )
    model.to(device)
    model.eval()
    return model


def build_prompt(sample: dict[str, Any], image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """从统一 messages 样本中构造只包含 user 的推理 prompt。"""

    user_content = sample["messages"][0]["content"]
    if image_token not in user_content:
        user_content = f"{image_token}\n{user_content}"
    return f"{IM_START}user\n{user_content}{IM_END}\n{IM_START}assistant\n"


@torch.no_grad()
def generate_one(
    *,
    model: QwenSiglipVLM,
    helper: VLMDataCollator,
    sample: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """对单条样本生成回答。"""

    prompt = build_prompt(sample, image_token=helper.image_token)
    encoded = helper.tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    image_batch = helper.image_processor.process_batch([sample["image_path"]])
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = image_batch.pixel_values.to(device)

    visual_embeds, _ = model.encode_images(
        pixel_values=pixel_values,
        image_infos=image_batch.infos,
    )
    multimodal_inputs = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        visual_embeds=visual_embeds,
    )

    do_sample = temperature > 0
    generation_kwargs = {
        "inputs_embeds": multimodal_inputs["inputs_embeds"],
        "attention_mask": multimodal_inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "eos_token_id": helper.tokenizer.eos_token_id,
        "pad_token_id": helper.tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    output_ids = model.language_model.generate(**generation_kwargs)
    text = helper.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if IM_END in text:
        text = text.split(IM_END, 1)[0]
    return text.strip()


def normalize_answer(text: str) -> str:
    """VQA 常用答案归一化：小写、去标点、去冠词、合并空格。"""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def exact_match(prediction: str, references: list[str]) -> float:
    """多个参考答案中任意一个匹配即可。"""

    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))


def token_f1(prediction: str, references: list[str]) -> float:
    """计算 prediction 和多个 reference 的最大 token F1。"""

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def parse_number(text: str) -> float | None:
    """从字符串中提取第一个数字。"""

    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def relaxed_numeric_accuracy(prediction: str, references: list[str]) -> float:
    """ChartQA 常用 relaxed accuracy：数字允许 5% 相对误差。"""

    pred_num = parse_number(prediction)
    if pred_num is None:
        return exact_match(prediction, references)
    for ref in references:
        ref_num = parse_number(ref)
        if ref_num is None:
            continue
        if math.isclose(pred_num, ref_num, rel_tol=0.05, abs_tol=1e-4):
            return 1.0
    return 0.0


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出中尽量解析 JSON 对象。"""

    text = text.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def flatten_json(obj: Any, prefix: str = "") -> dict[str, str]:
    """把嵌套 JSON 展平成 key_path -> value，便于粗粒度字段比较。"""

    output: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_json(value, child_prefix))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_prefix = f"{prefix}[{index}]"
            output.update(flatten_json(value, child_prefix))
    else:
        output[prefix] = clean_scalar(obj)
    return output


def clean_scalar(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def json_field_f1(prediction: str, references: list[str]) -> tuple[float, float]:
    """返回 JSON parse success 和字段级 token F1。"""

    pred_obj = extract_json(prediction)
    if pred_obj is None:
        return 0.0, 0.0
    best_f1 = 0.0
    for ref in references:
        ref_obj = extract_json(ref)
        if ref_obj is None:
            continue
        pred_flat = flatten_json(pred_obj)
        ref_flat = flatten_json(ref_obj)
        if not ref_flat:
            continue
        key_scores = []
        for key, ref_value in ref_flat.items():
            pred_value = pred_flat.get(key, "")
            key_scores.append(token_f1(pred_value, [ref_value]))
        if key_scores:
            best_f1 = max(best_f1, sum(key_scores) / len(key_scores))
    return 1.0, best_f1


def score_prediction(prediction: str, sample: dict[str, Any]) -> dict[str, float]:
    """按样本任务类型计算指标。"""

    references = sample.get("answers") or [
        sample["messages"][-1]["content"]
    ]
    metric = (sample.get("eval") or {}).get("metric", "vqa")
    scores = {
        "em": exact_match(prediction, references),
        "f1": token_f1(prediction, references),
        "relaxed_acc": 0.0,
        "json_parse": 0.0,
        "json_field_f1": 0.0,
    }
    if metric == "chartqa":
        scores["relaxed_acc"] = relaxed_numeric_accuracy(prediction, references)
    if metric == "json":
        parse_success, field_f1 = json_field_f1(prediction, references)
        scores["json_parse"] = parse_success
        scores["json_field_f1"] = field_f1
    return scores


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合整体和按 source/task 的指标。"""

    metric_keys = ["em", "f1", "relaxed_acc", "json_parse", "json_field_f1"]

    def summarize(subset: list[dict[str, Any]]) -> dict[str, float | int]:
        if not subset:
            return {"count": 0, **{key: 0.0 for key in metric_keys}}
        result: dict[str, float | int] = {"count": len(subset)}
        for key in metric_keys:
            result[key] = sum(item["scores"][key] for item in subset) / len(subset)
        return result

    by_source: dict[str, list[dict[str, Any]]] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
        by_task.setdefault(row["task"], []).append(row)

    return {
        "overall": summarize(rows),
        "by_source": {key: summarize(value) for key, value in sorted(by_source.items())},
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
    """写一个便于快速浏览的 Markdown 汇总。"""

    lines = ["# Stage 3 Domain Evaluation", ""]
    lines.append("| checkpoint | count | EM | F1 | Chart relaxed | JSON parse | JSON field F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metrics in all_metrics:
        overall = metrics["overall"]
        lines.append(
            f"| {metrics['checkpoint']} | {overall['count']} | "
            f"{overall['em']:.4f} | {overall['f1']:.4f} | "
            f"{overall['relaxed_acc']:.4f} | {overall['json_parse']:.4f} | "
            f"{overall['json_field_f1']:.4f} |"
        )
    lines.append("")
    lines.append("## By Source")
    for metrics in all_metrics:
        lines.append("")
        lines.append(f"### {metrics['checkpoint']}")
        lines.append("| source | count | EM | F1 | Chart relaxed | JSON parse | JSON field F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for source, row in metrics["by_source"].items():
            lines.append(
                f"| {source} | {row['count']} | {row['em']:.4f} | {row['f1']:.4f} | "
                f"{row['relaxed_acc']:.4f} | {row['json_parse']:.4f} | "
                f"{row['json_field_f1']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DomainMixDataset(
        annotation_path=args.test_annotation_path,
        verify_images=False,
        max_samples=args.max_samples,
    )
    samples = [dataset[i] for i in range(len(dataset))]
    print(f"[data] test samples: {len(samples)}")

    specs = [spec for spec in default_checkpoint_specs(args) if checkpoint_exists(spec)]
    if not specs:
        raise FileNotFoundError("没有找到任何可评估 checkpoint。")
    print("[eval] checkpoints:", ", ".join(spec.name for spec in specs))

    all_metrics = []
    all_predictions = []
    for spec in specs:
        metrics, rows = evaluate_checkpoint(args, spec, samples, device)
        all_metrics.append(metrics)
        all_predictions.extend(rows)

        # 每评估完一个模型就落盘，避免长评估中断后完全丢结果。
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        write_report(output_dir / "report.md", all_metrics)

    print(f"[done] metrics -> {output_dir / 'metrics.json'}")
    print(f"[done] predictions -> {output_dir / 'predictions.json'}")
    print(f"[done] report -> {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

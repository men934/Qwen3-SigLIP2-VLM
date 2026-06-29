"""Qualitative comparison for Stage 1 / Stage 2 checkpoints.

    1. 固定抽取一批验证样本；
    2. 对多个 checkpoint 生成回答；
    3. 输出 JSON 和 Markdown。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.conversation import DEFAULT_IMAGE_TOKEN, IM_END, IM_START
from vlm.data.llava_instruct_dataset import LlavaInstructDataset
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


@dataclass(frozen=True)
class CheckpointSpec:
    """需要评估的 checkpoint 配置。"""

    name: str
    projector_path: str
    lora_path: str | None = None
    vision_path: str | None = None
    dynamic_resolution: bool = False
    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 Stage 1/Stage 2 checkpoint")
    parser.add_argument("--qwen-path", default="/root/autodl-tmp/hf_models/Qwen3-1.7B")
    parser.add_argument(
        "--siglip-path",
        default="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
    )
    parser.add_argument(
        "--annotation-path",
        default="/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/splits/val.json",
    )
    parser.add_argument(
        "--image-root",
        default="/root/autodl-tmp/hf_datasets/coco/train2014",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="first 取前 N 条；random 用固定 seed 随机抽 N 条。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoints",
        default="stage2_lora_20k,stage2_lora_150k_r32",
        help="逗号分隔 checkpoint 名称；传 all 表示评估默认列表里的所有 checkpoint。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--output-json",
        default="/root/autodl-tmp/checkpoints/qual_eval/stage2_compare.json",
    )
    parser.add_argument(
        "--output-md",
        default="/root/autodl-tmp/checkpoints/qual_eval/stage2_compare.md",
    )
    return parser.parse_args()


def build_prompt(question: str, image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """构造单轮问答 prompt。"""

    return (
        f"{IM_START}user\n"
        f"{image_token}\n{question}"
        f"{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def extract_first_qa(sample: dict[str, Any]) -> tuple[str, str]:
    """从一条 LLaVA-Instruct 样本中取第一轮 user/assistant。"""

    messages = sample["messages"]
    question = None
    answer = None
    for message in messages:
        if message["role"] == "user" and question is None:
            question = message["content"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        elif message["role"] == "assistant" and question is not None:
            answer = message["content"].strip()
            break

    if not question or not answer:
        raise ValueError(f"样本 {sample.get('id')} 没有可用的首轮 QA。")
    return question, answer


def infer_question_type(question: str) -> str:
    """Infer a coarse question type from text."""

    q = question.lower()
    if "how many" in q or "number of" in q or re.search(r"\bcount\b", q):
        return "counting"
    if "color" in q or "colour" in q:
        return "color"
    if "where" in q or "position" in q or "located" in q:
        return "location"
    if "sign" in q or "text" in q or "word" in q or "read" in q or "letter" in q:
        return "ocr_or_sign"
    if "doing" in q or "activity" in q or "playing" in q or "riding" in q:
        return "action"
    if "why" in q or "are the" in q or "is the" in q and " or " in q:
        return "reasoning"
    if q.startswith("what type") or q.startswith("what animal") or q.startswith("what is"):
        return "object_or_scene"
    return "general"


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """加载固定数量样本。

    qualitative eval 必须可复现，所以 random 模式也使用固定 seed。
    """

    dataset = LlavaInstructDataset(
        annotation_path=args.annotation_path,
        image_root=args.image_root,
        max_samples=args.num_samples if args.sample_mode == "first" else None,
    )
    if args.sample_mode == "first":
        return [dataset[i] for i in range(len(dataset))]

    if args.num_samples <= 0:
        raise ValueError(f"num-samples 必须为正数，当前为 {args.num_samples}。")
    if args.num_samples > len(dataset):
        raise ValueError(
            f"num-samples={args.num_samples} 超过数据集大小 {len(dataset)}。"
        )

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.num_samples))
    return [dataset[i] for i in indices]


def load_model(
    args: argparse.Namespace,
    helper: VLMDataCollator,
    checkpoint: CheckpointSpec,
    device: torch.device,
) -> QwenSiglipVLM:
    """加载一个待评估 checkpoint。"""

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path=args.qwen_path,
            siglip_path=args.siglip_path,
            image_token_id=helper.image_token_id,
            tokenizer_length=len(helper.tokenizer),
            freeze_vision_encoder=True,
            freeze_language_model=True,
            use_siglip_abs_pos_embedding=checkpoint.use_siglip_abs_pos_embedding,
            use_siglip_qk_2d_rope=checkpoint.use_siglip_qk_2d_rope,
            torch_dtype=args.torch_dtype,
        )
    )

    projector_path = Path(checkpoint.projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"projector 不存在：{projector_path}")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    if checkpoint.vision_path is not None:
        vision_path = Path(checkpoint.vision_path)
        if not vision_path.is_file():
            raise FileNotFoundError(f"vision 权重不存在：{vision_path}")
        vision_state = torch.load(vision_path, map_location="cpu")
        missing, unexpected = model.vision_encoder.load_state_dict(
            vision_state,
            strict=False,
        )
        if unexpected:
            raise RuntimeError(f"加载 vision 权重出现 unexpected keys: {unexpected}")
        print(f"[load] vision trainable: {vision_path} (missing={len(missing)})")

    if checkpoint.lora_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("加载 LoRA checkpoint 需要 peft。") from exc

        lora_path = Path(checkpoint.lora_path)
        if not lora_path.is_dir():
            raise FileNotFoundError(f"LoRA adapter 不存在：{lora_path}")
        model.language_model = PeftModel.from_pretrained(model.language_model, lora_path)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_answer(
    model: QwenSiglipVLM,
    helper: VLMDataCollator,
    image_path: str,
    question: str,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    """对单个样本生成回答。"""

    processed = helper.image_processor.process_image(Path(image_path))
    pixel_values = processed.pixel_values.unsqueeze(0).to(device)

    prompt = build_prompt(question, image_token=helper.image_token)
    encoded = helper.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    visual_embeds, _ = model.encode_images(
        pixel_values,
        image_infos=[processed.info],
    )
    multimodal_inputs = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        visual_embeds=visual_embeds,
    )

    do_sample = args.temperature > 0
    generation_kwargs = {
        "inputs_embeds": multimodal_inputs["inputs_embeds"],
        "attention_mask": multimodal_inputs["attention_mask"],
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": helper.tokenizer.eos_token_id,
        "pad_token_id": helper.tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    output_ids = model.language_model.generate(**generation_kwargs)
    text = helper.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if IM_END in text:
        text = text.split(IM_END, 1)[0]
    return text.strip()


def default_checkpoints() -> list[CheckpointSpec]:
    """Default checkpoints for qualitative eval."""

    return [
        CheckpointSpec(
            name="stage1_50k",
            projector_path="/root/autodl-tmp/checkpoints/stage1_align_50k/step_003000/projector.pt",
        ),
        CheckpointSpec(
            name="stage2_sanity_projector",
            projector_path="/root/autodl-tmp/checkpoints/stage2_sanity/step_000002/projector.pt",
        ),
        CheckpointSpec(
            name="stage2_lora_5k",
            projector_path="/root/autodl-tmp/checkpoints/stage2_lora_5k/step_000625/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_lora_5k/step_000625/lora_adapter",
        ),
        CheckpointSpec(
            name="stage2_lora_20k",
            projector_path="/root/autodl-tmp/checkpoints/stage2_lora_20k/step_002500/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_lora_20k/step_002500/lora_adapter",
        ),
        CheckpointSpec(
            name="stage2_lora_150k_r32",
            projector_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_lora_150k_r32/step_018000/lora_adapter",
        ),
        CheckpointSpec(
            name="stage2_dynamic_qk_rope_150k_r32_best",
            projector_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_015000/lora_adapter",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
        CheckpointSpec(
            name="stage2_dynamic_qk_rope_150k_r32_final",
            projector_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_019089/projector.pt",
            lora_path="/root/autodl-tmp/checkpoints/stage2_dynamic_qk_rope_150k_r32/step_019089/lora_adapter",
            vision_path="/root/autodl-tmp/checkpoints/stage1_dynamic_qk_rope_50k/step_003000/vision_encoder_trainable.pt",
            dynamic_resolution=True,
            use_siglip_abs_pos_embedding=False,
            use_siglip_qk_2d_rope=True,
        ),
    ]


def select_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    """根据 --checkpoints 参数筛选要评估的 checkpoint。"""

    checkpoints = default_checkpoints()
    if args.checkpoints.strip().lower() == "all":
        return checkpoints

    wanted = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    by_name = {checkpoint.name: checkpoint for checkpoint in checkpoints}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(
            f"未知 checkpoint: {missing}；可选值为 {sorted(by_name)}，或传 all。"
        )
    return [by_name[name] for name in wanted]


def write_markdown_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    checkpoints: list[CheckpointSpec],
    args: argparse.Namespace,
) -> None:
    """Write qualitative eval results as Markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_names = [checkpoint.name for checkpoint in checkpoints]

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["id"]
        if key not in grouped:
            grouped[key] = {
                "id": row["id"],
                "question_type": row["question_type"],
                "image_path": row["image_path"],
                "question": row["question"],
                "gt_answer": row["gt_answer"],
                "predictions": {},
            }
        grouped[key]["predictions"][row["checkpoint"]] = row["prediction"]

    type_counts: dict[str, int] = {}
    for item in grouped.values():
        type_counts[item["question_type"]] = type_counts.get(item["question_type"], 0) + 1

    lines = [
        "# Stage 2 Qualitative Eval",
        "",
        "## 配置",
        "",
        f"- annotation: `{args.annotation_path}`",
        f"- sample_mode: `{args.sample_mode}`",
        f"- seed: `{args.seed}`",
        f"- num_samples: `{args.num_samples}`",
        f"- checkpoints: `{', '.join(checkpoint_names)}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        f"- temperature: `{args.temperature}`",
        "",
        "## 样本类型分布",
        "",
    ]
    for name, count in sorted(type_counts.items()):
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## 逐样本结果", ""])
    for idx, item in enumerate(grouped.values(), start=1):
        lines.extend(
            [
                f"### {idx}. {item['id']} [{item['question_type']}]",
                "",
                f"- image: `{item['image_path']}`",
                f"- question: {item['question']}",
                f"- reference: {item['gt_answer']}",
                "",
            ]
        )
        for checkpoint_name in checkpoint_names:
            pred = item["predictions"].get(checkpoint_name, "")
            lines.extend([f"**{checkpoint_name}**", "", pred, ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    helper = VLMDataCollator(
        CollatorConfig(
            tokenizer_path=args.qwen_path,
            image_processor_path=args.siglip_path,
            image_size=384,
            dynamic_resolution=False,
            max_length=1024,
        )
    )
    samples = load_samples(args)
    checkpoints = select_checkpoints(args)

    results: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        print("\n" + "=" * 100)
        print(f"CHECKPOINT: {checkpoint.name}")
        helper.image_processor.dynamic_resolution = checkpoint.dynamic_resolution
        model = load_model(args, helper, checkpoint, device)

        for sample in samples:
            question, gt_answer = extract_first_qa(sample)
            question_type = infer_question_type(question)
            pred = generate_answer(
                model=model,
                helper=helper,
                image_path=sample["image_path"],
                question=question,
                args=args,
                device=device,
            )

            row = {
                "checkpoint": checkpoint.name,
                "id": sample["id"],
                "question_type": question_type,
                "image_path": sample["image_path"],
                "question": question,
                "gt_answer": gt_answer,
                "prediction": pred,
            }
            results.append(row)

            print("-" * 100)
            print("id:", sample["id"])
            print("type:", question_type)
            print("question:", question)
            print("gt:", gt_answer)
            print("pred:", pred)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存：", output_path)

    md_path = Path(args.output_md)
    write_markdown_report(md_path, results, checkpoints, args)
    print("Markdown 报告已保存：", md_path)


if __name__ == "__main__":
    main()

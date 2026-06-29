"""Inference script for a Stage 1 projector checkpoint.

流程：
    image
        -> SigLIP2 image processor
        -> QwenSiglipVLM.encode_images
        -> 把 prompt 里的 <image> 替换成 visual embeddings
        -> Qwen3.generate
        -> decode 文本
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.conversation import DEFAULT_IMAGE_TOKEN, IM_END, IM_START
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 projector 推理脚本")

    parser.add_argument(
        "--qwen-path",
        default="/root/autodl-tmp/hf_models/Qwen3-1.7B",
        help="本地 Qwen3 模型路径。",
    )
    parser.add_argument(
        "--siglip-path",
        default="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
        help="本地 SigLIP2 模型/processor 路径。",
    )
    parser.add_argument(
        "--projector-path",
        default="/root/autodl-tmp/checkpoints/stage1_align_2k/step_000100/projector.pt",
        help="Stage 1 训练得到的 projector.pt 路径。",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="要推理的图片路径。如果不传，会自动从 LLaVA-Pretrain 中找一张图。",
    )
    parser.add_argument(
        "--question",
        default="Describe this image briefly.",
        help="用户问题。Stage 1 建议使用描述类 prompt。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")

    return parser.parse_args()


def find_default_image() -> Path:
    """Find one default image."""

    from vlm.data.image_processing import find_first_image

    image_root = Path("/root/autodl-tmp/hf_datasets/LLaVA-Pretrain")
    image_path = find_first_image(image_root)
    if image_path is None:
        raise FileNotFoundError(f"没有在 {image_root} 下找到图片。")
    return image_path


def build_inference_prompt(question: str, image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """构造 Qwen ChatML 推理 prompt。

    输出会以 ``<|im_start|>assistant\n`` 结尾，让模型从这里继续生成。
    """

    user_content = f"{image_token}\n{question}"
    return f"{IM_START}user\n{user_content}{IM_END}\n{IM_START}assistant\n"


def load_tokenizer_and_image_processor(args: argparse.Namespace) -> VLMDataCollator:
    """Load tokenizer and image processor through VLMDataCollator."""

    return VLMDataCollator(
        CollatorConfig(
            tokenizer_path=args.qwen_path,
            image_processor_path=args.siglip_path,
            image_size=384,
            max_length=512,
        )
    )


def load_model(
    args: argparse.Namespace,
    image_token_id: int,
    tokenizer_length: int,
    device: torch.device,
) -> QwenSiglipVLM:
    """加载 QwenSiglipVLM，并载入 Stage 1 projector 权重。"""

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path=args.qwen_path,
            siglip_path=args.siglip_path,
            image_token_id=image_token_id,
            tokenizer_length=tokenizer_length,
            freeze_vision_encoder=True,
            freeze_language_model=True,
            torch_dtype=args.torch_dtype,
        )
    )

    projector_path = Path(args.projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"projector checkpoint 不存在：{projector_path}")

    state_dict = torch.load(projector_path, map_location="cpu")
    model.projector.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_caption(
    model: QwenSiglipVLM,
    tokenizer,
    pixel_values: torch.Tensor,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """生成文本并 decode。"""

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = pixel_values.to(device)

    visual_embeds, visual_grid = model.encode_images(pixel_values)
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
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    output_ids = model.language_model.generate(**generation_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # generate(inputs_embeds=...) 返回的通常只有生成部分，但不同 transformers 版本行为
    # 可能略有差异。这里做一个轻量清理，去掉 ChatML 结束符之后的内容。
    if IM_END in text:
        text = text.split(IM_END, 1)[0]

    print("visual_grid_size:", visual_grid.as_tuple())
    print("visual_token_count:", visual_embeds.shape[1])
    return text.strip()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    image_path = Path(args.image) if args.image is not None else find_default_image()
    print("image:", image_path)
    print("question:", args.question)
    print("projector:", args.projector_path)

    helper = load_tokenizer_and_image_processor(args)
    processed = helper.image_processor.process_image(image_path)
    pixel_values = processed.pixel_values.unsqueeze(0)

    prompt = build_inference_prompt(args.question, image_token=helper.image_token)
    print("\nprompt:")
    print(prompt)

    model = load_model(
        args=args,
        image_token_id=helper.image_token_id,
        tokenizer_length=len(helper.tokenizer),
        device=device,
    )

    output_text = generate_caption(
        model=model,
        tokenizer=helper.tokenizer,
        pixel_values=pixel_values,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n生成结果:")
    print(output_text)


if __name__ == "__main__":
    # 示例：
    #
    #   PYTHONPATH=/root/qwen3_siglip2_vlm/src \
    #   python -m vlm.inference.infer_stage1 \
    #     --projector-path /root/autodl-tmp/checkpoints/stage1_align_2k/step_000100/projector.pt \
    #     --image /path/to/image.jpg \
    #     --question "Describe this image briefly."
    main()

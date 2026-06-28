"""Stage 2 checkpoint 的简单多模态问答推理脚本。

和 Stage 1 推理脚本不同，Stage 2 主要检查：

    image + question -> answer

支持加载：
    - Stage 2 projector
    - 可选 Qwen LoRA adapter

如果 Stage 2 是 projector-only 训练，直接传 ``--projector-path`` 即可。
如果 Stage 2 开启了 LoRA，还需要传 ``--lora-path``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vlm.data.collator import CollatorConfig, VLMDataCollator
from vlm.data.conversation import DEFAULT_IMAGE_TOKEN, IM_END, IM_START
from vlm.models.vlm_model import QwenSiglipVLM, VLMModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 多模态问答推理脚本")

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
        default="/root/autodl-tmp/checkpoints/stage2_llava_instruct/step_000100/projector.pt",
        help="Stage 2 projector.pt 路径。",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="可选 LoRA adapter 目录，例如 step_xxxxxx/lora_adapter。",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="要推理的图片路径。如果不传，会使用 COCO train2014 中一张默认图片。",
    )
    parser.add_argument(
        "--question",
        default="What is in this image?",
        help="用户问题。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")

    return parser.parse_args()


def find_default_image() -> Path:
    """找一张默认 COCO 图片，方便直接运行脚本。"""

    image_root = Path("/root/autodl-tmp/hf_datasets/coco/train2014")
    for path in image_root.glob("*.jpg"):
        return path
    raise FileNotFoundError(f"没有在 {image_root} 下找到图片。")


def build_inference_prompt(question: str, image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """构造 Qwen ChatML 推理 prompt。"""

    user_content = f"{image_token}\n{question}"
    return f"{IM_START}user\n{user_content}{IM_END}\n{IM_START}assistant\n"


def load_tokenizer_and_image_processor(args: argparse.Namespace) -> VLMDataCollator:
    """复用 collator 中已经写好的 tokenizer/image_processor 加载逻辑。"""

    return VLMDataCollator(
        CollatorConfig(
            tokenizer_path=args.qwen_path,
            image_processor_path=args.siglip_path,
            image_size=384,
            max_length=1024,
        )
    )


def load_model(
    args: argparse.Namespace,
    image_token_id: int,
    tokenizer_length: int,
    device: torch.device,
) -> QwenSiglipVLM:
    """加载 QwenSiglipVLM、projector，以及可选 LoRA。"""

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

    if args.lora_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "你传入了 --lora-path，但当前环境没有安装 peft。"
            ) from exc
        lora_path = Path(args.lora_path)
        if not lora_path.is_dir():
            raise FileNotFoundError(f"LoRA adapter 目录不存在：{lora_path}")
        model.language_model = PeftModel.from_pretrained(model.language_model, lora_path)
        print(f"[init] 已加载 LoRA adapter: {lora_path}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_answer(
    model: QwenSiglipVLM,
    tokenizer,
    pixel_values: torch.Tensor,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """生成回答文本。"""

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
    if args.lora_path:
        print("lora:", args.lora_path)

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

    output_text = generate_answer(
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
    main()

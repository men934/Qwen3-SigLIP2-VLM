"""VLM 训练用的 batch collator。

传统 CV 项目的 collator 通常只需要做：

    images -> stack
    labels -> tensor

但 VLM 的 collator 要复杂很多，因为一个样本同时包含图片和文本对话：

    image_path + messages
        -> pixel_values
        -> Qwen ChatML prompt
        -> input_ids
        -> labels
        -> attention_mask

这个文件第一版支持 Stage 1 LLaVA-Pretrain 对齐训练，核心目标是：

    1. 读取图片并得到 SigLIP2 pixel_values
    2. 把 messages 格式化成 Qwen prompt
    3. tokenize 文本
    4. 构造 labels，只对 assistant answer 部分计算 loss
    5. padding 成 batch

注意：
    这个 collator 暂时还没有把 ``<image>`` 展开成多个 visual patch token。
    第一版先保留 ``<image>`` 为单个特殊 token，后续写 ``vlm_model.py`` 时再负责
    将这个位置替换/扩展为 PatchMerger + Projector 得到的 visual embeddings。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

try:
    from .conversation import (
        IM_END,
        IM_START,
        DEFAULT_IMAGE_TOKEN,
        ensure_image_token,
        format_qwen_conversation,
        format_qwen_turn,
        normalize_messages,
    )
    from .image_processing import SiglipImageProcessor
except ImportError:  # 允许直接 python src/vlm/data/collator.py 运行 sanity check
    from conversation import (
        IM_END,
        IM_START,
        DEFAULT_IMAGE_TOKEN,
        ensure_image_token,
        format_qwen_conversation,
        format_qwen_turn,
        normalize_messages,
    )
    from image_processing import SiglipImageProcessor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class CollatorConfig:
    """VLMDataCollator 的主要配置。"""

    tokenizer_path: str
    image_processor_path: str
    image_size: int = 384
    dynamic_resolution: bool = False
    min_pixels: int = 384 * 384
    max_pixels: int = 672 * 672
    image_token: str = DEFAULT_IMAGE_TOKEN
    max_length: Optional[int] = 2048
    padding_side: str = "right"
    local_files_only: bool = True


class VLMDataCollator:
    """把多模态样本列表整理成模型训练所需的 batch。

    输入样本来自 Dataset，格式类似：

        {
            "id": "...",
            "image_path": "...",
            "messages": [
                {"role": "user", "content": "<image>\\nDescribe this image."},
                {"role": "assistant", "content": "a cat on a sofa"},
            ],
            "source": "llava_pretrain",
        }

    输出 batch：

        {
            "input_ids":      LongTensor[B, L],
            "attention_mask": LongTensor[B, L],
            "labels":         LongTensor[B, L],
            "pixel_values":   FloatTensor[B, 3, 384, 384],
            "image_paths":    list[str],
            "ids":            list[str],
            "sources":        list[str],
        }

    labels 的规则：
        - prompt/user/image/system 部分为 IGNORE_INDEX
        - assistant answer 和 <|im_end|> 为正常 token id
        - padding 部分为 IGNORE_INDEX
    """

    def __init__(self, config: CollatorConfig) -> None:
        self.config = config
        self.tokenizer = self._load_tokenizer(
            config.tokenizer_path,
            local_files_only=config.local_files_only,
            padding_side=config.padding_side,
        )
        self.image_processor = SiglipImageProcessor(
            processor_path=config.image_processor_path,
            image_size=config.image_size,
            dynamic_resolution=config.dynamic_resolution,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )

        self.image_token = config.image_token
        self.image_token_id = self._ensure_image_token(self.image_token)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("VLMDataCollator 收到了空 examples。")

        image_paths = [self._require_str(example, "image_path") for example in examples]
        image_batch = self.image_processor.process_batch(image_paths)

        tokenized_examples = [self._tokenize_example(example) for example in examples]
        text_batch = self._pad_text_batch(tokenized_examples)

        return {
            **text_batch,
            "pixel_values": image_batch.pixel_values,
            "image_infos": image_batch.infos,
            "image_paths": image_paths,
            "ids": [example.get("id", "") for example in examples],
            "sources": [example.get("source", "") for example in examples],
            # Stage 3 的定量评估需要保留这些元信息。
            # 训练时模型不会读取它们，所以对 Stage 1/Stage 2 没有行为影响。
            "tasks": [example.get("task", "") for example in examples],
            "answers": [example.get("answers", []) for example in examples],
            "evals": [example.get("eval", {}) for example in examples],
            "image_token_id": self.image_token_id,
        }

    @staticmethod
    def _load_tokenizer(tokenizer_path: str, local_files_only: bool, padding_side: str):
        """加载 Qwen tokenizer。"""

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "VLMDataCollator 需要 transformers。安装命令："
                "python -m pip install transformers"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        tokenizer.padding_side = padding_side

        # Qwen tokenizer 通常没有单独 pad_token。训练时常用 eos 作为 pad。
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _ensure_image_token(self, image_token: str) -> int:
        """确保 ``<image>`` 是 tokenizer 里的单个特殊 token。

        如果不这样做，Qwen tokenizer 可能把 ``<image>`` 拆成多个 subword token。
        后续模型替换 visual embeddings 时，单个特殊 token 更容易定位。

        注意：
            如果这里新增了 token，后面加载 Qwen 模型时必须调用
            ``resize_token_embeddings(len(tokenizer))``，否则 embedding 表大小不匹配。
        """

        token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        unk_id = self.tokenizer.unk_token_id
        token_missing = token_id is None or token_id == unk_id

        if token_missing:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [image_token]}
            )
            if num_added <= 0:
                raise RuntimeError(f"尝试添加 image token 失败：{image_token!r}")

        token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"无法获得 image token id：{image_token!r}")

        return int(token_id)

    def _tokenize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """格式化并 tokenize 单条样本，同时构造未 padding 的 labels。"""

        raw_messages = example.get("messages")
        if not isinstance(raw_messages, list):
            raise TypeError("example['messages'] 必须是 list。")

        messages = ensure_image_token(raw_messages, image_token=self.image_token)
        normalized_messages = normalize_messages(messages)

        input_ids: list[int] = []
        labels: list[int] = []
        prompt_parts: list[str] = []

        # 多轮 SFT 的 label mask：
        #   - system/user 的完整 ChatML 片段不计算 loss
        #   - assistant header 不计算 loss
        #   - assistant 内容 + <|im_end|>\n 计算 loss
        #
        # 这样一条多轮样本中的每个 assistant 回复都会提供监督信号。
        for message in normalized_messages:
            if message.role == "assistant":
                header_text = f"{IM_START}assistant\n"
                answer_text = message.content + f"{IM_END}\n"

                header_ids = self._encode_text(header_text)
                answer_ids = self._encode_text(answer_text)

                input_ids.extend(header_ids)
                input_ids.extend(answer_ids)
                labels.extend([IGNORE_INDEX] * len(header_ids))
                labels.extend(answer_ids.copy())

                prompt_parts.append(header_text)
                prompt_parts.append(answer_text)
            else:
                turn_text = format_qwen_turn(message.role, message.content)
                turn_ids = self._encode_text(turn_text)

                input_ids.extend(turn_ids)
                labels.extend([IGNORE_INDEX] * len(turn_ids))
                prompt_parts.append(turn_text)

        if self.config.max_length is not None and len(input_ids) > self.config.max_length:
            input_ids, labels = self._truncate(input_ids, labels, self.config.max_length)

        image_token_count = input_ids.count(self.image_token_id)
        if image_token_count != 1:
            raise ValueError(
                "当前第一版 collator 要求每条样本恰好包含 1 个 image token。"
                f"实际 image_token_count={image_token_count}, id={example.get('id')}。"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt": "".join(prompt_parts),
            "prompt_without_answer": "",
            "assistant_text": "",
        }

    def _encode_text(self, text: str) -> list[int]:
        """不添加额外 BOS/EOS，直接把 ChatML 文本编码成 token ids。"""

        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    def _truncate(
        self,
        input_ids: list[int],
        labels: list[int],
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """截断过长样本。

        第一版采用右截断，保留开头的 user prompt 和前半段 answer。
        这对 Stage 1 caption 数据通常够用。后面做长文档/多轮对话时，再考虑更精细的
        截断策略。
        """

        if max_length <= 0:
            raise ValueError(f"max_length 必须为正数，当前为 {max_length}。")

        return input_ids[:max_length], labels[:max_length]

    def _pad_text_batch(self, tokenized_examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        """对 input_ids / labels / attention_mask 做 padding。"""

        max_len = max(len(item["input_ids"]) for item in tokenized_examples)
        pad_token_id = int(self.tokenizer.pad_token_id)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in tokenized_examples:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)

            if self.tokenizer.padding_side == "right":
                padded_input_ids = input_ids + [pad_token_id] * pad_len
                attention_mask = [1] * len(input_ids) + [0] * pad_len
                padded_labels = labels + [IGNORE_INDEX] * pad_len
            elif self.tokenizer.padding_side == "left":
                padded_input_ids = [pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + [1] * len(input_ids)
                padded_labels = [IGNORE_INDEX] * pad_len + labels
            else:
                raise ValueError(f"不支持的 padding_side：{self.tokenizer.padding_side!r}")

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    @staticmethod
    def _require_str(example: dict[str, Any], key: str) -> str:
        value = example.get(key)
        if not isinstance(value, str):
            raise TypeError(f"example[{key!r}] 必须是字符串，当前为 {value!r}。")
        return value


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 这里组合前面已经写好的：
    #   LlavaPretrainDataset
    #   SiglipImageProcessor
    #   Qwen tokenizer
    #   VLMDataCollator
    #
    # 检查最终 batch 是否包含：
    #   input_ids: [B, L]
    #   attention_mask: [B, L]
    #   labels: [B, L]
    #   pixel_values: [B, 3, 384, 384]
    try:
        from .llava_pretrain_dataset import LlavaPretrainDataset
    except ImportError:
        from llava_pretrain_dataset import LlavaPretrainDataset

    dataset = LlavaPretrainDataset(
        annotation_path="/root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        image_root="/root/autodl-tmp/hf_datasets/LLaVA-Pretrain",
        verify_images=False,
        max_samples=2,
    )

    collator = VLMDataCollator(
        CollatorConfig(
            tokenizer_path="/root/autodl-tmp/hf_models/Qwen3-1.7B",
            image_processor_path="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
            image_size=384,
            max_length=512,
        )
    )

    examples = [dataset[0], dataset[1]]
    batch = collator(examples)

    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
    print("pixel_values shape:", tuple(batch["pixel_values"].shape))
    print("image_token_id:", batch["image_token_id"])
    print("image token counts:", (batch["input_ids"] == batch["image_token_id"]).sum(dim=1).tolist())
    print("参与 loss 的 token 数:", (batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist())

    first_labels = batch["labels"][0]
    first_input_ids = batch["input_ids"][0]
    loss_positions = first_labels != IGNORE_INDEX
    decoded_loss_text = collator.tokenizer.decode(
        first_input_ids[loss_positions],
        skip_special_tokens=False,
    )
    print("\n第 1 条样本参与 loss 的文本:")
    print(decoded_loss_text)

    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert tuple(batch["pixel_values"].shape[:2]) == (2, 3)
    assert tuple(batch["pixel_values"].shape[-2:]) == (384, 384)
    assert (batch["input_ids"] == batch["image_token_id"]).sum(dim=1).tolist() == [1, 1]
    print("\nVLMDataCollator sanity check 通过。")

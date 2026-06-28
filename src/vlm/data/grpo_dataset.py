"""GRPO 训练用数据集读取脚本。

SFT 数据里 ``messages`` 通常包含：

    user -> assistant

因为 SFT 要直接对 assistant answer 计算交叉熵 loss。

GRPO 数据不一样。它只需要给模型一个 prompt，让模型自己采样多个回答，然后根据
``reward`` 字段打分。因此 GRPO 样本通常只有 user 消息：

    {
      "id": "abo_B000_color_grpo",
      "image_path": "/abs/path/to/image.jpg",
      "messages": [
        {"role": "user", "content": "<image>\\nWhat is the product color?"}
      ],
      "answer": "Black",
      "reward": {
        "type": "normalized_exact_match",
        "answers": ["Black"],
        "target_field": "color"
      }
    }

这个 Dataset 只负责读取和校验，不做 tokenize、不读图片。后续 GRPO trainer 可以复用
collator 里的 tokenizer/image_processor 逻辑，但需要自己构造 generation prompt，而不是
构造 assistant label。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset


class GRPODataset(Dataset):
    """读取 Stage 4 GRPO JSON。"""

    def __init__(
        self,
        annotation_path: str | Path,
        verify_images: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"GRPO 标注文件不存在：{self.annotation_path}")

        self.samples = self._load_json(self.annotation_path)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples 必须为正数，当前为 {max_samples}。")
            self.samples = self.samples[:max_samples]
        if verify_images:
            self.samples = self._filter_existing_images(self.samples)
        if not self.samples:
            raise ValueError(f"GRPO 数据集为空：{self.annotation_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._to_grpo_sample(self.samples[index])
        self._validate_sample(sample)
        return sample

    @staticmethod
    def _load_json(annotation_path: Path) -> list[dict[str, Any]]:
        """读取 JSON list。"""

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(
                "GRPO 标注文件顶层应该是 list，"
                f"但实际是 {type(data).__name__}。"
            )
        return data

    @staticmethod
    def _filter_existing_images(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """过滤图片不存在的样本。"""

        kept: list[dict[str, Any]] = []
        missing = 0
        for sample in samples:
            image_path = sample.get("image_path")
            if isinstance(image_path, str) and Path(image_path).is_file():
                kept.append(sample)
            else:
                missing += 1
        if missing:
            print(f"[GRPODataset] 过滤掉 {missing} 条图片缺失样本。")
        return kept

    @staticmethod
    def _to_grpo_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """把 SFT 两轮样本按需转成 GRPO prompt 样本。

        Stage4 的 SFT 数据是 ``user -> assistant`` 两轮格式；GRPO 训练只需要 user
        prompt，让模型自己生成多个候选答案，再用规则 reward 打分。因此这里在读取时做
        一个轻量转换：

        - 如果样本已经只有 1 条 user message，说明它本来就是 GRPO 格式，直接返回。
        - 如果样本有 2 条消息，则保留 user prompt，把 assistant 内容或 answers 作为
          reward.references。

        这样训练脚本可以直接复用 SFT 的 balanced 数据，避免 GRPO 只覆盖短答案任务而
        漏掉 title_generation / attribute_summary。
        """

        messages = sample.get("messages")
        if not isinstance(messages, list):
            return sample
        if len(messages) == 1:
            return sample
        if len(messages) != 2:
            return sample

        user_message, assistant_message = messages
        answers = sample.get("answers")
        if not isinstance(answers, list) or not answers:
            answers = [assistant_message.get("content", "")]
        converted = dict(sample)
        converted["messages"] = [user_message]
        converted["answer"] = answers[0]
        converted["reward"] = {
            "type": "task_adaptive",
            "answers": answers,
            "target_field": sample.get("task", "unknown"),
            "normalization": "lower_strip_punct_space",
        }
        return converted

    @staticmethod
    def _validate_sample(sample: dict[str, Any]) -> None:
        """校验 GRPO 样本的必要字段。"""

        sample_id = sample.get("id", "<unknown>")
        image_path = sample.get("image_path")
        messages = sample.get("messages")
        reward = sample.get("reward")

        if not isinstance(image_path, str):
            raise TypeError(f"样本 {sample_id} 的 image_path 必须是字符串。")
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(f"样本 {sample_id} 的 messages 应该只包含 1 条 user prompt。")
        if messages[0].get("role") != "user":
            raise ValueError(f"样本 {sample_id} 的唯一消息必须是 user。")
        if not isinstance(reward, dict):
            raise TypeError(f"样本 {sample_id} 缺少 reward 配置。")
        answers = reward.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"样本 {sample_id} 的 reward.answers 必须是非空 list。")


if __name__ == "__main__":
    default_path = Path("/root/autodl-tmp/hf_datasets/stage4_ecommerce/stage4_abo/grpo/train.json")
    dataset = GRPODataset(default_path, verify_images=True, max_samples=3)
    print("样本数:", len(dataset))
    for item in dataset:
        print(item["id"], item["task"], item["reward"])
        print(item["messages"][0]["content"])

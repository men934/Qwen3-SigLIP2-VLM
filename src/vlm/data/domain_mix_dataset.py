"""Stage 3 垂域混合数据集读取脚本。

Stage 3 的原始数据来自多个公开数据集：

    DocVQA / InfographicVQA / TextVQA / ChartQA / CORD / FUNSD

这些数据集的字段命名、图片存储方式、答案格式都不一样。为了让训练脚本保持简单，
我们先用 ``build_stage3_domain_mix.py`` 做一次离线转换，生成统一 JSON：

    {
      "id": "docvqa_train_00000001",
      "image_path": "/abs/path/to/image.png",
      "messages": [
        {"role": "user", "content": "<image>\\nQuestion..."},
        {"role": "assistant", "content": "answer"}
      ],
      "source": "docvqa",
      "task": "document_qa",
      "answers": ["answer", "..."],
      "eval": {"metric": "vqa"}
    }

这个 Dataset 只负责读取统一 JSON 和做轻量校验。图片读取、tokenize、label mask 仍然交给
``VLMDataCollator``，这样 Stage 2 和 Stage 3 的训练路径可以共用同一套 batch 格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset


class DomainMixDataset(Dataset):
    """读取 Stage 3 统一格式 JSON。

    Args:
        annotation_path:
            ``build_stage3_domain_mix.py`` 生成的 train/val/test JSON。

        verify_images:
            如果为 True，初始化时过滤掉图片文件不存在的样本。全量训练时为了启动更快，
            可以先关闭；做数据检查时建议打开。

        max_samples:
            可选，只取前 N 条，方便快速验证训练脚本和评估脚本。
    """

    def __init__(
        self,
        annotation_path: str | Path,
        verify_images: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.verify_images = verify_images

        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Stage 3 标注文件不存在：{self.annotation_path}")

        self.samples = self._load_json(self.annotation_path)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples 必须为正数，当前为 {max_samples}。")
            self.samples = self.samples[:max_samples]

        if verify_images:
            self.samples = self._filter_existing_images(self.samples)

        if not self.samples:
            raise ValueError(f"Stage 3 数据集为空：{self.annotation_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        self._validate_sample(sample)
        return sample

    @staticmethod
    def _load_json(annotation_path: Path) -> list[dict[str, Any]]:
        """读取统一 JSON，并检查顶层结构。"""

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(
                "Stage 3 标注文件顶层应该是 list，"
                f"但实际是 {type(data).__name__}。"
            )
        return data

    @staticmethod
    def _filter_existing_images(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """过滤图片缺失样本。"""

        kept: list[dict[str, Any]] = []
        missing = 0
        for sample in samples:
            image_path = sample.get("image_path")
            if isinstance(image_path, str) and Path(image_path).is_file():
                kept.append(sample)
            else:
                missing += 1
        if missing:
            print(f"[DomainMixDataset] 过滤掉 {missing} 条图片缺失样本。")
        return kept

    @staticmethod
    def _validate_sample(sample: dict[str, Any]) -> None:
        """做最小校验，尽早暴露格式错误。"""

        sample_id = sample.get("id", "<unknown>")
        image_path = sample.get("image_path")
        messages = sample.get("messages")

        if not isinstance(image_path, str):
            raise TypeError(f"样本 {sample_id} 的 image_path 必须是字符串。")
        if not isinstance(messages, list) or len(messages) < 2:
            raise TypeError(f"样本 {sample_id} 的 messages 至少需要 user/assistant 两轮。")
        if messages[0].get("role") != "user":
            raise ValueError(f"样本 {sample_id} 第一轮必须是 user。")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"样本 {sample_id} 最后一轮必须是 assistant。")


if __name__ == "__main__":
    # 临时 sanity check，方便直接检查构建后的 JSON。
    default_path = Path("/root/autodl-tmp/hf_datasets/domain_mix/stage3_mix/train.json")
    dataset = DomainMixDataset(default_path, verify_images=True, max_samples=3)
    print("样本数:", len(dataset))
    for item in dataset:
        print(item["id"], item["source"], item["task"], item["image_path"])
        print(item["messages"][0]["content"][:120].replace("\n", "\\n"))
        print(item["messages"][1]["content"][:120].replace("\n", "\\n"))

"""LLaVA-Instruct 数据集读取脚本。

Stage 2 的目标是多模态指令微调，不再只是让模型根据图片生成 caption，而是训练：

    image + user question -> assistant answer

LLaVA-Instruct 的原始样本通常长这样：

    {
      "id": "000000033471",
      "image": "000000033471.jpg",
      "conversations": [
        {"from": "human", "value": "<image> What are the colors of the bus?"},
        {"from": "gpt", "value": "The bus is white and red."},
        {"from": "human", "value": "What feature can be seen on the back?"},
        {"from": "gpt", "value": "There is an advertisement."}
      ]
    }

这个 Dataset 仍然遵守项目当前的数据边界：

    Dataset:
        只读取 JSON，解析图片路径，转换 role。

    Collator:
        负责读取图片、tokenize、多轮 assistant label mask。

    Model:
        负责把 <image> token 替换成视觉 embedding。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset


ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
}


@dataclass(frozen=True)
class LlavaInstructExample:
    """Dataset 对外返回的统一样本结构。"""

    id: str
    image_path: str
    messages: list[dict[str, str]]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "messages": self.messages,
            "source": self.source,
        }


class LlavaInstructDataset(Dataset):
    """读取 LLaVA-Instruct JSON，并返回统一 messages 格式。

    Args:
        annotation_path:
            例如 ``llava_instruct_150k.json``。

        image_root:
            图片根目录。对我们当前下载的数据，推荐使用
            ``/root/autodl-tmp/hf_datasets/coco/train2014``。

        image_token:
            图片占位符，默认 ``<image>``。

        verify_images:
            如果为 True，初始化时过滤掉图片缺失样本。

        max_samples:
            可选，只取前 N 条，方便小规模 sanity check。
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        image_token: str = "<image>",
        verify_images: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.image_token = image_token
        self.verify_images = verify_images

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"标注文件不存在：{self.annotation_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"图片根目录不存在：{self.image_root}")

        self.samples = self._load_json(self.annotation_path)

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples 必须为正数，当前为 {max_samples}。")
            self.samples = self.samples[:max_samples]

        if verify_images:
            self.samples = self._filter_existing_images(self.samples)

        if not self.samples:
            raise ValueError("LlavaInstructDataset 没有可用样本。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._convert_raw_example(self.samples[index]).to_dict()

    @staticmethod
    def _load_json(annotation_path: Path) -> list[dict[str, Any]]:
        """读取 LLaVA-Instruct JSON。"""

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(
                "LLaVA-Instruct 标注文件顶层应该是 list，"
                f"但实际是 {type(data).__name__}。"
            )
        return data

    def _filter_existing_images(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """过滤图片文件不存在的样本。"""

        kept: list[dict[str, Any]] = []
        missing = 0
        for sample in samples:
            image_rel_path = sample.get("image")
            if not isinstance(image_rel_path, str):
                missing += 1
                continue
            if self._resolve_image_path(image_rel_path).is_file():
                kept.append(sample)
            else:
                missing += 1

        if missing:
            print(f"[LlavaInstructDataset] 过滤掉 {missing} 条图片缺失样本。")
        return kept

    def _convert_raw_example(self, raw: dict[str, Any]) -> LlavaInstructExample:
        """把原始样本转换成项目内部统一格式。"""

        sample_id = self._require_str(raw, "id")
        image_rel_path = self._require_str(raw, "image")
        conversations = raw.get("conversations")

        if not isinstance(conversations, list):
            raise TypeError(f"样本 {sample_id} 的 conversations 必须是 list。")

        messages: list[dict[str, str]] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                raise TypeError(f"样本 {sample_id} 中存在非 dict 对话轮次：{turn!r}")

            raw_role = self._require_str(turn, "from")
            content = self._require_str(turn, "value")
            role = self._map_role(raw_role, sample_id)
            messages.append({"role": role, "content": content})

        self._validate_messages(messages, sample_id)

        return LlavaInstructExample(
            id=sample_id,
            image_path=str(self._resolve_image_path(image_rel_path)),
            messages=messages,
            source="llava_instruct",
        )

    def _resolve_image_path(self, image_rel_path: str) -> Path:
        """把 LLaVA 的 image 字段解析成本地图片路径。

        当前常见情况：
            - llava_instruct_150k.json: ``000000033471.jpg``
            - COCO train2014 实际文件名: ``COCO_train2014_000000033471.jpg``
            - llava_v1_5_mix665k.json: ``coco/train2017/000000033471.jpg``

        因此这里会依次尝试：
            1. image_root / 原始相对路径
            2. image_root / basename
            3. image_root / COCO_train2014_basename
        """

        raw_path = Path(image_rel_path)
        candidates = [
            self.image_root / raw_path,
            self.image_root / raw_path.name,
        ]

        name = raw_path.name
        if name and not name.startswith("COCO_train2014_"):
            candidates.append(self.image_root / f"COCO_train2014_{name}")

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        # 不立即报错，留给 image processor 在真正读取时暴露具体路径。
        return candidates[-1]

    @staticmethod
    def _require_str(obj: dict[str, Any], key: str) -> str:
        value = obj.get(key)
        if not isinstance(value, str):
            raise TypeError(f"字段 {key!r} 必须是字符串，当前为 {value!r}。")
        return value

    @staticmethod
    def _map_role(raw_role: str, sample_id: str) -> str:
        try:
            return ROLE_MAP[raw_role]
        except KeyError as exc:
            raise ValueError(
                f"样本 {sample_id} 中出现未知角色 {raw_role!r}，"
                f"支持的角色是 {sorted(ROLE_MAP)}。"
            ) from exc

    def _validate_messages(self, messages: list[dict[str, str]], sample_id: str) -> None:
        """检查 LLaVA-Instruct 样本是否适合当前 VLM 训练。"""

        if len(messages) < 2:
            raise ValueError(f"样本 {sample_id} 至少需要 user 和 assistant 两轮消息。")
        if messages[0]["role"] != "user":
            raise ValueError(f"样本 {sample_id} 第一轮必须是 user。")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"样本 {sample_id} 最后一轮必须是 assistant。")

        user_text = "\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        image_count = user_text.count(self.image_token)
        if image_count == 0:
            # collator 会自动把 <image> 补到第一轮 user 前面。
            return
        if image_count > 1:
            raise ValueError(
                f"样本 {sample_id} 当前包含多个 {self.image_token}，"
                "第一版模型只支持单图输入。"
            )


if __name__ == "__main__":
    dataset = LlavaInstructDataset(
        annotation_path="/root/autodl-tmp/hf_datasets/LLaVA-Instruct-150K/llava_instruct_150k.json",
        image_root="/root/autodl-tmp/hf_datasets/coco/train2014",
        max_samples=3,
    )

    print(f"dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        sample = dataset[i]
        print("=" * 80)
        print("id:", sample["id"])
        print("image_path:", sample["image_path"])
        print("image_exists:", Path(sample["image_path"]).is_file())
        print("source:", sample["source"])
        print("messages:")
        for message in sample["messages"]:
            content = message["content"].replace("\n", " ")
            print(f"  {message['role']}: {content[:180]}")

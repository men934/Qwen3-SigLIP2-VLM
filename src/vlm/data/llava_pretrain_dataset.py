"""LLaVA-Pretrain 数据集读取脚本。

这个 Dataset 负责读取 Stage 1 视觉-语言对齐数据：

    /root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json

LLaVA-Pretrain 的每条样本大致长这样：

    {
      "id": "004539375",
      "image": "00453/004539375.jpg",
      "conversations": [
        {
          "from": "human",
          "value": "Render a clear and concise summary of the photo.\\n<image>"
        },
        {
          "from": "gpt",
          "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
        }
      ]
    }

这里有一个重要设计：Dataset 不直接返回 tokenized input_ids，也不直接返回
pixel_values。它只把原始数据规整成统一格式：

    {
        "id": "...",
        "image_path": "...",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ],
        "source": "llava_pretrain",
    }

为什么这样设计？
    多模态训练里，tokenizer、image processor、image token 占位、label mask 都会在
    collator 里统一处理。如果每个 Dataset 自己 tokenize，后面混合 LLaVA、DocVQA、
    TextVQA、CORD 等数据时会很难维护。
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
class LlavaPretrainExample:
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


class LlavaPretrainDataset(Dataset):
    """读取 LLaVA-Pretrain JSON，并返回统一 messages 格式。

    Args:
        annotation_path:
            LLaVA-Pretrain 的 JSON 标注文件路径。

        image_root:
            图片根目录。样本里的 ``image`` 字段是相对路径，需要和这个根目录拼接。

        image_token:
            图片占位符。LLaVA 原始数据里通常使用 ``<image>``。

        verify_images:
            如果为 True，初始化时会过滤掉图片文件不存在的样本。
            这会扫描 55 万条样本，稍微慢一点，但训练前更稳。

            第一版默认 False，因为我们已经解压并核对过图片数量。

        max_samples:
            可选，只取前 N 条样本。用于快速 sanity check 或小规模调试。
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
            raise ValueError("LlavaPretrainDataset 没有可用样本。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.samples[index]
        example = self._convert_raw_example(raw)
        return example.to_dict()

    @staticmethod
    def _load_json(annotation_path: Path) -> list[dict[str, Any]]:
        """读取 LLaVA-Pretrain JSON 文件。"""

        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(
                "LLaVA-Pretrain 标注文件顶层应该是 list，"
                f"但实际是 {type(data).__name__}。"
            )

        return data

    def _filter_existing_images(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """过滤掉图片文件不存在的样本。"""

        kept: list[dict[str, Any]] = []
        missing = 0
        for sample in samples:
            image_rel_path = sample.get("image")
            if not isinstance(image_rel_path, str):
                missing += 1
                continue

            if (self.image_root / image_rel_path).is_file():
                kept.append(sample)
            else:
                missing += 1

        if missing:
            print(f"[LlavaPretrainDataset] 过滤掉 {missing} 条图片缺失样本。")

        return kept

    def _convert_raw_example(self, raw: dict[str, Any]) -> LlavaPretrainExample:
        """把 LLaVA 原始样本转换成项目内部统一格式。"""

        sample_id = self._require_str(raw, "id")
        image_rel_path = self._require_str(raw, "image")
        conversations = raw.get("conversations")

        if not isinstance(conversations, list):
            raise TypeError(f"样本 {sample_id} 的 conversations 必须是 list。")

        image_path = self.image_root / image_rel_path

        messages = []
        for turn in conversations:
            if not isinstance(turn, dict):
                raise TypeError(f"样本 {sample_id} 中存在非 dict 对话轮次：{turn!r}")

            raw_role = self._require_str(turn, "from")
            content = self._require_str(turn, "value")
            role = self._map_role(raw_role, sample_id)

            # 这里暂时保留 LLaVA 原始 prompt 里的 <image> 位置。
            # 有些样本是 "<image>\\nWhat is this?"，有些是 "xxx\\n<image>"。
            # 后面的 conversation/collator 会统一处理 image token 位置。
            messages.append({"role": role, "content": content})

        self._validate_messages(messages, sample_id)

        return LlavaPretrainExample(
            id=sample_id,
            image_path=str(image_path),
            messages=messages,
            source="llava_pretrain",
        )

    @staticmethod
    def _require_str(obj: dict[str, Any], key: str) -> str:
        """读取必需字符串字段。"""

        value = obj.get(key)
        if not isinstance(value, str):
            raise TypeError(f"字段 {key!r} 必须是字符串，当前为 {value!r}。")
        return value

    @staticmethod
    def _map_role(raw_role: str, sample_id: str) -> str:
        """把 LLaVA 的 human/gpt 角色映射成 user/assistant。"""

        try:
            return ROLE_MAP[raw_role]
        except KeyError as exc:
            raise ValueError(
                f"样本 {sample_id} 中出现未知角色 {raw_role!r}，"
                f"支持的角色是 {sorted(ROLE_MAP)}。"
            ) from exc

    def _validate_messages(self, messages: list[dict[str, str]], sample_id: str) -> None:
        """做一些轻量级格式检查，尽早发现坏样本。"""

        if len(messages) < 2:
            raise ValueError(f"样本 {sample_id} 至少需要 user 和 assistant 两轮消息。")

        if messages[0]["role"] != "user":
            raise ValueError(f"样本 {sample_id} 第一轮必须是 user。")

        if messages[-1]["role"] != "assistant":
            raise ValueError(f"样本 {sample_id} 最后一轮必须是 assistant。")

        user_text = "\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        if self.image_token not in user_text:
            raise ValueError(f"样本 {sample_id} 的 user 消息里没有 {self.image_token}。")


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 这里读取前 3 条 LLaVA-Pretrain 样本，确认：
    #   1. JSON 能正常加载
    #   2. 相对图片路径能拼成绝对路径
    #   3. human/gpt 能映射成 user/assistant
    #   4. 输出格式统一为 image_path + messages
    annotation_path = Path(
        "/root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    )
    image_root = Path("/root/autodl-tmp/hf_datasets/LLaVA-Pretrain")

    dataset = LlavaPretrainDataset(
        annotation_path=annotation_path,
        image_root=image_root,
        verify_images=True,
        max_samples=3,
    )

    print("Dataset 长度:", len(dataset))
    for i in range(len(dataset)):
        item = dataset[i]
        print("\n--- 样本", i)
        print("id:", item["id"])
        print("source:", item["source"])
        print("image_path:", item["image_path"])
        print("image_exists:", Path(item["image_path"]).is_file())
        print("messages:")
        for message in item["messages"]:
            print(f"  - {message['role']}: {message['content']}")

    print("\nLLaVA-Pretrain Dataset sanity check 通过。")

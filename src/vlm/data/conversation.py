"""多模态对话样本的 prompt 构造工具。

Dataset 返回的是统一的 messages 格式：

    [
        {"role": "user", "content": "<image>\\nWhat is this?"},
        {"role": "assistant", "content": "a dragon kite flying in the blue sky"},
    ]

但语言模型训练需要的是一段完整文本。对 Qwen 系列模型来说，常用格式是 ChatML：

    <|im_start|>user
    <image>
    What is this?<|im_end|>
    <|im_start|>assistant
    a dragon kite flying in the blue sky<|im_end|>

这个文件的职责是：

    1. 规范 role 和 content
    2. 构造 Qwen 风格 prompt
    3. 保留 assistant answer 的文本边界信息
    4. 为后续 collator 做 label mask 打基础

注意：
    这个脚本目前只处理文本层面的 prompt 拼接，不做 tokenizer，也不做图片读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


Role = Literal["system", "user", "assistant"]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<image>"


@dataclass(frozen=True)
class Message:
    """一轮对话消息。"""

    role: Role
    content: str


@dataclass(frozen=True)
class FormattedConversation:
    """格式化后的对话文本，以及训练所需的文本边界信息。"""

    prompt: str
    assistant_text: str
    prompt_without_answer: str
    messages: list[Message]


def normalize_messages(raw_messages: list[dict[str, Any]]) -> list[Message]:
    """把 Dataset 输出的 dict messages 转成强约束的 Message 列表。

    这里做轻量校验，尽早发现坏样本。后续 collator 可以假设 messages 已经满足：

        - role 只可能是 system/user/assistant
        - content 一定是字符串
        - 至少有一轮 user 和一轮 assistant
        - 最后一轮是 assistant
    """

    if not isinstance(raw_messages, list):
        raise TypeError(f"raw_messages 必须是 list，当前为 {type(raw_messages).__name__}。")

    messages: list[Message] = []
    for i, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise TypeError(f"第 {i} 轮 message 必须是 dict，当前为 {raw!r}。")

        role = raw.get("role")
        content = raw.get("content")

        if role not in ("system", "user", "assistant"):
            raise ValueError(
                f"第 {i} 轮 message role 不合法：{role!r}。"
                "只支持 system/user/assistant。"
            )
        if not isinstance(content, str):
            raise TypeError(f"第 {i} 轮 message content 必须是字符串，当前为 {content!r}。")

        messages.append(Message(role=role, content=content))

    validate_messages(messages)
    return messages


def validate_messages(messages: list[Message]) -> None:
    """检查对话结构是否适合监督微调。"""

    if len(messages) < 2:
        raise ValueError("messages 至少需要包含 user 和 assistant 两轮。")

    roles = [message.role for message in messages]
    if "user" not in roles:
        raise ValueError("messages 中必须包含 user 消息。")
    if "assistant" not in roles:
        raise ValueError("messages 中必须包含 assistant 消息。")
    if messages[-1].role != "assistant":
        raise ValueError("当前训练样本要求最后一轮必须是 assistant。")

    # Stage 2 的 LLaVA-Instruct 样本经常是多轮对话：
    #   user -> assistant -> user -> assistant ...
    # 因此这里不再限制 assistant 只能出现一次。真正的 label mask 会在 collator
    # 中按每个 assistant span 分别构造。


def format_qwen_turn(role: Role, content: str) -> str:
    """把单轮消息格式化成 Qwen ChatML 片段。"""

    return f"{IM_START}{role}\n{content}{IM_END}\n"


def format_qwen_conversation(
    raw_messages: list[dict[str, Any]],
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> FormattedConversation:
    """把 messages 转成 Qwen ChatML prompt。

    Args:
        raw_messages:
            Dataset 返回的 messages。

        system_prompt:
            可选 system prompt。Stage 1 对齐训练通常不需要。

        add_generation_prompt:
            推理时可以设为 True，让 prompt 以 ``<|im_start|>assistant\\n`` 结尾，
            等模型继续生成。训练时应该保持 False，因为 assistant answer 已经在
            messages 里。

    Returns:
        ``FormattedConversation``。

    训练时最重要的是：
        ``prompt_without_answer`` + ``assistant_text`` 可以还原完整 ``prompt``。
        后续 collator 会用这个边界来决定哪些 token 计算 loss。
    """

    messages = normalize_messages(raw_messages)

    if system_prompt is not None:
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt 必须是字符串或 None。")
        messages = [Message(role="system", content=system_prompt), *messages]

    if add_generation_prompt:
        if messages[-1].role == "assistant":
            raise ValueError("add_generation_prompt=True 时，messages 最后一轮不应已有 assistant。")
        prompt = "".join(format_qwen_turn(message.role, message.content) for message in messages)
        prompt += f"{IM_START}assistant\n"
        return FormattedConversation(
            prompt=prompt,
            assistant_text="",
            prompt_without_answer=prompt,
            messages=messages,
        )

    assistant_message = messages[-1]
    if assistant_message.role != "assistant":
        raise ValueError("训练格式化要求最后一轮是 assistant。")

    prefix_messages = messages[:-1]
    prompt_without_answer = "".join(
        format_qwen_turn(message.role, message.content) for message in prefix_messages
    )
    prompt_without_answer += f"{IM_START}assistant\n"

    assistant_text = assistant_message.content
    prompt = prompt_without_answer + assistant_text + f"{IM_END}\n"

    return FormattedConversation(
        prompt=prompt,
        assistant_text=assistant_text,
        prompt_without_answer=prompt_without_answer,
        messages=messages,
    )


def ensure_image_token(
    messages: list[dict[str, Any]],
    image_token: str = DEFAULT_IMAGE_TOKEN,
) -> list[dict[str, str]]:
    """确保 user 消息中含有 image token。

    LLaVA-Pretrain 原始数据本身已经带 ``<image>``，但后续我们处理 DocVQA、CORD
    这类数据时，可能需要手动把 image token 插进去。

    这个函数会返回一个新的 messages 列表，不会原地修改传入对象。
    """

    normalized = normalize_messages(messages)
    has_image = any(
        message.role == "user" and image_token in message.content
        for message in normalized
    )
    if has_image:
        return [{"role": message.role, "content": message.content} for message in normalized]

    output: list[dict[str, str]] = []
    inserted = False
    for message in normalized:
        content = message.content
        if message.role == "user" and not inserted:
            content = f"{image_token}\n{content}"
            inserted = True
        output.append({"role": message.role, "content": content})

    if not inserted:
        raise ValueError("没有找到 user 消息，无法插入 image token。")

    return output


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 这里模拟 LLaVA-Pretrain Dataset 输出的一条样本，检查：
    #   1. role/content 规范化
    #   2. Qwen ChatML prompt 拼接
    #   3. assistant answer 边界
    #   4. image token 自动插入逻辑
    sample_messages = [
        {"role": "user", "content": "<image>\nWhat is this?"},
        {"role": "assistant", "content": "a dragon kite flying in the blue sky"},
    ]

    formatted = format_qwen_conversation(sample_messages)

    print("完整 prompt:")
    print(formatted.prompt)
    print("prompt_without_answer:")
    print(formatted.prompt_without_answer)
    print("assistant_text:")
    print(formatted.assistant_text)

    assert formatted.prompt.startswith("<|im_start|>user\n")
    assert formatted.prompt_without_answer.endswith("<|im_start|>assistant\n")
    assert formatted.assistant_text == "a dragon kite flying in the blue sky"
    assert formatted.prompt == (
        formatted.prompt_without_answer + formatted.assistant_text + "<|im_end|>\n"
    )

    no_image_messages = [
        {"role": "user", "content": "Describe this image briefly."},
        {"role": "assistant", "content": "a small object on a table"},
    ]
    with_image_messages = ensure_image_token(no_image_messages)

    print("\n自动插入 image token 后:")
    for message in with_image_messages:
        print(f"- {message['role']}: {message['content']}")

    assert with_image_messages[0]["content"].startswith("<image>\n")
    print("\nConversation sanity check 通过。")

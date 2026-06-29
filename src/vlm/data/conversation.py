"""Prompt formatting utilities for multimodal conversations.

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
    """Normalize raw dict messages and validate their roles/content."""

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
    """Format messages as Qwen ChatML."""

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
    """Return a copy of messages with an image token in the first user turn."""

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
    # Quick prompt formatting check.
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
    print("\nConversation quick check passed.")

"""把视觉特征投影到语言模型的 hidden space。

经过 SigLIP2 和 PatchMerger 之后，视觉 token 仍然处在视觉模型自己的特征空间里。
Qwen 不能直接消费这些特征，因为 Qwen 的 token embedding 位于另一套 hidden space。

Projector 的作用就是做这层桥接：

    SigLIP2 patch features
        -> 2x2 PatchMerger
        -> merged visual tokens: [B, N, 4 * vision_hidden_size]
        -> MLPProjector
        -> Qwen-space visual tokens: [B, N, qwen_hidden_size]

按我们当前选择的模型，典型维度是：

    SigLIP2 SO400M hidden size: 1152
    2x2 merge 后输入维度:       4 * 1152 = 4608
    Qwen3-1.7B hidden size:     2048

所以第一版真实 projector 大概率会是：

    MLPProjector(input_dim=4608, hidden_dim=2048, output_dim=2048)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


ActivationName = Literal["gelu", "silu", "relu", "tanh"]


def build_activation(name: ActivationName) -> nn.Module:
    """根据配置字符串构造激活函数模块。"""

    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()

    # 正常情况下 Literal 类型会阻止走到这里；保留这个分支便于防御式报错。
    raise ValueError(f"不支持的激活函数：{name!r}。")


class MLPProjector(nn.Module):
    """用于视觉-语言对齐的两层 MLP projector。

    这是 LLaVA 风格 VLM 中非常常见、也比较稳妥的 projector 选择。
    它有足够的表达能力把视觉特征对齐到 LLM embedding space，同时参数量又不大，
    Stage 1 只训练它和 PatchMerger 时成本比较低。

    Shape:
        输入：
            ``x`` 的 shape 是 ``[B, N, input_dim]``。

        输出：
            Tensor 的 shape 是 ``[B, N, output_dim]``。

    Args:
        input_dim:
            merged visual tokens 的特征维度。

        hidden_dim:
            MLP 中间层宽度。第一版通常可以直接设成 Qwen hidden size。

        output_dim:
            语言模型 hidden size。Qwen3-1.7B 是 2048。

        activation:
            两个 Linear 层之间使用的非线性激活函数。

        bias:
            Linear 层是否使用 bias。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: ActivationName = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._validate_dim("input_dim", input_dim)
        self._validate_dim("hidden_dim", hidden_dim)
        self._validate_dim("output_dim", output_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            build_activation(activation),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        """把 merged visual tokens 投影到 Qwen hidden space。"""

        if x.ndim != 3:
            raise ValueError(
                "MLPProjector 期望 x 的 shape 为 [B, N, input_dim]，"
                f"但实际得到 {tuple(x.shape)}。"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "x 的最后一维和 projector input_dim 不匹配："
                f"x.shape[-1]={x.shape[-1]}, input_dim={self.input_dim}。"
            )

        return self.net(x)

    @staticmethod
    def _validate_dim(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} 必须为正数，当前为 {value}。")


class LinearProjector(nn.Module):
    """单层 Linear projector，用于后续消融实验。

    我们主线会使用 ``MLPProjector``。这里保留一个简单线性 projector，是为了之后
    做实验对比：

        Linear projector vs 2-layer MLP projector

    如果 MLP 在 Stage 1/2 里明显更好，我们就能在项目报告里解释额外参数量的价值。
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        MLPProjector._validate_dim("input_dim", input_dim)
        MLPProjector._validate_dim("output_dim", output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "LinearProjector 期望 x 的 shape 为 [B, N, input_dim]，"
                f"但实际得到 {tuple(x.shape)}。"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "x 的最后一维和 projector input_dim 不匹配："
                f"x.shape[-1]={x.shape[-1]}, input_dim={self.input_dim}。"
            )

        return self.proj(x)


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 假设 PatchMerger 已经从一个很小的假特征图里产出了 visual tokens。
    # 例如：
    #   B = 2
    #   N = 6 个 merged visual tokens
    #   input_dim = 12，可以理解成 toy C=3 时的 2*2*C
    #
    # Projector 会独立映射每个 visual token：
    #   [B, N, input_dim] -> [B, N, output_dim]
    torch.manual_seed(0)

    batch_size = 2
    num_visual_tokens = 6
    input_dim = 12
    hidden_dim = 16
    output_dim = 8

    x = torch.randn(batch_size, num_visual_tokens, input_dim)

    mlp_projector = MLPProjector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        activation="gelu",
    )
    y = mlp_projector(x)

    print("MLPProjector")
    print("输入:", tuple(x.shape))
    print("输出:", tuple(y.shape))
    print("参数量:", sum(p.numel() for p in mlp_projector.parameters()))

    linear_projector = LinearProjector(input_dim=input_dim, output_dim=output_dim)
    y_linear = linear_projector(x)

    print("\nLinearProjector")
    print("输入:", tuple(x.shape))
    print("输出:", tuple(y_linear.shape))
    print("参数量:", sum(p.numel() for p in linear_projector.parameters()))

    assert tuple(y.shape) == (batch_size, num_visual_tokens, output_dim)
    assert tuple(y_linear.shape) == (batch_size, num_visual_tokens, output_dim)
    print("\nProjector sanity check 通过。")

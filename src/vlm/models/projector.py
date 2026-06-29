"""Project visual features into the language model hidden space.

Expected production shape:
    [B, N, 4 * 1152] -> [B, N, 2048]
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

    raise ValueError(f"不支持的激活函数：{name!r}。")


class MLPProjector(nn.Module):
    """Two-layer MLP projector.

    Input: ``[B, N, input_dim]``.
    Output: ``[B, N, output_dim]``.
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
    """Single-layer projector used for ablation."""

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
    # Quick shape check for MLPProjector and LinearProjector.
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
    print("\nProjector quick check passed.")

"""ViT/SigLIP 视觉 token 的 Patch Merger。

这个模块实现一个很小但很关键的操作：在把视觉 token 送入语言模型之前，
先把相邻的图像 patch token 做局部合并，从而减少视觉 token 数量。

在 ViT/SigLIP 这类视觉编码器里，一张图片通常会被表示成二维 patch 网格：

    [B, H, W, C]

其中：
    B: batch size
    H: patch 网格高度
    W: patch 网格宽度
    C: 视觉特征维度

2x2 Patch Merger 会把每个局部 2x2 patch 块合并成 1 个 token。这里采用
concat 方式，也就是把 4 个 patch 的特征在通道维拼接起来：

    [B, H, W, C] -> [B, H/2, W/2, 4*C]

再把二维网格 flatten 之后，输出会变成：

    [B, (H/2)*(W/2), 4*C]

这样视觉 token 数会降到原来的 1/4，同时局部 2x2 patch 的信息仍然被保留。
后面的 projector 再负责把 4*C 维特征映射到 Qwen 的 hidden size。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GridSize:
    """二维 patch 网格大小的简单容器。"""

    height: int
    width: int

    def as_tuple(self) -> tuple[int, int]:
        return self.height, self.width


class PatchMerger(nn.Module):
    """把局部 m x m 视觉 patch 合并成更少的视觉 token。

    默认配置是 ``merge_size=2``，也就是最常见的 2x2 patch merge。

    支持两种输入格式：
        1. ``[B, H, W, C]``
           patch 的二维空间网格已经显式存在。

        2. ``[B, N, C]`` 加 ``grid_size=(H, W)``
           Hugging Face 的 ViT/SigLIP 输出经常是这种 flatten 后的格式。
           这种情况下必须满足 ``N == H * W``。

    输出：
        ``merged_tokens``:
            shape 为 ``[B, merged_H * merged_W, merge_size^2 * C]`` 的 Tensor。

        ``merged_grid_size``:
            ``GridSize(merged_H, merged_W)``。

    为什么用 concat，而不是 average？
        average 会保持通道维 C 不变，但会丢掉一些局部细节。
        concat 会保留 m*m 个 patch 的原始特征，然后交给后面的 MLP projector
        学习怎么融合它们。
    """

    def __init__(self, merge_size: int = 2, allow_truncate: bool = False) -> None:
        """创建 PatchMerger。

        Args:
            merge_size:
                空间合并大小。``2`` 表示每个 2x2 patch 块合并成 1 个 token。
                ``merge_size`` 必须是正整数。

            allow_truncate:
                如果为 False，H 和 W 必须能被 ``merge_size`` 整除。
                如果为 True，会丢弃底部/右侧无法组成完整 merge block 的多余行列。

                VLM 训练早期建议保持 False，让 shape 问题尽早暴露。
                后续做动态分辨率时，resize 逻辑应该保证 H/W 能被
                ``patch_size * merge_size`` 对齐。
        """

        super().__init__()
        if merge_size <= 0:
            raise ValueError(f"merge_size 必须为正数，当前为 {merge_size}。")

        self.merge_size = merge_size
        self.allow_truncate = allow_truncate

    def forward(
        self,
        x: Tensor,
        grid_size: Optional[tuple[int, int] | GridSize] = None,
    ) -> tuple[Tensor, GridSize]:
        """合并视觉 patch token。

        Args:
            x:
                输入可以是 ``[B, H, W, C]``，也可以是 ``[B, N, C]``。

            grid_size:
                只有当 ``x`` 是 ``[B, N, C]`` 时才需要传入。
                它告诉 merger 如何把 flatten 后的一维 token 序列还原成二维网格。

        Returns:
            ``(merged_tokens, merged_grid_size)``。
        """

        if x.ndim == 4:
            # 输入已经是显式二维空间格式：[B, H, W, C]。
            x_4d = x
        elif x.ndim == 3:
            # 输入是 flatten 后的 patch-token 格式：[B, N, C]。
            if grid_size is None:
                raise ValueError("当 x 的 shape 为 [B, N, C] 时必须传入 grid_size。")

            grid = self._normalize_grid_size(grid_size)
            batch_size, num_tokens, channels = x.shape
            expected_tokens = grid.height * grid.width

            if num_tokens != expected_tokens:
                raise ValueError(
                    "flatten 后的 token 数和 grid_size 不匹配："
                    f"N={num_tokens}, H*W={expected_tokens}, "
                    f"grid_size={grid.as_tuple()}。"
                )

            x_4d = x.reshape(batch_size, grid.height, grid.width, channels)
        else:
            raise ValueError(
                "PatchMerger 期望 x 的 shape 为 [B, H, W, C] 或 [B, N, C]，"
                f"但实际得到 {tuple(x.shape)}。"
            )

        return self._merge_4d(x_4d)

    def _merge_4d(self, x: Tensor) -> tuple[Tensor, GridSize]:
        """合并显式二维格式的 ``[B, H, W, C]`` patch 网格。"""

        batch_size, height, width, channels = x.shape
        merge_size = self.merge_size

        if height < merge_size or width < merge_size:
            raise ValueError(
                "patch 网格比 merge_size 还小："
                f"grid=({height}, {width}), merge_size={merge_size}。"
            )

        if height % merge_size != 0 or width % merge_size != 0:
            if not self.allow_truncate:
                raise ValueError(
                    "patch 网格的高度和宽度必须能被 merge_size 整除。"
                    f"当前 grid=({height}, {width}), merge_size={merge_size}。"
                    "可以设置 allow_truncate=True 丢弃底部/右侧多余 patch，"
                    "或者在图像预处理阶段把尺寸 resize 到可整除。"
                )

            # 丢弃底部和右侧无法组成完整 merge block 的 patch。
            new_height = (height // merge_size) * merge_size
            new_width = (width // merge_size) * merge_size
            x = x[:, :new_height, :new_width, :]
            height, width = new_height, new_width

        merged_height = height // merge_size
        merged_width = width // merge_size

        # 下面三步是 patch merge 的核心。
        #
        # 第 1 步：
        #   把 H 和 W 按 merge_size 分组。
        #
        #   [B, H, W, C]
        #   -> [B, H/m, m, W/m, m, C]
        #
        # 当 m=2 时，每个新的空间位置都对应原来的一个 2x2 局部 patch 块。
        x = x.reshape(
            batch_size,
            merged_height,
            merge_size,
            merged_width,
            merge_size,
            channels,
        )

        # 第 2 步：
        #   把两个局部 merge 维度移动到通道维附近。
        #
        #   [B, H/m, m, W/m, m, C]
        #   -> [B, H/m, W/m, m, m, C]
        #
        # 这样更方便下一步把局部 m*m 个 patch 拼到特征维。
        x = x.permute(0, 1, 3, 2, 4, 5)

        # 第 3 步：
        #   flatten 合并后的二维网格，并把每个 m*m 局部块 concat 到通道维。
        #
        #   [B, H/m, W/m, m, m, C]
        #   -> [B, (H/m)*(W/m), m*m*C]
        merged_tokens = x.reshape(
            batch_size,
            merged_height * merged_width,
            merge_size * merge_size * channels,
        )

        return merged_tokens, GridSize(merged_height, merged_width)

    @staticmethod
    def _normalize_grid_size(grid_size: tuple[int, int] | GridSize) -> GridSize:
        """把支持的 grid_size 输入统一转换成 ``GridSize``。"""

        if isinstance(grid_size, GridSize):
            return grid_size

        if not isinstance(grid_size, tuple) or len(grid_size) != 2:
            raise TypeError(
                "grid_size 必须是 tuple[int, int] 或 GridSize，"
                f"当前为 {grid_size!r}。"
            )

        height, width = grid_size
        if height <= 0 or width <= 0:
            raise ValueError(f"grid_size 的值必须为正数，当前为 {grid_size}。")

        return GridSize(height=height, width=width)


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 这里构造一个很小的假 patch 网格：
    #   B = 2
    #   H = 4
    #   W = 6
    #   C = 3
    #
    # 经过 2x2 merge 后：
    #   merged_H = 2
    #   merged_W = 3
    #   输出 token 数 = 2 * 3 = 6
    #   输出特征维 = 2 * 2 * 3 = 12
    #
    # 因此期望输出 shape 是：
    #   [2, 6, 12]
    torch.manual_seed(0)

    merger = PatchMerger(merge_size=2)
    x_4d = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)

    merged_from_4d, merged_grid = merger(x_4d)

    print("输入 [B, H, W, C]:", tuple(x_4d.shape))
    print("合并后 tokens:", tuple(merged_from_4d.shape))
    print("合并后网格:", merged_grid.as_tuple())

    # 同一批数据也可以用 ViT/SigLIP 常见的 flatten token 格式传入：
    #   [B, N, C]，其中 N = H * W。
    x_3d = x_4d.reshape(2, 4 * 6, 3)
    merged_from_3d, merged_grid_3d = merger(x_3d, grid_size=(4, 6))

    print("\n输入 [B, N, C]:", tuple(x_3d.shape))
    print("合并后 tokens:", tuple(merged_from_3d.shape))
    print("合并后网格:", merged_grid_3d.as_tuple())

    # 两种输入路径应该得到完全一致的 merged tokens。
    max_diff = (merged_from_4d - merged_from_3d).abs().max().item()
    print("\n4D 路径和 3D 路径的最大差异:", max_diff)

    assert tuple(merged_from_4d.shape) == (2, 6, 12)
    assert merged_grid.as_tuple() == (2, 3)
    assert max_diff == 0.0
    print("\nPatchMerger sanity check 通过。")

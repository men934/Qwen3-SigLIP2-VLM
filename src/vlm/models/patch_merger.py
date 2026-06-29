"""Patch merge for ViT/SigLIP visual tokens.

Default 2x2 merge:
    [B, H, W, C] -> [B, (H/2) * (W/2), 4 * C]
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
    """Merge local m x m patch blocks by channel concat.

    Inputs:
        ``[B, H, W, C]`` or ``[B, N, C]`` with ``grid_size=(H, W)``.

    Output:
        ``[B, merged_H * merged_W, merge_size^2 * C]`` and merged grid size.
    """

    def __init__(self, merge_size: int = 2, allow_truncate: bool = False) -> None:
        """Create a patch merger.

        ``allow_truncate=True`` drops the bottom/right remainder when H or W is
        not divisible by ``merge_size``.
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
            x_4d = x
        elif x.ndim == 3:
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

            new_height = (height // merge_size) * merge_size
            new_width = (width // merge_size) * merge_size
            x = x[:, :new_height, :new_width, :]
            height, width = new_height, new_width

        merged_height = height // merge_size
        merged_width = width // merge_size

        # [B, H, W, C] -> [B, H/m, m, W/m, m, C]
        x = x.reshape(
            batch_size,
            merged_height,
            merge_size,
            merged_width,
            merge_size,
            channels,
        )

        # [B, H/m, m, W/m, m, C] -> [B, H/m, W/m, m, m, C]
        x = x.permute(0, 1, 3, 2, 4, 5)

        # [B, H/m, W/m, m, m, C] -> [B, (H/m)*(W/m), m*m*C]
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
    # Quick shape check for both supported input formats.
    torch.manual_seed(0)

    merger = PatchMerger(merge_size=2)
    x_4d = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)

    merged_from_4d, merged_grid = merger(x_4d)

    print("输入 [B, H, W, C]:", tuple(x_4d.shape))
    print("合并后 tokens:", tuple(merged_from_4d.shape))
    print("合并后网格:", merged_grid.as_tuple())

    x_3d = x_4d.reshape(2, 4 * 6, 3)
    merged_from_3d, merged_grid_3d = merger(x_3d, grid_size=(4, 6))

    print("\n输入 [B, N, C]:", tuple(x_3d.shape))
    print("合并后 tokens:", tuple(merged_from_3d.shape))
    print("合并后网格:", merged_grid_3d.as_tuple())

    max_diff = (merged_from_4d - merged_from_3d).abs().max().item()
    print("\n4D 路径和 3D 路径的最大差异:", max_diff)

    assert tuple(merged_from_4d.shape) == (2, 6, 12)
    assert merged_grid.as_tuple() == (2, 3)
    assert max_diff == 0.0
    print("\nPatchMerger quick check passed.")

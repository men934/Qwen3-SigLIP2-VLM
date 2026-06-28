"""SigLIP2 + Qwen VLM 的图像读取与预处理。

传统 CV 项目里，我们经常直接把图像 transform 写在 ``dataset.py`` 里。
但在 VLM 项目中，把图像预处理单独拆出来会更清晰，因为同一套逻辑会被多个阶段复用：

    1. Stage 1 视觉-语言对齐训练
    2. Stage 2 多模态指令微调
    3. Stage 3 垂域微调
    4. inference / demo 脚本

第一版先故意保持固定分辨率，和我们下载的 SigLIP2 checkpoint 对齐：

    google/siglip2-so400m-patch14-384

所以这个模块当前做的事情是：

    image path
        -> PIL RGB image
        -> SigLIP2 processor
        -> pixel_values tensor [3, 384, 384]

现在开始加入动态分辨率。这里先不魔改 Qwen3 的 RoPE，也不改 SigLIP2 的内部结构，
而是在进入 processor 之前做一层“长宽比保留 + token 数受控”的 resize：

    image path
        -> PIL RGB image
        -> smart_resize 得到动态 H/W
        -> SigLIP2 processor 只做 rescale/normalize
        -> pixel_values tensor [3, dynamic_h, dynamic_w]

为什么不直接让 Hugging Face processor resize？
    默认 SigLIP2 processor 会强制把所有图片 resize 成 384x384。这样实现简单，
    但会破坏长宽比，也无法复现 Qwen2.5-VL 这类动态分辨率思想。所以动态模式下，
    我们自己控制尺寸，然后关闭 processor 内部的 resize。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image, ImageFile
from torch import Tensor


# 大规模网页图像数据里经常会有少量截断/不完整图片。
# PIL 开启这个选项后，很多轻微损坏的图片仍然可以被解码。
# 这样可以避免训练过程中因为单张脏图导致整个任务失败。
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ImageInfo:
    """单张图片处理前后的元信息。"""

    path: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    patch_size: int = 14
    merge_size: int = 2
    patch_grid_width: int = 0
    patch_grid_height: int = 0
    merged_grid_width: int = 0
    merged_grid_height: int = 0
    num_image_tokens: int = 0
    dynamic_resolution: bool = False


@dataclass(frozen=True)
class ProcessedImage:
    """单张图片预处理后的结果。"""

    pixel_values: Tensor
    info: ImageInfo


@dataclass(frozen=True)
class ProcessedImageBatch:
    """一批图片预处理后的结果。"""

    pixel_values: Tensor
    infos: list[ImageInfo]


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """从磁盘读取图片，并统一转换成 RGB。

    为什么一定转成 RGB？
        SigLIP2 期望输入是 3 通道 RGB。真实数据里可能出现灰度图、调色板图、
        CMYK JPEG，或者带 alpha 通道的图片。这里统一转换一次，下游代码就不用
        关心这些格式差异。
    """

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"图片路径不是文件：{path}")

    with Image.open(path) as image:
        return image.convert("RGB")


def round_by_factor(value: int, factor: int) -> int:
    """把整数四舍五入到最接近的 factor 倍数。"""

    return max(factor, int(round(value / factor) * factor))


def ceil_by_factor(value: int, factor: int) -> int:
    """把整数向上取整到 factor 倍数。"""

    return max(factor, int(math.ceil(value / factor) * factor))


def floor_by_factor(value: int, factor: int) -> int:
    """把整数向下取整到 factor 倍数。"""

    return max(factor, int(math.floor(value / factor) * factor))


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """按 Qwen2.5-VL 的思路计算动态分辨率尺寸。

    这一步只返回新尺寸，不直接改图片内容。

    设计目标：
        1. 尽量保持原图长宽比。
        2. 高和宽都对齐到 ``factor`` 的倍数。
        3. 总像素数限制在 ``[min_pixels, max_pixels]`` 内。

    为什么 factor 默认应该是 28？
        我们当前视觉塔是 SigLIP2 patch14，后面接 2x2 Patch Merger。
        如果输入高宽是 28 的倍数，那么 patch grid 一定是 2 的倍数，
        Patch Merger 就可以稳定地把 2x2 patch 合并成 1 个视觉 token。
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"图片尺寸必须为正数，当前 height={height}, width={width}。")
    if factor <= 0:
        raise ValueError(f"factor 必须为正数，当前为 {factor}。")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError(
            f"min_pixels/max_pixels 必须为正数，当前为 {min_pixels}/{max_pixels}。"
        )
    if min_pixels > max_pixels:
        raise ValueError(
            f"min_pixels 不能大于 max_pixels，当前为 {min_pixels} > {max_pixels}。"
        )

    resized_height = round_by_factor(height, factor)
    resized_width = round_by_factor(width, factor)

    current_pixels = resized_height * resized_width
    original_pixels = height * width

    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / original_pixels)
        resized_height = floor_by_factor(int(height * scale), factor)
        resized_width = floor_by_factor(int(width * scale), factor)
    elif current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / original_pixels)
        resized_height = ceil_by_factor(int(height * scale), factor)
        resized_width = ceil_by_factor(int(width * scale), factor)

    return resized_height, resized_width


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """按 PIL 的 ``(width, height)`` 约定 resize 图片。"""

    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize(size, resample=resample)


class SiglipImageProcessor:
    """Hugging Face SigLIP2 processor 的项目内封装。

    这个 wrapper 给我们提供一个稳定的项目内部 API。后面如果把固定分辨率替换成
    动态分辨率，dataset 和 collator 仍然可以继续调用这个类，而不需要直接依赖
    Hugging Face processor 的细节。
    """

    def __init__(
        self,
        processor_path: str | Path,
        image_size: int = 384,
        dynamic_resolution: bool = False,
        patch_size: int = 14,
        merge_size: int = 2,
        min_pixels: int = 384 * 384,
        max_pixels: int = 672 * 672,
    ) -> None:
        """创建图像预处理器。

        Args:
            processor_path:
                本地 Hugging Face model/processor 目录。当前机器上是：

                ``/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384``

            image_size:
                第一版使用的固定正方形图像尺寸。

            dynamic_resolution:
                是否开启动态分辨率。关闭时完全保持旧版固定 384 逻辑，方便继续加载
                已经训练好的 Stage 1/Stage 2 checkpoint。

            patch_size:
                SigLIP2 的 patch 大小。我们当前用的是 patch14，所以默认 14。

            merge_size:
                Patch Merger 的空间合并大小。我们当前设计是 2x2，所以默认 2。

            min_pixels/max_pixels:
                动态分辨率的像素数上下界。第一版先给一个保守范围，避免视觉 token
                数量暴涨导致显存不可控。
        """

        if image_size <= 0:
            raise ValueError(f"image_size 必须为正数，当前为 {image_size}。")
        if patch_size <= 0:
            raise ValueError(f"patch_size 必须为正数，当前为 {patch_size}。")
        if merge_size <= 0:
            raise ValueError(f"merge_size 必须为正数，当前为 {merge_size}。")

        # 旧版代码是这样写的：
        #
        # if dynamic_resolution:
        #     raise NotImplementedError(
        #         "dynamic_resolution 已预留，但当前 image_processing.py 第一版还未实现。"
        #         "请先使用固定 384 分辨率。"
        #     )
        #
        # 现在我们把这个报错注释掉，改为真正支持动态分辨率。这样做的原因是：
        #   1. Stage 3 前需要先验证动态视觉 token 是否能跑通。
        #   2. 下游模型需要真实 patch grid，不能只拿固定 384 的 pixel_values。
        #   3. 默认 dynamic_resolution=False，所以旧 checkpoint 的路径仍然不变。

        self.processor_path = str(processor_path)
        self.image_size = image_size
        self.dynamic_resolution = dynamic_resolution
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.size_factor = patch_size * merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = self._load_hf_processor(self.processor_path)

    @staticmethod
    def _load_hf_processor(processor_path: str):
        """加载 Hugging Face processor，并给出清晰的依赖错误提示。"""

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SiglipImageProcessor 需要 transformers。安装命令："
                "python -m pip install transformers"
            ) from exc

        return AutoProcessor.from_pretrained(processor_path, local_files_only=True)

    def process_image(self, image_path: str | Path) -> ProcessedImage:
        """读取并预处理单张图片。

        Returns:
            ``ProcessedImage``，其中 ``pixel_values`` 的 shape 是 ``[3, H, W]``。
            当前固定 SigLIP2 processor 下，H=W=384。
        """

        image = load_rgb_image(image_path)
        original_width, original_height = image.size

        if self.dynamic_resolution:
            image = self._resize_dynamic(image)
            encoded = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=False,
            )
        else:
            # 旧版固定分辨率逻辑如下，先保留在注释里，方便对照：
            #
            # encoded = self.processor(images=image, return_tensors="pt")
            #
            # 它会调用 SigLIP2 processor 默认配置，把所有图强制 resize 到 384x384。
            # 现在固定模式仍然使用这条路径，因为我们已经有基于固定 384 训练出的
            # Stage 1/Stage 2 checkpoint。
            encoded = self.processor(images=image, return_tensors="pt")

        # Hugging Face processor 即使处理单张图片，也会返回 batch 维度：
        #   pixel_values: [1, 3, H, W]
        # 这里去掉第 0 维，让 dataset 返回单张图片时保持：
        #   [3, H, W]
        pixel_values = encoded["pixel_values"].squeeze(0)

        if pixel_values.ndim != 3:
            raise ValueError(
                "预期 pixel_values 的 shape 为 [3, H, W]，"
                f"但实际得到 {tuple(pixel_values.shape)}。"
            )

        processed_height = int(pixel_values.shape[-2])
        processed_width = int(pixel_values.shape[-1])
        if self.dynamic_resolution:
            self._validate_processed_size(processed_height, processed_width)
        patch_grid_height = processed_height // self.patch_size
        patch_grid_width = processed_width // self.patch_size
        merged_grid_height = patch_grid_height // self.merge_size
        merged_grid_width = patch_grid_width // self.merge_size

        info = ImageInfo(
            path=str(image_path),
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            patch_size=self.patch_size,
            merge_size=self.merge_size,
            patch_grid_width=patch_grid_width,
            patch_grid_height=patch_grid_height,
            merged_grid_width=merged_grid_width,
            merged_grid_height=merged_grid_height,
            num_image_tokens=merged_grid_height * merged_grid_width,
            dynamic_resolution=self.dynamic_resolution,
        )
        return ProcessedImage(pixel_values=pixel_values, info=info)

    def process_batch(self, image_paths: Iterable[str | Path]) -> ProcessedImageBatch:
        """读取并预处理多张图片，然后 stack 成一个 batch。"""

        processed = [self.process_image(path) for path in image_paths]
        if not processed:
            raise ValueError("process_batch 没有收到任何图片路径。")

        if self.dynamic_resolution:
            pixel_values = self._pad_and_stack_dynamic_images(
                [item.pixel_values for item in processed]
            )
        else:
            # 旧版固定 384 路径可以直接 stack：
            #
            # pixel_values = torch.stack([item.pixel_values for item in processed], dim=0)
            #
            # 动态分辨率下，不同图片可能是 [3, 392, 672]、[3, 672, 392]，
            # 直接 stack 会失败，所以动态模式会先 padding 到 batch 内最大 H/W。
            pixel_values = torch.stack([item.pixel_values for item in processed], dim=0)

        infos = [item.info for item in processed]
        return ProcessedImageBatch(pixel_values=pixel_values, infos=infos)

    def _resize_dynamic(self, image: Image.Image) -> Image.Image:
        """动态模式下保留长宽比 resize。"""

        original_width, original_height = image.size
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=self.size_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return resize_image(image, (resized_width, resized_height))

    def _validate_processed_size(self, height: int, width: int) -> None:
        """确保处理后尺寸能被视觉塔和 Patch Merger 正常消费。"""

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "处理后图片尺寸必须能被 patch_size 整除，"
                f"当前 H/W={height}/{width}, patch_size={self.patch_size}。"
            )

        patch_grid_height = height // self.patch_size
        patch_grid_width = width // self.patch_size
        if (
            patch_grid_height % self.merge_size != 0
            or patch_grid_width % self.merge_size != 0
        ):
            raise ValueError(
                "patch grid 必须能被 merge_size 整除，"
                f"当前 grid={patch_grid_height}x{patch_grid_width}, "
                f"merge_size={self.merge_size}。"
            )

    @staticmethod
    def _pad_and_stack_dynamic_images(pixel_values_list: list[Tensor]) -> Tensor:
        """把动态分辨率图片 padding 到 batch 内最大 H/W 后再 stack。

        注意：
            这里的 padding 只是为了让 PyTorch 能组成 batch。真实有效区域仍然由
            ``ImageInfo.processed_height/processed_width`` 记录，后面的模型会根据
            grid 元信息丢掉 padding 区域对应的视觉 token。
        """

        max_height = max(int(pixel_values.shape[-2]) for pixel_values in pixel_values_list)
        max_width = max(int(pixel_values.shape[-1]) for pixel_values in pixel_values_list)

        padded_images = []
        for pixel_values in pixel_values_list:
            if pixel_values.ndim != 3:
                raise ValueError(
                    "预期每张图片 pixel_values 为 [3, H, W]，"
                    f"但实际得到 {tuple(pixel_values.shape)}。"
                )

            pad_height = max_height - int(pixel_values.shape[-2])
            pad_width = max_width - int(pixel_values.shape[-1])
            padded = torch.nn.functional.pad(
                pixel_values,
                pad=(0, pad_width, 0, pad_height),
                mode="constant",
                value=0.0,
            )
            padded_images.append(padded)

        return torch.stack(padded_images, dim=0)


def find_first_image(root: str | Path) -> Optional[Path]:
    """在目录下找一张图片，用于快速 sanity check。"""

    root_path = Path(root)
    if not root_path.exists():
        return None

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_suffixes:
            return path
    return None


if __name__ == "__main__":
    # 临时 sanity check，方便学习和调试。
    #
    # 这个脚本会使用我们之前下载到本地的 SigLIP2 processor，以及一张
    # LLaVA-Pretrain 图片。它验证：
    #   1. 图片读取是否正常
    #   2. RGB 转换是否正常
    #   3. Hugging Face processor 是否返回 [3, 384, 384] tensor
    #   4. batch stack 是否正常
    default_processor_path = Path(
        "/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384"
    )
    default_image_root = Path("/root/autodl-tmp/hf_datasets/LLaVA-Pretrain")

    image_path = find_first_image(default_image_root)
    if image_path is None:
        raise FileNotFoundError(f"在目录下没有找到图片：{default_image_root}")

    fixed_processor = SiglipImageProcessor(
        processor_path=default_processor_path,
        image_size=384,
        dynamic_resolution=False,
    )

    one = fixed_processor.process_image(image_path)
    print("固定分辨率单张图片")
    print("路径:", one.info.path)
    print("原始尺寸:", (one.info.original_width, one.info.original_height))
    print("处理后尺寸:", (one.info.processed_width, one.info.processed_height))
    print("patch grid:", (one.info.patch_grid_height, one.info.patch_grid_width))
    print("merged grid:", (one.info.merged_grid_height, one.info.merged_grid_width))
    print("视觉 token 数:", one.info.num_image_tokens)
    print("pixel_values shape:", tuple(one.pixel_values.shape))
    print("pixel_values dtype:", one.pixel_values.dtype)
    print("pixel_values min/max:", float(one.pixel_values.min()), float(one.pixel_values.max()))

    batch = fixed_processor.process_batch([image_path, image_path])
    print("\n固定分辨率 Batch")
    print("pixel_values shape:", tuple(batch.pixel_values.shape))
    print("info 数量:", len(batch.infos))

    assert tuple(one.pixel_values.shape) == (3, 384, 384)
    assert tuple(batch.pixel_values.shape) == (2, 3, 384, 384)

    dynamic_processor = SiglipImageProcessor(
        processor_path=default_processor_path,
        image_size=384,
        dynamic_resolution=True,
        min_pixels=384 * 384,
        max_pixels=672 * 672,
    )
    dynamic_one = dynamic_processor.process_image(image_path)
    print("\n动态分辨率单张图片")
    print("处理后尺寸:", (dynamic_one.info.processed_width, dynamic_one.info.processed_height))
    print(
        "patch grid:",
        (dynamic_one.info.patch_grid_height, dynamic_one.info.patch_grid_width),
    )
    print(
        "merged grid:",
        (dynamic_one.info.merged_grid_height, dynamic_one.info.merged_grid_width),
    )
    print("视觉 token 数:", dynamic_one.info.num_image_tokens)
    print("pixel_values shape:", tuple(dynamic_one.pixel_values.shape))

    print("\nImage processing sanity check 通过。")

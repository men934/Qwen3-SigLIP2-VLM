"""Image loading and preprocessing for SigLIP2 + Qwen VLM.

    image path
        -> PIL RGB image
        -> optional dynamic resize
        -> SigLIP2 processor
        -> pixel_values [3, H, W]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image, ImageFile
from torch import Tensor


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
    """Load an image and convert it to RGB."""

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
    """Compute aspect-ratio-preserving dynamic resolution.

    The result is aligned to ``factor`` and constrained by
    ``[min_pixels, max_pixels]``.
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
    """Project wrapper around the Hugging Face SigLIP processor."""

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
        """Create an image processor."""

        if image_size <= 0:
            raise ValueError(f"image_size 必须为正数，当前为 {image_size}。")
        if patch_size <= 0:
            raise ValueError(f"patch_size 必须为正数，当前为 {patch_size}。")
        if merge_size <= 0:
            raise ValueError(f"merge_size 必须为正数，当前为 {merge_size}。")
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
        """Load and preprocess one image."""

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
            encoded = self.processor(images=image, return_tensors="pt")

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
        """Pad dynamic-resolution images to batch max H/W and stack."""

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
    """Find one image under a directory."""

    root_path = Path(root)
    if not root_path.exists():
        return None

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_suffixes:
            return path
    return None


if __name__ == "__main__":
    # Quick fixed/dynamic preprocessing check.
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

    print("\nImage processing quick check passed.")

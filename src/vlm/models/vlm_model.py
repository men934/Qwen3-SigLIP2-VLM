"""SigLIP2 + PatchMerger + Projector + Qwen3 VLM.

    pixel_values
        -> SigLIP2 vision tower
        -> patch features [B, N, vision_hidden]
        -> PatchMerger
        -> merged visual tokens [B, N_merged, 4 * vision_hidden]
        -> MLPProjector
        -> 可选 2D visual RoPE-style 位置旋转
        -> Qwen-space visual tokens [B, N_merged, qwen_hidden]
        -> 替换文本中的 <image> token
        -> Qwen3ForCausalLM
        -> loss / logits
"""

from __future__ import annotations

from types import MethodType
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

try:
    from .patch_merger import GridSize, PatchMerger
    from .projector import MLPProjector
except ImportError:  # 允许直接 python src/vlm/models/vlm_model.py 运行
    from patch_merger import GridSize, PatchMerger
    from projector import MLPProjector


IGNORE_INDEX = -100


@dataclass(frozen=True)
class VLMModelConfig:
    """Qwen-SigLIP VLM 的最小配置。"""

    qwen_path: str
    siglip_path: str
    image_token_id: int
    tokenizer_length: int
    vision_hidden_size: int = 1152
    qwen_hidden_size: int = 2048
    merge_size: int = 2
    projector_hidden_size: int = 2048
    patch_size: int = 14
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = True
    patch_merger_allow_truncate: bool = True
    use_siglip_abs_pos_embedding: bool = True
    use_siglip_qk_2d_rope: bool = False
    siglip_rope_base: float = 10000.0
    siglip_rope_dim: Optional[int] = None
    local_files_only: bool = True
    torch_dtype: str = "bfloat16"


class QwenSiglipVLM(nn.Module):
    """Trainable SigLIP2 + Qwen3 VLM wrapper."""

    def __init__(self, config: VLMModelConfig) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = config.image_token_id

        self.vision_encoder = self._load_vision_encoder(config)
        self.language_model = self._load_language_model(config)

        if config.use_siglip_qk_2d_rope:
            self._patch_siglip_attention_for_2d_rope()

        # SigLIP2 patch14-384 produces a 27x27 patch grid.
        # 27 不能被 2 整除，所以 2x2 merge 时需要裁掉最后一行和最后一列：
        #   27x27 -> 26x26 -> 13x13 merged tokens
        self.patch_merger = PatchMerger(
            merge_size=config.merge_size,
            allow_truncate=config.patch_merger_allow_truncate,
        )
        self.projector = MLPProjector(
            input_dim=config.merge_size * config.merge_size * config.vision_hidden_size,
            hidden_dim=config.projector_hidden_size,
            output_dim=config.qwen_hidden_size,
        )
        self.projector.to(dtype=self._dtype_from_string(config.torch_dtype))

        self._resize_language_embeddings_if_needed(config.tokenizer_length)
        self._apply_freezing()

    @staticmethod
    def _dtype_from_string(dtype_name: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"不支持的 torch_dtype：{dtype_name!r}")

    def _load_vision_encoder(self, config: VLMModelConfig) -> nn.Module:
        """加载 SigLIP2 vision tower。"""

        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "QwenSiglipVLM 需要 transformers。安装命令："
                "python -m pip install transformers"
            ) from exc

        dtype = self._dtype_from_string(config.torch_dtype)
        return SiglipVisionModel.from_pretrained(
            config.siglip_path,
            local_files_only=config.local_files_only,
            torch_dtype=dtype,
        )

    def _load_language_model(self, config: VLMModelConfig) -> nn.Module:
        """加载 Qwen3 CausalLM。"""

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "QwenSiglipVLM 需要 transformers。安装命令："
                "python -m pip install transformers"
            ) from exc

        dtype = self._dtype_from_string(config.torch_dtype)
        return AutoModelForCausalLM.from_pretrained(
            config.qwen_path,
            local_files_only=config.local_files_only,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    def _patch_siglip_attention_for_2d_rope(self) -> None:
        """Patch SigLIP2 attention to apply 2D RoPE on Q/K.

        Only the attention computation changes; q/k/v/out projection weights
        are unchanged.
        """

        outer_model = self

        def forward_with_2d_rope(
            attention_module: nn.Module,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            visual_rope_positions: Optional[tuple[Tensor, Tensor]] = None,
            **_: Any,
        ) -> tuple[Tensor, Optional[Tensor]]:
            """SigLIP attention forward，区别是对 Q/K 应用 2D RoPE。"""

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attention_module.head_dim)

            queries = attention_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            keys = attention_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            values = attention_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if visual_rope_positions is not None:
                row_positions, col_positions = visual_rope_positions
                queries = outer_model._apply_2d_rope_to_attention_states(
                    queries,
                    row_positions=row_positions,
                    col_positions=col_positions,
                    base=outer_model.config.siglip_rope_base,
                    rope_dim=outer_model.config.siglip_rope_dim,
                )
                keys = outer_model._apply_2d_rope_to_attention_states(
                    keys,
                    row_positions=row_positions,
                    col_positions=col_positions,
                    base=outer_model.config.siglip_rope_base,
                    rope_dim=outer_model.config.siglip_rope_dim,
                )

            attn_weights = torch.matmul(queries, keys.transpose(-2, -1))
            attn_weights = attn_weights * attention_module.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(queries.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=0.0 if not attention_module.training else attention_module.dropout,
                training=attention_module.training,
            )

            attn_output = torch.matmul(attn_weights, values)
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = attention_module.out_proj(attn_output)
            return attn_output, None

        for layer in self.vision_encoder.encoder.layers:
            layer.self_attn.forward = MethodType(forward_with_2d_rope, layer.self_attn)

    def _resize_language_embeddings_if_needed(self, tokenizer_length: int) -> None:
        """Resize Qwen embeddings when the added image token exceeds the table."""

        current_size = self.language_model.get_input_embeddings().num_embeddings
        required_size = max(tokenizer_length, self.image_token_id + 1)

        if required_size > current_size:
            self.language_model.resize_token_embeddings(required_size)

    def _apply_freezing(self) -> None:
        """按配置冻结 vision encoder 和 language model。"""

        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if self.config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # PatchMerger 没有参数，但保留语义。
        for param in self.projector.parameters():
            param.requires_grad = True

    def encode_images(
        self,
        pixel_values: Tensor,
        image_infos: Optional[list[Any]] = None,
    ) -> tuple[Tensor | list[Tensor], GridSize | list[GridSize]]:
        """把图片编码成 Qwen hidden space 中的 visual embeddings。

        Args:
            pixel_values:
                shape 为 ``[B, 3, H, W]`` 的图片 tensor。固定分辨率时通常是
                ``[B, 3, 384, 384]``；动态分辨率 + batch padding 后，H/W 是当前
                batch 内的最大高宽。

            image_infos:
                可选的图像元信息列表，来自 ``image_processing.py``。动态分辨率下，
                每张图真实的 patch grid 可能不同，必须依赖它裁掉 padding 区域。

        Returns:
            visual_embeds:
                固定视觉 token 数时返回 ``[B, N_merged, qwen_hidden_size]``。
                动态分辨率导致 batch 内视觉 token 数不同时，返回 ``list[Tensor]``，
                每个元素是 ``[N_i, qwen_hidden_size]``。

            merged_grid:
                merge 后的二维网格大小，例如 13x13。动态 batch 下返回 list。
        """

        vision_dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        full_grid = self._grid_size_from_pixel_values(pixel_values)

        patch_tokens = self._encode_with_siglip(
            pixel_values=pixel_values,
            full_grid=full_grid,
        )

        sample_patch_tokens = self._crop_patch_tokens_to_real_grids(
            patch_tokens=patch_tokens,
            full_grid=full_grid,
            image_infos=image_infos,
        )

        visual_embeds_list: list[Tensor] = []
        merged_grids: list[GridSize] = []
        for sample_tokens, sample_grid in sample_patch_tokens:
            merged_tokens, merged_grid = self.patch_merger(
                sample_tokens.unsqueeze(0),
                grid_size=sample_grid,
            )
            sample_visual_embeds = self.projector(merged_tokens).squeeze(0)

            visual_embeds_list.append(sample_visual_embeds)
            merged_grids.append(merged_grid)

        if self._all_visual_lengths_equal(visual_embeds_list):
            return torch.stack(visual_embeds_list, dim=0), merged_grids[0]

        return visual_embeds_list, merged_grids

    def _grid_size_from_pixel_values(self, pixel_values: Tensor) -> GridSize:
        """根据输入图片尺寸计算 SigLIP patch grid。

        SigLIP2 的 patch embedding 是 ``Conv2d(kernel=14, stride=14)``，所以 patch
        grid 不是四舍五入，而是卷积意义上的向下取整：

            patch_grid_h = image_h // patch_size
            patch_grid_w = image_w // patch_size
        """

        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values 应为 [B, 3, H, W]，"
                f"实际为 {tuple(pixel_values.shape)}。"
            )

        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
        return GridSize(
            height=height // self.config.patch_size,
            width=width // self.config.patch_size,
        )

    def _needs_siglip_position_interpolation(self, grid_size: GridSize) -> bool:
        """判断 SigLIP2 absolute position embedding 是否需要插值。"""

        num_current_positions = grid_size.height * grid_size.width
        position_embedding = self.vision_encoder.embeddings.position_embedding
        num_pretrained_positions = int(position_embedding.weight.shape[0])
        return num_current_positions != num_pretrained_positions

    def _encode_with_siglip(self, pixel_values: Tensor, full_grid: GridSize) -> Tensor:
        """Run SigLIP2 with absolute position embedding or Q/K 2D RoPE."""

        if self.config.use_siglip_abs_pos_embedding:
            interpolate_pos_encoding = self._needs_siglip_position_interpolation(full_grid)
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            return vision_outputs.last_hidden_state

        if not self.config.use_siglip_qk_2d_rope:
            raise ValueError(
                "关闭 use_siglip_abs_pos_embedding 后，需要启用 "
                "use_siglip_qk_2d_rope，否则 SigLIP2 内部没有任何二维位置信息。"
            )

        embeddings = self.vision_encoder.embeddings
        target_dtype = embeddings.patch_embedding.weight.dtype

        # Avoid embeddings.forward because it always adds absolute position embedding.
        patch_embeds = embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        hidden_states = patch_embeds.flatten(2).transpose(1, 2)

        row_positions, col_positions = self._build_2d_positions(
            grid_size=full_grid,
            device=hidden_states.device,
        )
        encoder_outputs = self.vision_encoder.encoder(
            inputs_embeds=hidden_states,
            visual_rope_positions=(row_positions, col_positions),
        )
        last_hidden_state = self.vision_encoder.post_layernorm(
            encoder_outputs.last_hidden_state
        )
        return last_hidden_state

    def _crop_patch_tokens_to_real_grids(
        self,
        patch_tokens: Tensor,
        full_grid: GridSize,
        image_infos: Optional[list[Any]],
    ) -> list[tuple[Tensor, GridSize]]:
        """Crop SigLIP patch tokens back to each image's real patch grid."""

        if patch_tokens.ndim != 3:
            raise ValueError(
                "vision encoder 输出的 patch_tokens 应为 [B, N, C]，"
                f"实际为 {tuple(patch_tokens.shape)}。"
            )

        batch_size, num_tokens, channels = patch_tokens.shape
        expected_tokens = full_grid.height * full_grid.width
        if num_tokens != expected_tokens:
            raise ValueError(
                "SigLIP 输出 token 数和输入图片 grid 不匹配："
                f"N={num_tokens}, grid={full_grid.as_tuple()}。"
            )

        tokens_4d = patch_tokens.reshape(
            batch_size,
            full_grid.height,
            full_grid.width,
            channels,
        )

        real_grids = self._real_patch_grids_from_image_infos(
            image_infos=image_infos,
            batch_size=batch_size,
            fallback_grid=full_grid,
        )

        output: list[tuple[Tensor, GridSize]] = []
        for batch_idx, real_grid in enumerate(real_grids):
            if real_grid.height > full_grid.height or real_grid.width > full_grid.width:
                raise ValueError(
                    "真实 patch grid 不能大于 padded batch grid："
                    f"real={real_grid.as_tuple()}, full={full_grid.as_tuple()}。"
                )

            sample_tokens = tokens_4d[
                batch_idx,
                : real_grid.height,
                : real_grid.width,
                :,
            ].reshape(real_grid.height * real_grid.width, channels)
            output.append((sample_tokens, real_grid))

        return output

    def _real_patch_grids_from_image_infos(
        self,
        image_infos: Optional[list[Any]],
        batch_size: int,
        fallback_grid: GridSize,
    ) -> list[GridSize]:
        """从 ImageInfo/dict 中读取每张图真实 patch grid。"""

        if image_infos is None:
            return [fallback_grid for _ in range(batch_size)]

        if len(image_infos) != batch_size:
            raise ValueError(
                "image_infos 数量必须等于 batch size："
                f"len(image_infos)={len(image_infos)}, batch_size={batch_size}。"
            )

        grids: list[GridSize] = []
        for info in image_infos:
            patch_grid_height = self._get_info_value(info, "patch_grid_height")
            patch_grid_width = self._get_info_value(info, "patch_grid_width")

            if patch_grid_height is None or patch_grid_width is None:
                processed_height = self._get_info_value(info, "processed_height")
                processed_width = self._get_info_value(info, "processed_width")
                if processed_height is None or processed_width is None:
                    raise ValueError(
                        "image_infos 需要包含 patch_grid_height/patch_grid_width，"
                        "或者 processed_height/processed_width。"
                    )
                patch_grid_height = int(processed_height) // self.config.patch_size
                patch_grid_width = int(processed_width) // self.config.patch_size

            grids.append(
                GridSize(height=int(patch_grid_height), width=int(patch_grid_width))
            )

        return grids

    @staticmethod
    def _get_info_value(info: Any, name: str) -> Any:
        """兼容 dataclass ImageInfo 和 dict 两种元信息格式。"""

        if isinstance(info, dict):
            return info.get(name)
        return getattr(info, name, None)

    @staticmethod
    def _all_visual_lengths_equal(visual_embeds_list: list[Tensor]) -> bool:
        """判断 batch 内每张图的视觉 token 数是否一致。"""

        if not visual_embeds_list:
            return True
        first_len = visual_embeds_list[0].shape[0]
        return all(item.shape[0] == first_len for item in visual_embeds_list)

    def _apply_2d_rope_to_attention_states(
        self,
        states: Tensor,
        *,
        row_positions: Tensor,
        col_positions: Tensor,
        base: float,
        rope_dim: Optional[int],
    ) -> Tensor:
        """Apply 2D RoPE to Q/K states shaped ``[B, heads, N, head_dim]``."""

        if states.ndim != 4:
            raise ValueError(
                "attention states 应为 [B, num_heads, N, head_dim]，"
                f"实际为 {tuple(states.shape)}。"
            )

        head_dim = states.shape[-1]
        effective_rope_dim = rope_dim or head_dim
        effective_rope_dim = min(effective_rope_dim, head_dim)
        effective_rope_dim = (effective_rope_dim // 4) * 4
        if effective_rope_dim <= 0:
            return states

        axis_dim = effective_rope_dim // 2
        rope_part = states[..., :effective_rope_dim]
        pass_part = states[..., effective_rope_dim:]

        row_part = rope_part[..., :axis_dim]
        col_part = rope_part[..., axis_dim:]

        row_part = self._apply_1d_rope_to_attention_axis(
            row_part,
            row_positions,
            base=base,
        )
        col_part = self._apply_1d_rope_to_attention_axis(
            col_part,
            col_positions,
            base=base,
        )
        return torch.cat([row_part, col_part, pass_part], dim=-1)

    @staticmethod
    def _build_2d_positions(grid_size: GridSize, device: torch.device) -> tuple[Tensor, Tensor]:
        """生成 flatten 后视觉 token 对应的 row/col 坐标。"""

        rows = torch.arange(grid_size.height, device=device)
        cols = torch.arange(grid_size.width, device=device)
        row_grid = rows[:, None].expand(grid_size.height, grid_size.width)
        col_grid = cols[None, :].expand(grid_size.height, grid_size.width)
        return row_grid.reshape(-1), col_grid.reshape(-1)

    @staticmethod
    def _apply_1d_rope_to_attention_axis(
        x: Tensor,
        positions: Tensor,
        base: float,
    ) -> Tensor:
        """对 Q/K 的一个轴向子空间应用标准 1D RoPE。

        ``x`` 的 shape 是 ``[B, num_heads, N, axis_dim]``，positions 的 shape 是
        ``[N]``。这里会把 cos/sin broadcast 到 batch 和 head 维度。
        """

        if x.shape[-1] % 2 != 0:
            raise ValueError(f"RoPE 维度必须是偶数，当前为 {x.shape[-1]}。")

        dtype = x.dtype
        x_float = x.float()
        positions_float = positions.float()
        dim = x.shape[-1]

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, device=x.device, dtype=torch.float32)
                / dim
            )
        )
        angles = positions_float[:, None] * inv_freq[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated = torch.empty_like(x_float)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        return rotated.to(dtype=dtype)

    def prepare_multimodal_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor],
        visual_embeds: Tensor | list[Tensor],
    ) -> dict[str, Tensor]:
        """Replace one ``<image>`` token with projected visual embeddings."""

        batch_size = input_ids.shape[0]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        new_embeds: list[Tensor] = []
        new_attention_masks: list[Tensor] = []
        new_labels: list[Tensor] = []

        for batch_idx in range(batch_size):
            image_positions = (input_ids[batch_idx] == self.image_token_id).nonzero(
                as_tuple=False
            )
            if image_positions.numel() != 1:
                raise ValueError(
                    "每条样本必须恰好包含 1 个 image token。"
                    f"batch_idx={batch_idx}, count={image_positions.numel()}。"
                )

            image_pos = int(image_positions.item())
            sample_visual_embeds = visual_embeds[batch_idx]
            visual_len = sample_visual_embeds.shape[0]

            before_embeds = text_embeds[batch_idx, :image_pos]
            after_embeds = text_embeds[batch_idx, image_pos + 1 :]
            sample_embeds = torch.cat(
                [before_embeds, sample_visual_embeds, after_embeds],
                dim=0,
            )
            new_embeds.append(sample_embeds)

            before_attention = attention_mask[batch_idx, :image_pos]
            after_attention = attention_mask[batch_idx, image_pos + 1 :]
            visual_attention = torch.ones(
                visual_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            sample_attention = torch.cat(
                [before_attention, visual_attention, after_attention],
                dim=0,
            )
            new_attention_masks.append(sample_attention)

            if labels is not None:
                before_labels = labels[batch_idx, :image_pos]
                after_labels = labels[batch_idx, image_pos + 1 :]
                visual_labels = torch.full(
                    (visual_len,),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                sample_labels = torch.cat(
                    [before_labels, visual_labels, after_labels],
                    dim=0,
                )
                new_labels.append(sample_labels)

        padded_embeds = self._pad_embeds(new_embeds)
        padded_attention_mask = self._pad_1d(new_attention_masks, pad_value=0)

        output = {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_attention_mask,
        }

        if labels is not None:
            output["labels"] = self._pad_1d(new_labels, pad_value=IGNORE_INDEX)

        return output

    @staticmethod
    def _pad_embeds(sequences: list[Tensor]) -> Tensor:
        """把不同长度的 embedding 序列右 padding 成 batch。"""

        max_len = max(sequence.shape[0] for sequence in sequences)
        hidden_size = sequences[0].shape[-1]
        batch_size = len(sequences)
        dtype = sequences[0].dtype
        device = sequences[0].device

        output = torch.zeros(
            batch_size,
            max_len,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for i, sequence in enumerate(sequences):
            output[i, : sequence.shape[0]] = sequence
        return output

    @staticmethod
    def _pad_1d(sequences: list[Tensor], pad_value: int) -> Tensor:
        """把不同长度的一维序列右 padding 成 batch。"""

        max_len = max(sequence.shape[0] for sequence in sequences)
        batch_size = len(sequences)
        dtype = sequences[0].dtype
        device = sequences[0].device

        output = torch.full(
            (batch_size, max_len),
            pad_value,
            dtype=dtype,
            device=device,
        )
        for i, sequence in enumerate(sequences):
            output[i, : sequence.shape[0]] = sequence
        return output

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        labels: Optional[Tensor] = None,
        image_infos: Optional[list[Any]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """训练 forward。

        Returns:
            一个 dict，包含：
                loss / logits / visual_grid_size / visual_token_count
        """

        visual_embeds, visual_grid = self.encode_images(
            pixel_values=pixel_values,
            image_infos=image_infos,
        )
        multimodal_inputs = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            visual_embeds=visual_embeds,
        )

        outputs = self.language_model(
            inputs_embeds=multimodal_inputs["inputs_embeds"],
            attention_mask=multimodal_inputs["attention_mask"],
            labels=multimodal_inputs.get("labels"),
            use_cache=False,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "visual_grid_size": visual_grid,
            "visual_token_count": self._visual_token_count_for_output(visual_embeds),
            "expanded_attention_mask": multimodal_inputs["attention_mask"],
            "expanded_labels": multimodal_inputs.get("labels"),
        }

    @staticmethod
    def _visual_token_count_for_output(visual_embeds: Tensor | list[Tensor]) -> int | list[int]:
        """返回便于日志打印的视觉 token 数。"""

        if isinstance(visual_embeds, Tensor):
            return int(visual_embeds.shape[1])
        return [int(item.shape[0]) for item in visual_embeds]

    def print_trainable_parameters(self) -> None:
        """打印当前可训练参数量，便于确认冻结策略是否正确。"""

        trainable = 0
        total = 0
        for _, param in self.named_parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count

        ratio = 100 * trainable / total if total else 0
        print(f"可训练参数: {trainable:,} / {total:,} ({ratio:.4f}%)")


if __name__ == "__main__":
    # Quick end-to-end forward check. This loads local Qwen and SigLIP weights.
    try:
        from vlm.data.collator import CollatorConfig, VLMDataCollator
        from vlm.data.llava_pretrain_dataset import LlavaPretrainDataset
    except ImportError:
        import sys
        from pathlib import Path

        project_src = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_src))
        from vlm.data.collator import CollatorConfig, VLMDataCollator
        from vlm.data.llava_pretrain_dataset import LlavaPretrainDataset

    dataset = LlavaPretrainDataset(
        annotation_path="/root/autodl-tmp/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        image_root="/root/autodl-tmp/hf_datasets/LLaVA-Pretrain",
        max_samples=1,
    )
    collator = VLMDataCollator(
        CollatorConfig(
            tokenizer_path="/root/autodl-tmp/hf_models/Qwen3-1.7B",
            image_processor_path="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
            max_length=512,
        )
    )
    batch = collator([dataset[0]])

    model = QwenSiglipVLM(
        VLMModelConfig(
            qwen_path="/root/autodl-tmp/hf_models/Qwen3-1.7B",
            siglip_path="/root/autodl-tmp/hf_models/siglip2-so400m-patch14-384",
            image_token_id=batch["image_token_id"],
            tokenizer_length=len(collator.tokenizer),
            freeze_vision_encoder=True,
            freeze_language_model=True,
            torch_dtype="bfloat16",
        )
    )
    model.eval()
    model.print_trainable_parameters()

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pixel_values=batch["pixel_values"],
        )

    print("loss:", float(outputs["loss"]))
    print("visual_grid_size:", outputs["visual_grid_size"].as_tuple())
    print("visual_token_count:", outputs["visual_token_count"])
    print("expanded_attention_mask shape:", tuple(outputs["expanded_attention_mask"].shape))
    print("expanded_labels shape:", tuple(outputs["expanded_labels"].shape))
    print("\nQwenSiglipVLM quick check passed.")

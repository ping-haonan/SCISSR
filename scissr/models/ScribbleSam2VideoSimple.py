"""
ScribbleSam2VideoSimple - Stage 1 训练模型

设计特点：
1. 支持双通道 scribble 输入（正/负）
2. 支持 Mask Decoder LoRA 微调（可选）
3. 支持迭代推理
4. 空输入检测：当 scribble 全零时返回全零 embedding

迭代逻辑（符合 ScribblePrompt 原文）：
- 初始阶段：只有正 scribble（在 GT 前景区域）
- 修正阶段：在 FN 区域生成正 scribble，在 FP 区域生成负 scribble
- 修正 scribble 使用标准 scribble generator（Line/Centerline/Contour）

Author: ScribblePrompt Team
"""

from typing import Optional, Tuple, List, Dict
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

# 导入统一的 ScribbleEncoder
from scissr.models.scribble_encoder import ScribbleEncoder

# 导入 scribble generators
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scissr.interactions.adaptive_scribble import (
    AdaptiveScribble, CorrectionConfig
)
from scissr.interactions.scribbles import LineScribble


class ScribbleSam2VideoSimple(SAM2VideoPredictor):
    """
    简化版 ScribbleSam2Video
    
    特点：
    - 支持单通道或双通道 scribble 输入
    - 支持 Mask Decoder LoRA 微调（可选）
    - 支持迭代推理
    """
    
    def __init__(
        self,
        config_file: str,
        ckpt_path: str,
        device: str = None,
        scribble_channels: int = 1,  # 1 或 2
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 构建基础 SAM2VideoPredictor
        sam2_model = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
        )
        
        # 初始化父类
        super().__init__(
            image_encoder=sam2_model.image_encoder,
            memory_attention=sam2_model.memory_attention,
            memory_encoder=sam2_model.memory_encoder,
            fill_hole_area=sam2_model.fill_hole_area,
            non_overlap_masks=sam2_model.non_overlap_masks,
        )
        
        # 复制所有属性
        for attr in vars(sam2_model):
            setattr(self, attr, getattr(sam2_model, attr))
        
        self.to(device)
        self._device = device
        self.input_size = (1024, 1024)
        self.scribble_channels = scribble_channels
        
        # 修正 scribble 生成器（混合策略）：
        # FN（漏检）→ AdaptiveScribble：沿结构骨架精准引导（对 Thread 至关重要）
        #   使用 LowResCorrectionConfig：阈值=1，因为在 256 降分辨率上生成
        # FP（误检）→ LineScribble：轻量快速，模拟用户"随手划掉"行为
        lowres_config = CorrectionConfig()
        lowres_config.MIN_AREA_THRESHOLD = 1   # 256 上不过滤，交给 try/except 兜底
        lowres_config.MIN_COMPONENT_AREA = 1   # 256 上不过滤
        self.correction_pos_generator = AdaptiveScribble(
            config=lowres_config,
        )
        self.correction_neg_generator = LineScribble(
            thickness=3, warp=False,
        )
        
        # 初始化 ScribbleEncoder（统一版本，支持空输入检测）
        self.scribble_encoder = ScribbleEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
            in_channels=scribble_channels,
        ).to(device)
        
        print(f"[ScribbleSam2VideoSimple] Model initialized on {device}, scribble_channels={scribble_channels}")
    
    def freeze_pretrained(self):
        """冻结所有预训练参数，只保留 ScribbleEncoder 可训练"""
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻 ScribbleEncoder
        for param in self.scribble_encoder.parameters():
            param.requires_grad = True
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Freeze] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def enable_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list = None,
    ):
        """
        为 Mask Decoder 启用 LoRA
        
        Args:
            rank: LoRA rank（低秩维度）
            alpha: LoRA alpha（缩放因子）
            dropout: dropout rate
            target_modules: 要应用 LoRA 的模块名称，默认 ['q_proj', 'v_proj']
            
        Returns:
            LoRA 统计信息
        """
        from scissr.models.lora import (
            apply_lora_to_mask_decoder, print_lora_summary
        )
        
        # 应用 LoRA 到 Mask Decoder
        stats = apply_lora_to_mask_decoder(
            self.sam_mask_decoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
        
        print(f"\n[LoRA] Applied to Mask Decoder:")
        print(f"  - Rank: {rank}, Alpha: {alpha}")
        print(f"  - LoRA modules: {stats['total_lora_modules']}")
        print(f"  - LoRA params: {stats['total_lora_params']:,}")
        print(f"  - Locations: {', '.join(stats['locations'])}")
        
        # 打印总结
        print_lora_summary(self)
        
        self._lora_enabled = True
        self._lora_stats = stats
        
        return stats
    
    def get_trainable_params(self) -> list:
        """
        获取所有可训练参数（ScribbleEncoder + LoRA）
        """
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def get_trainable_param_groups(self, base_lr: float = 1e-4) -> list:
        """
        获取参数组（可以为不同模块设置不同学习率）
        
        Returns:
            [
                {'params': scribble_encoder_params, 'lr': base_lr},
                {'params': lora_params, 'lr': base_lr * 0.1},  # LoRA 用更小的学习率
            ]
        """
        scribble_params = []
        lora_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'scribble_encoder' in name:
                scribble_params.append(param)
            elif 'lora_' in name:
                lora_params.append(param)
        
        param_groups = [
            {'params': scribble_params, 'lr': base_lr, 'name': 'scribble_encoder'},
        ]
        
        if lora_params:
            # LoRA 参数使用稍小的学习率
            param_groups.append({
                'params': lora_params, 
                'lr': base_lr * 0.5,  # LoRA 学习率是 ScribbleEncoder 的一半
                'name': 'lora'
            })
        
        return param_groups
    
    def forward_single_image(
        self,
        image: torch.Tensor,
        scribble: torch.Tensor,
        prev_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单图推理（用于训练）
        
        Args:
            image: (B, 3, H, W) 输入图像，需要已经 ImageNet 归一化
            scribble: (B, C, H, W) scribble，C=1或2
                      如果 C=2: 通道0=正scribble，通道1=负scribble
            prev_mask: (B, 1, H, W) 上一次预测的 mask（可选，用于迭代）
            
        Returns:
            low_res_masks: (B, 1, 256, 256) 低分辨率 logits
            high_res_masks: (B, 1, 1024, 1024) 高分辨率 logits
        """
        B = image.shape[0]
        device = image.device
        
        # 1. 图像编码
        backbone_out = self.forward_image(image)
        
        # 获取 backbone features
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # 获取最高层特征
        pix_feat = vision_feats[-1]
        pix_feat = pix_feat.permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        
        # 高分辨率特征
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(B, x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # 2. Scribble 编码
        scribble_embedding = self.scribble_encoder(scribble)
        
        # 3. Prompt 编码（只使用 scribble，不使用 point/box）
        # 创建空的 point prompt
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        
        # mask prompt（如果有 prev_mask）
        sam_mask_prompt = None
        if prev_mask is not None:
            if prev_mask.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    prev_mask.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                sam_mask_prompt = prev_mask
        
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        
        # 4. 融合 scribble embedding
        assert scribble_embedding.shape == dense_embeddings.shape, \
            f"Shape mismatch: scribble {scribble_embedding.shape} vs dense {dense_embeddings.shape}"
        dense_embeddings = dense_embeddings + scribble_embedding
        
        # 5. Mask Decoder
        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # 6. 上采样到高分辨率
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        
        return low_res_masks, high_res_masks
    
    def forward_with_features(
        self,
        pix_feat: torch.Tensor,
        high_res_features: List[torch.Tensor],
        scribble: torch.Tensor,
        prev_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用已计算的图像特征进行推理（避免重复编码图像）
        
        Args:
            pix_feat: (B, C, H, W) 图像特征
            high_res_features: 高分辨率特征列表
            scribble: (B, C, H, W) scribble
            prev_mask: (B, 1, H, W) 上一次的 mask logits（可选）
            
        Returns:
            low_res_masks, high_res_masks
        """
        B = pix_feat.shape[0]
        device = pix_feat.device
        
        # Scribble 编码
        scribble_embedding = self.scribble_encoder(scribble)
        
        # 创建空的 point prompt
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        
        # mask prompt（如果有 prev_mask）
        sam_mask_prompt = None
        if prev_mask is not None:
            if prev_mask.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    prev_mask.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                sam_mask_prompt = prev_mask
        
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        
        # 融合 scribble embedding
        dense_embeddings = dense_embeddings + scribble_embedding
        
        # Mask Decoder
        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # 上采样到高分辨率
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        
        return low_res_masks, high_res_masks
    
    def forward_iterative(
        self,
        image: torch.Tensor,
        initial_scribble: torch.Tensor,
        gt_mask: torch.Tensor,
        num_iterations: int = 2,
        correction_threshold: float = 0.5,
        correction_n_scribbles: int = 1,
        correction_min_area: int = 0,
        correction_min_area_final: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        迭代推理（用于训练）
        
        符合 ScribblePrompt 原文逻辑：
        1. 第一次推理：使用初始 scribble（只有正 scribble，负通道为空）
        2. 计算错误区域：FN（漏检）和 FP（误检）
        3. 生成修正 scribble（使用标准 scribble generator）：
           - 在 FN 区域生成正 scribble（模型漏掉了，需要告诉它这里有目标）
           - 在 FP 区域生成负 scribble（模型误检了，需要告诉它这里没有目标）
        4. 第二次推理：使用累积的 scribble + 上一次的 low_res_mask
        
        Args:
            image: (B, 3, H, W) 输入图像
            initial_scribble: (B, 2, H, W) 初始 scribble（双通道）
                - 通道0: 正 scribble（前景区域）
                - 通道1: 负 scribble（初始为空）
            gt_mask: (B, 1, H, W) GT mask
            num_iterations: 迭代次数
            correction_threshold: 二值化阈值
            correction_n_scribbles: 修正 scribble 数量
            
        Returns:
            Dict 包含：
                - 'masks_0', 'masks_1', ...: 每次迭代的高分辨率 mask
                - 'low_res_masks_0', 'low_res_masks_1', ...: 低分辨率 mask
                - 'scribbles_0', 'scribbles_1', ...: 每次使用的 scribble
                - 'corrections_pos_1', 'corrections_neg_1', ...: 修正 scribble
        """
        B = image.shape[0]
        device = image.device
        outputs = {}
        
        # 1. 图像编码（只做一次）
        backbone_out = self.forward_image(image)
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        pix_feat = vision_feats[-1]
        pix_feat = pix_feat.permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(B, x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # 当前 scribble（会累积）
        current_scribble = initial_scribble.clone()
        prev_low_res_mask = None
        
        for iter_idx in range(num_iterations):
            # 保存当前 scribble
            outputs[f'scribbles_{iter_idx}'] = current_scribble.clone()
            
            # 推理
            low_res_masks, high_res_masks = self.forward_with_features(
                pix_feat=pix_feat,
                high_res_features=high_res_features,
                scribble=current_scribble,
                prev_mask=prev_low_res_mask,
            )
            
            outputs[f'low_res_masks_{iter_idx}'] = low_res_masks
            outputs[f'masks_{iter_idx}'] = high_res_masks
            
            # 如果还有下一次迭代，生成修正 scribble
            if iter_idx < num_iterations - 1:
                with torch.no_grad():
                    # 计算错误区域
                    pred_binary = (torch.sigmoid(high_res_masks) > correction_threshold).float()
                    gt_binary = (gt_mask > correction_threshold).float()
                    
                    # FN: GT=1, Pred=0 (漏检，需要正 scribble 告诉模型"这里有目标")
                    fn_region = gt_binary * (1 - pred_binary)  # (B, 1, H, W)
                    
                    # FP: GT=0, Pred=1 (误检，需要负 scribble 告诉模型"这里没有目标")
                    fp_region = (1 - gt_binary) * pred_binary  # (B, 1, H, W)
                    
                    # 动态 min_area：最后一轮修正使用更小阈值
                    is_final_correction = (iter_idx == num_iterations - 2)
                    min_area = correction_min_area
                    if is_final_correction and correction_min_area_final is not None:
                        min_area = correction_min_area_final
                    
                    # 混合策略：FN 用 Adaptive（精准引导），FP 用 Line（快速擦除）
                    pos_correction = self._generate_correction_scribble(
                        fn_region, n_scribbles=correction_n_scribbles,
                        is_positive=True,
                    )
                    neg_correction = self._generate_correction_scribble(
                        fp_region, n_scribbles=correction_n_scribbles,
                        is_positive=False,
                    )
                    
                    outputs[f'corrections_pos_{iter_idx+1}'] = pos_correction
                    outputs[f'corrections_neg_{iter_idx+1}'] = neg_correction
                    outputs[f'fn_region_{iter_idx}'] = fn_region
                    outputs[f'fp_region_{iter_idx}'] = fp_region
                    
                    # 累积 scribble（使用 max 避免覆盖）
                    current_scribble = torch.stack([
                        torch.max(current_scribble[:, 0], pos_correction.squeeze(1)),
                        torch.max(current_scribble[:, 1], neg_correction.squeeze(1)),
                    ], dim=1)
                    
                    # 更新 prev_mask 用于下一次迭代
                    prev_low_res_mask = low_res_masks.detach()
        
        return outputs
    
    # Correction scribble 生成分辨率（降分辨率加速骨架提取）
    # 256: 面积缩 16 倍，thinning 速度 ~3ms vs ~80ms@1024
    # 配合 max_pool 下采样（保留细线）+ 阈值=1（不过滤任何组件）
    CORRECTION_GEN_SIZE = 256
    
    def _generate_correction_scribble(
        self,
        error_region: torch.Tensor,
        n_scribbles: int = 1,
        is_positive: bool = True,
    ) -> torch.Tensor:
        """
        混合策略生成修正 scribble（降分辨率加速）：
        - FN（is_positive=True）→ AdaptiveScribble：降到 256×256 做骨架提取，再上采样
        - FP（is_positive=False）→ LineScribble：轻量快速
        
        降分辨率理由：cv2.ximgproc.thinning 在 1024×1024 上很慢，
        在 512×512 上生成（面积缩 4 倍，骨架提取快 ~4 倍），再 nearest 上采样回去。
        用 max_pool 下采样保证 Thread 等 1-2px 细线不会消失。
        """
        if error_region.sum() < 1:
            return torch.zeros_like(error_region)
        
        _, _, H, W = error_region.shape
        gen_size = self.CORRECTION_GEN_SIZE
        
        try:
            if is_positive:
                # FN: 降分辨率 → AdaptiveScribble → 上采样
                if H > gen_size:
                    # 下采样用 max_pool：保证 1-2px 的细线（Thread）不会消失
                    pool_k = H // gen_size  # 1024/256 = 4
                    error_small = F.max_pool2d(
                        error_region.float(), kernel_size=pool_k, stride=pool_k,
                    )
                    correction_small = self.correction_pos_generator(
                        error_small, n_scribbles=n_scribbles
                    )
                    # 上采样用 nearest：scribble 是二值信号，不需要平滑
                    correction = F.interpolate(
                        correction_small.float(), size=(H, W),
                        mode='nearest',
                    )
                else:
                    correction = self.correction_pos_generator(
                        error_region, n_scribbles=n_scribbles
                    )
            else:
                # FP: LineScribble 本身就很快，不需要降分辨率
                correction = self.correction_neg_generator(
                    error_region, n_scribbles=n_scribbles
                )
            return correction
        except Exception as e:
            return torch.zeros_like(error_region)
    
    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

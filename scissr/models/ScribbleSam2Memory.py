"""
ScribbleSam2Memory: Memory-Driven Interactive Segmentation

核心思想：
- 把 SAM2 的"帧间传播"能力转换为"迭代修正"能力
- 在同一张图上进行多轮推理，每轮推理后将结果存入 Memory Bank
- 下一轮推理时，Memory Attention 可以从 Memory Bank 中提取上一轮的特征信息

设计决策：
- Scribble 注入位置：Memory Attention Query (使用 Gated Residual Fusion)
- Round 0 Prompt：Noisy Box
- 迭代次数：3 轮 (R0 初始 + R1, R2 两轮修正)
- Round 0 计算 Loss：是
- Memory Bank：只保留上一轮 (FIFO, Size=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import random

from sam2.sam2_video_predictor import SAM2VideoPredictor


class SpatialGatedFusion(nn.Module):
    """
    空间门控融合模块 - 使用 Large Kernel DW-Conv 实现"墨水扩散"效果
    
    核心改进（相比原始 GatedResidualFusion 的 Conv1x1）：
    1. 7x7 Depthwise Conv：感受野从 1x1 扩展到 7x7，实现空间扩散
    2. Hard Gate (Binary)：判断是否有 scribble 输入，作用于输入端
    3. Soft Gate (Learnable Alpha)：零初始化，平滑接入残差，作用于输出端
    
    设计理念：
    - "墨水扩散效应"：scribble 的影响应该基于空间邻近性扩散开来
    - "零初始化"：训练初期 alpha=0，模型等价于原始 SAM2，渐进式引入 scribble
    - "保护预训练权重"：通过 learnable alpha 避免"特征休克"
    
    公式: 
        output = image_features + alpha * has_scribble * SpatialMix(Concat(image, scribble))
    """
    
    def __init__(self, feature_dim: int = 256, scribble_dim: int = 256, kernel_size: int = 7):
        super().__init__()
        self.feature_dim = feature_dim
        self.scribble_dim = scribble_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 1. 降维融合：Concat(image, scribble) → feature_dim
        # 使用 GroupNorm 替代 BatchNorm2d（batch_size=2 时 BN 统计量不稳定）
        self.proj_in = nn.Sequential(
            nn.Conv2d(feature_dim + scribble_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=feature_dim),
            nn.GELU(),
        )
        
        # 2. 空间扩散层 (核心改进)
        # Depthwise Conv: 大感受野 (7x7)，参数量小 (256 * 49 = 12544 params)
        # Pointwise Conv: 通道交互
        self.spatial_mix = nn.Sequential(
            # Depthwise Conv: 每个通道独立卷积，实现空间扩散
            nn.Conv2d(feature_dim, feature_dim, kernel_size=kernel_size,
                      padding=padding, groups=feature_dim, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=feature_dim),
            nn.GELU(),
            # Pointwise Conv: 通道间信息交互
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=feature_dim),
        )
        
        # 3. Soft Gate: 可学习的缩放因子，零初始化
        # 训练初期 alpha=0，模型完全等价于原始 SAM2
        # 随着训练，alpha 逐渐增大，平滑引入 scribble 信号
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        权重初始化策略：
        1. Conv 使用 Kaiming 初始化
        2. GroupNorm 使用标准初始化 (weight=1, bias=0)
        3. alpha 已经在定义时初始化为 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self, 
        image_features: torch.Tensor,  # (B, C, H, W) or (HW, B, C)
        scribble_embed: torch.Tensor,  # (B, C, H, W)
        scribble_input: torch.Tensor = None,  # 原始 scribble，用于计算 hard gate
    ) -> torch.Tensor:
        """
        Args:
            image_features: 图像特征，可能是 (B, C, H, W) 或 (HW, B, C)
            scribble_embed: Scribble 编码后的 embedding (B, C, H, W)
            scribble_input: 原始 scribble 输入，用于判断是否为空
            
        Returns:
            fused_features: 融合后的特征，形状与 image_features 相同
        """
        # 处理输入维度: (HW, B, C) → (B, C, H, W)
        need_permute = False
        
        if image_features.dim() == 3:
            need_permute = True
            HW, B, C = image_features.shape
            H = W = int(np.sqrt(HW))
            image_features = image_features.permute(1, 2, 0).view(B, C, H, W)
        
        B, C, H, W = image_features.shape
        
        # ========== 1. Hard Gate: 判断是否有 scribble (Binary Mask) ==========
        # 作用于输入端，确保空 scribble 时输入纯净
        if scribble_input is not None:
            scribble_density = scribble_input.abs().sum(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
            has_scribble = (scribble_density > 1e-6).float()  # (B, 1, 1, 1)
        else:
            has_scribble = torch.ones(B, 1, 1, 1, device=image_features.device)
        
        # 确保 scribble_embed 尺寸匹配
        if scribble_embed.shape[-2:] != (H, W):
            scribble_embed = F.interpolate(
                scribble_embed, size=(H, W), mode='bilinear', align_corners=False
            )
        
        # ========== 2. 空间融合 ==========
        # Hard gate 作用于输入: 空 scribble 时，concat 的一半是 0
        scribble_feat = scribble_embed * has_scribble
        
        # Concat + 降维
        concat = torch.cat([image_features, scribble_feat], dim=1)  # (B, 2C, H, W)
        x = self.proj_in(concat)  # (B, C, H, W)
        
        # 空间扩散 (7x7 DW-Conv)
        delta = self.spatial_mix(x)  # (B, C, H, W)
        
        # ========== 3. Soft Gate: 残差连接 ==========
        # alpha 控制融合力度（可学习，初始化为 0）
        # has_scribble 控制开关（binary）
        # 公式: output = image_features + alpha * has_scribble * delta
        fused_features = image_features + self.alpha * has_scribble * delta
        
        # 恢复原始形状: (B, C, H, W) → (HW, B, C)
        if need_permute:
            fused_features = fused_features.view(B, C, -1).permute(2, 0, 1)
        
        return fused_features


# 保留旧名称作为别名，以便兼容
GatedResidualFusion = SpatialGatedFusion


# 导入统一的 ScribbleEncoder（支持空输入检测，避免 bias 噪声）
from scissr.models.scribble_encoder import ScribbleEncoder


class NoisyBoxGenerator:
    """
    生成带噪声的边界框
    
    用于 Round 0 模拟用户提供的不精确初始定位
    """
    
    def __init__(
        self,
        jitter_ratio: float = 0.1,    # 位置抖动比例
        scale_range: Tuple[float, float] = (0.8, 1.2),  # 缩放范围
    ):
        self.jitter_ratio = jitter_ratio
        self.scale_range = scale_range
    
    def generate_noisy_box(
        self,
        gt_mask: torch.Tensor,  # (B, 1, H, W) 或 (B, H, W)
    ) -> torch.Tensor:
        """
        从 GT mask 生成带噪声的边界框
        
        Args:
            gt_mask: Ground Truth mask
            
        Returns:
            noisy_boxes: (B, 4) 格式为 [x1, y1, x2, y2]，归一化到 [0, 1]
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        
        B, _, H, W = gt_mask.shape
        device = gt_mask.device
        
        boxes = []
        for b in range(B):
            mask = gt_mask[b, 0]  # (H, W)
            
            # 找到 mask 的边界框
            if mask.sum() > 0:
                rows = torch.any(mask > 0.5, dim=1)
                cols = torch.any(mask > 0.5, dim=0)
                
                y_indices = torch.where(rows)[0]
                x_indices = torch.where(cols)[0]
                
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y1, y2 = y_indices[0].item(), y_indices[-1].item()
                    x1, x2 = x_indices[0].item(), x_indices[-1].item()
                else:
                    # 默认中心区域
                    x1, y1 = W // 4, H // 4
                    x2, y2 = 3 * W // 4, 3 * H // 4
            else:
                # 空 mask，使用中心区域
                x1, y1 = W // 4, H // 4
                x2, y2 = 3 * W // 4, 3 * H // 4
            
            # 计算中心和尺寸
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_w = x2 - x1
            box_h = y2 - y1
            
            # 添加噪声
            # 1. 位置抖动
            jitter_x = random.uniform(-self.jitter_ratio, self.jitter_ratio) * box_w
            jitter_y = random.uniform(-self.jitter_ratio, self.jitter_ratio) * box_h
            cx += jitter_x
            cy += jitter_y
            
            # 2. 尺寸缩放
            scale = random.uniform(*self.scale_range)
            box_w *= scale
            box_h *= scale
            
            # 计算新的边界框
            new_x1 = max(0, cx - box_w / 2)
            new_y1 = max(0, cy - box_h / 2)
            new_x2 = min(W, cx + box_w / 2)
            new_y2 = min(H, cy + box_h / 2)
            
            # 归一化到 [0, 1]
            boxes.append([new_x1 / W, new_y1 / H, new_x2 / W, new_y2 / H])
        
        return torch.tensor(boxes, device=device, dtype=torch.float32)


class CorrectionScribbleGenerator:
    """
    混合策略修正 Scribble 生成器
    
    FN（漏检）→ AdaptiveScribble：沿结构骨架精准引导（Thread 等细结构需要）
    FP（误检）→ LineScribble：轻量快速，模拟用户"随手划掉"行为
    """
    
    def __init__(
        self,
        min_area: int = 10,         # 最小错误区域面积
        thickness: int = 3,         # FP scribble 线宽
        max_scribbles: int = 3,     # 最大 scribble 数量
    ):
        from scissr.interactions.adaptive_scribble import (
            AdaptiveScribble, CorrectionConfig
        )
        from scissr.interactions.scribbles import LineScribble
        
        self.min_area = min_area
        self.thickness = thickness
        self.max_scribbles = max_scribbles
        
        # FN: AdaptiveScribble（精准引导）
        self.fn_generator = AdaptiveScribble(config=CorrectionConfig())
        # FP: LineScribble（快速擦除）
        self.fp_generator = LineScribble(thickness=thickness, warp=False)
    
    def generate(
        self,
        pred_mask: torch.Tensor,   # (B, 1, H, W) logits
        gt_mask: torch.Tensor,     # (B, 1, H, W)
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合策略生成正/负修正 scribble
        
        FN → AdaptiveScribble（沿结构骨架精准引导）
        FP → LineScribble（快速擦除）
        
        Args:
            pred_mask: 预测 mask (logits)
            gt_mask: Ground Truth mask
            threshold: 二值化阈值
            
        Returns:
            pos_scribble: (B, 1, H, W) 正 scribble (FN 区域)
            neg_scribble: (B, 1, H, W) 负 scribble (FP 区域)
        """
        B, _, H, W = pred_mask.shape
        
        # 二值化
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
        gt_binary = (gt_mask > threshold).float()
        
        # FN: GT=1, Pred=0 (漏检) → 需要正 scribble
        fn_region = gt_binary * (1 - pred_binary)
        # FP: GT=0, Pred=1 (误检) → 需要负 scribble
        fp_region = (1 - gt_binary) * pred_binary
        
        # FN → AdaptiveScribble（精准引导）
        pos_scribbles = torch.zeros_like(pred_mask)
        if fn_region.sum() > 1:
            try:
                pos_scribbles = self.fn_generator(fn_region, n_scribbles=1)
            except Exception:
                pass
        
        # FP → LineScribble（快速擦除）
        neg_scribbles = torch.zeros_like(pred_mask)
        if fp_region.sum() > 1:
            try:
                neg_scribbles = self.fp_generator(fp_region, n_scribbles=1)
            except Exception:
                pass
        
        return pos_scribbles, neg_scribbles


class ScribbleSam2Memory(SAM2VideoPredictor):
    """
    Memory-Driven Interactive Segmentation Model
    
    基于 SAM2VideoPredictor，添加：
    1. Scribble Encoder
    2. Gated Residual Fusion (注入到 Memory Attention Query)
    3. Memory-Driven 迭代推理
    """
    
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        scribble_channels: int = 2,
        image_size: int = 1024,
        **kwargs,
    ):
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            image_size=image_size,
            **kwargs,
        )
        
        self.scribble_channels = scribble_channels
        self.image_size = image_size
        
        # Scribble Encoder（统一版本，支持空输入检测）
        self.scribble_encoder = ScribbleEncoder(
            embed_dim=self.hidden_dim,  # 256
            in_channels=scribble_channels,
        )
        
        # Spatial Gated Fusion (用于 Query 注入)
        # 使用 7x7 DW-Conv 实现"墨水扩散"效果
        self.query_fusion = SpatialGatedFusion(
            feature_dim=self.hidden_dim,
            scribble_dim=self.hidden_dim,
            kernel_size=7,  # 7x7 感受野
        )
        
        # 辅助工具
        self.noisy_box_generator = NoisyBoxGenerator()
        self.correction_generator = CorrectionScribbleGenerator()
        
        # Memory Bank (只保留上一轮)
        self._memory_bank = None
        self._memory_pos_bank = None
        
        # LoRA 状态
        self._memory_lora_enabled = False
        self._memory_lora_info = None
    
    def reset_memory_bank(self):
        """重置 Memory Bank"""
        self._memory_bank = None
        self._memory_pos_bank = None
    
    def enable_memory_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: List[str] = None,
        apply_to_self_attn: bool = True,
        apply_to_cross_attn: bool = True,
        apply_to_ffn: bool = False,
    ) -> Dict:
        """
        为 Memory Attention 启用 LoRA
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha (缩放因子)
            target_modules: 目标模块名称 (默认 ['q_proj', 'v_proj'])
            apply_to_self_attn: 是否应用到 Self Attention
            apply_to_cross_attn: 是否应用到 Cross Attention
            apply_to_ffn: 是否应用到 FFN
            
        Returns:
            LoRA 统计信息
        """
        from scissr.models.memory_attention_lora import (
            apply_lora_to_memory_attention,
            freeze_memory_attention_original_params,
            print_memory_attention_trainable_params,
        )
        
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']
        
        # 应用 LoRA
        info = apply_lora_to_memory_attention(
            self.memory_attention,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            apply_to_self_attn=apply_to_self_attn,
            apply_to_cross_attn=apply_to_cross_attn,
            apply_to_ffn=apply_to_ffn,
        )
        
        # 冻结原始 Memory Attention 参数
        freeze_memory_attention_original_params(self.memory_attention)
        
        print(f"\n[Memory LoRA] Applied to Memory Attention:")
        print(f"  - Rank: {rank}, Alpha: {alpha}")
        print(f"  - Target modules: {target_modules}")
        print(f"  - LoRA params: {info['total_lora_params']:,}")
        print_memory_attention_trainable_params(self.memory_attention)
        
        self._memory_lora_enabled = True
        self._memory_lora_info = info
        
        return info
    
    def enable_mask_decoder_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: List[str] = None,
    ) -> Dict:
        """
        为 Mask Decoder 启用 LoRA
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha (缩放因子)
            dropout: dropout rate
            target_modules: 目标模块名称 (默认 ['q_proj', 'v_proj'])
            
        Returns:
            LoRA 统计信息
        """
        from scissr.models.lora import (
            apply_lora_to_mask_decoder,
            print_lora_summary,
        )
        
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']
        
        # 应用 LoRA 到 Mask Decoder
        stats = apply_lora_to_mask_decoder(
            self.sam_mask_decoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
        
        print(f"\n[Mask Decoder LoRA] Applied:")
        print(f"  - Rank: {rank}, Alpha: {alpha}")
        print(f"  - LoRA modules: {stats['total_lora_modules']}")
        print(f"  - LoRA params: {stats['total_lora_params']:,}")
        print(f"  - Locations: {', '.join(stats['locations'])}")
        
        self._mask_decoder_lora_enabled = True
        self._mask_decoder_lora_info = stats
        
        return stats
    
    def freeze_pretrained(self):
        """冻结预训练权重，只训练新添加的模块"""
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻 Scribble Encoder
        for param in self.scribble_encoder.parameters():
            param.requires_grad = True
        
        # 解冻 Query Fusion
        for param in self.query_fusion.parameters():
            param.requires_grad = True
        
        # 如果启用了 Memory LoRA，解冻 LoRA 参数
        if self._memory_lora_enabled:
            for name, param in self.memory_attention.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
        
        # 如果启用了 Mask Decoder LoRA，解冻 LoRA 参数
        if hasattr(self, '_mask_decoder_lora_enabled') and self._mask_decoder_lora_enabled:
            for name, param in self.sam_mask_decoder.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
        
        # 打印统计
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Freeze] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取所有可训练参数"""
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def get_trainable_param_groups(
        self, 
        base_lr: float = 1e-4,
        scribble_lr: float = None,
        mask_decoder_lora_lr: float = None,
        alpha_lr: float = None,
    ) -> List[Dict]:
        """
        获取参数组（可以为不同模块设置不同学习率）
        
        Args:
            base_lr: 基础学习率 (用于 query_fusion 和 memory_lora)
            scribble_lr: ScribbleEncoder 学习率 (默认 = base_lr * 0.1)
            mask_decoder_lora_lr: MaskDecoder LoRA 学习率 (默认 = base_lr * 0.1)
            alpha_lr: SpatialGatedFusion 的 alpha 参数学习率 (默认 = base_lr * 0.5)
            
        Returns:
            参数组列表，格式: [{'params': [...], 'lr': ..., 'name': ...}, ...]
        """
        if scribble_lr is None:
            scribble_lr = base_lr * 0.1  # 默认 1e-5
        if mask_decoder_lora_lr is None:
            mask_decoder_lora_lr = base_lr * 0.1  # 默认 1e-5
        if alpha_lr is None:
            alpha_lr = base_lr * 0.5  # 默认 5e-5，比其他参数更稳定
        
        scribble_params = []
        fusion_params = []
        fusion_alpha_params = []  # Alpha 单独分离
        memory_lora_params = []
        mask_decoder_lora_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'scribble_encoder' in name:
                scribble_params.append(param)
            elif 'query_fusion' in name:
                # Alpha 参数单独处理
                if 'alpha' in name:
                    fusion_alpha_params.append(param)
                else:
                    fusion_params.append(param)
            elif 'memory_attention' in name and 'lora_' in name:
                memory_lora_params.append(param)
            elif 'sam_mask_decoder' in name and 'lora_' in name:
                mask_decoder_lora_params.append(param)
        
        param_groups = []
        
        # ScribbleEncoder: 低学习率（已在 Stage 1 训练）
        if scribble_params:
            param_groups.append({
                'params': scribble_params,
                'lr': scribble_lr,
                'name': 'scribble_encoder',
            })
        
        # QueryFusion (不含 alpha): 高学习率（新模块）
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': base_lr,
                'name': 'query_fusion',
            })
        
        # QueryFusion Alpha: 单独学习率 + 无 weight_decay（防止被正则化压回 0）
        if fusion_alpha_params:
            param_groups.append({
                'params': fusion_alpha_params,
                'lr': alpha_lr,
                'weight_decay': 0.0,  # 关键：防止 alpha 被正则化压回 0
                'name': 'query_fusion_alpha',
            })
        
        # Memory Attention LoRA: 高学习率（新模块）
        if memory_lora_params:
            param_groups.append({
                'params': memory_lora_params,
                'lr': base_lr,
                'name': 'memory_lora',
            })
        
        # Mask Decoder LoRA: 低学习率（已在 Stage 1 训练）
        if mask_decoder_lora_params:
            param_groups.append({
                'params': mask_decoder_lora_params,
                'lr': mask_decoder_lora_lr,
                'name': 'mask_decoder_lora',
            })
        
        return param_groups
    
    def _encode_scribble(self, scribble: torch.Tensor) -> torch.Tensor:
        """
        编码 Scribble
        
        Args:
            scribble: (B, 2, H, W) 或 None
            
        Returns:
            scribble_embed: (B, 256, 64, 64) 或 None
        """
        if scribble is None:
            return None
        return self.scribble_encoder(scribble)
    
    def _prepare_memory_conditioned_features_with_scribble(
        self,
        current_vision_feats: List[torch.Tensor],
        current_vision_pos_embeds: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        scribble: torch.Tensor = None,
        use_memory: bool = False,
    ) -> torch.Tensor:
        """
        准备带 Scribble 融合的 Memory 条件特征
        
        这是核心方法：将 Scribble Embedding 注入到 Memory Attention 的 Query
        
        注意：
        - 初始帧 (use_memory=False): 直接添加 no_mem_embed，不经过 Memory Attention
        - 后续帧 (use_memory=True): 使用 Memory Attention 融合历史记忆
        
        Args:
            current_vision_feats: 当前帧视觉特征
            current_vision_pos_embeds: 位置编码
            feat_sizes: 特征尺寸
            scribble: (B, 2, H, W) scribble 输入
            use_memory: 是否使用 Memory Bank
            
        Returns:
            pix_feat_with_mem: (B, C, H, W) 融合后的特征
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        
        # 1. 编码 Scribble
        scribble_embed = self._encode_scribble(scribble)  # (B, 256, 64, 64) 或 None
        
        # 2. 准备 Query (vision_feats + scribble_embed)
        # current_vision_feats[-1] 形状: (HW, B, C)
        query_feats = current_vision_feats[-1]  # (HW, B, C)
        
        if scribble_embed is not None:
            # 融合 scribble 到 query
            query_feats = self.query_fusion(
                image_features=query_feats,
                scribble_embed=scribble_embed,
                scribble_input=scribble,
            )
        
        # 3. 根据是否使用 Memory 选择不同的处理方式
        if use_memory and self._memory_bank is not None:
            # ========== 后续帧：使用 Memory Attention ==========
            # Memory Bank 存储的是 mem_dim=64 维度的特征
            memory = self._memory_bank      # (HW, B, mem_dim)
            memory_pos = self._memory_pos_bank  # (HW, B, mem_dim)
            
            # Memory Attention
            pix_feat_with_mem = self.memory_attention(
                curr=[query_feats],  # (HW, B, hidden_dim)
                curr_pos=current_vision_pos_embeds[-1:],
                memory=memory,       # (HW, B, mem_dim)
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0,
            )
        else:
            # ========== 初始帧：直接添加 no_mem_embed，跳过 Memory Attention ==========
            # 这是 SAM2 的默认行为 (directly_add_no_mem_embed=True)
            pix_feat_with_mem = query_feats + self.no_mem_embed  # (HW, B, C)
        
        # 4. Reshape: (HW, B, C) → (B, C, H, W)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        
        return pix_feat_with_mem
    
    def _encode_to_memory(
        self,
        vision_feats: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将预测结果编码为 Memory
        
        使用原始 backbone 特征（而非 scribble 融合后的特征），原因：
        1. Track 1 已显式保留所有历史 scribble → Memory 无需重复编码
        2. Scribble 意图已通过 predicted mask 间接传递（mask 充当 feature selector）
        3. Memory Encoder 是冻结的，使用原始分布避免 distribution shift
        
        Args:
            vision_feats: backbone 输出特征列表
            feat_sizes: 特征尺寸列表
            pred_mask: (B, 1, H, W) 预测的 mask (高分辨率)
            
        Returns:
            memory_feat: (HW, B, mem_dim) Memory 特征
            memory_pos: (HW, B, mem_dim) Memory 位置编码
        """
        B = vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # 64, 64
        
        # 获取 top-level 特征: (HW, B, C) → (B, C, H, W)
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        
        # 准备 mask: 应用 sigmoid
        mask_for_mem = torch.sigmoid(pred_mask)
        
        # 调用 memory_encoder（输入原始 backbone 特征 + predicted mask）
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        
        maskmem_features = maskmem_out["vision_features"]  # (B, mem_dim, H, W)
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]    # list of (B, mem_dim, H, W)
        
        # 转换为 Memory Attention 期望的格式
        # (B, mem_dim, H, W) → (HW, B, mem_dim)
        memory_feat = maskmem_features.flatten(2).permute(2, 0, 1)  # (HW, B, mem_dim)
        memory_pos = maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)  # (HW, B, mem_dim)
        
        return memory_feat, memory_pos
    
    def forward_single_round(
        self,
        image: torch.Tensor,
        latest_scribble: torch.Tensor = None,
        accumulated_scribble: torch.Tensor = None,
        box: torch.Tensor = None,
        use_memory: bool = False,
        update_memory: bool = True,
        backbone_out: dict = None,
        vision_feats: List[torch.Tensor] = None,
        vision_pos_embeds: List[torch.Tensor] = None,
        feat_sizes: List[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        双轨单轮推理
        
        Track 1（累积 scribble → dense_embeddings）：
            ScribbleEncoder 编码所有历史 scribble，加到 Mask Decoder 的 dense input 上。
            承载"用户到底想分割什么"的完整意图，与 Stage 1 路径一致。
            
        Track 2（最新 scribble → Memory Attention Query）：
            ScribbleEncoder 编码本轮最新 scribble，通过 SpatialGatedFusion 注入到
            Memory Attention 的 Query 中。引导 Memory Attention "关注哪里需要修正"。
        
        Args:
            image: (B, 3, H, W) 输入图像
            latest_scribble: (B, 2, H, W) 本轮最新 scribble → Track 2 (可选)
            accumulated_scribble: (B, 2, H, W) 累积历史 scribble → Track 1 (可选)
            box: (B, 4) 边界框 [x1, y1, x2, y2] 归一化 (可选)
            use_memory: 是否使用 Memory Bank
            update_memory: 是否更新 Memory Bank
            backbone_out/vision_feats/...: 预计算的特征 (避免重复编码)
            
        Returns:
            low_res_masks: (B, 1, 256, 256) 低分辨率 mask
            high_res_masks: (B, 1, 1024, 1024) 高分辨率 mask
            cache: 缓存的中间结果
        """
        B = image.shape[0]
        device = image.device
        
        # 1. 图像编码 (如果未提供预计算特征)
        if backbone_out is None:
            backbone_out = self.forward_image(image)
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # 2. Track 2: 最新 scribble → QueryFusion → Memory Attention
        #    引导 Memory Attention 关注本轮修正区域
        pix_feat_with_mem = self._prepare_memory_conditioned_features_with_scribble(
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            scribble=latest_scribble,
            use_memory=use_memory,
        )
        
        # 3. 准备 Prompt
        if box is not None:
            H, W = self.image_size, self.image_size
            sam_boxes = box.clone()
            sam_boxes[:, [0, 2]] *= W
            sam_boxes[:, [1, 3]] *= H
            sam_boxes = sam_boxes.unsqueeze(1)
            
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                points=None,
                boxes=sam_boxes,
                masks=None,
            )
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
            
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels),
                boxes=None,
                masks=None,
            )
        
        # 4. Track 1: 累积 scribble → ScribbleEncoder → 加到 dense_embeddings
        #    承载完整的用户意图历史（与 Stage 1 路径一致）
        if accumulated_scribble is not None:
            accumulated_embed = self.scribble_encoder(accumulated_scribble)
            dense_embeddings = dense_embeddings + accumulated_embed
        
        # 5. 高分辨率特征
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(B, x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # 6. Mask Decoder
        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # 7. 上采样到高分辨率
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        
        # 8. 更新 Memory Bank（使用原始 backbone 特征，scribble 意图通过 mask 间接传递）
        if update_memory:
            memory_feat, memory_pos = self._encode_to_memory(
                vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_mask=high_res_masks,
            )
            self._memory_bank = memory_feat
            self._memory_pos_bank = memory_pos
        
        # 9. 返回缓存
        cache = {
            'backbone_out': backbone_out,
            'vision_feats': vision_feats,
            'vision_pos_embeds': vision_pos_embeds,
            'feat_sizes': feat_sizes,
        }
        
        return low_res_masks, high_res_masks, cache
    
    def forward_iterative(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        initial_scribble: torch.Tensor = None,
        num_rounds: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        双轨 Memory-Driven 迭代推理
        
        双轨设计：
        - Track 1（accumulated → dense_embeddings）：保留完整用户意图
        - Track 2（latest → QueryFusion → Memory Query）：聚焦本轮修正
        
        流程：
        - Round 0: Noisy Box + 初始 scribble → 初始预测 → 存入 Memory
        - Round 1+: 累积 scribble(Track 1) + 最新 correction(Track 2) + Memory → 预测
        
        Args:
            image: (B, 3, H, W) 输入图像
            gt_mask: (B, 1, H, W) Ground Truth mask
            initial_scribble: (B, 2, H, W) 初始 scribble (可选，Round 0 使用)
            num_rounds: 迭代轮数 (默认 3)
            
        Returns:
            outputs: Dict 包含每轮的输出
        """
        B = image.shape[0]
        device = image.device
        
        # 重置 Memory Bank
        self.reset_memory_bank()
        
        outputs = {}
        
        # 图像编码 (只做一次)
        backbone_out = self.forward_image(image)
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        cache = {
            'backbone_out': backbone_out,
            'vision_feats': vision_feats,
            'vision_pos_embeds': vision_pos_embeds,
            'feat_sizes': feat_sizes,
        }
        
        # 双轨 scribble 状态
        accumulated_scribble = initial_scribble  # Track 1: 累积所有历史
        latest_scribble = initial_scribble        # Track 2: 只有本轮最新
        
        for round_idx in range(num_rounds):
            if round_idx == 0:
                # ============ Round 0: Noisy Box + 初始 scribble ============
                noisy_box = self.noisy_box_generator.generate_noisy_box(gt_mask)
                
                # R0: Track 1 和 Track 2 相同（都是初始 scribble）
                low_res, high_res, cache = self.forward_single_round(
                    image=image,
                    latest_scribble=latest_scribble,
                    accumulated_scribble=accumulated_scribble,
                    box=noisy_box,
                    use_memory=False,
                    update_memory=True,
                    **cache,
                )
                
                outputs[f'box_{round_idx}'] = noisy_box
            else:
                # ============ Round 1+: Correction Scribble ============
                with torch.no_grad():
                    prev_high_res = outputs[f'masks_{round_idx-1}']
                    pos_correction, neg_correction = self.correction_generator.generate(
                        pred_mask=prev_high_res,
                        gt_mask=gt_mask,
                    )
                    
                    # Track 2: 只有本轮的 correction（聚焦修正意图）
                    latest_scribble = torch.cat([pos_correction, neg_correction], dim=1)
                    
                    # Track 1: 累积所有历史（保留完整用户意图）
                    if accumulated_scribble is not None:
                        accumulated_scribble = torch.stack([
                            torch.max(accumulated_scribble[:, 0], pos_correction.squeeze(1)),
                            torch.max(accumulated_scribble[:, 1], neg_correction.squeeze(1)),
                        ], dim=1)
                    else:
                        accumulated_scribble = latest_scribble.clone()
                    
                    outputs[f'pos_correction_{round_idx}'] = pos_correction
                    outputs[f'neg_correction_{round_idx}'] = neg_correction
                
                # 修正推理（使用 Memory + 双轨 scribble）
                low_res, high_res, cache = self.forward_single_round(
                    image=image,
                    latest_scribble=latest_scribble,
                    accumulated_scribble=accumulated_scribble,
                    box=None,
                    use_memory=True,
                    update_memory=(round_idx < num_rounds - 1),
                    **cache,
                )
            
            # 保存输出
            outputs[f'low_res_masks_{round_idx}'] = low_res
            outputs[f'masks_{round_idx}'] = high_res
            outputs[f'accumulated_scribble_{round_idx}'] = accumulated_scribble.clone() if accumulated_scribble is not None else None
            outputs[f'latest_scribble_{round_idx}'] = latest_scribble.clone() if latest_scribble is not None else None
        
        return outputs


def build_scribble_sam2_memory(
    config_file: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    ckpt_path: str = None,
    device: str = "cuda",
    scribble_channels: int = 2,
) -> ScribbleSam2Memory:
    """
    构建 ScribbleSam2Memory 模型
    
    Args:
        config_file: SAM2 配置文件
        ckpt_path: 预训练权重路径
        device: 设备
        scribble_channels: Scribble 通道数
        
    Returns:
        model: ScribbleSam2Memory 实例
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import os
    
    # 清理并重新初始化 Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # SAM2 配置目录 - 使用 ScribblePrompt 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sam2_config_dir = os.path.join(project_root, "sam2", "configs")
    
    if not os.path.exists(sam2_config_dir):
        # 尝试从 CWD 寻找
        sam2_config_dir = os.path.join(os.getcwd(), "sam2", "configs")
    
    initialize_config_dir(config_dir=sam2_config_dir, version_base="1.2")
    
    # 使用 SAM2 的构建方式
    from sam2.build_sam import build_sam2_video_predictor
    
    # 处理配置文件路径：需要相对于 sam2/configs 目录
    # 例如: "configs/sam2.1/sam2.1_hiera_t.yaml" -> "sam2.1/sam2.1_hiera_t.yaml"
    if config_file.startswith("configs/"):
        config_file_rel = config_file[len("configs/"):]
    else:
        config_file_rel = config_file
    
    # 先构建标准的 SAM2VideoPredictor
    base_predictor = build_sam2_video_predictor(
        config_file=config_file_rel,
        ckpt_path=ckpt_path,
        device=device,
        mode="train",
    )
    
    # 提取组件
    model = ScribbleSam2Memory(
        image_encoder=base_predictor.image_encoder,
        memory_attention=base_predictor.memory_attention,
        memory_encoder=base_predictor.memory_encoder,
        scribble_channels=scribble_channels,
        image_size=base_predictor.image_size,
        # 继承其他必要参数
        num_maskmem=base_predictor.num_maskmem,
        sigmoid_scale_for_mem_enc=base_predictor.sigmoid_scale_for_mem_enc,
        sigmoid_bias_for_mem_enc=base_predictor.sigmoid_bias_for_mem_enc,
        use_high_res_features_in_sam=base_predictor.use_high_res_features_in_sam,
    )
    
    # 复制其他必要属性
    model.sam_prompt_encoder = base_predictor.sam_prompt_encoder
    model.sam_mask_decoder = base_predictor.sam_mask_decoder
    model.no_mem_embed = base_predictor.no_mem_embed
    model.no_mem_pos_enc = base_predictor.no_mem_pos_enc
    model.hidden_dim = base_predictor.hidden_dim
    model.mem_dim = base_predictor.mem_dim
    
    model = model.to(device)
    
    return model


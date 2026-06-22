"""
ScribbleEncoder - 统一的 Scribble 编码器

设计特点：
1. 结构与 SAM2 PromptEncoder.mask_downscaling 一致，可复用预训练权重
2. 支持空输入检测：当 scribble 全零时返回全零 embedding（避免 bias 噪声）
3. 支持从 SAM2 mask_downscaling 权重初始化

输入: (B, 2, H, W) - 正scribble + 负scribble
输出: (B, embed_dim, 64, 64) - 与 dense_embeddings 相同

Author: ScribblePrompt Team
"""

from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import LayerNorm2d


class ScribbleEncoder(nn.Module):
    """
    Scribble Prompt 编码器
    
    结构与 SAM2 PromptEncoder.mask_downscaling 一致，可复用预训练权重。
    
    处理流程:
    1. 检测 scribble 是否为空（全零）
    2. 如果为空，返回全零 embedding（避免 bias 噪声）
    3. 如果非空，通过下采样网络编码
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,  # 与 SAM2 一致
        in_channels: int = 2,     # 2通道：正/负 scribble
        activation: Type[nn.Module] = nn.GELU,
    ):
        """
        初始化 ScribbleEncoder

        Args:
            embed_dim: 输出 embedding 维度（256）
            image_embedding_size: 输出空间尺寸 (64, 64)
            input_image_size: 输入图像尺寸 (1024, 1024)
            mask_in_chans: 中间隐藏通道数（默认16，与SAM2一致）
            in_channels: 输入通道数（默认2：正/负scribble）
            activation: 激活函数
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.in_channels = in_channels
        self.mask_in_chans = mask_in_chans
        
        # 与 SAM2 mask_downscaling 一致的输入尺寸
        self.mask_input_size = (
            4 * image_embedding_size[0],  # 256
            4 * image_embedding_size[1],  # 256
        )
        
        # 下采样网络 - 结构与 SAM2 mask_downscaling 完全一致
        # 只是第一层输入通道从 1 改为 2
        self.scribble_downscaling = nn.Sequential(
            # 256 → 128
            nn.Conv2d(in_channels, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            # 128 → 64
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            # 64 → 64 (保持尺寸)
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        # 空输入检测阈值
        self.empty_threshold = 1e-6
        
        # 初始化权重
        self._init_weights()
        
        print(f"[ScribbleEncoder] Initialized: in_channels={in_channels}, embed_dim={embed_dim}")

    def _init_weights(self):
        """使用 trunc_normal 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_from_mask_encoder(self, mask_downscaling_state_dict: dict):
        """
        从 SAM2 的 mask_downscaling 权重初始化
        
        处理第一层通道不匹配：将 (4, 1, 2, 2) 扩展为 (4, 2, 2, 2)
        通过复制权重到两个输入通道（正/负 scribble）

        Args:
            mask_downscaling_state_dict: SAM2 PromptEncoder.mask_downscaling 的 state_dict
        """
        our_state = self.scribble_downscaling.state_dict()
        
        for name, param in mask_downscaling_state_dict.items():
            if name in our_state:
                if name == '0.weight':  # 第一层 Conv2d weight
                    # mask_downscaling: (out_ch, 1, k, k)
                    # scribble: (out_ch, 2, k, k)
                    # 策略：将权重复制到两个输入通道，并除以2保持数值稳定
                    expanded_weight = param.repeat(1, self.in_channels, 1, 1) / self.in_channels
                    our_state[name].copy_(expanded_weight)
                    print(f"  Expanded {name}: {param.shape} → {expanded_weight.shape}")
                elif our_state[name].shape == param.shape:
                    our_state[name].copy_(param)
                    print(f"  Copied {name}: {param.shape}")
                else:
                    print(f"  Skipped {name}: shape mismatch {param.shape} vs {our_state[name].shape}")
        
        self.scribble_downscaling.load_state_dict(our_state)
        print("[ScribbleEncoder] Initialized from mask_downscaling weights!")

    def _is_empty(self, scribble: torch.Tensor) -> torch.Tensor:
        """
        检测每个 batch 的 scribble 是否为空
        
        Args:
            scribble: (B, C, H, W)
            
        Returns:
            is_empty: (B,) bool tensor，True 表示该 batch 的 scribble 为空
        """
        # 计算每个 batch 的 scribble 总和
        scribble_sum = scribble.abs().sum(dim=(1, 2, 3))  # (B,)
        return scribble_sum < self.empty_threshold

    def forward(self, scribbles: Optional[torch.Tensor]) -> torch.Tensor:
        """
        编码 scribble 输入

        Args:
            scribbles: (B, 2, H, W) scribble mask，或 None
                       通道0: 正scribble（前景）
                       通道1: 负scribble（背景）

        Returns:
            (B, embed_dim, H', W') dense embedding
            - 如果 scribble 为 None 或全零，返回全零 embedding
            - 否则返回编码后的 embedding
        """
        # 处理 None 输入
        if scribbles is None:
            # 返回需要知道 batch size，这里假设 batch_size=1
            return torch.zeros(
                1, self.embed_dim, 
                self.image_embedding_size[0], self.image_embedding_size[1],
                device=next(self.parameters()).device
            )
        
        # 处理维度
        if scribbles.dim() == 3:
            scribbles = scribbles.unsqueeze(1)
        
        B = scribbles.shape[0]
        device = scribbles.device
        
        # 下采样到 256x256（与 SAM2 mask_input_size 一致）
        if scribbles.shape[-2:] != self.mask_input_size:
            scribbles = F.interpolate(
                scribbles,
                size=self.mask_input_size,
                mode='bilinear',
                align_corners=False
            )
        
        # 检测空输入
        is_empty = self._is_empty(scribbles)  # (B,)
        
        # 编码
        scribble_embedding = self.scribble_downscaling(scribbles)  # (B, embed_dim, 64, 64)
        
        # 对空输入的 batch，将 embedding 置零（避免 bias 噪声）
        if is_empty.any():
            # 创建 mask: (B, 1, 1, 1)
            zero_mask = is_empty.view(B, 1, 1, 1).float()
            # 空输入的 batch 置零
            scribble_embedding = scribble_embedding * (1 - zero_mask)
        
        return scribble_embedding


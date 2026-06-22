"""
LoRA (Low-Rank Adaptation) 模块

用于在不修改原始权重的情况下微调模型。
LoRA 公式: y = Wx + BAx * scaling
其中 W 是原始权重（冻结），B 和 A 是低秩矩阵（可训练）

Author: ScribblePrompt Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA 改造的 Linear 层
    
    原始: y = Wx + b
    LoRA: y = Wx + b + BA * x * scaling
    
    其中:
        W: (out_features, in_features) - 原始权重，冻结
        A: (rank, in_features) - 低秩矩阵，可训练
        B: (out_features, rank) - 低秩矩阵，可训练
        scaling: alpha / rank - 缩放因子
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 获取原始 Linear 的设备
        device = original_linear.weight.device
        
        # 冻结原始权重
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # LoRA 矩阵 - 在同一设备上创建
        self.lora_A = nn.Linear(self.in_features, rank, bias=False, device=device)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False, device=device)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        LoRA 初始化策略:
        - A: 正态分布初始化
        - B: 零初始化 (保证训练开始时 LoRA 输出为 0)
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_out = self.original_linear(x)
        
        # LoRA 增量
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self) -> nn.Linear:
        """
        将 LoRA 权重合并到原始权重中，返回普通 Linear
        用于推理时减少计算量
        """
        merged = nn.Linear(self.in_features, self.out_features, 
                          bias=self.original_linear.bias is not None)
        
        # W_merged = W + BA * scaling
        merged.weight.data = self.original_linear.weight.data + \
            (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        
        if self.original_linear.bias is not None:
            merged.bias.data = self.original_linear.bias.data
        
        return merged


def apply_lora_to_attention(
    attention_module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list = None,
) -> int:
    """
    对 Attention 模块应用 LoRA
    
    Args:
        attention_module: SAM2 的 Attention 模块
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: dropout rate
        target_modules: 要替换的模块名称列表，默认 ['q_proj', 'v_proj']
        
    Returns:
        替换的模块数量
    """
    if target_modules is None:
        # 默认只对 q_proj 和 v_proj 应用 LoRA
        # 这是 LoRA 论文中推荐的做法
        target_modules = ['q_proj', 'v_proj']
    
    count = 0
    for name in target_modules:
        if hasattr(attention_module, name):
            original_linear = getattr(attention_module, name)
            if isinstance(original_linear, nn.Linear):
                lora_linear = LoRALinear(
                    original_linear,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attention_module, name, lora_linear)
                count += 1
    
    return count


def apply_lora_to_mask_decoder(
    mask_decoder: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list = None,
) -> dict:
    """
    对 SAM2 的 Mask Decoder 应用 LoRA
    
    Args:
        mask_decoder: SAM2 的 MaskDecoder 模块
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: dropout rate
        target_modules: 要替换的模块名称列表
        
    Returns:
        统计信息字典
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']
    
    stats = {
        'total_lora_modules': 0,
        'total_lora_params': 0,
        'locations': []
    }
    
    # 获取 transformer
    transformer = mask_decoder.transformer
    
    # 1. 对 TwoWayTransformer 的每个 layer 应用 LoRA
    for layer_idx, layer in enumerate(transformer.layers):
        # self_attn
        count = apply_lora_to_attention(
            layer.self_attn, rank, alpha, dropout, target_modules
        )
        if count > 0:
            stats['total_lora_modules'] += count
            stats['locations'].append(f'layer_{layer_idx}.self_attn')
        
        # cross_attn_token_to_image
        count = apply_lora_to_attention(
            layer.cross_attn_token_to_image, rank, alpha, dropout, target_modules
        )
        if count > 0:
            stats['total_lora_modules'] += count
            stats['locations'].append(f'layer_{layer_idx}.cross_attn_token_to_image')
        
        # cross_attn_image_to_token
        count = apply_lora_to_attention(
            layer.cross_attn_image_to_token, rank, alpha, dropout, target_modules
        )
        if count > 0:
            stats['total_lora_modules'] += count
            stats['locations'].append(f'layer_{layer_idx}.cross_attn_image_to_token')
    
    # 2. 对 final_attn_token_to_image 应用 LoRA
    count = apply_lora_to_attention(
        transformer.final_attn_token_to_image, rank, alpha, dropout, target_modules
    )
    if count > 0:
        stats['total_lora_modules'] += count
        stats['locations'].append('final_attn_token_to_image')
    
    # 计算 LoRA 参数量
    for name, param in mask_decoder.named_parameters():
        if 'lora_' in name and param.requires_grad:
            stats['total_lora_params'] += param.numel()
    
    return stats


def get_lora_params(model: nn.Module) -> list:
    """
    获取模型中所有 LoRA 参数
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def print_lora_summary(model: nn.Module):
    """
    打印 LoRA 参数摘要
    """
    lora_params = 0
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if 'lora_' in name:
            lora_params += param.numel()
    
    print(f"\n{'='*60}")
    print("LoRA Summary")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"LoRA parameters:      {lora_params:,} ({100*lora_params/total_params:.4f}%)")
    print(f"{'='*60}\n")


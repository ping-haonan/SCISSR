"""
Memory Attention LoRA 模块

功能：
1. 为 Memory Attention 中的 Attention 层添加 LoRA
2. 支持选择性地为 Q, K, V, Out 添加 LoRA
3. Zero Init 保证训练初期不影响原有行为

目标层：
- MemoryAttentionLayer.self_attn (RoPEAttention)
- MemoryAttentionLayer.cross_attn_image (RoPEAttention) ← 最重要！
- MemoryAttentionLayer.linear1, linear2 (FFN)

Author: ScribblePrompt Team
"""

import torch
import torch.nn as nn
from typing import List, Optional, Set


class LoRALayer(nn.Module):
    """
    LoRA 层：低秩适应
    
    原理：y = Wx + BAx = (W + BA)x
    - W: 原始权重（冻结）
    - B: 初始化为 0（保证训练初期 ΔW = 0）
    - A: 随机初始化
    - scale = alpha / rank（控制 LoRA 影响强度）
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA 分解：W = BA, where B: (out, r), A: (r, in)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)  # B 初始化为 0 → 初始 ΔW = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 LoRA 的增量"""
        return self.scale * self.lora_B(self.lora_A(x))


class LoRALinear(nn.Module):
    """
    带 LoRA 的 Linear 层
    
    包装原始 Linear 层，添加 LoRA 分支
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        
        self.original = original_linear
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
        )
        
        # 将 LoRA 层移动到与原始 Linear 相同的设备
        device = original_linear.weight.device
        self.lora = self.lora.to(device)
        
        # 冻结原始权重
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA 增量
        return self.original(x) + self.lora(x)
    
    @property
    def weight(self):
        """兼容性：返回原始权重"""
        return self.original.weight
    
    @property
    def bias(self):
        """兼容性：返回原始 bias"""
        return self.original.bias


def apply_lora_to_attention(
    attention_module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ['q_proj', 'v_proj'],
) -> Set[str]:
    """
    为 Attention 模块添加 LoRA
    
    Args:
        attention_module: RoPEAttention 或 Attention 模块
        rank: LoRA 秩
        alpha: LoRA alpha（scale = alpha / rank）
        target_modules: 要添加 LoRA 的投影层名称
                       可选: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    
    Returns:
        modified_names: 被修改的层名称集合
    """
    modified_names = set()
    
    for name in target_modules:
        if hasattr(attention_module, name):
            original_linear = getattr(attention_module, name)
            if isinstance(original_linear, nn.Linear):
                lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha)
                setattr(attention_module, name, lora_linear)
                modified_names.add(name)
    
    return modified_names


def apply_lora_to_memory_attention(
    memory_attention: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ['q_proj', 'v_proj'],
    apply_to_self_attn: bool = True,
    apply_to_cross_attn: bool = True,
    apply_to_ffn: bool = False,
) -> dict:
    """
    为 MemoryAttention 模块添加 LoRA
    
    Args:
        memory_attention: MemoryAttention 模块
        rank: LoRA 秩
        alpha: LoRA alpha
        target_modules: Attention 层中要添加 LoRA 的投影
        apply_to_self_attn: 是否对 self_attn 添加 LoRA
        apply_to_cross_attn: 是否对 cross_attn_image 添加 LoRA（推荐！）
        apply_to_ffn: 是否对 FFN 的 linear1/linear2 添加 LoRA
    
    Returns:
        info: 修改信息字典
    """
    info = {
        'num_layers': 0,
        'self_attn_modified': [],
        'cross_attn_modified': [],
        'ffn_modified': [],
        'total_lora_params': 0,
    }
    
    # 遍历 MemoryAttention 的所有层
    for layer_idx, layer in enumerate(memory_attention.layers):
        info['num_layers'] += 1
        
        # 1. Self-Attention LoRA
        if apply_to_self_attn and hasattr(layer, 'self_attn'):
            modified = apply_lora_to_attention(
                layer.self_attn, 
                rank=rank, 
                alpha=alpha,
                target_modules=target_modules,
            )
            if modified:
                info['self_attn_modified'].append((layer_idx, modified))
        
        # 2. Cross-Attention LoRA（最重要！）
        if apply_to_cross_attn and hasattr(layer, 'cross_attn_image'):
            modified = apply_lora_to_attention(
                layer.cross_attn_image,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
            )
            if modified:
                info['cross_attn_modified'].append((layer_idx, modified))
        
        # 3. FFN LoRA
        if apply_to_ffn:
            if hasattr(layer, 'linear1') and isinstance(layer.linear1, nn.Linear):
                layer.linear1 = LoRALinear(layer.linear1, rank=rank, alpha=alpha)
                info['ffn_modified'].append((layer_idx, 'linear1'))
            if hasattr(layer, 'linear2') and isinstance(layer.linear2, nn.Linear):
                layer.linear2 = LoRALinear(layer.linear2, rank=rank, alpha=alpha)
                info['ffn_modified'].append((layer_idx, 'linear2'))
    
    # 统计 LoRA 参数量
    total_params = 0
    for name, param in memory_attention.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            total_params += param.numel()
    info['total_lora_params'] = total_params
    
    return info


def get_memory_attention_lora_params(memory_attention: nn.Module) -> List[nn.Parameter]:
    """
    获取 Memory Attention 中所有 LoRA 参数
    
    用于单独设置优化器参数组
    """
    lora_params = []
    for name, param in memory_attention.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params.append(param)
    return lora_params


def freeze_memory_attention_original_params(memory_attention: nn.Module):
    """
    冻结 Memory Attention 的原始参数，只保留 LoRA 可训练
    """
    for name, param in memory_attention.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False


def print_memory_attention_trainable_params(memory_attention: nn.Module):
    """
    打印 Memory Attention 的可训练参数信息
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in memory_attention.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
    
    print(f"[Memory Attention] Total params: {total_params:,}")
    print(f"[Memory Attention] Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"[Memory Attention] LoRA params: {lora_params:,}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.sam.transformer import RoPEAttention
    
    print("=" * 60)
    print("测试 Memory Attention LoRA")
    print("=" * 60)
    
    # 创建一个 MemoryAttentionLayer
    layer = MemoryAttentionLayer(
        activation='relu',
        cross_attention=RoPEAttention(
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=64,
            rope_k_repeat=True,
        ),
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=RoPEAttention(
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
        ),
    )
    
    # 创建 MemoryAttention
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=4,
    )
    
    print(f"\n原始 Memory Attention:")
    print(f"  Layers: {len(memory_attention.layers)}")
    print(f"  cross_attn_image.q_proj type: {type(memory_attention.layers[0].cross_attn_image.q_proj)}")
    
    # 应用 LoRA
    print(f"\n应用 LoRA (rank=8, alpha=16)...")
    info = apply_lora_to_memory_attention(
        memory_attention,
        rank=8,
        alpha=16.0,
        target_modules=['q_proj', 'v_proj'],
        apply_to_self_attn=True,
        apply_to_cross_attn=True,
        apply_to_ffn=False,
    )
    
    print(f"\nLoRA 应用结果:")
    print(f"  处理的层数: {info['num_layers']}")
    print(f"  Self-Attn 修改: {info['self_attn_modified']}")
    print(f"  Cross-Attn 修改: {info['cross_attn_modified']}")
    print(f"  LoRA 总参数量: {info['total_lora_params']:,}")
    
    print(f"\n应用后:")
    print(f"  cross_attn_image.q_proj type: {type(memory_attention.layers[0].cross_attn_image.q_proj)}")
    
    # 冻结原始参数
    freeze_memory_attention_original_params(memory_attention)
    
    print(f"\n参数统计:")
    print_memory_attention_trainable_params(memory_attention)
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    B, HW, C = 2, 4096, 256
    mem_dim = 64  # Memory 的维度（kv_in_dim）
    
    curr = torch.randn(HW, B, C)
    memory = torch.randn(100, B, mem_dim)  # Memory tokens（维度要与 kv_in_dim 匹配）
    curr_pos = torch.randn(HW, B, C)
    memory_pos = torch.randn(100, B, mem_dim)
    
    output = None
    try:
        output = memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
        )
        print(f"  输入 curr shape: {curr.shape}")
        print(f"  输入 memory shape: {memory.shape}")
        print(f"  输出 shape: {output.shape}")
        print(f"  ✅ 前向传播成功！")
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试梯度
    if output is not None:
        print(f"\n测试梯度回传...")
        loss = output.sum()
        loss.backward()
        
        lora_grads = []
        for name, param in memory_attention.named_parameters():
            if 'lora' in name.lower() and param.grad is not None:
                lora_grads.append((name, param.grad.abs().mean().item()))
        
        print(f"  LoRA 参数梯度 (前5个):")
        for name, grad in lora_grads[:5]:
            print(f"    {name}: {grad:.6f}")
        print(f"  ✅ 梯度回传成功！")
    else:
        print(f"\n跳过梯度测试（前向传播失败）")


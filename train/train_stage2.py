"""
Stage 2 渐进式训练脚本

核心设计：
1. 使用 Memory-Driven 迭代推理（ScribbleSam2Memory）
2. 渐进式训练：从 100% box 逐渐过渡到纯 scribble
3. 只用最新的 scribble，不累积历史（Memory 负责记忆）
4. AdaptiveScribble + 扰动增强鲁棒性
5. 加载 Stage 1 权重：ScribbleEncoder + MaskDecoder LoRA

训练阶段：
- Phase 1 (epoch 1-2): 100% noisy box (R0), 修正 scribble (R1,R2)
- Phase 2 (epoch 3-6): box 概率 80%→60%→40%→20%
- Phase 3 (epoch 7-10): 纯 scribble

Author: ScribblePrompt Team
"""

import os
import sys
import functools
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
import albumentations as A
import cv2

# 确保 print 立即输出
print = functools.partial(print, flush=True)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化 Hydra（SAM2 需要）
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir=os.path.join(os.getcwd(), "sam2", "configs"),
    version_base="1.2",
)

from scissr.models.ScribbleSam2Memory import (
    ScribbleSam2Memory,
    build_scribble_sam2_memory,
)
from scissr.interactions.scribbles import LineScribble
from scissr.interactions.adaptive_scribble import AdaptiveScribble


GLOBAL_SEED = 42

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Scribble 生成器
# =============================================================================

class TwoChannelScribbleGenerator:
    """
    双通道 scribble 生成器（混合策略，与 Stage 1 / 模型 correction 对齐）
    
    正 scribble（前景）→ AdaptiveScribble：逐连通域几何分析 + 自适应策略
    负 scribble（背景）→ LineScribble：轻量快速
    """
    
    def __init__(self, neg_prob: float = 0.5):
        self.pos_generator = AdaptiveScribble()
        self.neg_generator = LineScribble(thickness=3, warp=False)
        self.neg_prob = neg_prob
    
    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (B, 1, H, W) GT 前景 mask
            n_scribbles: 正 scribble 数量
        Returns:
            (B, 2, H, W) 双通道 scribble [正, 负]
        """
        B, _, H, W = mask.shape
        
        # 正 scribble: AdaptiveScribble
        try:
            pos_scribble = self.pos_generator(mask, n_scribbles=n_scribbles)
        except:
            pos_scribble = torch.zeros_like(mask)
        
        # 负 scribble: LineScribble（在背景区域）
        neg_scribble = torch.zeros_like(mask)
        if self.neg_prob > 0 and random.random() < self.neg_prob:
            bg_mask = 1.0 - mask
            if bg_mask.sum() > 100:
                try:
                    neg_scribble = self.neg_generator(bg_mask, n_scribbles=1)
                except:
                    neg_scribble = torch.zeros_like(mask)
        
        return torch.cat([pos_scribble, neg_scribble], dim=1)


class CorrectionScribbleGenerator:
    """
    混合策略修正 Scribble 生成器
    
    FN（漏检）→ AdaptiveScribble：沿结构骨架精准引导
    FP（误检）→ LineScribble：轻量快速，模拟"随手划掉"
    """
    
    def __init__(self):
        from scissr.interactions.adaptive_scribble import CorrectionConfig
        self.fn_generator = AdaptiveScribble(config=CorrectionConfig())
        self.fp_generator = LineScribble(thickness=3, warp=False)
    
    def generate(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        is_final_round: bool = False,
    ) -> tuple:
        """
        生成正/负修正 scribble
        
        Returns:
            pos_scribble: (B, 1, H, W) FN 区域的正 scribble
            neg_scribble: (B, 1, H, W) FP 区域的负 scribble
        """
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
        gt_binary = (gt_mask > threshold).float()
        
        fn_region = gt_binary * (1 - pred_binary)
        fp_region = (1 - gt_binary) * pred_binary
        
        # FN → AdaptiveScribble
        pos_scribble = torch.zeros_like(pred_mask)
        if fn_region.sum() > 1:
            try:
                pos_scribble = self.fn_generator(fn_region, n_scribbles=1)
            except:
                pass
        
        # FP → LineScribble
        neg_scribble = torch.zeros_like(pred_mask)
        if fp_region.sum() > 1:
            try:
                neg_scribble = self.fp_generator(fp_region, n_scribbles=1)
            except:
                pass
        
        return pos_scribble, neg_scribble


# =============================================================================
# Noisy Box 生成器
# =============================================================================

class NoisyBoxGenerator:
    """生成带噪声的边界框"""
    
    def __init__(
        self,
        jitter_ratio: float = 0.1,
        scale_range: tuple = (0.8, 1.2),
    ):
        self.jitter_ratio = jitter_ratio
        self.scale_range = scale_range
    
    def generate(self, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        从 GT mask 生成带噪声的边界框
        
        Args:
            gt_mask: (B, 1, H, W) 或 (B, H, W)
            
        Returns:
            boxes: (B, 4) 格式为 [x1, y1, x2, y2]，归一化到 [0, 1]
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        
        B, _, H, W = gt_mask.shape
        device = gt_mask.device
        
        boxes = []
        for b in range(B):
            mask = gt_mask[b, 0]
            
            if mask.sum() > 0:
                rows = torch.any(mask > 0.5, dim=1)
                cols = torch.any(mask > 0.5, dim=0)
                
                y_indices = torch.where(rows)[0]
                x_indices = torch.where(cols)[0]
                
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y1, y2 = y_indices[0].item(), y_indices[-1].item()
                    x1, x2 = x_indices[0].item(), x_indices[-1].item()
                else:
                    x1, y1 = W // 4, H // 4
                    x2, y2 = 3 * W // 4, 3 * H // 4
            else:
                x1, y1 = W // 4, H // 4
                x2, y2 = 3 * W // 4, 3 * H // 4
            
            # 添加噪声
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_w = x2 - x1
            box_h = y2 - y1
            
            # 位置抖动
            jitter_x = random.uniform(-self.jitter_ratio, self.jitter_ratio) * box_w
            jitter_y = random.uniform(-self.jitter_ratio, self.jitter_ratio) * box_h
            cx += jitter_x
            cy += jitter_y
            
            # 尺寸缩放
            scale = random.uniform(*self.scale_range)
            box_w *= scale
            box_h *= scale
            
            # 计算新边界框
            new_x1 = max(0, cx - box_w / 2)
            new_y1 = max(0, cy - box_h / 2)
            new_x2 = min(W, cx + box_w / 2)
            new_y2 = min(H, cy + box_h / 2)
            
            # 归一化
            boxes.append([new_x1 / W, new_y1 / H, new_x2 / W, new_y2 / H])
        
        return torch.tensor(boxes, device=device, dtype=torch.float32)


# =============================================================================
# 数据集
# =============================================================================

# RGB 到类别的映射（全局定义）
COLOR_TO_CLASS = {
    (0, 0, 0): 0, (0, 255, 0): 1, (0, 255, 255): 2, (125, 255, 12): 3,
    (255, 55, 0): 4, (24, 55, 125): 5, (187, 155, 25): 6, (0, 255, 125): 7,
    (255, 255, 125): 8, (123, 15, 175): 9, (124, 155, 5): 10, (12, 255, 141): 11,
}

ENDOVISION18_CLASSES = {
    0: 'background-tissue', 1: 'instrument-shaft', 2: 'instrument-clasper',
    3: 'instrument-wrist', 4: 'kidney-parenchyma', 5: 'covered-kidney',
    6: 'thread', 7: 'clamps', 8: 'suturing-needle', 9: 'suction-instrument',
    10: 'small-intestine', 11: 'ultrasound-probe',
}


def load_train_val_split(split_json_path: str = None):
    """加载训练/验证集划分文件"""
    if split_json_path is None:
        split_json_path = 'dataset/Endovision18/train_val_split.json'
    
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    
    return set(split_data['train_sequences']), set(split_data['val_sequences'])


def get_train_augmentation():
    """
    训练数据增强管线（与 Stage 1 完全一致，参考 ScribblePrompt 原文 Table 4）
    应用于 image + mask（scribble 在增强后生成，保证一致性）
    """
    return A.Compose([
        # --- 几何增强 ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.25,
        ),
        # --- 光照增强 ---
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])


class Endovision18Dataset(Dataset):
    """Endovision18 数据集 - 训练用（每帧随机一个类别）"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        split: str = 'train',
        split_json_path: str = None,
        transform: A.Compose = None,  # 数据增强管线
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        # 从 JSON 文件加载训练/验证集划分
        train_seqs, val_seqs = load_train_val_split(split_json_path)
        target_seqs = train_seqs if split == 'train' else val_seqs
        
        # 收集样本
        self.samples = []
        for seq_dir in sorted(self.data_dir.glob('seq_*')):
            if not seq_dir.is_dir():
                continue
            if seq_dir.name not in target_seqs:
                continue
            
            images_dir = seq_dir / 'left_frames'
            labels_dir = seq_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            for img_path in sorted(images_dir.glob('*.png')):
                frame_name = img_path.stem
                label_path = labels_dir / f"{frame_name}.png"
                
                if label_path.exists():
                    self.samples.append({
                        'image': str(img_path),
                        'label': str(label_path),
                        'seq': seq_dir.name,
                    })
        
        print(f"[{split.upper()} Dataset] Sequences: {len(target_seqs)}, Samples: {len(self.samples)}")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image)
        
        # 加载标签 - 必须用 RGB 模式！
        label_rgb = Image.open(sample['label']).convert('RGB')
        label_rgb = label_rgb.resize((self.image_size, self.image_size), Image.NEAREST)
        label_rgb = np.array(label_rgb)
        
        # 找出标签中存在的所有前景类别
        foreground_classes = []
        for color, class_id in COLOR_TO_CLASS.items():
            if class_id == 0:
                continue
            color_match = np.all(label_rgb == color, axis=2)
            if color_match.sum() > 0:
                foreground_classes.append((class_id, color))
        
        if len(foreground_classes) == 0:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            chosen_class, chosen_color = random.choice(foreground_classes)
            mask = np.all(label_rgb == chosen_color, axis=2).astype(np.float32)
        
        # --- 数据增强（增强后再生成 scribble，保证一致性）---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为 tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'path': sample['image'],
        }


class Endovision18ValDataset(Dataset):
    """验证数据集 - 全类别验证（每帧的每个类别都作为一个样本）"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        split_json_path: str = None,  # 使用预定义的 split 文件
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # 从 JSON 文件加载验证集划分
        _, val_seqs = load_train_val_split(split_json_path)
        
        # 收集验证集帧
        val_frames = []
        for seq_dir in sorted(self.data_dir.glob('seq_*')):
            if not seq_dir.is_dir():
                continue
            if seq_dir.name not in val_seqs:
                continue
            
            images_dir = seq_dir / 'left_frames'
            labels_dir = seq_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            for img_path in sorted(images_dir.glob('*.png')):
                frame_name = img_path.stem
                label_path = labels_dir / f"{frame_name}.png"
                
                if label_path.exists():
                    val_frames.append({
                        'image': str(img_path),
                        'label': str(label_path),
                        'seq': seq_dir.name,
                    })
        
        # 展开为 (帧, 类别) 对
        self.samples = []
        for frame in val_frames:
            label_rgb = np.array(Image.open(frame['label']).convert('RGB'))
            
            for color, class_id in COLOR_TO_CLASS.items():
                if class_id == 0:
                    continue
                color_match = np.all(label_rgb == color, axis=2)
                if color_match.sum() > 10:  # 与 Stage 1 对齐：阈值 > 10（仅排除标注噪声）
                    self.samples.append({
                        'image': frame['image'],
                        'label': frame['label'],
                        'seq': frame['seq'],
                        'class_id': class_id,
                        'class_color': color,
                        'class_name': ENDOVISION18_CLASSES[class_id],
                    })
        
        # 统计类别分布
        class_counts = {}
        for s in self.samples:
            name = s['class_name']
            class_counts[name] = class_counts.get(name, 0) + 1
        
        print(f"[VAL Dataset - All Classes] Total samples: {len(self.samples)}")
        print(f"  - Unique frames: {len(val_frames)}")
        print(f"  - Classes: {len(class_counts)}")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image)
        
        # 加载标签
        label_rgb = Image.open(sample['label']).convert('RGB')
        label_rgb = label_rgb.resize((self.image_size, self.image_size), Image.NEAREST)
        label_rgb = np.array(label_rgb)
        
        # 提取指定类别的 mask
        class_color = sample['class_color']
        mask = np.all(label_rgb == class_color, axis=2).astype(np.float32)
        
        # 转换为 tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'path': sample['image'],
            'class_id': sample['class_id'],
            'class_name': sample['class_name'],
        }


# =============================================================================
# 损失函数
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation"""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class IterativeLoss(nn.Module):
    """Focal Loss + Dice Loss（与 Stage 1 对齐）"""
    
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:], mode='nearest')
        
        focal = self.focal(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return (intersection / (union + 1e-6)).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    
    return (2 * intersection / (total + 1e-6)).item()


# =============================================================================
# 训练循环
# =============================================================================

def get_box_probability(epoch: int, total_epochs: int = 10) -> float:
    """
    获取当前 epoch 的 box 使用概率
    
    全程 0%：直接使用 scribble，不用 box。
    理由：
    1. Stage 1 已预训练好 ScribbleEncoder + MaskDecoder LoRA，scribble 路径 day 1 即可工作
    2. SpatialGatedFusion alpha=0 初始化保证 Track 2 不会破坏模型
    3. 10 个 epoch 有限，全部用于训练 scribble→memory→correction 完整管线
    """
    return 0.0


def forward_iterative_progressive(
    model: ScribbleSam2Memory,
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    scribble_gen: TwoChannelScribbleGenerator,
    correction_gen: CorrectionScribbleGenerator,
    box_gen: NoisyBoxGenerator,
    use_box_prob: float,
    num_rounds: int = 3,
    disable_memory: bool = False,
) -> dict:
    """
    双轨渐进式迭代推理
    
    Track 1（accumulated → dense_embeddings）：保留完整用户意图
    Track 2（latest → QueryFusion → Memory Query）：聚焦本轮修正
    
    Args:
        model: ScribbleSam2Memory 模型
        image: (B, 3, H, W) 输入图像
        gt_mask: (B, 1, H, W) GT mask
        scribble_gen: 初始 scribble 生成器
        correction_gen: 修正 scribble 生成器
        box_gen: noisy box 生成器
        use_box_prob: 使用 box 的概率
        num_rounds: 迭代轮数
        disable_memory: 消融实验 - 禁用 Memory（所有轮次 use_memory=False）
        
    Returns:
        outputs: Dict 包含每轮的输出
    """
    B = image.shape[0]
    device = image.device
    
    # 重置 Memory Bank
    model.reset_memory_bank()
    
    outputs = {}
    
    # 图像编码（只做一次）
    backbone_out = model.forward_image(image)
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    
    cache = {
        'backbone_out': backbone_out,
        'vision_feats': vision_feats,
        'vision_pos_embeds': vision_pos_embeds,
        'feat_sizes': feat_sizes,
    }
    
    # 双轨 scribble 状态
    accumulated_scribble = None  # Track 1: 累积所有历史
    latest_scribble = None       # Track 2: 只有本轮最新
    
    for round_idx in range(num_rounds):
        if round_idx == 0:
            # ============ Round 0 ============
            use_box = random.random() < use_box_prob
            
            if use_box:
                noisy_box = box_gen.generate(gt_mask)
                # Box 模式：两轨都为 None（R0 由 Box 提供定位）
                accumulated_scribble = None
                latest_scribble = None
                outputs[f'box_{round_idx}'] = noisy_box
                outputs[f'use_box_{round_idx}'] = True
            else:
                # Scribble 模式：生成初始 scribble
                init_scribble = scribble_gen(gt_mask, n_scribbles=1).to(device)
                accumulated_scribble = init_scribble
                latest_scribble = init_scribble
                noisy_box = None
                outputs[f'use_box_{round_idx}'] = False
            
            # R0: 不使用 Memory，但存入 Memory（disable_memory 时也不存）
            low_res, high_res, cache = model.forward_single_round(
                image=image,
                latest_scribble=latest_scribble,
                accumulated_scribble=accumulated_scribble,
                box=noisy_box,
                use_memory=False,
                update_memory=(not disable_memory),
                **cache,
            )
        else:
            # ============ Round 1+: Dual-Track Correction ============
            with torch.no_grad():
                prev_high_res = outputs[f'masks_{round_idx-1}']
                is_final_round = (round_idx == num_rounds - 1)
                pos_correction, neg_correction = correction_gen.generate(
                    pred_mask=prev_high_res,
                    gt_mask=gt_mask,
                    is_final_round=is_final_round,
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
            
            # 修正推理（disable_memory 时所有轮次都不用 Memory）
            use_mem = (not disable_memory)
            update_mem = (not disable_memory) and (round_idx < num_rounds - 1)
            low_res, high_res, cache = model.forward_single_round(
                image=image,
                latest_scribble=latest_scribble,
                accumulated_scribble=accumulated_scribble,
                box=None,
                use_memory=use_mem,
                update_memory=update_mem,
                **cache,
            )
        
        # 保存输出
        outputs[f'low_res_masks_{round_idx}'] = low_res
        outputs[f'masks_{round_idx}'] = high_res
        outputs[f'accumulated_scribble_{round_idx}'] = accumulated_scribble.clone() if accumulated_scribble is not None else None
        outputs[f'latest_scribble_{round_idx}'] = latest_scribble.clone() if latest_scribble is not None else None
    
    return outputs


def train_one_epoch(
    model: ScribbleSam2Memory,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    scribble_gen: TwoChannelScribbleGenerator,
    correction_gen: CorrectionScribbleGenerator,
    box_gen: NoisyBoxGenerator,
    num_rounds: int = 3,
    total_epochs: int = 10,
    scaler: GradScaler = None,
    disable_memory: bool = False,
):
    """训练一个 epoch（支持 AMP + 消融）"""
    model.train()
    use_amp = scaler is not None
    
    box_prob = get_box_probability(epoch, total_epochs)
    
    iter_losses = {i: [] for i in range(num_rounds)}
    iter_ious = {i: [] for i in range(num_rounds)}
    box_count = 0
    scribble_count = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch} (box_prob={box_prob:.0%})")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        if masks.sum() < 10 * masks.shape[0]:
            continue
        
        optimizer.zero_grad()
        
        try:
            with autocast('cuda', enabled=use_amp):
                outputs = forward_iterative_progressive(
                    model=model,
                    image=images,
                    gt_mask=masks,
                    scribble_gen=scribble_gen,
                    correction_gen=correction_gen,
                    box_gen=box_gen,
                    use_box_prob=box_prob,
                    num_rounds=num_rounds,
                    disable_memory=disable_memory,
                )
                
                # 统计 box 使用
                if outputs.get('use_box_0', False):
                    box_count += 1
                else:
                    scribble_count += 1
                
                # 计算 loss（递增权重：后面迭代更重要）
                total_loss = 0
                weights = [(i + 1) for i in range(num_rounds)]
                weight_sum = sum(weights)
                
                for i in range(num_rounds):
                    iter_loss = loss_fn(outputs[f'masks_{i}'], masks)
                    iter_losses[i].append(iter_loss.item())
                    
                    iou = compute_iou(outputs[f'masks_{i}'], masks)
                    iter_ious[i].append(iou)
                    
                    total_loss = total_loss + (weights[i] / weight_sum) * iter_loss
            
            # AMP 反向传播
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        except Exception as e:
            print(f"[Warning] Batch {batch_idx} error: {e}")
            continue
        
        # 更新进度条
        loss_str = ', '.join([f'L{i}:{np.mean(iter_losses[i][-20:]):.4f}' for i in range(num_rounds)])
        iou_str = ', '.join([f'I{i}:{np.mean(iter_ious[i][-20:]):.3f}' for i in range(num_rounds)])
        pbar.set_postfix_str(f'{loss_str} | {iou_str}')
    
    print(f"  R0 使用: Box={box_count}, Scribble={scribble_count}")
    
    return {
        'losses': {i: np.mean(iter_losses[i]) if iter_losses[i] else 0 for i in range(num_rounds)},
        'ious': {i: np.mean(iter_ious[i]) if iter_ious[i] else 0 for i in range(num_rounds)},
        'box_ratio': box_count / max(1, box_count + scribble_count),
    }


@torch.no_grad()
def validate(
    model: ScribbleSam2Memory,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    scribble_gen: TwoChannelScribbleGenerator,
    correction_gen: CorrectionScribbleGenerator,
    num_rounds: int = 3,
    use_amp: bool = False,
    disable_memory: bool = False,
):
    """验证 - 只用 scribble（不用 box），支持 AMP + 消融"""
    model.eval()
    
    iter_losses = {i: [] for i in range(num_rounds)}
    iter_ious = {i: [] for i in range(num_rounds)}
    iter_dices = {i: [] for i in range(num_rounds)}
    
    dummy_box_gen = NoisyBoxGenerator()
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        if masks.sum() < 10 * masks.shape[0]:
            continue
        
        try:
            with autocast('cuda', enabled=use_amp):
                outputs = forward_iterative_progressive(
                    model=model,
                    image=images,
                    gt_mask=masks,
                    scribble_gen=scribble_gen,
                    correction_gen=correction_gen,
                    box_gen=dummy_box_gen,
                    use_box_prob=0.0,
                    num_rounds=num_rounds,
                    disable_memory=disable_memory,
                )
        except Exception:
            continue
        
        for i in range(num_rounds):
            iter_loss = loss_fn(outputs[f'masks_{i}'], masks)
            iter_losses[i].append(iter_loss.item())
            
            iou = compute_iou(outputs[f'masks_{i}'], masks)
            dice = compute_dice(outputs[f'masks_{i}'], masks)
            iter_ious[i].append(iou)
            iter_dices[i].append(dice)
        
        pbar.set_postfix_str(f"mIoU_R2: {np.mean(iter_ious[num_rounds-1]):.3f}")
    
    return {
        'losses': {i: np.mean(iter_losses[i]) if iter_losses[i] else 0 for i in range(num_rounds)},
        'ious': {i: np.mean(iter_ious[i]) if iter_ious[i] else 0 for i in range(num_rounds)},
        'dices': {i: np.mean(iter_dices[i]) if iter_dices[i] else 0 for i in range(num_rounds)},
    }


# =============================================================================
# 模型构建
# =============================================================================

def build_model(args) -> ScribbleSam2Memory:
    """构建并初始化模型（支持消融实验配置）"""
    
    disable_memory = getattr(args, 'disable_memory', False)
    disable_sgf = getattr(args, 'disable_sgf', False)
    
    # 打印消融配置
    if disable_memory or disable_sgf:
        ablation_parts = []
        if disable_memory:
            ablation_parts.append("Memory DISABLED")
        if disable_sgf:
            ablation_parts.append("SGF DISABLED")
        print(f"\n[Ablation] {', '.join(ablation_parts)}")
    
    print("\n[Build Model] Creating ScribbleSam2Memory...")
    
    # 1. 构建基础模型
    model = build_scribble_sam2_memory(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device='cuda',
        scribble_channels=2,
    )
    
    # 2. 启用 Mask Decoder LoRA（所有配置都需要）
    print("\n[Build Model] Enabling Mask Decoder LoRA...")
    model.enable_mask_decoder_lora(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    
    # 3. 启用 Memory Attention LoRA（仅在不禁用 Memory 时）
    if not disable_memory:
        print("\n[Build Model] Enabling Memory Attention LoRA...")
        model.enable_memory_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
    else:
        print("\n[Build Model] Memory LoRA SKIPPED (--disable_memory)")
    
    # 4. 加载 Stage 1 权重
    print(f"\n[Build Model] Loading Stage 1 weights from: {args.stage1_ckpt}")
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location='cpu', weights_only=False)
    
    print(f"  Stage 1 info: epoch={stage1_ckpt.get('epoch', 'N/A')}, best_val_iou={stage1_ckpt.get('best_val_iou', 'N/A'):.4f}")
    
    # 加载 ScribbleEncoder 和 MaskDecoder LoRA 权重
    model_state = model.state_dict()
    loaded_count = 0
    loaded_names = []
    
    for name, param in stage1_ckpt['model_state_dict'].items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded_count += 1
            loaded_names.append(name)
    
    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {loaded_count} parameters from Stage 1")
    
    # 打印加载的参数类别
    scribble_loaded = sum(1 for n in loaded_names if 'scribble_encoder' in n)
    mask_decoder_loaded = sum(1 for n in loaded_names if 'sam_mask_decoder' in n)
    print(f"    - ScribbleEncoder: {scribble_loaded}")
    print(f"    - MaskDecoder LoRA: {mask_decoder_loaded}")
    
    # 5. 冻结预训练权重
    model.freeze_pretrained()
    
    # 6. 消融：禁用 SpatialGatedFusion（冻结 alpha=0 + 冻结所有 query_fusion 参数）
    if disable_sgf:
        print("\n[Ablation] Freezing SpatialGatedFusion (alpha=0, all params frozen)...")
        with torch.no_grad():
            model.query_fusion.alpha.fill_(0.0)
        for param in model.query_fusion.parameters():
            param.requires_grad = False
        print(f"  query_fusion alpha = {model.query_fusion.alpha.item():.4f} (frozen)")
    
    return model


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2 Progressive Training')
    
    # 模型配置
    parser.add_argument('--config_file', type=str,
                        default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--ckpt_path', type=str,
                        default='checkpoints/sam2.1_hiera_tiny.pt')
    parser.add_argument('--stage1_ckpt', type=str,
                        default='trained_models/lora_comparison/20260207_070833_latest/with_lora/best_model.pt')
    
    # LoRA 配置
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='Learning rate for new modules (query_fusion, memory_lora)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                        help='Learning rate for Stage 1 modules (scribble_encoder, mask_decoder_lora)')
    parser.add_argument('--num_rounds', type=int, default=3)
    
    # 数据配置
    parser.add_argument('--data_dir', type=str,
                        default='dataset/Endovision18/raw/Train_Data')
    
    # 保存配置
    parser.add_argument('--save_dir', type=str, default='trained_models/stage2_progressive')
    
    # 消融实验配置
    parser.add_argument('--disable_memory', action='store_true',
                        help='Ablation: 禁用 Memory 机制（所有轮次 use_memory=False）')
    parser.add_argument('--disable_sgf', action='store_true',
                        help='Ablation: 禁用 SpatialGatedFusion（alpha 固定为 0）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 确定消融实验名称（用于保存目录）
    if args.disable_memory and args.disable_sgf:
        ablation_name = "ablation_baseline"
    elif args.disable_memory:
        ablation_name = "ablation_sgf_only"
    elif args.disable_sgf:
        ablation_name = "ablation_memory_only"
    else:
        ablation_name = None
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if ablation_name:
        save_dir = os.path.join(args.save_dir, f"{ablation_name}_{timestamp}")
    else:
        save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 70)
    if ablation_name:
        print(f"Stage 2 Ablation: {ablation_name}")
    else:
        print("Stage 2 Progressive Training")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Base LR (new modules): {args.base_lr}")
    print(f"Finetune LR (Stage 1 modules): {args.finetune_lr}")
    print(f"Iterations: {args.num_rounds}")
    print(f"Stage 1 checkpoint: {args.stage1_ckpt}")
    print(f"Memory: {'DISABLED' if args.disable_memory else 'ENABLED'}")
    print(f"SGF: {'DISABLED' if args.disable_sgf else 'ENABLED'}")
    print(f"Save dir: {save_dir}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    model = build_model(args)
    
    # 创建数据集（使用预定义的序列划分）
    print("\n[Data] Loading datasets...")
    train_transform = get_train_augmentation()
    train_dataset = Endovision18Dataset(
        data_dir=args.data_dir,
        image_size=1024,
        split='train',
        transform=train_transform,
    )
    
    # 使用全类别验证数据集
    val_dataset = Endovision18ValDataset(
        data_dir=args.data_dir,
        image_size=1024,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 优化器
    param_groups = model.get_trainable_param_groups(
        base_lr=args.base_lr,
        scribble_lr=args.finetune_lr,
        mask_decoder_lora_lr=args.finetune_lr,
    )
    
    print("\n[Optimizer] Parameter groups:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  - {pg['name']}: {n_params:,} params, lr={pg['lr']}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.finetune_lr * 0.1,
    )
    
    # 损失函数（Focal + Dice，与 Stage 1 对齐）
    loss_fn = IterativeLoss()
    
    # Scribble 生成器（与 Stage 1 对齐）
    scribble_gen = TwoChannelScribbleGenerator(neg_prob=0.5)
    correction_gen = CorrectionScribbleGenerator()
    box_gen = NoisyBoxGenerator(
        jitter_ratio=0.1,
        scale_range=(0.8, 1.2),
    )
    
    # 训练历史
    history = {
        'train_losses': [],
        'train_ious': [],
        'val_losses': [],
        'val_ious': [],
        'val_dices': [],
        'box_ratios': [],
        'alpha_values': [],  # 监控 Soft Gate 学习情况
    }
    
    best_val_iou = 0
    
    # AMP 混合精度（激活 Flash Attention，大幅提速）
    scaler = GradScaler('cuda')
    print(f"[AMP] GradScaler 已创建，混合精度训练已启用")
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            scribble_gen=scribble_gen,
            correction_gen=correction_gen,
            box_gen=box_gen,
            num_rounds=args.num_rounds,
            total_epochs=args.epochs,
            scaler=scaler,
            disable_memory=args.disable_memory,
        )
        
        # 更新学习率
        scheduler.step()
        
        # 获取 Alpha 值（监控 Soft Gate 的学习情况）
        alpha_value = model.query_fusion.alpha.item() if not args.disable_sgf else 0.0
        
        # 验证（每 2 个 epoch 验证一次，节省时间）
        val_every = 2
        run_val = (epoch % val_every == 0) or (epoch == args.epochs)
        
        if run_val:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                scribble_gen=scribble_gen,
                correction_gen=correction_gen,
                num_rounds=args.num_rounds,
                use_amp=True,
                disable_memory=args.disable_memory,
            )
        
        # 记录历史
        history['train_losses'].append(train_metrics['losses'])
        history['train_ious'].append(train_metrics['ious'])
        history['val_losses'].append(val_metrics['losses'] if run_val else None)
        history['val_ious'].append(val_metrics['ious'] if run_val else None)
        history['val_dices'].append(val_metrics['dices'] if run_val else None)
        history['box_ratios'].append(train_metrics['box_ratio'])
        history['alpha_values'].append(alpha_value)
        
        # 打印结果
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Box ratio: {train_metrics['box_ratio']:.1%}")
        if not args.disable_sgf:
            print(f"  Alpha (Soft Gate): {alpha_value:.4f}")
        else:
            print(f"  Alpha (Soft Gate): DISABLED")
        print(f"  Train: " + ", ".join([f"L{i}={train_metrics['losses'][i]:.4f}" for i in range(args.num_rounds)]))
        print(f"         " + ", ".join([f"IoU{i}={train_metrics['ious'][i]:.4f}" for i in range(args.num_rounds)]))
        if run_val:
            print(f"  Val:   " + ", ".join([f"L{i}={val_metrics['losses'][i]:.4f}" for i in range(args.num_rounds)]))
            print(f"         " + ", ".join([f"IoU{i}={val_metrics['ious'][i]:.4f}" for i in range(args.num_rounds)]))
            print(f"         " + ", ".join([f"Dice{i}={val_metrics['dices'][i]:.4f}" for i in range(args.num_rounds)]))
        else:
            print(f"  Val:   skipped (next val at epoch {epoch + val_every - epoch % val_every})")
        
        # 保存最佳模型（只在有验证的 epoch）
        if run_val:
            final_iou = val_metrics['ious'][args.num_rounds - 1]
            if final_iou > best_val_iou:
                best_val_iou = final_iou
                
                trainable_state = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_state[name] = param.data.clone()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_iou': best_val_iou,
                    'history': history,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  * Saved best model (Val mIoU: {best_val_iou:.4f})")
    
    # 保存最终模型
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.clone()
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': trainable_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_iou': best_val_iou,
        'history': history,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    # 保存训练历史（处理跳过验证的 epoch，对应值为 None）
    def _serialize_dict_list(lst):
        """将 dict 列表序列化，None 保持为 None"""
        return [
            {str(k): v for k, v in d.items()} if d is not None else None
            for d in lst
        ]
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        history_serializable = {
            'train_losses': _serialize_dict_list(history['train_losses']),
            'train_ious': _serialize_dict_list(history['train_ious']),
            'val_losses': _serialize_dict_list(history['val_losses']),
            'val_ious': _serialize_dict_list(history['val_ious']),
            'val_dices': _serialize_dict_list(history['val_dices']),
            'box_ratios': history['box_ratios'],
            'alpha_values': history['alpha_values'],  # 记录 Soft Gate 学习曲线
        }
        json.dump(history_serializable, f, indent=2)
    
    # 打印最终结果
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()


"""
LoRA 对比训练脚本

功能：
1. 分别训练无 LoRA 和有 LoRA 的模型
2. 使用完整 Endovision18 数据集
3. 划分训练集（80%）和验证集（20%）
4. 训练 5 个 epoch
5. 迭代推理 3 次
6. 训练后可视化对比
7. 分类别统计 Dice 和 IoU 指标

Author: ScribblePrompt Team
"""

import os
import sys
import functools

# 确保 print 立即输出（不被 conda run 缓冲）
print = functools.partial(print, flush=True)

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import defaultdict
import cv2
import albumentations as A


# Endovision18 类别定义
ENDOVISION18_CLASSES = {
    0: 'background-tissue',
    1: 'instrument-shaft',
    2: 'instrument-clasper',
    3: 'instrument-wrist',
    4: 'kidney-parenchyma',
    5: 'covered-kidney',
    6: 'thread',
    7: 'clamps',
    8: 'suturing-needle',
    9: 'suction-instrument',
    10: 'small-intestine',
    11: 'ultrasound-probe',
}

# 颜色到类别的映射（基于 RGB 值）
COLOR_TO_CLASS = {
    (0, 0, 0): 0,         # background-tissue
    (0, 255, 0): 1,       # instrument-shaft
    (0, 255, 255): 2,     # instrument-clasper
    (125, 255, 12): 3,    # instrument-wrist
    (255, 55, 0): 4,      # kidney-parenchyma
    (24, 55, 125): 5,     # covered-kidney
    (187, 155, 25): 6,    # thread
    (0, 255, 125): 7,     # clamps
    (255, 255, 125): 8,   # suturing-needle
    (123, 15, 175): 9,    # suction-instrument
    (124, 155, 5): 10,    # small-intestine
    (12, 255, 141): 11,   # ultrasound-probe
}

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scissr.models.ScribbleSam2VideoSimple import ScribbleSam2VideoSimple
from scissr.interactions.adaptive_scribble import AdaptiveScribble
from scissr.interactions.scribbles import LineScribble


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: 允许 benchmark 选择最快算法（大幅提速）
    # 固定种子已保证绝大部分可复现性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# 全局随机种子（与 Stage 2 保持一致）
GLOBAL_SEED = 42


def get_correction_min_area(epoch: int, warmup_epochs: int = 5) -> tuple:
    """
    修正 scribble 的最小面积阈值
    
    已取消动态阈值（原 ScribblePrompt 没有此限制）。
    设为 0 让所有 error region 都能生成修正 scribble，
    生成器内部会自然处理过小区域（try/except 返回空）。
    这对 Thread 等细线目标尤为重要。
    """
    return 0, None


def get_train_augmentation():
    """
    训练数据增强管线（参考 ScribblePrompt 原文 Table 4）
    
    包含几何增强和光照增强，应用于 image + mask（scribble 在增强后生成）。
    """
    return A.Compose([
        # --- 几何增强 ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,           # scale ∈ [0.9, 1.1]
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,          # sigma=6, 模拟器官形变
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
        A.GaussianBlur(
            blur_limit=(3, 7),          # 模拟对焦不准
            p=0.3,
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),     # 模拟内窥镜噪点
            p=0.3,
        ),
    ])


class TwoChannelScribbleGenerator:
    """
    双通道 scribble 生成器（混合策略，与模型的 correction 逻辑对齐）
    
    正 scribble（前景）→ AdaptiveScribble：
    - 逐连通域分析几何特征 → 选策略 → 自适应粗细
    - 对 Thread 等细结构至关重要
    
    负 scribble（背景）→ LineScribble：
    - 轻量快速，模拟用户"随手标注背景"行为
    - 背景区域大且无规则，不需要骨架分析
    """
    
    def __init__(self, neg_prob: float = 0.0):
        """
        Args:
            neg_prob: R0 阶段生成负 scribble 的概率（0.0=不生成, 0.5=50%概率）
        """
        self.pos_generator = AdaptiveScribble()
        self.neg_generator = LineScribble(thickness=3, warp=False)  # 背景用轻量 Line
        self.neg_prob = neg_prob
    
    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (B, 1, H, W) GT 前景 mask
            n_scribbles: 正 scribble 数量
        
        Returns:
            (B, 2, H, W) 双通道 scribble
                通道0: 正 scribble（前景区域）
                通道1: 负 scribble（背景区域）
        """
        B, _, H, W = mask.shape
        
        # 正 scribble: AdaptiveScribble（逐连通域分析 + 选策略）
        try:
            pos_scribble = self.pos_generator(mask, n_scribbles=n_scribbles)
        except:
            pos_scribble = torch.zeros_like(mask)
        
        # 负 scribble: LineScribble（快速，在背景区域画线）
        neg_scribble = torch.zeros_like(mask)
        if self.neg_prob > 0 and random.random() < self.neg_prob:
            bg_mask = 1.0 - mask  # 背景区域
            if bg_mask.sum() > 100:
                try:
                    neg_scribble = self.neg_generator(bg_mask, n_scribbles=1)
                except:
                    neg_scribble = torch.zeros_like(mask)
        
        return torch.cat([pos_scribble, neg_scribble], dim=1)


def load_train_val_split(split_json_path: str = None):
    """加载训练/验证集划分文件"""
    if split_json_path is None:
        split_json_path = 'dataset/Endovision18/train_val_split.json'
    
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    
    return set(split_data['train_sequences']), set(split_data['val_sequences'])


class Endovision18Dataset(Dataset):
    """完整 Endovision18 数据集 - 训练用（每帧随机一个类别）"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        split: str = 'train',  # 'train' or 'val'
        split_json_path: str = None,  # 使用预定义的 split 文件
        transform: A.Compose = None,  # 数据增强管线
        neg_scribble_prob: float = 0.0,  # R0 负 scribble 概率
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.scribble_generator = TwoChannelScribbleGenerator(neg_prob=neg_scribble_prob)
        
        # 从 JSON 文件加载训练/验证集划分
        self.train_seqs, self.val_seqs = load_train_val_split(split_json_path)
        
        # 收集所有样本
        all_samples = []
        for seq_dir in sorted(self.data_dir.glob('seq_*')):
            if not seq_dir.is_dir():
                continue
            
            images_dir = seq_dir / 'left_frames'
            labels_dir = seq_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            for img_path in sorted(images_dir.glob('*.png')):
                frame_name = img_path.stem
                label_path = labels_dir / f"{frame_name}.png"
                
                if label_path.exists():
                    all_samples.append({
                        'image': str(img_path),
                        'label': str(label_path),
                        'seq': seq_dir.name,
                    })
        
        if split == 'train':
            self.samples = [s for s in all_samples if s['seq'] in self.train_seqs]
        else:
            self.samples = [s for s in all_samples if s['seq'] in self.val_seqs]
        
        print(f"[{split.upper()} Dataset] Sequences: {len(self.train_seqs if split == 'train' else self.val_seqs)}, Samples: {len(self.samples)}")
        
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
        
        chosen_class = 0
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
        
        # 生成 scribble（极小阈值 10px，仅排除标注噪声）
        if mask.sum() > 10:
            scribble = self.scribble_generator(mask.unsqueeze(0), n_scribbles=1)
            scribble = scribble.squeeze(0)
        else:
            scribble = torch.zeros(2, self.image_size, self.image_size)
        
        return {
            'image': image,
            'mask': mask,
            'scribble': scribble,
            'path': sample['image'],
            'class_id': chosen_class,
        }


class Endovision18ValDataset(Dataset):
    """验证数据集 - 全类别验证（每帧的每个类别都作为一个样本）"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        split_json_path: str = None,  # 使用预定义的 split 文件
        neg_scribble_prob: float = 0.5,  # 验证集也用负 scribble 匹配训练分布
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.scribble_generator = TwoChannelScribbleGenerator(neg_prob=neg_scribble_prob)
        
        # 从 JSON 文件加载训练/验证集划分
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
                if color_match.sum() > 10:  # 至少 10 像素（仅排除标注噪声）
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
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")
        
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
        
        # 生成 scribble（极小阈值 10px，仅排除标注噪声）
        if mask.sum() > 10:
            scribble = self.scribble_generator(mask.unsqueeze(0), n_scribbles=1)
            scribble = scribble.squeeze(0)
        else:
            scribble = torch.zeros(2, self.image_size, self.image_size)
        
        return {
            'image': image,
            'mask': mask,
            'scribble': scribble,
            'path': sample['image'],
            'class_id': sample['class_id'],
            'class_name': sample['class_name'],
        }


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    
    对简单样本（背景）降权，迫使模型关注难样本（前景 Thread 等小目标）。
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: 聚焦参数，越大越关注难样本。默认 2.0
        alpha: 前景类权重。默认 0.25（原始论文推荐值）
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 逐像素 BCE（不 reduce）
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # p_t: 正确分类的概率
        p_t = torch.exp(-bce)
        # focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        # alpha balancing: 前景用 alpha，背景用 1-alpha
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # 最终 focal loss
        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


class IterativeLoss(nn.Module):
    """
    迭代训练损失函数: Focal Loss + Soft Dice Loss
    
    参考 ScribblePrompt 原文使用 Dice + Focal 的组合。
    默认权重 focal_weight=20, dice_weight=1，因为 Focal Loss 数值通常较小。
    """
    
    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
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
    """计算 IoU"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return (intersection / (union + 1e-6)).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算 Dice 系数"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    
    return (2 * intersection / (total + 1e-6)).item()


class MetricsTracker:
    """分类别指标追踪器"""
    
    def __init__(self, class_names: dict = None):
        self.class_names = class_names or ENDOVISION18_CLASSES
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        # 每个迭代、每个类别的指标
        self.class_metrics = defaultdict(lambda: defaultdict(list))
        # 总体指标
        self.overall_metrics = defaultdict(list)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, 
               class_id: int, iteration: int):
        """更新指标"""
        iou = compute_iou(pred, target)
        dice = compute_dice(pred, target)
        
        # 按类别记录
        self.class_metrics[iteration][class_id].append({
            'iou': iou,
            'dice': dice,
        })
        
        # 总体记录
        self.overall_metrics[iteration].append({
            'iou': iou,
            'dice': dice,
            'class_id': class_id,
        })
    
    def get_class_metrics(self, iteration: int) -> dict:
        """获取每个类别的平均指标"""
        results = {}
        for class_id, metrics_list in self.class_metrics[iteration].items():
            if len(metrics_list) > 0:
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                results[class_name] = {
                    'iou': np.mean([m['iou'] for m in metrics_list]),
                    'dice': np.mean([m['dice'] for m in metrics_list]),
                    'count': len(metrics_list),
                }
        return results
    
    def get_overall_metrics(self, iteration: int) -> dict:
        """获取总体平均指标"""
        metrics_list = self.overall_metrics[iteration]
        if len(metrics_list) == 0:
            return {'mIoU': 0, 'mDice': 0}
        
        return {
            'mIoU': np.mean([m['iou'] for m in metrics_list]),
            'mDice': np.mean([m['dice'] for m in metrics_list]),
        }
    
    def get_class_averaged_metrics(self, iteration: int) -> dict:
        """获取类别平均指标（每个类别权重相同）"""
        class_metrics = self.get_class_metrics(iteration)
        if len(class_metrics) == 0:
            return {'cIoU': 0, 'cDice': 0}
        
        ious = [m['iou'] for m in class_metrics.values()]
        dices = [m['dice'] for m in class_metrics.values()]
        
        return {
            'cIoU': np.mean(ious),  # class-averaged IoU
            'cDice': np.mean(dices),  # class-averaged Dice
        }
    
    def print_summary(self, iteration: int, prefix: str = ""):
        """打印指标摘要"""
        class_metrics = self.get_class_metrics(iteration)
        overall = self.get_overall_metrics(iteration)
        class_avg = self.get_class_averaged_metrics(iteration)
        
        print(f"\n{prefix}Iteration {iteration} Metrics:")
        print(f"  {'Class':<25} {'IoU':>8} {'Dice':>8} {'Count':>6}")
        print(f"  {'-'*50}")
        
        for class_name in sorted(class_metrics.keys()):
            m = class_metrics[class_name]
            print(f"  {class_name:<25} {m['iou']:>8.4f} {m['dice']:>8.4f} {m['count']:>6}")
        
        print(f"  {'-'*50}")
        print(f"  {'Sample Average (mIoU/mDice)':<25} {overall['mIoU']:>8.4f} {overall['mDice']:>8.4f}")
        print(f"  {'Class Average (cIoU/cDice)':<25} {class_avg['cIoU']:>8.4f} {class_avg['cDice']:>8.4f}")


def train_one_epoch(
    model: ScribbleSam2VideoSimple,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    num_iterations: int = 3,
    correction_min_area: int = 0,
    correction_min_area_final: int = None,
    scaler: GradScaler = None,
):
    """训练一个 epoch（支持 AMP 混合精度）"""
    model.train()
    use_amp = scaler is not None
    
    iter_losses = {i: [] for i in range(num_iterations)}
    iter_ious = {i: [] for i in range(num_iterations)}
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        scribbles = batch['scribble'].to(device)
        
        optimizer.zero_grad()
        
        # AMP 自动混合精度前向（激活 Flash Attention）
        with autocast(enabled=use_amp):
            # 迭代推理（传递动态阈值）
            outputs = model.forward_iterative(
                image=images,
                initial_scribble=scribbles,
                gt_mask=masks,
                num_iterations=num_iterations,
                correction_min_area=correction_min_area,
                correction_min_area_final=correction_min_area_final,
            )
            
            # 计算每次迭代的 loss，后面迭代权重更大
            total_loss = 0
            weights = [(i + 1) for i in range(num_iterations)]
            weight_sum = sum(weights)
            
            for i in range(num_iterations):
                iter_loss = loss_fn(outputs[f'masks_{i}'], masks)
                iter_losses[i].append(iter_loss.item())
                
                iou = compute_iou(outputs[f'masks_{i}'], masks)
                iter_ious[i].append(iou)
                
                total_loss = total_loss + (weights[i] / weight_sum) * iter_loss
        
        # AMP 反向传播
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        # 更新进度条
        loss_str = ', '.join([f'L{i}:{np.mean(iter_losses[i][-20:]):.4f}' for i in range(num_iterations)])
        iou_str = ', '.join([f'I{i}:{np.mean(iter_ious[i][-20:]):.3f}' for i in range(num_iterations)])
        pbar.set_postfix_str(f'{loss_str} | {iou_str}')
    
    return {
        'losses': {i: np.mean(iter_losses[i]) for i in range(num_iterations)},
        'ious': {i: np.mean(iter_ious[i]) for i in range(num_iterations)},
    }


@torch.no_grad()
def validate(
    model: ScribbleSam2VideoSimple,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_iterations: int = 3,
    verbose: bool = False,
    correction_min_area: int = 0,
    correction_min_area_final: int = None,
    use_amp: bool = False,
):
    """验证 - 支持分类别统计（支持 AMP 混合精度）"""
    model.eval()
    
    iter_losses = {i: [] for i in range(num_iterations)}
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        scribbles = batch['scribble'].to(device)
        class_ids = batch['class_id'].numpy()  # (B,)
        
        with autocast(enabled=use_amp):
            outputs = model.forward_iterative(
                image=images,
                initial_scribble=scribbles,
                gt_mask=masks,
                num_iterations=num_iterations,
                correction_min_area=correction_min_area,
                correction_min_area_final=correction_min_area_final,
            )
        
        batch_size = images.shape[0]
        
        for i in range(num_iterations):
            iter_loss = loss_fn(outputs[f'masks_{i}'], masks)
            iter_losses[i].append(iter_loss.item())
            
            # 对每个样本分别计算指标
            for b in range(batch_size):
                pred = outputs[f'masks_{i}'][b:b+1]
                target = masks[b:b+1]
                class_id = int(class_ids[b])
                
                metrics_tracker.update(pred, target, class_id, i)
        
        # 更新进度条
        overall = metrics_tracker.get_overall_metrics(num_iterations - 1)
        pbar.set_postfix_str(f"mIoU:{overall['mIoU']:.3f}, mDice:{overall['mDice']:.3f}")
    
    # 获取所有指标
    results = {
        'losses': {i: np.mean(iter_losses[i]) for i in range(num_iterations)},
        'metrics_tracker': metrics_tracker,
    }
    
    # 为兼容性保留 ious
    results['ious'] = {
        i: metrics_tracker.get_overall_metrics(i)['mIoU'] 
        for i in range(num_iterations)
    }
    results['dices'] = {
        i: metrics_tracker.get_overall_metrics(i)['mDice'] 
        for i in range(num_iterations)
    }
    
    # 详细打印
    if verbose:
        for i in range(num_iterations):
            metrics_tracker.print_summary(i, prefix="  ")
    
    return results


def visualize_results(
    model: ScribbleSam2VideoSimple,
    dataset: Dataset,
    device: torch.device,
    save_dir: str,
    num_samples: int = 8,
    num_iterations: int = 3,
):
    """可视化结果"""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        scribble = sample['scribble'].unsqueeze(0).to(device)
        
        with torch.no_grad(), autocast(enabled=True):
            outputs = model.forward_iterative(
                image=image,
                initial_scribble=scribble,
                gt_mask=mask,
                num_iterations=num_iterations,
            )
        
        # 反归一化图像
        img_vis = image[0].cpu() * std + mean
        img_vis = img_vis.permute(1, 2, 0).numpy().clip(0, 1)
        
        # 创建可视化
        fig, axes = plt.subplots(2, num_iterations + 2, figsize=(4 * (num_iterations + 2), 8))
        
        # 第一行：原图、GT、各迭代预测
        axes[0, 0].imshow(img_vis)
        axes[0, 0].set_title('Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask[0, 0].cpu().numpy(), cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        for i in range(num_iterations):
            pred = torch.sigmoid(outputs[f'masks_{i}'][0, 0]).cpu().numpy()
            iou = compute_iou(outputs[f'masks_{i}'], mask)
            axes[0, i + 2].imshow(pred, cmap='gray')
            axes[0, i + 2].set_title(f'Iter {i} (IoU: {iou:.3f})')
            axes[0, i + 2].axis('off')
        
        # 第二行：Scribble 叠加
        axes[1, 0].imshow(img_vis)
        pos_scribble = scribble[0, 0].cpu().numpy()
        neg_scribble = scribble[0, 1].cpu().numpy()
        scribble_overlay = np.zeros((*pos_scribble.shape, 3))
        scribble_overlay[:, :, 1] = pos_scribble  # Green for positive
        scribble_overlay[:, :, 0] = neg_scribble  # Red for negative
        axes[1, 0].imshow(scribble_overlay, alpha=0.7)
        axes[1, 0].set_title('Initial Scribble')
        axes[1, 0].axis('off')
        
        # GT 轮廓叠加在原图上
        axes[1, 1].imshow(img_vis)
        gt_mask = mask[0, 0].cpu().numpy()
        axes[1, 1].contour(gt_mask, levels=[0.5], colors='yellow', linewidths=2)
        axes[1, 1].set_title('GT Overlay')
        axes[1, 1].axis('off')
        
        for i in range(num_iterations):
            scrib = outputs[f'scribbles_{i}'][0].cpu().numpy()
            pos_s = scrib[0]
            neg_s = scrib[1]
            
            axes[1, i + 2].imshow(img_vis)
            scribble_vis = np.zeros((*pos_s.shape, 3))
            scribble_vis[:, :, 1] = pos_s
            scribble_vis[:, :, 0] = neg_s
            
            # 膨胀让 scribble 更明显
            from scipy.ndimage import binary_dilation
            if pos_s.sum() > 0:
                pos_dilated = binary_dilation(pos_s > 0.5, iterations=3)
                scribble_vis[:, :, 1] = pos_dilated.astype(float)
            if neg_s.sum() > 0:
                neg_dilated = binary_dilation(neg_s > 0.5, iterations=3)
                scribble_vis[:, :, 0] = neg_dilated.astype(float)
            
            axes[1, i + 2].imshow(scribble_vis, alpha=0.7)
            axes[1, i + 2].set_title(f'Scribble {i} (P:{pos_s.sum():.0f}, N:{neg_s.sum():.0f})')
            axes[1, i + 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{idx}.png'), dpi=150)
        plt.close()
    
    print(f"Saved {num_samples} visualization samples to {save_dir}")


def train_model(
    use_lora: bool,
    config: dict,
    save_dir: str,
):
    """训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "with_lora" if use_lora else "no_lora"
    print(f"\n{'='*70}")
    print(f"Training Model: {model_name.upper()}")
    print(f"{'='*70}")
    
    # 创建模型
    print(f"[1/6] Loading SAM2 model (config={config['config_file']}, ckpt={config['ckpt_path']})...")
    import time
    t0 = time.time()
    model = ScribbleSam2VideoSimple(
        config_file=config['config_file'],
        ckpt_path=config['ckpt_path'],
        device=str(device),
        scribble_channels=2,
    )
    print(f"[1/6] Model loaded in {time.time()-t0:.1f}s")
    
    # 冻结预训练参数
    print(f"[2/6] Freezing pretrained parameters...")
    model.freeze_pretrained()
    
    # 启用 LoRA（如果需要）
    if use_lora:
        print(f"[3/6] Enabling LoRA (rank={config['lora_rank']}, alpha={config['lora_alpha']})...")
        model.enable_lora(
            rank=config['lora_rank'],
            alpha=config['lora_alpha'],
        )
    else:
        print(f"[3/6] LoRA disabled, training ScribbleEncoder only")
    
    # 创建数据集（使用预定义的序列划分）
    print(f"[4/6] Creating datasets...")
    # 训练集使用数据增强 + R0 负 scribble
    train_transform = get_train_augmentation()
    train_dataset = Endovision18Dataset(
        data_dir=config['data_dir'],
        image_size=1024,
        split='train',
        transform=train_transform,
        neg_scribble_prob=config.get('neg_scribble_prob', 0.5),
    )
    
    # 验证集：无增强，但保持 R0 负 scribble 以匹配训练分布
    val_dataset = Endovision18ValDataset(
        data_dir=config['data_dir'],
        image_size=1024,
        neg_scribble_prob=config.get('neg_scribble_prob', 0.5),
    )
    
    print(f"[4/6] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 优化器
    print(f"[5/6] Setting up optimizer and loss...")
    if use_lora:
        param_groups = model.get_trainable_param_groups(base_lr=config['lr'])
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=config['lr'],
            weight_decay=0.01,
        )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01,
    )
    
    # 损失函数: Focal Loss + Dice Loss（参考 ScribblePrompt 原文）
    loss_fn = IterativeLoss(
        focal_weight=config.get('focal_weight', 20.0),
        dice_weight=config.get('dice_weight', 1.0),
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', 0.25),
    )
    
    # 创建保存目录
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 训练历史
    history = {
        'train_losses': [],
        'train_ious': [],
        'train_dices': [],
        'val_losses': [],
        'val_ious': [],
        'val_dices': [],
    }
    
    best_val_iou = 0
    final_val_metrics = None
    
    # AMP 混合精度（激活 Flash Attention，大幅提速）
    scaler = GradScaler()
    print(f"[6/6] Starting training ({config['epochs']} epochs, {len(train_loader)} batches/epoch, AMP=ON)...")
    print(f"{'='*70}")
    
    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        # 获取当前 epoch 的动态阈值
        min_area, min_area_final = get_correction_min_area(epoch, warmup_epochs=5)
        print(f"\n--- Epoch {epoch}/{config['epochs']} (min_area={min_area}, min_area_final={min_area_final}) ---")
        
        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            num_iterations=config['num_iterations'],
            correction_min_area=min_area,
            correction_min_area_final=min_area_final,
            scaler=scaler,
        )
        
        # 验证 - 最后一个 epoch 打印详细的分类别指标
        verbose = (epoch == config['epochs'])
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_iterations=config['num_iterations'],
            verbose=verbose,
            correction_min_area=min_area,
            correction_min_area_final=min_area_final,
            use_amp=True,
        )
        final_val_metrics = val_metrics
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_losses'].append(train_metrics['losses'])
        history['train_ious'].append(train_metrics['ious'])
        history['train_dices'].append({i: 0 for i in range(config['num_iterations'])})  # 训练时不计算 Dice
        history['val_losses'].append(val_metrics['losses'])
        history['val_ious'].append(val_metrics['ious'])
        history['val_dices'].append(val_metrics.get('dices', {i: 0 for i in range(config['num_iterations'])}))
        
        # 打印结果
        last_iter = config['num_iterations'] - 1
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: " + ", ".join([f"L{i}={train_metrics['losses'][i]:.4f}" for i in range(config['num_iterations'])]))
        print(f"         " + ", ".join([f"IoU{i}={train_metrics['ious'][i]:.4f}" for i in range(config['num_iterations'])]))
        print(f"  Val:   " + ", ".join([f"L{i}={val_metrics['losses'][i]:.4f}" for i in range(config['num_iterations'])]))
        print(f"         " + ", ".join([f"mIoU{i}={val_metrics['ious'][i]:.4f}" for i in range(config['num_iterations'])]))
        print(f"         " + ", ".join([f"mDice{i}={val_metrics['dices'][i]:.4f}" for i in range(config['num_iterations'])]))
        
        # === Per-Class 验证指标（最后一轮迭代 R2）===
        metrics_tracker = val_metrics.get('metrics_tracker')
        if metrics_tracker is not None:
            class_metrics = metrics_tracker.get_class_metrics(last_iter)
            class_avg = metrics_tracker.get_class_averaged_metrics(last_iter)
            
            print(f"\n  Per-Class Val Metrics (R{last_iter}):")
            print(f"    {'Class':<25} {'IoU':>8} {'Dice':>8} {'Count':>6}")
            print(f"    {'-'*50}")
            for class_name in sorted(class_metrics.keys()):
                m = class_metrics[class_name]
                print(f"    {class_name:<25} {m['iou']:>8.4f} {m['dice']:>8.4f} {m['count']:>6}")
            print(f"    {'-'*50}")
            print(f"    {'Sample-Avg (mIoU/mDice)':<25} {val_metrics['ious'][last_iter]:>8.4f} {val_metrics['dices'][last_iter]:>8.4f}")
            print(f"    {'Class-Avg  (cIoU/cDice)':<25} {class_avg['cIoU']:>8.4f} {class_avg['cDice']:>8.4f}")
        
        # 计算迭代改进
        train_improvement = (train_metrics['ious'][last_iter] - train_metrics['ious'][0]) / (train_metrics['ious'][0] + 1e-6) * 100
        val_improvement = (val_metrics['ious'][last_iter] - val_metrics['ious'][0]) / (val_metrics['ious'][0] + 1e-6) * 100
        print(f"  Iterative Improvement: Train={train_improvement:.1f}%, Val={val_improvement:.1f}%")
        
        # 保存最佳模型（完整版：ScribbleEncoder + LoRA 参数）
        final_iou = val_metrics['ious'][last_iter]
        if final_iou > best_val_iou:
            best_val_iou = final_iou
            # 收集所有可训练参数
            trainable_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_state[name] = param.data.clone()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_state,  # 完整的可训练参数
                'scribble_encoder_state_dict': model.scribble_encoder.state_dict(),  # 兼容旧格式
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'use_lora': use_lora,
            }, os.path.join(model_save_dir, 'best_model.pt'))
            print(f"  * Saved best model (Val mIoU: {best_val_iou:.4f})")
    
    # 保存最终模型（完整版：ScribbleEncoder + LoRA 参数）
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.clone()
    
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': trainable_state,  # 完整的可训练参数
        'scribble_encoder_state_dict': model.scribble_encoder.state_dict(),  # 兼容旧格式
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_iou': best_val_iou,
        'use_lora': use_lora,
        'history': history,
    }, os.path.join(model_save_dir, 'final_model.pt'))
    
    # 保存训练历史
    with open(os.path.join(model_save_dir, 'history.json'), 'w') as f:
        # 转换为可序列化格式
        history_serializable = {
            'train_losses': [{str(k): v for k, v in d.items()} for d in history['train_losses']],
            'train_ious': [{str(k): v for k, v in d.items()} for d in history['train_ious']],
            'train_dices': [{str(k): v for k, v in d.items()} for d in history['train_dices']],
            'val_losses': [{str(k): v for k, v in d.items()} for d in history['val_losses']],
            'val_ious': [{str(k): v for k, v in d.items()} for d in history['val_ious']],
            'val_dices': [{str(k): v for k, v in d.items()} for d in history['val_dices']],
        }
        json.dump(history_serializable, f, indent=2)
    
    # 保存最终的分类别指标
    if final_val_metrics and 'metrics_tracker' in final_val_metrics:
        metrics_tracker = final_val_metrics['metrics_tracker']
        last_iter = config['num_iterations'] - 1
        
        class_metrics = metrics_tracker.get_class_metrics(last_iter)
        overall = metrics_tracker.get_overall_metrics(last_iter)
        class_avg = metrics_tracker.get_class_averaged_metrics(last_iter)
        
        final_metrics = {
            'per_class': {
                class_name: {
                    'iou': float(m['iou']),
                    'dice': float(m['dice']),
                    'count': int(m['count']),
                }
                for class_name, m in class_metrics.items()
            },
            'sample_average': {
                'mIoU': float(overall['mIoU']),
                'mDice': float(overall['mDice']),
            },
            'class_average': {
                'cIoU': float(class_avg['cIoU']),
                'cDice': float(class_avg['cDice']),
            },
        }
        
        with open(os.path.join(model_save_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Final Per-Class Metrics (Iteration {last_iter}):")
        print(f"{'='*60}")
        metrics_tracker.print_summary(last_iter)
    
    # 可视化
    print(f"\nGenerating visualizations...")
    visualize_results(
        model=model,
        dataset=val_dataset,
        device=device,
        save_dir=os.path.join(model_save_dir, 'visualizations'),
        num_samples=10,
        num_iterations=config['num_iterations'],
    )
    
    return history, best_val_iou


def plot_comparison(histories: dict, save_path: str, num_iterations: int = 3):
    """绘制对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'no_lora': 'blue', 'with_lora': 'red'}
    epochs = range(1, len(list(histories.values())[0]['train_losses']) + 1)
    
    # 训练 Loss
    for model_name, history in histories.items():
        for i in range(num_iterations):
            losses = [h[i] for h in history['train_losses']]
            linestyle = ['-', '--', ':'][i]
            axes[0, 0].plot(epochs, losses, 
                          color=colors[model_name], 
                          linestyle=linestyle,
                          label=f'{model_name} Iter{i}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练 IoU
    for model_name, history in histories.items():
        for i in range(num_iterations):
            ious = [h[i] for h in history['train_ious']]
            linestyle = ['-', '--', ':'][i]
            axes[0, 1].plot(epochs, ious, 
                          color=colors[model_name], 
                          linestyle=linestyle,
                          label=f'{model_name} Iter{i}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Training IoU')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 验证 Loss
    for model_name, history in histories.items():
        for i in range(num_iterations):
            losses = [h[i] for h in history['val_losses']]
            linestyle = ['-', '--', ':'][i]
            axes[1, 0].plot(epochs, losses, 
                          color=colors[model_name], 
                          linestyle=linestyle,
                          label=f'{model_name} Iter{i}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 验证 IoU
    for model_name, history in histories.items():
        for i in range(num_iterations):
            ious = [h[i] for h in history['val_ious']]
            linestyle = ['-', '--', ':'][i]
            axes[1, 1].plot(epochs, ious, 
                          color=colors[model_name], 
                          linestyle=linestyle,
                          label=f'{model_name} Iter{i}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Validation IoU')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('LoRA vs No-LoRA Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='LoRA Comparison Training')
    parser.add_argument('--data_dir', type=str, 
                        default='dataset/Endovision18/raw/Train_Data')
    parser.add_argument('--config_file', type=str,
                        default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--ckpt_path', type=str,
                        default='checkpoints/sam2.1_hiera_tiny.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--save_dir', type=str, default='trained_models/lora_comparison')
    parser.add_argument('--only_lora', action='store_true', help='Only train with LoRA')
    parser.add_argument('--only_no_lora', action='store_true', help='Only train without LoRA')
    # P0 新增参数
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal Loss alpha (foreground weight)')
    parser.add_argument('--focal_weight', type=float, default=20.0, help='Focal Loss weight in combined loss')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Dice Loss weight in combined loss')
    parser.add_argument('--neg_scribble_prob', type=float, default=0.5, help='R0 negative scribble probability')
    args = parser.parse_args()
    
    # 设置随机种子（使用全局常量，与 Stage 2 保持一致）
    set_seed(GLOBAL_SEED)
    print(f"[Seed] Using global seed: {GLOBAL_SEED}")
    
    # 配置
    config = {
        'data_dir': args.data_dir,
        'config_file': args.config_file,
        'ckpt_path': args.ckpt_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_iterations': args.num_iterations,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        # P0: Focal Loss 参数
        'focal_gamma': args.focal_gamma,
        'focal_alpha': args.focal_alpha,
        'focal_weight': args.focal_weight,
        'dice_weight': args.dice_weight,
        # P0: 负 scribble 概率
        'neg_scribble_prob': args.neg_scribble_prob,
    }
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print("LoRA Comparison Training")
    print("="*70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Iterations: {args.num_iterations}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Loss: {args.focal_weight}*Focal(gamma={args.focal_gamma}, alpha={args.focal_alpha}) + {args.dice_weight}*Dice")
    print(f"Neg scribble prob: {args.neg_scribble_prob}")
    print(f"Data augmentation: ON (train only)")
    print(f"Save dir: {save_dir}")
    print("="*70)
    
    histories = {}
    best_ious = {}
    
    # 训练无 LoRA 模型
    if not args.only_lora:
        print("\n" + "="*70)
        print("PHASE 1: Training WITHOUT LoRA")
        print("="*70)
        history_no_lora, best_iou_no_lora = train_model(
            use_lora=False,
            config=config,
            save_dir=save_dir,
        )
        histories['no_lora'] = history_no_lora
        best_ious['no_lora'] = best_iou_no_lora
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
    
    # 训练有 LoRA 模型
    if not args.only_no_lora:
        print("\n" + "="*70)
        print("PHASE 2: Training WITH LoRA")
        print("="*70)
        history_with_lora, best_iou_with_lora = train_model(
            use_lora=True,
            config=config,
            save_dir=save_dir,
        )
        histories['with_lora'] = history_with_lora
        best_ious['with_lora'] = best_iou_with_lora
    
    # 绘制对比图
    if len(histories) == 2:
        plot_comparison(
            histories=histories,
            save_path=os.path.join(save_dir, 'comparison.png'),
            num_iterations=args.num_iterations,
        )
    
    # 打印最终结果
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for model_name, best_iou in best_ious.items():
        print(f"{model_name}: Best Val IoU = {best_iou:.4f}")
    
    if len(best_ious) == 2:
        improvement = (best_ious['with_lora'] - best_ious['no_lora']) / best_ious['no_lora'] * 100
        print(f"\nLoRA Improvement: {improvement:+.2f}%")
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()


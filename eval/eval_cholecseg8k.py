"""
OOD (Out-of-Distribution) Test on CholecSeg8k - video01
Compare: SAM2 Tiny (baseline) vs ScribbleSam2Memory (ours)

两个模型都基于 SAM2 Tiny 预训练权重:
- SAM2: 原始预训练权重, scribble 转 point 输入, 无 memory, 使用 prev_mask
- ScribbleSam2Memory: Stage 2 训练权重, 双轨 scribble, memory-driven 迭代

Scribble 策略与参数与 eval/eval_endovis18.py 完全一致:
1. TwoChannelScribbleGenerator: AdaptiveScribble(正) + LineScribble(负, neg_prob=0.5)
2. CorrectionScribbleGenerator: AdaptiveScribble(FN) + LineScribble(FP)
3. 双轨设计: accumulated(Track 1) + latest(Track 2)
4. R0 不用 Memory, R1+ 使用 Memory
5. mask 阈值 > 10
6. AMP 混合精度
7. 3 轮迭代推理

Author: ScribblePrompt Team
"""

import os
import sys
import random
import functools
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import glob
import logging

# 抑制 SAM3 的 verbose INFO 日志（set_image 每次调用都会打）
logging.getLogger("root").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# 确保 flush 立即输出
print = functools.partial(print, flush=True)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化 Hydra（SAM2 需要）
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir=os.path.join(os.getcwd(), "sam2", "configs"),
    version_base="1.2",
)

from scissr.models.ScribbleSam2Memory import build_scribble_sam2_memory
from scissr.interactions.scribbles import LineScribble
from scissr.interactions.adaptive_scribble import AdaptiveScribble
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# NOTE: SAM 3 is only used as an optional baseline (see build_sam3_baseline).
# It is NOT a public package, so we import it lazily inside that function and
# the script runs fine without it when SAM 3 evaluation is skipped (--no_sam3).


# =============================================================================
# 随机种子 & 状态管理
# =============================================================================

def set_seed(seed=42):
    """设置随机种子，确保测试结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed] 随机种子已设置为 {seed}")




# =============================================================================
# CholecSeg8k 类别映射
# =============================================================================

# =============================================================================
# CholecSeg8k 类别映射 (共 13 类，使用 color_mask RGB 映射)
#
# 官方标注文件: processed/out_ann_file_test_noback.json (12 前景类)
# 通过 watershed_mask ↔ color_mask 像素对齐验证得到 RGB 映射
# =============================================================================

RGB_TO_CLASSID = {
    # ---- Background（不参与测试）----
    # (127, 127, 127): -1,  # Black Background (内窥镜黑边)
    # (255, 255, 255): -1,  # 边界/未标注
    # ---- Foreground (12 类) ----
    (210, 140, 140): 0,   # Abdominal Wall
    (255, 114, 114): 1,   # Liver
    (231, 70, 156):  2,   # Gastrointestinal Tract
    (186, 183, 75):  3,   # Fat
    (170, 255, 0):   4,   # Grasper
    (255, 85, 0):    5,   # Connective Tissue
    (255, 0, 0):     6,   # Blood
    (255, 255, 0):   7,   # Cystic Duct
    (169, 255, 184): 8,   # L-hook Electrocautery
    (255, 160, 165): 9,   # Gallbladder
    (0, 50, 128):    10,  # Hepatic Vein
    (111, 74, 0):    11,  # Liver Ligament
}

CLASSID_TO_NAME = {
    0:  "Abdominal Wall",
    1:  "Liver",
    2:  "Gastrointestinal Tract",
    3:  "Fat",
    4:  "Grasper",
    5:  "Connective Tissue",
    6:  "Blood",
    7:  "Cystic Duct",
    8:  "L-hook Electrocautery",
    9:  "Gallbladder",
    10: "Hepatic Vein",
    11: "Liver Ligament",
}

ALL_CLASSES = list(range(12))            # 0-11 全部 12 类
FOREGROUND_CLASSES = list(range(12))     # 0-11 全部参与测试和统计


def get_class_name(class_id):
    return CLASSID_TO_NAME.get(class_id, f"class_{class_id}")


def color_mask_to_class_mask(color_rgb):
    """
    将 RGB color_mask 转换为 class mask (-1 表示未映射/背景)

    Args:
        color_rgb: (H, W, 3) numpy array, RGB color mask
    Returns:
        class_mask: (H, W) numpy array, int32, 像素值为 class_id 或 -1
    """
    H, W, _ = color_rgb.shape
    class_mask = np.full((H, W), -1, dtype=np.int32)
    for rgb, class_id in RGB_TO_CLASSID.items():
        match = np.all(color_rgb == rgb, axis=2)
        class_mask[match] = class_id
    return class_mask


# =============================================================================
# Scribble 生成器 —— 与 eval/eval_endovis18.py / train/train_stage2.py 完全一致
# =============================================================================

class TwoChannelScribbleGenerator:
    """
    双通道 scribble 生成器
    正 scribble（前景）→ AdaptiveScribble
    负 scribble（背景）→ LineScribble
    """
    def __init__(self, neg_prob: float = 0.5):
        self.pos_generator = AdaptiveScribble()
        self.neg_generator = LineScribble(thickness=3, warp=False)
        self.neg_prob = neg_prob

    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        B, _, H, W = mask.shape
        try:
            pos_scribble = self.pos_generator(mask, n_scribbles=n_scribbles)
        except:
            pos_scribble = torch.zeros_like(mask)

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
    FN（漏检）→ AdaptiveScribble
    FP（误检）→ LineScribble
    """
    def __init__(self):
        from scissr.interactions.adaptive_scribble import CorrectionConfig
        self.fn_generator = AdaptiveScribble(config=CorrectionConfig())
        self.fp_generator = LineScribble(thickness=3, warp=False)

    def generate(self, pred_mask, gt_mask, threshold=0.5):
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
        gt_binary = (gt_mask > threshold).float()

        fn_region = gt_binary * (1 - pred_binary)
        fp_region = (1 - gt_binary) * pred_binary

        pos_scribble = torch.zeros_like(pred_mask)
        if fn_region.sum() > 1:
            try:
                pos_scribble = self.fn_generator(fn_region, n_scribbles=1)
            except:
                pass

        neg_scribble = torch.zeros_like(pred_mask)
        if fp_region.sum() > 1:
            try:
                neg_scribble = self.fp_generator(fp_region, n_scribbles=1)
            except:
                pass

        return pos_scribble, neg_scribble


# =============================================================================
# Scribble → Points 转换（用于 SAM2 baseline）
# =============================================================================

def _farthest_point_sampling(points, n_samples):
    """
    最远点采样 (Farthest Point Sampling, FPS)
    从 points 中选出 n_samples 个点，使其空间覆盖最均匀。

    Args:
        points: (N, 2) numpy array
        n_samples: 要选的点数

    Returns:
        selected_indices: list of int
    """
    N = len(points)
    if N <= n_samples:
        return list(range(N))

    # 第 1 个点: 选最接近质心的像素（确定性，位置居中）
    centroid = points.mean(axis=0)
    dists_to_center = np.sum((points - centroid) ** 2, axis=1)
    selected = [int(np.argmin(dists_to_center))]
    distances = np.full(N, np.inf)

    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))

    return selected


def _sample_points_from_mask(binary_mask, n_points):
    """
    从二值 mask 中采样点。

    策略:
    - n_points=None: 返回全部像素
    - n_points=-1: 每个连通域取 1 个质心点 (center-per-CC)
    - n_points=N (>0): 用 Farthest Point Sampling (FPS) 选 N 个点，
                        保证空间覆盖均匀 + 连通域自然覆盖

    Returns:
        coords: list of [x, y], 或空 list
    """
    ys, xs = np.where(binary_mask)
    if len(ys) == 0:
        return []

    # 不限制 → 全部像素
    if n_points is None:
        return [[xs[i], ys[i]] for i in range(len(ys))]

    # Center-per-CC: 每个连通域取质心最近的有效像素
    if n_points == -1:
        from scipy.ndimage import label
        labeled, num_cc = label(binary_mask)
        centers = []
        for cc_id in range(1, num_cc + 1):
            cc_ys, cc_xs = np.where(labeled == cc_id)
            cy, cx = cc_ys.mean(), cc_xs.mean()
            dists = (cc_ys - cy) ** 2 + (cc_xs - cx) ** 2
            best_idx = int(np.argmin(dists))
            centers.append([int(cc_xs[best_idx]), int(cc_ys[best_idx])])
        return centers

    # FPS 采样: 天然保证空间均匀覆盖
    # 不同连通域之间距离远，FPS 会自动优先覆盖不同连通域
    points = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)
    selected = _farthest_point_sampling(points, n_points)

    return [[xs[i], ys[i]] for i in selected]


def scribble_to_points(scribble_2ch, gt_mask_np=None, n_points_per_channel=None):
    """
    将双通道 scribble 转换为 SAM2 point prompts

    每通道独立采样，保证连通域最小覆盖:
    - n_points_per_channel=None: 全部像素
    - n_points_per_channel=N: 每通道最多 N 点，每个连通域至少 1 点 (when possible)

    Args:
        scribble_2ch: (1, 2, H, W) tensor [positive, negative]
        gt_mask_np: (H, W) numpy GT mask, 用于兜底
        n_points_per_channel: 每通道最大采样点数, None=全部像素

    Returns:
        point_coords: (N, 2) numpy array, (X, Y) 像素坐标
        point_labels: (N,) numpy array, 1=前景, 0=背景
    """
    pos_mask = scribble_2ch[0, 0].cpu().numpy() > 0.5
    neg_mask = scribble_2ch[0, 1].cpu().numpy() > 0.5

    all_coords = []
    all_labels = []

    # 正 scribble → 前景点（连通域感知采样）
    pos_coords = _sample_points_from_mask(pos_mask, n_points_per_channel)
    if pos_coords:
        all_coords.extend(pos_coords)
        all_labels.extend([1] * len(pos_coords))
    elif gt_mask_np is not None:
        # 兜底：正 scribble 为空，从 GT mask 采 1 个前景点
        fg_ys, fg_xs = np.where(gt_mask_np > 0.5)
        if len(fg_ys) > 0:
            idx = np.random.choice(len(fg_ys))
            all_coords.append([fg_xs[idx], fg_ys[idx]])
            all_labels.append(1)

    # 负 scribble → 背景点（连通域感知采样）
    neg_coords = _sample_points_from_mask(neg_mask, n_points_per_channel)
    if neg_coords:
        all_coords.extend(neg_coords)
        all_labels.extend([0] * len(neg_coords))

    if len(all_coords) == 0:
        return None, None

    return np.array(all_coords, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# =============================================================================
# 指标计算
# =============================================================================

def compute_metrics(pred, gt):
    """计算 IoU 和 Dice"""
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)
    return iou, dice


# =============================================================================
# 数据加载: CholecSeg8k (使用 processed test split)
# =============================================================================

def load_cholecseg8k_test(data_root, video_filter=None):
    """
    加载 CholecSeg8k 测试集（processed/images/test/ 下的 PNG 图像）

    目录结构（扁平目录，无子文件夹）:
        processed/images/test/video01_00080frame_101_endo.png

    文件命名映射到 raw mask:
        raw/video01/video01_00080/frame_101_endo_color_mask.png

    Args:
        data_root: CholecSeg8k 数据集根目录
        video_filter: 指定视频名 (如 'video01')、视频列表 (如 ['video01','video09'])，
                      或 None 表示加载所有测试视频
    """
    test_img_dir = os.path.join(data_root, 'processed', 'images', 'test')
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_img_dir}")

    # 扁平目录，按 video 前缀筛选
    all_img_files = sorted(glob.glob(os.path.join(test_img_dir, '*.png')))

    if video_filter:
        if isinstance(video_filter, list):
            all_img_files = [f for f in all_img_files
                             if any(os.path.basename(f).startswith(v + '_') for v in video_filter)]
        else:
            all_img_files = [f for f in all_img_files if os.path.basename(f).startswith(video_filter + '_')]

    # 统计各 video 图像数
    video_counts = {}
    for f in all_img_files:
        vname = os.path.basename(f).split('_')[0]  # 'video01'
        video_counts[vname] = video_counts.get(vname, 0) + 1

    print(f"\n加载 CholecSeg8k 测试集 (processed test split, PNG)...")
    for vname in sorted(video_counts.keys()):
        print(f"  {vname}: {video_counts[vname]} 张图像")
    print(f"  总计: {len(all_img_files)} 张图像")

    test_samples = []

    for img_path in tqdm(all_img_files, desc="扫描测试帧"):
        filename = os.path.basename(img_path)  # e.g. video01_00080frame_101_endo.png

        # 解析文件名 → 定位 raw color_mask (使用 RGB 映射，避免灰度碰撞)
        videoId = filename.split('frame')[0]           # 'video01_00080'
        videoMajor = videoId.split('_')[0]              # 'video01'
        frameId = filename.split('_')[2]                # '101'
        mask_filename = f"frame_{frameId}_endo_color_mask.png"
        mask_path = os.path.join(data_root, 'raw', videoMajor, videoId, mask_filename)

        if not os.path.exists(mask_path):
            continue

        # 加载 color_mask 获取存在的类别 (RGB → class_id)
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        class_mask = color_mask_to_class_mask(mask_rgb)

        present_classes = [int(c) for c in np.unique(class_mask) if c >= 0]

        for class_id in present_classes:
            binary_mask = (class_mask == class_id).astype(np.float32)
            # 与训练一致：阈值 > 10（仅排除标注噪声）
            if binary_mask.sum() > 10:
                test_samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'class_id': class_id,
                    'video': videoMajor,
                    'frame_name': filename,
                })

    return test_samples


# =============================================================================
# 单样本测试: SAM2 Baseline
# =============================================================================

def test_sam2_single_sample(
    predictor, image_np, gt_mask, gt_mask_tensor,
    scribble_gen, correction_gen, device,
    num_rounds=3, n_points_per_channel=None,
    skip_set_image=False,
):
    """
    SAM2 Baseline 单样本测试
    - scribble 转 point 输入
    - 无 memory
    - prev_mask 通道输入上一轮结果

    Points 累积策略:
        每轮只从本轮新增 scribble 中采样，然后追加到历史 points。
        n_points_per_channel 限制的是每轮每通道的新增点数，不是总量。
        None = 不限制，使用本轮新增 scribble 的全部像素。
        -1 = center-per-CC (每连通域质心 1 点)

    Returns:
        round_metrics: list of (iou, dice) per round, or None if failed
    """
    H, W = 1024, 1024

    # 生成初始 scribble（与训练完全一致的策略）
    if gt_mask_tensor.sum() > 10:
        initial_scribble = scribble_gen(gt_mask_tensor, n_scribbles=1).to(device)
    else:
        initial_scribble = torch.zeros(1, 2, H, W, device=device)

    try:
        # 使用 bfloat16 autocast 以启用 Flash Attention（否则 float32 回退到慢路径）
        if not skip_set_image:
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # 设置图像（计算 backbone 特征，只做一次）
                predictor.set_image(image_np)

        # 跨轮累积的 points
        accumulated_coords = None  # (N, 2) numpy
        accumulated_labels = None  # (N,) numpy

        prev_low_res_mask = None
        round_metrics = []
        round_outputs = {}

        for round_idx in range(num_rounds):
            if round_idx == 0:
                # R0: 从初始 scribble 采样 points
                new_pts, new_lbls = scribble_to_points(
                    initial_scribble, gt_mask_np=gt_mask,
                    n_points_per_channel=n_points_per_channel,
                )
                if new_pts is None:
                    return None

                accumulated_coords = new_pts
                accumulated_labels = new_lbls

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    masks, ious, low_res_masks = predictor.predict(
                        point_coords=accumulated_coords,
                        point_labels=accumulated_labels,
                        mask_input=None,
                        multimask_output=False,
                        return_logits=True,
                        normalize_coords=True,
                    )

                prev_low_res_mask = low_res_masks
                high_res_tensor = torch.from_numpy(masks).unsqueeze(0).float().to(device)
                round_outputs[f'masks_{round_idx}'] = high_res_tensor

            else:
                # R1+: 从本轮 correction scribble 采样新 points，追加到历史
                prev_high_res = round_outputs[f'masks_{round_idx - 1}']
                pos_correction, neg_correction = correction_gen.generate(
                    pred_mask=prev_high_res,
                    gt_mask=gt_mask_tensor,
                )

                # 本轮 correction scribble
                correction_scribble = torch.cat([pos_correction, neg_correction], dim=1)

                # 只从本轮新增 scribble 中采样
                new_pts, new_lbls = scribble_to_points(
                    correction_scribble, gt_mask_np=gt_mask,
                    n_points_per_channel=n_points_per_channel,
                )

                # 追加到累积 points
                if new_pts is not None and len(new_pts) > 0:
                    accumulated_coords = np.concatenate([accumulated_coords, new_pts], axis=0)
                    accumulated_labels = np.concatenate([accumulated_labels, new_lbls], axis=0)

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    masks, ious, low_res_masks = predictor.predict(
                        point_coords=accumulated_coords,
                        point_labels=accumulated_labels,
                        mask_input=prev_low_res_mask,
                        multimask_output=False,
                        return_logits=True,
                        normalize_coords=True,
                    )

                prev_low_res_mask = low_res_masks
                high_res_tensor = torch.from_numpy(masks).unsqueeze(0).float().to(device)
                round_outputs[f'masks_{round_idx}'] = high_res_tensor

            # 计算指标
            pred_logits = masks[0]  # (H, W) logits
            pred = 1.0 / (1.0 + np.exp(-pred_logits))  # sigmoid
            pred_binary = (pred > 0.5).astype(np.float32)
            iou, dice = compute_metrics(pred_binary, gt_mask)
            round_metrics.append((iou, dice))

        return round_metrics

    except Exception as e:
        return None


# =============================================================================
# 单样本测试: ScribbleSam2Memory (Ours)
# =============================================================================

def test_ours_single_sample(
    model, image_tensor, gt_mask, gt_mask_tensor,
    scribble_gen, correction_gen, device,
    num_rounds=3, use_amp=True,
):
    """
    ScribbleSam2Memory 单样本测试
    - 双轨 scribble 策略（与 eval/eval_endovis18.py 完全一致）
    - R0 不用 Memory, R1+ 使用 Memory

    Returns:
        round_metrics: list of (iou, dice) per round, or None if failed
    """
    H, W = 1024, 1024

    # 生成初始 scribble（与训练完全一致的策略）
    if gt_mask_tensor.sum() > 10:
        initial_scribble = scribble_gen(gt_mask_tensor, n_scribbles=1).to(device)
    else:
        initial_scribble = torch.zeros(1, 2, H, W, device=device)

    try:
        with autocast('cuda', enabled=use_amp):
            # 重置 Memory Bank
            model.reset_memory_bank()

            # 图像编码（只做一次）
            backbone_out = model.forward_image(image_tensor)
            _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

            cache = {
                'backbone_out': backbone_out,
                'vision_feats': vision_feats,
                'vision_pos_embeds': vision_pos_embeds,
                'feat_sizes': feat_sizes,
            }

            # 双轨 scribble 状态
            accumulated_scribble = initial_scribble   # Track 1: 累积所有历史
            latest_scribble = initial_scribble         # Track 2: 只有本轮最新

            round_metrics = []
            round_outputs = {}

            for round_idx in range(num_rounds):
                if round_idx == 0:
                    # R0: 纯 scribble，不用 Memory
                    low_res, high_res, cache = model.forward_single_round(
                        image=image_tensor,
                        latest_scribble=latest_scribble,
                        accumulated_scribble=accumulated_scribble,
                        box=None,
                        use_memory=False,
                        update_memory=True,
                        **cache,
                    )
                else:
                    # R1+: Correction Scribble + Memory
                    prev_high_res = round_outputs[f'masks_{round_idx - 1}']
                    pos_correction, neg_correction = correction_gen.generate(
                        pred_mask=prev_high_res,
                        gt_mask=gt_mask_tensor,
                    )

                    # Track 2: 只有本轮的 correction
                    latest_scribble = torch.cat([pos_correction, neg_correction], dim=1)

                    # Track 1: 累积所有历史
                    if accumulated_scribble is not None:
                        accumulated_scribble = torch.stack([
                            torch.max(accumulated_scribble[:, 0], pos_correction.squeeze(1)),
                            torch.max(accumulated_scribble[:, 1], neg_correction.squeeze(1)),
                        ], dim=1)
                    else:
                        accumulated_scribble = latest_scribble.clone()

                    # 修正推理（使用 Memory）
                    use_mem = True
                    update_mem = (round_idx < num_rounds - 1)
                    low_res, high_res, cache = model.forward_single_round(
                        image=image_tensor,
                        latest_scribble=latest_scribble,
                        accumulated_scribble=accumulated_scribble,
                        box=None,
                        use_memory=use_mem,
                        update_memory=update_mem,
                        **cache,
                    )

                round_outputs[f'masks_{round_idx}'] = high_res

                # 计算指标
                pred = torch.sigmoid(high_res[0, 0]).cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.float32)
                iou, dice = compute_metrics(pred_binary, gt_mask)
                round_metrics.append((iou, dice))

            return round_metrics

    except Exception as e:
        return None


# =============================================================================
# 结果打印
# =============================================================================

def print_results(results, model_name, num_rounds):
    """打印模型结果"""
    last_r = num_rounds - 1
    n_total = len(results['all'][f'iou_r{last_r}'])

    if n_total == 0:
        print(f"[{model_name}] 没有有效的测试样本！")
        return 0, {}, {}

    print(f"\n{'=' * 90}")
    print(f"测试结果 — {model_name}")
    print(f"{'=' * 90}")

    # 整体 Sample-Avg（只统计前景类样本）
    fg_ious = {r: [] for r in range(num_rounds)}
    fg_dices = {r: [] for r in range(num_rounds)}
    for cid in FOREGROUND_CLASSES:
        d = results['per_class'][cid]
        for r in range(num_rounds):
            fg_ious[r].extend(d[f'iou_r{r}'])
            fg_dices[r].extend(d[f'dice_r{r}'])

    n_fg = len(fg_ious[last_r])
    print(f"\n整体 Sample-Avg (N={n_fg}, foreground only):")
    iou_strs = [f"R{r}={np.mean(fg_ious[r]):.4f}" for r in range(num_rounds)]
    dice_strs = [f"R{r}={np.mean(fg_dices[r]):.4f}" for r in range(num_rounds)]
    print(f"  mIoU:  {', '.join(iou_strs)}")
    print(f"  mDice: {', '.join(dice_strs)}")

    # 整体 Class-Avg（只统计前景类，排除 Abdominal Wall）
    class_ious = {r: [] for r in range(num_rounds)}
    class_dices = {r: [] for r in range(num_rounds)}
    for cid in FOREGROUND_CLASSES:
        d = results['per_class'][cid]
        if len(d[f'iou_r{last_r}']) > 0:
            for r in range(num_rounds):
                class_ious[r].append(np.mean(d[f'iou_r{r}']))
                class_dices[r].append(np.mean(d[f'dice_r{r}']))

    n_classes = len(class_ious[last_r])
    if n_classes > 0:
        print(f"\n整体 Class-Avg ({n_classes} foreground classes, excl. Abdominal Wall):")
        ciou_strs = [f"R{r}={np.mean(class_ious[r]):.4f}" for r in range(num_rounds)]
        cdice_strs = [f"R{r}={np.mean(class_dices[r]):.4f}" for r in range(num_rounds)]
        print(f"  cIoU:  {', '.join(ciou_strs)}")
        print(f"  cDice: {', '.join(cdice_strs)}")

    # 按类别打印（全部类别都展示，方便参考）
    header = f"{'ID':<5} {'Name':<30} {'N':<8}"
    for r in range(num_rounds):
        header += f" {'R' + str(r) + '_IoU':<10}"
    header += f" {'R' + str(last_r) + '_Dice':<10}"
    print(f"\n{header}")
    print("-" * (55 + 10 * num_rounds + 10))

    for cid in ALL_CLASSES:
        d = results['per_class'][cid]
        n = len(d[f'iou_r{last_r}'])
        marker = " " if cid in FOREGROUND_CLASSES else "*"  # * 标记非前景类
        if n > 0:
            row = f"{cid:<5} {get_class_name(cid):<30} {n:<8}"
            for r in range(num_rounds):
                row += f" {np.mean(d[f'iou_r{r}']):<10.4f}"
            row += f" {np.mean(d[f'dice_r{last_r}']):<10.4f}"
            print(f"{marker}{row}")
        else:
            print(f"{marker}{cid:<5} {get_class_name(cid):<30} {'N/A':<8}")
    print("-" * (55 + 10 * num_rounds + 10))
    print("(* = not included in Class-Avg)")

    # 迭代改进（前景类）
    r0_mean = np.mean(fg_ious[0])
    r_last_mean = np.mean(fg_ious[last_r])
    improvement = (r_last_mean - r0_mean) / (r0_mean + 1e-6) * 100
    print(f"\n迭代改进: R0→R{last_r} = {r0_mean:.4f} → {r_last_mean:.4f} ({improvement:+.1f}%)")

    return n_total, class_ious, class_dices


# =============================================================================
# 模型构建
# =============================================================================

def build_sam2_baseline(ckpt_path, device):
    """构建 SAM2 Tiny baseline (使用 SAM2ImagePredictor 做单帧推理)"""
    print("\n[Build SAM2] Creating SAM2 Tiny (baseline)...")
    sam2_model = build_sam2(
        config_file="sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=ckpt_path,
        device=str(device),
        mode="eval",
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print("[Build SAM2] SAM2 Tiny loaded, wrapped with SAM2ImagePredictor")
    return predictor


def build_sam3_baseline(ckpt_path, device):
    """构建 SAM3 baseline (使用 SAM3InteractiveImagePredictor 做单帧推理)

    注意: build_sam3_image_model 的 enable_inst_interactivity 模式下
    tracker 没有独立 backbone (with_backbone=False), 导致 set_image 时
    self.backbone 为 None。这里直接构建一个带 backbone 的独立 tracker。
    """
    from sam3.model_builder import build_tracker, _load_checkpoint
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
    from iopath.common.file_io import g_pathmgr

    print("\n[Build SAM3] Creating SAM3 (baseline) with standalone backbone...")

    # 构建带 backbone 的 tracker
    tracker_model = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=True,  # 关键: 需要独立 backbone 做单帧推理
    )

    # 加载 checkpoint (只取 tracker 部分的权重)
    print(f"[Build SAM3] Loading checkpoint: {ckpt_path}")
    with g_pathmgr.open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    # tracker 权重 (tracker.* → *)
    tracker_state = {
        k.replace("tracker.", ""): v
        for k, v in ckpt.items() if k.startswith("tracker.")
    }
    # backbone 权重: 视频模型中 detector 和 tracker 共享 backbone，
    # checkpoint 只存在 detector.backbone.* 下，需要映射到 backbone.*
    backbone_state = {
        k.replace("detector.backbone.", "backbone."): v
        for k, v in ckpt.items() if k.startswith("detector.backbone.")
    }
    # 合并 (backbone 不覆盖已有 tracker 权重)
    combined_state = {**backbone_state, **tracker_state}
    missing, unexpected = tracker_model.load_state_dict(combined_state, strict=False)
    print(f"[Build SAM3] Loaded {len(tracker_state)} tracker + {len(backbone_state)} backbone keys")
    if missing:
        print(f"[Build SAM3] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[Build SAM3] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    tracker_model.to(device)
    tracker_model.eval()

    # 包装为 SAM3InteractiveImagePredictor
    predictor = SAM3InteractiveImagePredictor(tracker_model)
    print(f"[Build SAM3] Image size: {tracker_model.image_size}")
    return predictor


def build_our_model(args, device):
    """构建 ScribbleSam2Memory 并加载权重"""
    print("\n[Build Ours] Creating ScribbleSam2Memory...")
    model = build_scribble_sam2_memory(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device=str(device),
        scribble_channels=2,
    )

    # 启用 LoRA
    print("[Build Ours] Enabling Mask Decoder LoRA...")
    model.enable_mask_decoder_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    print("[Build Ours] Enabling Memory Attention LoRA...")
    model.enable_memory_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    # 冻结预训练权重
    model.freeze_pretrained()

    # 加载权重
    print(f"[Build Ours] Loading from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)

    epoch_info = ckpt.get('epoch', 'N/A')
    best_val = ckpt.get('best_val_iou', 'N/A')

    model_state = model.state_dict()
    loaded_count = 0
    loaded_categories = {
        'scribble_encoder': 0, 'mask_decoder_lora': 0,
        'memory_lora': 0, 'query_fusion': 0, 'other': 0,
    }

    for name, param in ckpt['model_state_dict'].items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded_count += 1
            if 'scribble_encoder' in name:
                loaded_categories['scribble_encoder'] += 1
            elif 'query_fusion' in name:
                loaded_categories['query_fusion'] += 1
            elif 'memory_attention' in name:
                loaded_categories['memory_lora'] += 1
            elif 'sam_mask_decoder' in name:
                loaded_categories['mask_decoder_lora'] += 1
            else:
                loaded_categories['other'] += 1

    model.load_state_dict(model_state, strict=False)

    print(f"  模型加载成功: Epoch={epoch_info}, Val mIoU={best_val}")
    print(f"  已加载 {loaded_count} 个参数:")
    for cat, count in loaded_categories.items():
        if count > 0:
            print(f"    - {cat}: {count}")

    alpha_value = model.query_fusion.alpha.item()
    print(f"  Alpha (Soft Gate): {alpha_value:.4f}")

    model.eval()
    return model, epoch_info, best_val


# =============================================================================
# 主函数
# =============================================================================

def _set_seed_quiet(seed):
    """静默设置 seed（不打印日志，用于逐样本 seed）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='OOD Test: CholecSeg8k')

    # 数据路径
    parser.add_argument('--data_root', type=str,
                        default='dataset/CholecSeg8k',
                        help='CholecSeg8k 数据集根目录')
    parser.add_argument('--video', type=str, default='video01',
                        help='测试视频名 (如 video01)，逗号分隔多视频 (如 video01,video09)，设为 all 则测所有测试视频')

    # 模型配置
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/scissr/stage2_best.pt',
                        help='ScribbleSam2Memory 权重路径')
    parser.add_argument('--config_file', type=str, default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/sam2.1_hiera_tiny.pt')
    parser.add_argument('--sam3_ckpt_path', type=str,
                        default='checkpoints/sam3.pt',
                        help='SAM3 checkpoint 路径')
    parser.add_argument('--no_sam3', action='store_true',
                        help='跳过 SAM3 baseline (checkpoint 不可用时使用)')

    # LoRA 配置（与训练一致）
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)

    # 测试配置
    parser.add_argument('--num_rounds', type=int, default=3, help='迭代轮数')
    parser.add_argument('--n_points', type=int, default=10,
                        help='SAM2 采样模式每通道点数 (另外还会跑全部像素模式)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_amp', action='store_true', help='禁用 AMP')
    parser.add_argument('--no_all_pixels', action='store_true',
                        help='跳过 SAM2 全像素点模式 (节省时间)')
    parser.add_argument('--sam3_only', action='store_true',
                        help='只跑 SAM3 baseline (跳过 SAM2 和 Ours)')
    parser.add_argument('--sam3_all_pixels', action='store_true',
                        help='同时跑 SAM3 全像素点模式')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')

    # 消融实验
    parser.add_argument('--ablation_pts', type=str, default=None,
                        help='消融实验：逗号分隔的点数 (如 "1,10,30,50")。1=每CC质心1点')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_amp = not args.no_amp
    run_all_pixels = not args.no_all_pixels
    run_sam3 = not args.no_sam3
    sam3_only = args.sam3_only
    run_sam3_all_pixels = args.sam3_all_pixels
    num_rounds = args.num_rounds
    if args.video == 'all':
        video_filter = None
    elif ',' in args.video:
        video_filter = [v.strip() for v in args.video.split(',')]
    else:
        video_filter = args.video

    # 消融模式
    ablation_mode = args.ablation_pts is not None
    ablation_pts_list = []
    if ablation_mode:
        ablation_pts_list = [int(p.strip()) for p in args.ablation_pts.split(',')]
        # 消融模式: 强制只跑 SAM2 + SAM3, 不跑 Ours 和 all-pixels
        run_sam3 = True
        sam3_only = False
        run_sam3_all_pixels = False
        run_all_pixels = False
        print(f"\n[Ablation Mode] 点数设置: {ablation_pts_list}")
        print(f"  1 = center-per-CC (每连通域质心 1 点)")

    # sam3_only 模式: 强制启用 SAM3, 禁用 SAM2 和 Ours
    if sam3_only:
        run_sam3 = True
    run_sam2 = not sam3_only if not ablation_mode else True
    run_ours = not sam3_only if not ablation_mode else False

    # ======================== 1. 加载数据 ========================
    # 使用 processed/images/test_jpg 下的标准测试划分
    test_samples = load_cholecseg8k_test(args.data_root, video_filter=video_filter)
    print(f"\n总测试样本数: {len(test_samples)}")

    class_counts = {}
    for s in test_samples:
        cid = s['class_id']
        class_counts[cid] = class_counts.get(cid, 0) + 1
    print("\n类别分布:")
    for cid in sorted(class_counts.keys()):
        print(f"  {cid:2d} - {get_class_name(cid):25s}: {class_counts[cid]} 样本")

    # ======================== 2. 构建模型 ========================
    sam2_predictor = None
    sam3_predictor = None
    our_model = None
    epoch_info = 'N/A'
    best_val = 'N/A'

    # SAM2 Baseline
    if run_sam2:
        sam2_predictor = build_sam2_baseline(args.ckpt_path, device)

    # SAM3 Baseline (可选)
    if run_sam3:
        if os.path.exists(args.sam3_ckpt_path):
            sam3_predictor = build_sam3_baseline(args.sam3_ckpt_path, device)
        else:
            print(f"\n[Warning] SAM3 checkpoint 不存在: {args.sam3_ckpt_path}")
            print(f"  跳过 SAM3 baseline。请下载 sam3.pt 到该路径。")
            run_sam3 = False

    # ScribbleSam2Memory (Ours)
    if run_ours:
        our_model, epoch_info, best_val = build_our_model(args, device)

    video_desc = video_filter if isinstance(video_filter, str) else (
        ','.join(video_filter) if isinstance(video_filter, list) else "all videos")
    n_points = args.n_points
    print(f"\n{'=' * 60}")
    print(f"测试配置:")
    print(f"  Dataset:  CholecSeg8k {video_desc} (OOD, processed test split)")
    print(f"  Rounds:   {num_rounds}")
    print(f"  AMP:      {'ON' if use_amp else 'OFF'}")
    print(f"  Seed:     {args.seed}")
    if ablation_mode:
        print(f"  Mode:     ABLATION (point density sweep)")
        print(f"  Pts/ch:   {ablation_pts_list}")
        if run_sam2:
            print(f"  SAM2:     Tiny ({args.ckpt_path})")
        if run_sam3:
            print(f"  SAM3:     {args.sam3_ckpt_path}")
    else:
        if run_sam2:
            print(f"  SAM2:     Tiny ({args.ckpt_path})")
            print(f"  SAM2 设置: (A) {n_points} pts/ch 采样" + ("  (B) 全部 scribble 像素" if run_all_pixels else ""))
        if run_sam3:
            print(f"  SAM3:     {args.sam3_ckpt_path}")
            print(f"  SAM3 设置: (A) {n_points} pts/ch 采样" + ("  (B) 全部 scribble 像素" if run_sam3_all_pixels else ""))
        if run_ours:
            print(f"  Ours:     {args.model_path}")
    print(f"{'=' * 60}")

    # ======================== 3. 共享 scribble 生成器 ========================
    scribble_gen = TwoChannelScribbleGenerator(neg_prob=0.5)
    correction_gen = CorrectionScribbleGenerator()

    # ImageNet 归一化（ScribbleSam2Memory 需要）
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ======================== 4. 结果容器 ========================
    # 三组实验:
    #   sam2_sampled: SAM2 + 采样 N 个 points/channel
    #   sam2_allpts:  SAM2 + 全部 scribble 像素作为 points
    #   ours:         ScribbleSam2Memory + 完整 scribble map
    def _make_results():
        return {
            'per_class': {
                cid: {f'iou_r{r}': [] for r in range(num_rounds)} |
                     {f'dice_r{r}': [] for r in range(num_rounds)}
                for cid in ALL_CLASSES
            },
            'all': {f'iou_r{r}': [] for r in range(num_rounds)} |
                   {f'dice_r{r}': [] for r in range(num_rounds)},
        }

    all_results = {}
    display_names = {}

    if ablation_mode:
        # 消融模式: 为每个 (model, pts) 创建结果容器
        def _pts_label(pts):
            return "1pt_cc" if pts == 1 else f"{pts}pts"
        def _pts_display(model_upper, pts):
            return f"{model_upper} (1pt/CC)" if pts == 1 else f"{model_upper} ({pts} pts/ch)"

        for pts in ablation_pts_list:
            label = _pts_label(pts)
            if run_sam2:
                key = f'sam2_{label}'
                all_results[key] = _make_results()
                display_names[key] = _pts_display("SAM2 Tiny", pts)
            if run_sam3:
                key = f'sam3_{label}'
                all_results[key] = _make_results()
                display_names[key] = _pts_display("SAM3", pts)
    else:
        # 正常模式
        if run_sam2:
            all_results[f'sam2_{n_points}pts'] = _make_results()
            display_names[f'sam2_{n_points}pts'] = f"SAM2 Tiny ({n_points} pts/ch)"
            if run_all_pixels:
                all_results['sam2_allpts'] = _make_results()
                display_names['sam2_allpts'] = "SAM2 Tiny (all pixels)"
        if run_sam3:
            all_results[f'sam3_{n_points}pts'] = _make_results()
            display_names[f'sam3_{n_points}pts'] = f"SAM3 ({n_points} pts/ch)"
            if run_sam3_all_pixels:
                all_results['sam3_allpts'] = _make_results()
                display_names['sam3_allpts'] = "SAM3 (all pixels)"
        if run_ours:
            all_results['ours'] = _make_results()
            display_names['ours'] = "ScribbleSam2Memory (Ours)"

    error_count = {k: 0 for k in all_results}
    last_r = num_rounds - 1

    # ======================== 5. 逐样本测试 ========================
    # 每个样本使用确定性 seed = base_seed + sample_idx
    # → 三组实验对同一样本获得完全相同的初始 scribble
    H = W = 1024

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(test_samples, desc="OOD Test")):
            img_path = sample['img_path']
            mask_path = sample['mask_path']
            class_id = sample['class_id']

            # ---- 公共数据加载 ----
            image_pil = Image.open(img_path).convert('RGB').resize((W, H), Image.BILINEAR)
            image_np = np.array(image_pil)  # uint8 HWC，SAM2 使用

            mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
            mask_pil = Image.fromarray(mask_rgb).resize((W, H), Image.NEAREST)
            mask_resized = np.array(mask_pil)

            class_mask = color_mask_to_class_mask(mask_resized)
            gt_mask = (class_mask == class_id).astype(np.float32)

            if gt_mask.sum() <= 10:
                continue

            # gt_mask_tensor 所有实验组都需要（scribble 生成用）
            gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)
            # image_tensor 只有 ScribbleSam2Memory 需要
            image_tensor = None
            if run_ours:
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                image_tensor = normalize(image_tensor).unsqueeze(0).to(device)

            sample_seed = args.seed + sample_idx

            if ablation_mode:
                # ============================================================
                # 消融模式: 对每个 model set_image 一次, 遍历所有 pts 设置
                # ============================================================

                def _run_ablation_for_model(predictor, model_tag):
                    """对单个模型跑所有 pts 设置（共享 image encoding）"""
                    if predictor is None:
                        return
                    # set_image 一次（image encoding 共享）
                    with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                        predictor.set_image(image_np)

                    for pts in ablation_pts_list:
                        _set_seed_quiet(sample_seed)
                        n_pts = -1 if pts == 1 else pts
                        label = _pts_label(pts)
                        key = f'{model_tag}_{label}'

                        metrics = test_sam2_single_sample(
                            predictor=predictor,
                            image_np=image_np,
                            gt_mask=gt_mask,
                            gt_mask_tensor=gt_mask_tensor,
                            scribble_gen=scribble_gen,
                            correction_gen=correction_gen,
                            device=device,
                            num_rounds=num_rounds,
                            n_points_per_channel=n_pts,
                            skip_set_image=True,
                        )
                        if metrics is not None:
                            for r, (iou, dice) in enumerate(metrics):
                                all_results[key]['per_class'][class_id][f'iou_r{r}'].append(iou)
                                all_results[key]['per_class'][class_id][f'dice_r{r}'].append(dice)
                                all_results[key]['all'][f'iou_r{r}'].append(iou)
                                all_results[key]['all'][f'dice_r{r}'].append(dice)
                        else:
                            error_count[key] += 1

                if run_sam2:
                    _run_ablation_for_model(sam2_predictor, 'sam2')
                if run_sam3:
                    _run_ablation_for_model(sam3_predictor, 'sam3')

            else:
                # ============================================================
                # 正常模式: 原有的 A/B/C/D 实验组
                # ============================================================

                # ---- (A) SAM2 采样 N pts/ch ----
                if run_sam2:
                    _set_seed_quiet(sample_seed)
                    metrics_a = test_sam2_single_sample(
                        predictor=sam2_predictor,
                        image_np=image_np,
                        gt_mask=gt_mask,
                        gt_mask_tensor=gt_mask_tensor,
                        scribble_gen=scribble_gen,
                        correction_gen=correction_gen,
                        device=device,
                        num_rounds=num_rounds,
                        n_points_per_channel=n_points,
                    )
                    key_a = f'sam2_{n_points}pts'
                    if metrics_a is not None:
                        for r, (iou, dice) in enumerate(metrics_a):
                            all_results[key_a]['per_class'][class_id][f'iou_r{r}'].append(iou)
                            all_results[key_a]['per_class'][class_id][f'dice_r{r}'].append(dice)
                            all_results[key_a]['all'][f'iou_r{r}'].append(iou)
                            all_results[key_a]['all'][f'dice_r{r}'].append(dice)
                    else:
                        error_count[key_a] += 1

                    # ---- (B) SAM2 全部 scribble 像素 (可选) ----
                    if run_all_pixels:
                        _set_seed_quiet(sample_seed)
                        metrics_b = test_sam2_single_sample(
                            predictor=sam2_predictor,
                            image_np=image_np,
                            gt_mask=gt_mask,
                            gt_mask_tensor=gt_mask_tensor,
                            scribble_gen=scribble_gen,
                            correction_gen=correction_gen,
                            device=device,
                            num_rounds=num_rounds,
                            n_points_per_channel=None,
                        )
                        if metrics_b is not None:
                            for r, (iou, dice) in enumerate(metrics_b):
                                all_results['sam2_allpts']['per_class'][class_id][f'iou_r{r}'].append(iou)
                                all_results['sam2_allpts']['per_class'][class_id][f'dice_r{r}'].append(dice)
                                all_results['sam2_allpts']['all'][f'iou_r{r}'].append(iou)
                                all_results['sam2_allpts']['all'][f'dice_r{r}'].append(dice)
                        else:
                            error_count['sam2_allpts'] += 1

                # ---- (C) SAM3 采样 N pts/ch (可选) ----
                if run_sam3:
                    _set_seed_quiet(sample_seed)
                    key_sam3 = f'sam3_{n_points}pts'
                    metrics_sam3 = test_sam2_single_sample(
                        predictor=sam3_predictor,
                        image_np=image_np,
                        gt_mask=gt_mask,
                        gt_mask_tensor=gt_mask_tensor,
                        scribble_gen=scribble_gen,
                        correction_gen=correction_gen,
                        device=device,
                        num_rounds=num_rounds,
                        n_points_per_channel=n_points,
                    )
                    if metrics_sam3 is not None:
                        for r, (iou, dice) in enumerate(metrics_sam3):
                            all_results[key_sam3]['per_class'][class_id][f'iou_r{r}'].append(iou)
                            all_results[key_sam3]['per_class'][class_id][f'dice_r{r}'].append(dice)
                            all_results[key_sam3]['all'][f'iou_r{r}'].append(iou)
                            all_results[key_sam3]['all'][f'dice_r{r}'].append(dice)
                    else:
                        error_count[key_sam3] += 1

                    # ---- (C2) SAM3 全部 scribble 像素 (可选) ----
                    if run_sam3_all_pixels:
                        _set_seed_quiet(sample_seed)
                        metrics_sam3b = test_sam2_single_sample(
                            predictor=sam3_predictor,
                            image_np=image_np,
                            gt_mask=gt_mask,
                            gt_mask_tensor=gt_mask_tensor,
                            scribble_gen=scribble_gen,
                            correction_gen=correction_gen,
                            device=device,
                            num_rounds=num_rounds,
                            n_points_per_channel=None,
                        )
                        if metrics_sam3b is not None:
                            for r, (iou, dice) in enumerate(metrics_sam3b):
                                all_results['sam3_allpts']['per_class'][class_id][f'iou_r{r}'].append(iou)
                                all_results['sam3_allpts']['per_class'][class_id][f'dice_r{r}'].append(dice)
                                all_results['sam3_allpts']['all'][f'iou_r{r}'].append(iou)
                                all_results['sam3_allpts']['all'][f'dice_r{r}'].append(dice)
                        else:
                            error_count['sam3_allpts'] += 1

                # ---- (D) ScribbleSam2Memory (Ours) ----
                if run_ours:
                    _set_seed_quiet(sample_seed)
                    metrics_c = test_ours_single_sample(
                        model=our_model,
                        image_tensor=image_tensor,
                        gt_mask=gt_mask,
                        gt_mask_tensor=gt_mask_tensor,
                        scribble_gen=scribble_gen,
                        correction_gen=correction_gen,
                        device=device,
                        num_rounds=num_rounds,
                        use_amp=use_amp,
                    )
                    if metrics_c is not None:
                        for r, (iou, dice) in enumerate(metrics_c):
                            all_results['ours']['per_class'][class_id][f'iou_r{r}'].append(iou)
                            all_results['ours']['per_class'][class_id][f'dice_r{r}'].append(dice)
                            all_results['ours']['all'][f'iou_r{r}'].append(iou)
                            all_results['ours']['all'][f'dice_r{r}'].append(dice)
                    else:
                        error_count['ours'] += 1

            # ---- 每 100 个样本打印中间结果 ----
            n_done = sample_idx + 1
            if n_done % 100 == 0 or n_done == len(test_samples):
                print(f"\n[Progress {n_done}/{len(test_samples)}] 当前 R{last_r} mIoU (foreground):")
                for key, name in display_names.items():
                    vals = []
                    for cid in FOREGROUND_CLASSES:
                        vals.extend(all_results[key]['per_class'][cid][f'iou_r{last_r}'])
                    avg = np.mean(vals) if vals else 0
                    print(f"  {name:<35} {avg:.4f} (n={len(vals)})")

    print(f"\n推理完成。错误数: {error_count}")

    # ======================== 6. 打印结果 ========================
    for key, name in display_names.items():
        print_results(all_results[key], name, num_rounds)

    # 对比摘要表
    print(f"\n{'=' * 100}")
    print(f"对比摘要 — OOD: CholecSeg8k {video_desc}")
    print(f"{'=' * 100}")

    def _fg_mean(results, metric, r):
        vals = []
        for cid in FOREGROUND_CLASSES:
            vals.extend(results['per_class'][cid][f'{metric}_r{r}'])
        return np.mean(vals) if vals else 0

    # mIoU 对比
    print(f"\nmIoU (Sample-Avg, foreground):")
    header = f"{'Method':<35}"
    for r in range(num_rounds):
        header += f" {'R' + str(r):<10}"
    print(header)
    print("-" * (35 + 10 * num_rounds))
    for key, name in display_names.items():
        row = f"{name:<35}"
        for r in range(num_rounds):
            row += f" {_fg_mean(all_results[key], 'iou', r):<10.4f}"
        print(row)
    print("-" * (35 + 10 * num_rounds))

    # mDice 对比
    print(f"\nmDice (Sample-Avg, foreground):")
    print(header)
    print("-" * (35 + 10 * num_rounds))
    for key, name in display_names.items():
        row = f"{name:<35}"
        for r in range(num_rounds):
            row += f" {_fg_mean(all_results[key], 'dice', r):<10.4f}"
        print(row)
    print("-" * (35 + 10 * num_rounds))

    # Per-class R2 对比
    print(f"\n按类别对比 (R{last_r} mIoU):")
    hdr = f"{'ID':<4} {'Name':<28}"
    for name in display_names.values():
        hdr += f" {name[:16]:<18}"
    print(hdr)
    print("-" * (32 + 18 * len(display_names)))
    for cid in FOREGROUND_CLASSES:
        row = f"{cid:<4} {get_class_name(cid):<28}"
        for key in display_names:
            d = all_results[key]['per_class'][cid]
            if len(d[f'iou_r{last_r}']) > 0:
                row += f" {np.mean(d[f'iou_r{last_r}']):<18.4f}"
            else:
                row += f" {'N/A':<18}"
        print(row)
    print("-" * (32 + 18 * len(display_names)))

    # ======================== 7. 保存结果 ========================
    if args.output_dir is None:
        if ablation_mode:
            args.output_dir = os.path.join(os.path.dirname(args.model_path), 'ablation')
        else:
            args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_tag = args.video.replace(',', '_') if args.video != 'all' else 'all'
    file_prefix = f'ablation_pts_{video_tag}' if ablation_mode else f'eval_cholecseg8k_{video_tag}'

    # === TXT ===
    txt_path = os.path.join(args.output_dir, f'{file_prefix}_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        if ablation_mode:
            f.write(f"Ablation: Point Density Sweep on CholecSeg8k {video_desc}\n")
            f.write(f"Points per channel: {ablation_pts_list}\n")
        else:
            f.write(f"OOD Test: CholecSeg8k {video_desc} (processed test split)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Rounds: {num_rounds}\n")
        f.write(f"AMP: {'ON' if use_amp else 'OFF'}\n")
        if run_sam2:
            if ablation_mode:
                f.write(f"SAM2 settings: ablation pts = {ablation_pts_list}\n")
            else:
                f.write(f"SAM2 settings: (A) {n_points} pts/ch sampled" + (", (B) all scribble pixels" if run_all_pixels else "") + "\n")
            f.write(f"SAM2 ckpt: {args.ckpt_path}\n")
        if run_sam3:
            if ablation_mode:
                f.write(f"SAM3 settings: ablation pts = {ablation_pts_list}\n")
            else:
                f.write(f"SAM3 settings: (A) {n_points} pts/ch sampled" + (", (B) all scribble pixels" if run_sam3_all_pixels else "") + "\n")
            f.write(f"SAM3 ckpt: {args.sam3_ckpt_path}\n")
        if run_ours:
            f.write(f"Ours model: {args.model_path}\n")
            f.write(f"Ours Epoch: {epoch_info}, Val mIoU: {best_val}\n")
        f.write(f"Total samples: {len(test_samples)}\n")
        f.write(f"Errors: {error_count}\n\n")

        for key, model_name in display_names.items():
            results = all_results[key]
            f.write(f"=== {model_name} ===\n")

            # Sample-Avg
            f.write(f"--- Sample-Avg (foreground) ---\n")
            for r in range(num_rounds):
                f.write(f"mIoU_R{r}: {_fg_mean(results, 'iou', r):.4f}\n")
                f.write(f"mDice_R{r}: {_fg_mean(results, 'dice', r):.4f}\n")

            # Class-Avg
            c_classes = [cid for cid in FOREGROUND_CLASSES
                         if len(results['per_class'][cid][f'iou_r{last_r}']) > 0]
            if c_classes:
                f.write(f"\n--- Class-Avg ({len(c_classes)} classes) ---\n")
                for r in range(num_rounds):
                    cr_ious = [np.mean(results['per_class'][cid][f'iou_r{r}']) for cid in c_classes]
                    cr_dices = [np.mean(results['per_class'][cid][f'dice_r{r}']) for cid in c_classes]
                    f.write(f"cIoU_R{r}: {np.mean(cr_ious):.4f}\n")
                    f.write(f"cDice_R{r}: {np.mean(cr_dices):.4f}\n")

            f.write("\n")

            # Per-class
            for cid in ALL_CLASSES:
                d = results['per_class'][cid]
                nc = len(d[f'iou_r{last_r}'])
                if nc > 0:
                    row = f"  {cid:<3} {get_class_name(cid):<28} N={nc:<6}"
                    for r in range(num_rounds):
                        row += f" R{r}={np.mean(d[f'iou_r{r}']):.4f}"
                    row += f" Dice={np.mean(d[f'dice_r{last_r}']):.4f}"
                    f.write(row + "\n")
            f.write("\n")

    # === JSON ===
    json_path = os.path.join(args.output_dir, f'{file_prefix}_{timestamp}.json')
    json_config = {
        'dataset': f'CholecSeg8k_{video_tag}',
        'test_split': 'processed_test',
        'test_type': 'ablation_point_density' if ablation_mode else 'out_of_distribution',
        'seed': args.seed,
        'num_rounds': num_rounds,
        'amp': use_amp,
        'sam2_ckpt': args.ckpt_path,
        'sam3_ckpt': args.sam3_ckpt_path if run_sam3 else None,
    }
    if ablation_mode:
        json_config['ablation_pts'] = ablation_pts_list
    else:
        json_config['sam2_sampled_n_points'] = n_points
        json_config['our_model_path'] = args.model_path
        json_config['our_epoch'] = epoch_info
        json_config['our_val_miou'] = float(best_val) if isinstance(best_val, (int, float)) else str(best_val)
    json_results = {'config': json_config}

    for key in display_names:
        results = all_results[key]

        fg_sample_avg = {}
        for r in range(num_rounds):
            fg_sample_avg[f'mIoU_R{r}'] = float(_fg_mean(results, 'iou', r))
            fg_sample_avg[f'mDice_R{r}'] = float(_fg_mean(results, 'dice', r))

        n_fg = sum(len(results['per_class'][cid][f'iou_r{last_r}']) for cid in FOREGROUND_CLASSES)

        model_json = {
            'overall': {
                'n_fg_samples': n_fg,
                'sample_avg': fg_sample_avg,
            },
            'per_class': {},
        }

        # Class-Avg
        c_classes = [cid for cid in FOREGROUND_CLASSES
                     if len(results['per_class'][cid][f'iou_r{last_r}']) > 0]
        if c_classes:
            model_json['overall']['class_avg'] = {}
            for r in range(num_rounds):
                cr = [np.mean(results['per_class'][cid][f'iou_r{r}']) for cid in c_classes]
                cd = [np.mean(results['per_class'][cid][f'dice_r{r}']) for cid in c_classes]
                model_json['overall']['class_avg'][f'cIoU_R{r}'] = float(np.mean(cr))
                model_json['overall']['class_avg'][f'cDice_R{r}'] = float(np.mean(cd))

        # Per-class
        for cid in ALL_CLASSES:
            d = results['per_class'][cid]
            nc = len(d[f'iou_r{last_r}'])
            if nc > 0:
                model_json['per_class'][get_class_name(cid)] = {
                    'class_id': cid,
                    'n_samples': nc,
                    **{f'IoU_R{r}': float(np.mean(d[f'iou_r{r}'])) for r in range(num_rounds)},
                    **{f'Dice_R{r}': float(np.mean(d[f'dice_r{r}'])) for r in range(num_rounds)},
                }

        json_results[key] = model_json

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n结果已保存:")
    print(f"  TXT:  {txt_path}")
    print(f"  JSON: {json_path}")


if __name__ == '__main__':
    main()

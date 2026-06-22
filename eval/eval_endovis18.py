"""
Stage 2 Memory 模型测试脚本 - 在测试集上测试所有帧的所有类别
使用与 Stage 2 训练完全一致的 scribble 生成、双轨迭代推理、correction 策略

关键对齐项（与 train/train_stage2.py 完全一致）：
1. TwoChannelScribbleGenerator: AdaptiveScribble(正) + LineScribble(负, neg_prob=0.5)
2. CorrectionScribbleGenerator: AdaptiveScribble(FN) + LineScribble(FP)
3. 双轨设计: accumulated(Track 1) + latest(Track 2)
4. R0 不用 Memory, R1+ 使用 Memory
5. forward_single_round 接口
6. mask 阈值 > 10
7. AMP 混合精度

扩展功能：
- --scribble_strategy: 测试时策略消融 (adaptive/centerline_only/wave_only/contour_only)
- --num_rounds: 支持任意轮数 (默认 3, 可设 5)
- --disable_memory / --disable_sgf: 架构消融
- per_sample 逐样本 IoU/Dice 记录

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

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir=os.path.join(os.getcwd(), "sam2", "configs"),
    version_base="1.2",
)

from scissr.models.ScribbleSam2Memory import (
    build_scribble_sam2_memory,
)
from scissr.interactions.scribbles import (
    LineScribble,
    CenterlineScribble,
    ContourScribble,
    WaveSkeletonScribble,
)
from scissr.interactions.adaptive_scribble import (
    AdaptiveScribble,
    AdaptiveConfig,
    CorrectionConfig,
    CenterlineOnlyConfig,
    CenterlineOnlyCorrectionConfig,
    WaveOnlyConfig,
    WaveOnlyCorrectionConfig,
    ContourOnlyConfig,
    ContourOnlyCorrectionConfig,
)


# =============================================================================
# 随机种子
# =============================================================================

def set_seed(seed=42):
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
# 类别映射（与训练完全一致）
# =============================================================================

COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (0, 255, 0): 1,
    (0, 255, 255): 2,
    (125, 255, 12): 3,
    (255, 55, 0): 4,
    (24, 55, 125): 5,
    (187, 155, 25): 6,
    (0, 255, 125): 7,
    (255, 255, 125): 8,
    (123, 15, 175): 9,
    (124, 155, 5): 10,
    (12, 255, 141): 11,
}

ENDOVISION18_CLASSES = {
    0: 'background-tissue', 1: 'instrument-shaft', 2: 'instrument-clasper',
    3: 'instrument-wrist', 4: 'kidney-parenchyma', 5: 'covered-kidney',
    6: 'thread', 7: 'clamps', 8: 'suturing-needle', 9: 'suction-instrument',
    10: 'small-intestine', 11: 'ultrasound-probe',
}

CLASS_TO_COLOR = {v: k for k, v in COLOR_TO_CLASS.items()}
FOREGROUND_CLASSES = list(range(1, 12))


def get_class_name(class_id):
    return ENDOVISION18_CLASSES.get(class_id, f"class_{class_id}")


def rgb_label_to_class_mask(label_rgb):
    H, W, _ = label_rgb.shape
    class_mask = np.zeros((H, W), dtype=np.int32)
    for color, class_id in COLOR_TO_CLASS.items():
        if class_id == 0:
            continue
        match = np.all(label_rgb == color, axis=2)
        class_mask[match] = class_id
    return class_mask


def get_binary_mask_for_class(label_rgb, class_id):
    color = CLASS_TO_COLOR.get(class_id)
    if color is None:
        return np.zeros(label_rgb.shape[:2], dtype=np.float32)
    return np.all(label_rgb == color, axis=2).astype(np.float32)


# =============================================================================
# Scribble 生成器（与 Stage 2 训练完全一致）
# =============================================================================

STRATEGY_CONFIGS = {
    'adaptive': (AdaptiveConfig, CorrectionConfig),
    'centerline_only': (CenterlineOnlyConfig, CenterlineOnlyCorrectionConfig),
    'wave_only': (WaveOnlyConfig, WaveOnlyCorrectionConfig),
    'contour_only': (ContourOnlyConfig, ContourOnlyCorrectionConfig),
}


class TwoChannelScribbleGenerator:
    """与 train/train_stage2.py 完全一致"""

    def __init__(self, neg_prob: float = 0.5, pos_config=None):
        self.pos_generator = AdaptiveScribble(config=pos_config) if pos_config else AdaptiveScribble()
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
    """与 train/train_stage2.py 完全一致"""

    def __init__(self, fn_config=None, strategy='adaptive'):
        self.fn_generator = AdaptiveScribble(config=fn_config) if fn_config else AdaptiveScribble(config=CorrectionConfig())
        if strategy == 'adaptive':
            self.fp_generator = LineScribble(thickness=3, warp=False)
        elif strategy == 'centerline_only':
            self.fp_generator = CenterlineScribble(dilate_kernel_size=3)
        elif strategy == 'wave_only':
            self.fp_generator = WaveSkeletonScribble(thickness=(3, 4))
        elif strategy == 'contour_only':
            self.fp_generator = ContourScribble(dilate_kernel_size=3)
        else:
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
# 指标计算（numpy，与旧脚本完全一致）
# =============================================================================

def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)
    return iou, dice


# =============================================================================
# 核心测试函数（保留旧脚本的直接文件遍历结构，不用 DataLoader）
# =============================================================================

def test_model(model, test_dir, device, scribble_gen, correction_gen,
               use_amp=True, num_rounds=3, disable_memory=False):
    # ======================== 1. 收集测试样本 ========================
    test_samples = []
    seq_dirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    print(f"\n加载测试数据 (RGB 模式)...")
    for seq_name in tqdm(seq_dirs, desc="扫描序列"):
        seq_path = os.path.join(test_dir, seq_name)
        img_dir = os.path.join(seq_path, 'left_frames')
        label_dir = os.path.join(seq_path, 'labels')
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            continue
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.endswith(('.png', '.jpg')):
                continue
            img_path = os.path.join(img_dir, img_name)
            label_name = img_name.replace('.jpg', '.png')
            label_path = os.path.join(label_dir, label_name)
            if not os.path.exists(label_path):
                continue
            label_rgb = np.array(Image.open(label_path).convert('RGB'))
            class_mask = rgb_label_to_class_mask(label_rgb)
            foreground_classes = [int(c) for c in np.unique(class_mask) if c > 0]
            for class_id in foreground_classes:
                binary_mask = (class_mask == class_id).astype(np.float32)
                if binary_mask.sum() > 10:
                    test_samples.append({
                        'img_path': img_path,
                        'label_rgb': label_rgb,
                        'class_id': class_id,
                        'seq_name': seq_name,
                        'frame_name': img_name,
                    })

    print(f"总测试样本数: {len(test_samples)}")
    class_counts = {}
    for sample in test_samples:
        cid = sample['class_id']
        class_counts[cid] = class_counts.get(cid, 0) + 1
    print("\n类别分布:")
    for cid in sorted(class_counts.keys()):
        print(f"  {cid:2d} - {get_class_name(cid):20s}: {class_counts[cid]} 样本")

    # 结果容器
    results = {
        'per_class': {
            cid: {f'iou_r{r}': [] for r in range(num_rounds)} |
                 {f'dice_r{r}': [] for r in range(num_rounds)}
            for cid in FOREGROUND_CLASSES
        },
        'all': {f'iou_r{r}': [] for r in range(num_rounds)} |
               {f'dice_r{r}': [] for r in range(num_rounds)},
    }
    per_sample_records = []

    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ======================== 2. 逐样本推理 ========================
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="测试中"):
            img_path = sample['img_path']
            label_rgb = sample['label_rgb']
            class_id = sample['class_id']

            image_pil = Image.open(img_path).convert('RGB').resize((1024, 1024), Image.BILINEAR)
            label_pil = Image.fromarray(label_rgb).resize((1024, 1024), Image.NEAREST)

            image_np = np.array(image_pil)
            label_rgb_resized = np.array(label_pil)

            H, W = 1024, 1024
            gt_mask = get_binary_mask_for_class(label_rgb_resized, class_id)

            if gt_mask.sum() <= 10:
                continue

            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = normalize(image_tensor).unsqueeze(0).to(device)
            gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)

            if gt_mask_tensor.sum() > 10:
                initial_scribble = scribble_gen(gt_mask_tensor, n_scribbles=1).to(device)
            else:
                initial_scribble = torch.zeros(1, 2, H, W, device=device)

            try:
                with autocast('cuda', enabled=use_amp):
                    model.reset_memory_bank()

                    backbone_out = model.forward_image(image_tensor)
                    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

                    cache = {
                        'backbone_out': backbone_out,
                        'vision_feats': vision_feats,
                        'vision_pos_embeds': vision_pos_embeds,
                        'feat_sizes': feat_sizes,
                    }

                    accumulated_scribble = initial_scribble
                    latest_scribble = initial_scribble

                    round_outputs = {}

                    for round_idx in range(num_rounds):
                        if round_idx == 0:
                            low_res, high_res, cache = model.forward_single_round(
                                image=image_tensor,
                                latest_scribble=latest_scribble,
                                accumulated_scribble=accumulated_scribble,
                                box=None,
                                use_memory=False,
                                update_memory=(not disable_memory),
                                **cache,
                            )
                        else:
                            prev_high_res = round_outputs[f'masks_{round_idx-1}']
                            pos_correction, neg_correction = correction_gen.generate(
                                pred_mask=prev_high_res,
                                gt_mask=gt_mask_tensor,
                            )

                            latest_scribble = torch.cat([pos_correction, neg_correction], dim=1)

                            if accumulated_scribble is not None:
                                accumulated_scribble = torch.stack([
                                    torch.max(accumulated_scribble[:, 0], pos_correction.squeeze(1)),
                                    torch.max(accumulated_scribble[:, 1], neg_correction.squeeze(1)),
                                ], dim=1)
                            else:
                                accumulated_scribble = latest_scribble.clone()

                            use_mem = not disable_memory
                            update_mem = (not disable_memory) and (round_idx < num_rounds - 1)
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

                    # ======== 计算每轮指标（numpy，与旧脚本一致）========
                    sample_ious = []
                    sample_dices = []
                    for r in range(num_rounds):
                        pred = torch.sigmoid(round_outputs[f'masks_{r}'][0, 0]).cpu().numpy()
                        pred_binary = (pred > 0.5).astype(np.float32)
                        iou, dice = compute_metrics(pred_binary, gt_mask)

                        results['per_class'][class_id][f'iou_r{r}'].append(iou)
                        results['per_class'][class_id][f'dice_r{r}'].append(dice)
                        results['all'][f'iou_r{r}'].append(iou)
                        results['all'][f'dice_r{r}'].append(dice)
                        sample_ious.append(float(iou))
                        sample_dices.append(float(dice))

                    per_sample_records.append({
                        'seq': sample['seq_name'],
                        'frame': sample['frame_name'],
                        'class_name': get_class_name(class_id),
                        'class_id': class_id,
                        'iou': sample_ious,
                        'dice': sample_dices,
                    })

            except Exception as e:
                print(f"\n[Warning] Error: {sample['frame_name']} class={get_class_name(class_id)}: {e}")
                continue

    return results, per_sample_records


# =============================================================================
# 模型构建与加载
# =============================================================================

def build_and_load_model(args, device):
    disable_memory = getattr(args, 'disable_memory', False)
    disable_sgf = getattr(args, 'disable_sgf', False)

    if disable_memory or disable_sgf:
        parts = []
        if disable_memory:
            parts.append("Memory DISABLED")
        if disable_sgf:
            parts.append("SGF DISABLED")
        print(f"\n[Ablation] {', '.join(parts)}")

    print("\n[Build Model] Creating ScribbleSam2Memory...")
    model = build_scribble_sam2_memory(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device=str(device),
        scribble_channels=2,
    )

    print("[Build Model] Enabling Mask Decoder LoRA...")
    model.enable_mask_decoder_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    if not disable_memory:
        print("[Build Model] Enabling Memory Attention LoRA...")
        model.enable_memory_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    else:
        print("[Build Model] Memory LoRA SKIPPED (--disable_memory)")

    model.freeze_pretrained()

    print(f"\n[Load Weights] Loading from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)

    epoch_info = ckpt.get('epoch', 'N/A')
    best_val = ckpt.get('best_val_iou', 'N/A')

    model_state = model.state_dict()
    loaded_count = 0
    for name, param in ckpt['model_state_dict'].items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded_count += 1

    model.load_state_dict(model_state, strict=False)

    print(f"\n  模型加载成功: Epoch={epoch_info}, Val mIoU={best_val}")
    print(f"  已加载 {loaded_count} 个参数")

    if disable_sgf:
        print("\n[Ablation] Freezing SpatialGatedFusion (alpha=0)")
        with torch.no_grad():
            model.query_fusion.alpha.fill_(0.0)
        for param in model.query_fusion.parameters():
            param.requires_grad = False
    else:
        alpha_value = model.query_fusion.alpha.item()
        print(f"  Alpha (Soft Gate): {alpha_value:.4f}")

    model.eval()
    return model, epoch_info, best_val


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Stage 2 Memory Model - All Classes')

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_file', type=str, default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/sam2.1_hiera_tiny.pt')

    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)

    parser.add_argument('--test_dir', type=str,
                        default='dataset/Endovision18/raw/Test_Data')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')

    parser.add_argument('--disable_memory', action='store_true')
    parser.add_argument('--disable_sgf', action='store_true')

    parser.add_argument('--scribble_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'centerline_only', 'wave_only', 'contour_only'])

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, epoch_info, best_val = build_and_load_model(args, device)

    use_amp = not args.no_amp
    print(f"AMP: {'ON' if use_amp else 'OFF'}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Strategy: {args.scribble_strategy}")
    print(f"Memory: {'DISABLED' if args.disable_memory else 'ENABLED'}")
    print(f"SGF: {'DISABLED' if args.disable_sgf else 'ENABLED'}")

    # --- Scribble generators ---
    pos_config_cls, corr_config_cls = STRATEGY_CONFIGS[args.scribble_strategy]
    if args.scribble_strategy == 'adaptive':
        scribble_gen = TwoChannelScribbleGenerator(neg_prob=0.5)
        correction_gen = CorrectionScribbleGenerator()
    else:
        scribble_gen = TwoChannelScribbleGenerator(neg_prob=0.5, pos_config=pos_config_cls())
        correction_gen = CorrectionScribbleGenerator(
            fn_config=corr_config_cls(),
            strategy=args.scribble_strategy,
        )

    results, per_sample_records = test_model(
        model, args.test_dir, device, scribble_gen, correction_gen,
        use_amp=use_amp, num_rounds=args.num_rounds,
        disable_memory=args.disable_memory,
    )

    num_rounds = args.num_rounds
    last_r = num_rounds - 1

    # ==================== 打印结果 ====================
    if args.disable_memory and args.disable_sgf:
        exp_name = "Baseline (no SGF, no Memory)"
    elif args.disable_memory:
        exp_name = "SGF Only (no Memory)"
    elif args.disable_sgf:
        exp_name = "Memory Only (no SGF)"
    else:
        exp_name = "Full Model (SGF + Memory)"

    print("\n" + "=" * 90)
    print(f"Stage 2 测试结果 - {exp_name} - Strategy: {args.scribble_strategy}")
    print("=" * 90)

    n_total = len(results['all'][f'iou_r{last_r}'])
    if n_total == 0:
        print("[Error] 没有有效的测试样本！")
        return

    # Sample-Avg
    print(f"\nSample-Avg (N={n_total}):")
    iou_strs = [f"R{r}={np.mean(results['all'][f'iou_r{r}']):.4f}" for r in range(num_rounds)]
    dice_strs = [f"R{r}={np.mean(results['all'][f'dice_r{r}']):.4f}" for r in range(num_rounds)]
    print(f"  mIoU:  {', '.join(iou_strs)}")
    print(f"  mDice: {', '.join(dice_strs)}")

    # Class-Avg
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
        print(f"\nClass-Avg ({n_classes} classes):")
        ciou_strs = [f"R{r}={np.mean(class_ious[r]):.4f}" for r in range(num_rounds)]
        cdice_strs = [f"R{r}={np.mean(class_dices[r]):.4f}" for r in range(num_rounds)]
        print(f"  cIoU:  {', '.join(ciou_strs)}")
        print(f"  cDice: {', '.join(cdice_strs)}")

    # Per-class
    print(f"\nPer-Class Detail:")
    header = f"{'ID':<5} {'Name':<25} {'N':<8}"
    for r in range(num_rounds):
        header += f" {'R'+str(r)+'_IoU':<10} {'R'+str(r)+'_Dice':<11}"
    print(header)
    print("-" * len(header))

    for cid in FOREGROUND_CLASSES:
        d = results['per_class'][cid]
        n = len(d[f'iou_r{last_r}'])
        if n > 0:
            row = f"{cid:<5} {get_class_name(cid):<25} {n:<8}"
            for r in range(num_rounds):
                row += f" {np.mean(d[f'iou_r{r}']):<10.4f} {np.mean(d[f'dice_r{r}']):<11.4f}"
            print(row)
    print("-" * len(header))

    # ==================== 保存结果 ====================
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_tag = f"_{args.scribble_strategy}" if args.scribble_strategy != 'adaptive' else ""
    rounds_tag = f"_R{num_rounds}" if num_rounds != 3 else ""

    # JSON
    per_class_summary = {}
    for cid in FOREGROUND_CLASSES:
        d = results['per_class'][cid]
        n = len(d[f'iou_r{last_r}'])
        if n > 0:
            entry = {'class_id': cid, 'n_samples': n}
            for r in range(num_rounds):
                entry[f'IoU_R{r}'] = float(np.mean(d[f'iou_r{r}']))
                entry[f'Dice_R{r}'] = float(np.mean(d[f'dice_r{r}']))
            per_class_summary[get_class_name(cid)] = entry

    sample_avg = {}
    class_avg_dict = {}
    for r in range(num_rounds):
        sample_avg[f'mIoU_R{r}'] = float(np.mean(results['all'][f'iou_r{r}']))
        sample_avg[f'mDice_R{r}'] = float(np.mean(results['all'][f'dice_r{r}']))
        if n_classes > 0:
            class_avg_dict[f'cIoU_R{r}'] = float(np.mean(class_ious[r]))
            class_avg_dict[f'cDice_R{r}'] = float(np.mean(class_dices[r]))

    json_results = {
        'config': {
            'model_path': args.model_path,
            'epoch': epoch_info,
            'val_miou_from_ckpt': float(best_val) if isinstance(best_val, (int, float)) else str(best_val),
            'seed': args.seed,
            'num_rounds': num_rounds,
            'amp': use_amp,
            'scribble_strategy': args.scribble_strategy,
            'disable_memory': args.disable_memory,
            'disable_sgf': args.disable_sgf,
            'timestamp': timestamp,
        },
        'overall': {
            'n_samples': n_total,
            'sample_avg': sample_avg,
            'class_avg': class_avg_dict,
        },
        'per_class': per_class_summary,
        'per_sample': per_sample_records,
    }

    json_path = os.path.join(args.output_dir, f'test_stage2_{timestamp}{strategy_tag}{rounds_tag}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # TXT
    txt_path = json_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Stage 2 测试结果 - {exp_name} - Strategy: {args.scribble_strategy}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Epoch: {epoch_info}\n")
        f.write(f"Val mIoU: {best_val}\n")
        f.write(f"Seed: {args.seed}, Rounds: {num_rounds}, AMP: {'ON' if use_amp else 'OFF'}\n")
        f.write(f"Strategy: {args.scribble_strategy}\n")
        f.write(f"Memory: {'DISABLED' if args.disable_memory else 'ENABLED'}\n")
        f.write(f"SGF: {'DISABLED' if args.disable_sgf else 'ENABLED'}\n")
        f.write(f"Samples: {n_total}\n\n")

        f.write("Sample-Avg:\n")
        for r in range(num_rounds):
            f.write(f"  R{r}: mIoU={sample_avg[f'mIoU_R{r}']:.4f}  mDice={sample_avg[f'mDice_R{r}']:.4f}\n")

        if n_classes > 0:
            f.write(f"\nClass-Avg ({n_classes} classes):\n")
            for r in range(num_rounds):
                f.write(f"  R{r}: cIoU={class_avg_dict[f'cIoU_R{r}']:.4f}  cDice={class_avg_dict[f'cDice_R{r}']:.4f}\n")

        f.write(f"\nPer-Class:\n")
        f.write(f"{'Class':<25s} {'N':>5s}")
        for r in range(num_rounds):
            f.write(f"  {'IoU_R'+str(r):>10s}  {'Dice_R'+str(r):>10s}")
        f.write("\n" + "-" * 80 + "\n")
        for cid in FOREGROUND_CLASSES:
            d = results['per_class'][cid]
            n = len(d[f'iou_r{last_r}'])
            if n > 0:
                f.write(f"{get_class_name(cid):<25s} {n:>5d}")
                for r in range(num_rounds):
                    f.write(f"  {np.mean(d[f'iou_r{r}']):>10.4f}  {np.mean(d[f'dice_r{r}']):>10.4f}")
                f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"TXT saved: {txt_path}")

    final_iou = np.mean(results['all'][f'iou_r{last_r}'])
    final_dice = np.mean(results['all'][f'dice_r{last_r}'])
    print(f"\nFINAL (R{last_r}): mIoU={final_iou:.4f}  mDice={final_dice:.4f}")


if __name__ == '__main__':
    main()

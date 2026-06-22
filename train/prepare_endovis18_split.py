#!/usr/bin/env python3
"""
Endovision18 数据集分析与划分脚本

功能：
1. 统计每个序列中的类别分布
2. 找出稀有类别所在的序列
3. 生成确保类别覆盖的 train/val split
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ============================================================
# 类别定义
# ============================================================
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

CLASS_NAMES = {
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

def analyze_sequence(seq_dir: Path) -> Dict[int, int]:
    """分析单个序列中每个类别出现的帧数"""
    class_counts = {i: 0 for i in range(12)}
    labels_dir = seq_dir / 'labels'
    
    if not labels_dir.exists():
        return class_counts
    
    for label_path in sorted(labels_dir.glob('*.png')):
        label_rgb = np.array(Image.open(label_path).convert('RGB'))
        
        for color, class_id in COLOR_TO_CLASS.items():
            if class_id == 0:  # 跳过背景
                continue
            if np.any(np.all(label_rgb == color, axis=2)):
                class_counts[class_id] += 1
    
    return class_counts


def get_sequence_frame_count(seq_dir: Path) -> int:
    """获取序列的总帧数"""
    images_dir = seq_dir / 'left_frames'
    if not images_dir.exists():
        return 0
    return len(list(images_dir.glob('*.png')))


def main():
    data_dir = Path('dataset/Endovision18/raw/Train_Data')
    output_dir = Path('dataset/Endovision18')
    
    # ============================================================
    # Step 1: 分析每个序列
    # ============================================================
    print("=" * 80)
    print("Step 1: 分析每个序列的类别分布")
    print("=" * 80)
    
    sequence_stats = {}
    all_seqs = sorted([d.name for d in data_dir.glob('seq_*')])
    
    for seq_name in all_seqs:
        seq_dir = data_dir / seq_name
        class_counts = analyze_sequence(seq_dir)
        frame_count = get_sequence_frame_count(seq_dir)
        sequence_stats[seq_name] = {
            'class_counts': class_counts,
            'total_frames': frame_count,
            'classes_present': [cid for cid, cnt in class_counts.items() if cnt > 0 and cid != 0]
        }
    
    # 打印每个序列的类别统计
    print(f"\n{'序列':<10} {'总帧数':>8} | 包含的前景类别")
    print("-" * 80)
    
    for seq_name in all_seqs:
        stats = sequence_stats[seq_name]
        classes_str = ", ".join([CLASS_NAMES[c] for c in sorted(stats['classes_present'])])
        print(f"{seq_name:<10} {stats['total_frames']:>8} | {classes_str}")
    
    # ============================================================
    # Step 2: 找出稀有类别所在的序列
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 2: 稀有类别分析")
    print("=" * 80)
    
    # 找出每个类别出现在哪些序列
    class_to_seqs: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    
    for seq_name, stats in sequence_stats.items():
        for class_id, count in stats['class_counts'].items():
            if count > 0 and class_id != 0:
                class_to_seqs[class_id].append((seq_name, count))
    
    # 打印每个类别的分布
    print(f"\n{'类别':<25} {'总帧数':>8} | 出现的序列")
    print("-" * 80)
    
    for class_id in range(1, 12):
        seqs_with_class = class_to_seqs[class_id]
        total_frames = sum(cnt for _, cnt in seqs_with_class)
        seqs_str = ", ".join([f"{s}({c})" for s, c in sorted(seqs_with_class)])
        
        # 标记稀有类别
        marker = " ⚠️ 稀有!" if len(seqs_with_class) <= 2 else ""
        print(f"{CLASS_NAMES[class_id]:<25} {total_frames:>8} | {seqs_str}{marker}")
    
    # 特别关注的类别
    print("\n关键发现:")
    suturing_seqs = [s for s, _ in class_to_seqs[8]]  # suturing-needle
    ultrasound_seqs = [s for s, _ in class_to_seqs[11]]  # ultrasound-probe
    
    print(f"  - suturing-needle (class 8) 仅出现在: {suturing_seqs}")
    print(f"  - ultrasound-probe (class 11) 仅出现在: {ultrasound_seqs}")
    
    # ============================================================
    # Step 3: 智能划分 Train/Val
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 3: 生成 Train/Val 划分")
    print("=" * 80)
    
    # 计算总帧数
    total_frames = sum(stats['total_frames'] for stats in sequence_stats.values())
    target_val_frames = total_frames / 5  # 目标 4:1 比例
    
    print(f"\n总帧数: {total_frames}")
    print(f"目标验证集帧数: ~{target_val_frames:.0f} (约 20%)")
    
    # 划分策略：
    # 1. 确保 suturing-needle 的序列在训练集
    # 2. 确保 ultrasound-probe 的序列在训练集
    # 3. 从剩余序列中选择适当数量作为验证集
    
    # 必须在训练集中的序列（包含稀有类别）
    must_train_seqs = set(suturing_seqs) | set(ultrasound_seqs)
    print(f"\n必须保留在训练集的序列（含稀有类别）: {sorted(must_train_seqs)}")
    
    # 可选的序列
    optional_seqs = [s for s in all_seqs if s not in must_train_seqs]
    print(f"可用于验证集的候选序列: {optional_seqs}")
    
    # 按帧数排序可选序列，选择合适的作为验证集
    optional_seqs_sorted = sorted(optional_seqs, 
                                   key=lambda s: sequence_stats[s]['total_frames'],
                                   reverse=True)
    
    # 贪心选择验证集序列，尽量接近目标比例
    val_seqs = []
    val_frames = 0
    
    for seq in optional_seqs_sorted:
        seq_frames = sequence_stats[seq]['total_frames']
        if val_frames + seq_frames <= target_val_frames * 1.2:  # 允许 20% 的余量
            val_seqs.append(seq)
            val_frames += seq_frames
        
        # 至少选择 2-3 个序列作为验证集
        if len(val_seqs) >= 3 and val_frames >= target_val_frames * 0.8:
            break
    
    train_seqs = [s for s in all_seqs if s not in val_seqs]
    train_frames = sum(sequence_stats[s]['total_frames'] for s in train_seqs)
    
    print(f"\n最终划分:")
    print(f"  训练集: {len(train_seqs)} 个序列, {train_frames} 帧")
    print(f"  验证集: {len(val_seqs)} 个序列, {val_frames} 帧")
    print(f"  比例: {train_frames/val_frames:.2f}:1")
    
    # ============================================================
    # Step 4: 验证划分的类别覆盖
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 4: 验证类别覆盖")
    print("=" * 80)
    
    train_class_counts = {i: 0 for i in range(12)}
    val_class_counts = {i: 0 for i in range(12)}
    
    for seq in train_seqs:
        for cid, cnt in sequence_stats[seq]['class_counts'].items():
            train_class_counts[cid] += cnt
    
    for seq in val_seqs:
        for cid, cnt in sequence_stats[seq]['class_counts'].items():
            val_class_counts[cid] += cnt
    
    print(f"\n{'类别':<25} {'训练集':>10} {'验证集':>10} | 状态")
    print("-" * 70)
    
    all_covered = True
    for class_id in range(1, 12):
        train_cnt = train_class_counts[class_id]
        val_cnt = val_class_counts[class_id]
        
        if train_cnt == 0:
            status = "❌ 训练集缺失!"
            all_covered = False
        elif val_cnt == 0:
            status = "⚠️ 验证集缺失 (可接受)"
        else:
            status = "✅"
        
        print(f"{CLASS_NAMES[class_id]:<25} {train_cnt:>10} {val_cnt:>10} | {status}")
    
    # ============================================================
    # Step 5: 保存 Split 文件
    # ============================================================
    print("\n" + "=" * 80)
    print("Step 5: 保存划分文件")
    print("=" * 80)
    
    split_data = {
        'train_sequences': sorted(train_seqs),
        'val_sequences': sorted(val_seqs),
        'statistics': {
            'total_sequences': len(all_seqs),
            'train_sequences_count': len(train_seqs),
            'val_sequences_count': len(val_seqs),
            'train_frames': train_frames,
            'val_frames': val_frames,
            'train_val_ratio': f"{train_frames/val_frames:.2f}:1"
        },
        'train_class_counts': {CLASS_NAMES[k]: v for k, v in train_class_counts.items() if k != 0},
        'val_class_counts': {CLASS_NAMES[k]: v for k, v in val_class_counts.items() if k != 0},
        'notes': [
            "划分原则: 按序列划分，不打乱帧顺序",
            "suturing-needle 必须在训练集中",
            "ultrasound-probe 必须在训练集中",
            f"目标比例 4:1，实际比例 {train_frames/val_frames:.2f}:1"
        ]
    }
    
    output_path = output_dir / 'train_val_split.json'
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 划分文件已保存到: {output_path}")
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    print(f"\n训练集序列 ({len(train_seqs)}): {sorted(train_seqs)}")
    print(f"验证集序列 ({len(val_seqs)}): {sorted(val_seqs)}")
    
    return split_data


if __name__ == '__main__':
    main()


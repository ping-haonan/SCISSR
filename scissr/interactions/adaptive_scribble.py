"""
Adaptive Scribble Generator

Automatically selects the optimal scribble strategy based on mask geometry,
generates scribbles per connected component with adaptive thickness.

Strategy Selection Logic:
- Thin structures (aspect_ratio > 4): Centerline (preserves elongated shape)
- Complex shapes (compactness < 0.6): Centerline or Contour (handles intricate boundaries)
- Simple/Compact shapes (compactness >= 0.6): WaveSkeleton or Contour (room for wave oscillations)
- Small targets: Centerline (most reliable for small regions)

Adaptive Thickness:
- Uses skeleton + distance transform to estimate typical object width per component
- Uses 25th percentile (conservative) to prevent overflow at narrow parts
- Clamped to [min_thickness, max_thickness]

Author: ScribblePrompt Team
"""

from typing import Union, Tuple, List, Optional, Dict
from warnings import warn
import numpy as np
import torch
import random
import cv2

from .geometry import GeometryAnalyzer
from .scribbles import (
    WarpScribble,
    LineScribble,
    CenterlineScribble, 
    ContourScribble,
    WaveSkeletonScribble,
)


# =============================================================================
# Adaptive Thickness
# =============================================================================

def get_adaptive_thickness(mask: np.ndarray, 
                           min_th: int = 2, 
                           max_th: int = 10,
                           safety_margin: float = 0.7,
                           percentile: float = 25) -> int:
    """
    根据连通域的形态学宽度自动计算 scribble 粗细。
    
    使用骨架 + 距离变换，取骨架上距离值的保守分位数，
    确保 scribble 不会溢出到目标边界外。
    
    Args:
        mask: (H, W) uint8 二值 mask (0 or 255)
        min_th: 最小粗细（防止 Encoder 看不见）
        max_th: 最大粗细（防止过粗）
        safety_margin: 安全系数（0.7 表示只占物体宽度的 70%）
        percentile: 分位数（25 = 偏保守，照顾细的部分）
    
    Returns:
        int: 推荐的 thickness
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    if np.sum(mask > 0) == 0:
        return min_th
    
    # 距离变换：每个前景像素到最近背景像素的距离
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 提取骨架（只统计 scribble 真正会经过的位置的宽度）
    skeleton = cv2.ximgproc.thinning(mask)
    skel_points = skeleton > 0
    
    if np.sum(skel_points) == 0:
        # 骨架为空（极小目标），用整体最大半径
        max_radius = np.max(dist_map)
        target = max_radius * 2 * safety_margin
        return int(np.clip(round(target), min_th, max_th))
    
    # 取骨架上距离值的保守分位数
    skel_distances = dist_map[skel_points]
    conservative_radius = np.percentile(skel_distances, percentile)
    
    # 物体典型宽度 = 半径 × 2
    object_width = conservative_radius * 2
    target_thickness = object_width * safety_margin
    
    return int(np.clip(round(target_thickness), min_th, max_th))


# =============================================================================
# Configuration
# =============================================================================

class AdaptiveConfig:
    """Configuration for adaptive scribble selection."""
    
    # Area thresholds
    LARGE_AREA_THRESHOLD: int = 50000
    MEDIUM_AREA_THRESHOLD: int = 3000
    MIN_AREA_THRESHOLD: int = 10        # 低于 10px 不生成 scribble（避免噪声）
    MIN_COMPONENT_AREA: int = 5         # 低于 5px 的连通域跳过（省去骨架提取开销）
    
    # Geometry thresholds
    THIN_ASPECT_RATIO: float = 4.0      # Above this → thin structure
    COMPACT_THRESHOLD: float = 0.4       # Above this → simple shape, below → complex
    
    # GT Preprocessing (remove annotation noise)
    # 关闭预处理：前景类标注质量较好，预处理反而可能损害细目标（thread等）
    ENABLE_MASK_PREPROCESSING: bool = False
    CLOSING_KERNEL_SIZE: int = 7        # 闭运算核大小（填充小缝隙）
    OPENING_KERNEL_SIZE: int = 5        # 开运算核大小（去除小噪点）
    EDGE_MARGIN: int = 5                # Ignore GT pixels near image edges
    
    # Adaptive thickness
    MIN_THICKNESS: int = 2
    MAX_THICKNESS: int = 4
    THICKNESS_SAFETY_MARGIN: float = 0.7
    THICKNESS_PERCENTILE: float = 25     # 25th percentile (conservative)
    
    # Strategy weights for random selection
    SIMPLE_SHAPE_WEIGHTS = {
        'wave_skeleton': 0.7,
        'contour': 0.3,
    }
    
    COMPLEX_SHAPE_WEIGHTS = {
        'centerline': 0.6,
        'contour': 0.4,
    }
    
    THIN_STRUCTURE_WEIGHTS = {
        'centerline': 1.0,
    }
    
    # Line removed: too random for small targets, Centerline always better
    SMALL_TARGET_WEIGHTS = {
        'centerline': 1.0,
    }


# =============================================================================
# Adaptive Scribble Generator
# =============================================================================

class AdaptiveScribble(WarpScribble):
    """
    Adaptive Scribble Generator
    
    Features:
    1. Geometry-based strategy selection (per component)
    2. Adaptive thickness based on skeleton + distance transform
    3. Per-component scribble generation (ensures all components are covered)
    4. Mask preprocessing (closing + opening + small component filtering)
    
    Usage:
        generator = AdaptiveScribble()
        scribbles = generator(mask)  # mask: (b, 1, H, W) or (1, H, W)
    """
    
    def __init__(self,
                 config: Optional[AdaptiveConfig] = None,
                 # Warp settings (shared across all strategies)
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 # Debug
                 verbose: bool = False,
                 ):
        super().__init__(
            warp=warp,
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        
        self.config = config or AdaptiveConfig()
        self.verbose = verbose
        
        # Shared warp kwargs for creating strategy instances on-the-fly
        self._warp_kwargs = {
            'warp': warp,
            'warp_smoothing': warp_smoothing,
            'warp_magnitude': warp_magnitude,
            'mask_smoothing': mask_smoothing,
        }
    
    def _create_strategy(self, strategy_name: str, thickness: int):
        """
        创建指定粗细的策略实例。
        每次按需创建，因为不同连通域的粗细不同。
        """
        kwargs = self._warp_kwargs.copy()
        
        if strategy_name == 'line':
            return LineScribble(thickness=thickness, **kwargs)
        elif strategy_name == 'centerline':
            return CenterlineScribble(dilate_kernel_size=thickness, **kwargs)
        elif strategy_name == 'contour':
            return ContourScribble(dilate_kernel_size=thickness, **kwargs)
        elif strategy_name == 'wave_skeleton':
            return WaveSkeletonScribble(thickness=(thickness, thickness + 1), **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        对 GT mask 进行形态学预处理：
        1. 清除边缘像素
        2. 闭运算（膨胀→腐蚀）填充小缝隙
        3. 开运算（腐蚀→膨胀）去除小噪点
        4. 过滤小连通域
        
        Args:
            mask: (H, W) uint8 mask (0 or 255)
            
        Returns:
            Preprocessed mask
        """
        if not self.config.ENABLE_MASK_PREPROCESSING:
            return mask
        
        H, W = mask.shape
        processed = mask.copy()
        
        # Step 1: Clear edge pixels
        margin = self.config.EDGE_MARGIN
        if margin > 0:
            processed[:margin, :] = 0
            processed[-margin:, :] = 0
            processed[:, :margin] = 0
            processed[:, -margin:] = 0
        
        # Step 2: 闭运算（填充小缝隙）
        closing_kernel = np.ones(
            (self.config.CLOSING_KERNEL_SIZE, self.config.CLOSING_KERNEL_SIZE), np.uint8
        )
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, closing_kernel)
        
        # Step 3: 开运算（去除小噪点）
        opening_kernel = np.ones(
            (self.config.OPENING_KERNEL_SIZE, self.config.OPENING_KERNEL_SIZE), np.uint8
        )
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, opening_kernel)
        
        # Step 4: 过滤小连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
        filtered = np.zeros_like(processed)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.MIN_COMPONENT_AREA:
                filtered[labels == i] = 255
        
        return filtered
    
    def select_strategy(self, geometry: Dict) -> str:
        """
        Select the optimal strategy based on mask geometry.
        
        Args:
            geometry: Dict with keys: area, aspect_ratio, compactness, solidity
            
        Returns:
            strategy_name: One of 'wave_skeleton', 'centerline', 'contour', 'line'
        """
        area = geometry['area']
        aspect_ratio = geometry['aspect_ratio']
        compactness = geometry['compactness']
        
        # 1. Thin structures → Centerline
        if aspect_ratio > self.config.THIN_ASPECT_RATIO:
            weights = self.config.THIN_STRUCTURE_WEIGHTS
            if self.verbose:
                print(f"    [Thin] AR={aspect_ratio:.1f} → Centerline")
        
        # 2. Small targets → Centerline or Line
        elif area < self.config.MEDIUM_AREA_THRESHOLD:
            weights = self.config.SMALL_TARGET_WEIGHTS
            if self.verbose:
                print(f"    [Small] Area={area} → Centerline/Line")
        
        # 3. Complex shapes → Centerline or Contour
        elif compactness < self.config.COMPACT_THRESHOLD:
            weights = self.config.COMPLEX_SHAPE_WEIGHTS
            if self.verbose:
                print(f"    [Complex] Comp={compactness:.2f} → Centerline/Contour")
        
        # 4. Simple/Compact shapes → WaveSkeleton or Contour
        else:
            weights = self.config.SIMPLE_SHAPE_WEIGHTS
            if self.verbose:
                print(f"    [Simple] Comp={compactness:.2f} → WaveSkeleton/Contour")
        
        # Weighted random selection
        strategies = list(weights.keys())
        probs = list(weights.values())
        selected = random.choices(strategies, weights=probs, k=1)[0]
        
        return selected
    
    def _generate_for_single_component(self, 
                                        component_mask_np: np.ndarray, 
                                        device: torch.device,
                                        n_scribbles: int = 1) -> Optional[torch.Tensor]:
        """
        对单个连通域生成 scribble：分析几何 → 选策略 → 自适应粗细 → 生成。
        
        Args:
            component_mask_np: (H, W) uint8 单连通域 mask (0 or 255)
            device: torch device
            n_scribbles: Number of scribbles
            
        Returns:
            (1, 1, H, W) scribble tensor, or None if component too small
        """
        # 分析几何特征
        geometry = GeometryAnalyzer.analyze_numpy(component_mask_np)
        
        if geometry['area'] < self.config.MIN_AREA_THRESHOLD:
            return None
        
        # 选择策略
        strategy_name = self.select_strategy(geometry)
        
        # 自适应粗细
        thickness = get_adaptive_thickness(
            component_mask_np,
            min_th=self.config.MIN_THICKNESS,
            max_th=self.config.MAX_THICKNESS,
            safety_margin=self.config.THICKNESS_SAFETY_MARGIN,
            percentile=self.config.THICKNESS_PERCENTILE,
        )
        
        if self.verbose:
            print(f"    Strategy={strategy_name}, Thickness={thickness}, "
                  f"Area={geometry['area']}, AR={geometry['aspect_ratio']:.1f}, "
                  f"Comp={geometry['compactness']:.2f}")
        
        # 创建策略实例
        strategy = self._create_strategy(strategy_name, thickness)
        
        # 转换为 tensor
        mask_tensor = torch.from_numpy(component_mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        mask_tensor = mask_tensor.to(device)
        
        # 生成 scribble
        try:
            scribble = strategy(mask_tensor, n_scribbles=n_scribbles)
            return scribble
        except Exception as e:
            if self.verbose:
                print(f"    Warning: {strategy_name} failed: {e}, falling back to line")
            # Fallback to simple line
            fallback = LineScribble(thickness=thickness, **self._warp_kwargs)
            try:
                return fallback(mask_tensor, n_scribbles=n_scribbles)
            except:
                return None
    
    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Generate scribbles for a batch of masks.
        
        Per-component processing:
        1. Preprocess mask (closing + opening + filter small components)
        2. Find connected components
        3. For each component: analyze geometry → select strategy → adaptive thickness → generate
        4. Merge all component scribbles
        
        Args:
            mask: (b, 1, H, W) mask in [0,1]
            n_scribbles: Number of scribbles per component
            
        Returns:
            scribble_mask: (b, 1, H, W) mask of scribbles in [0,1]
        """
        bs = mask.shape[0]
        device = mask.device
        scribbles = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(bs):
            mask_np = (mask[b, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            
            # Preprocess
            processed = self.preprocess_mask(mask_np)
            
            # If preprocessing removed everything, use original
            if np.sum(processed) == 0:
                processed = mask_np
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                processed, connectivity=8
            )
            
            if self.verbose:
                valid_count = sum(
                    1 for i in range(1, num_labels) 
                    if stats[i, cv2.CC_STAT_AREA] >= self.config.MIN_COMPONENT_AREA
                )
                print(f"  Batch {b}: {valid_count} valid components")
            
            # Generate scribble for each component
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.config.MIN_COMPONENT_AREA:
                    continue
                
                # Extract component mask
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # Generate scribble for this component
                comp_scribble = self._generate_for_single_component(
                    component_mask, device, n_scribbles
                )
                
                if comp_scribble is not None:
                    # Merge: take max to avoid overwriting
                    scribbles[b, ...] = torch.max(scribbles[b, ...], comp_scribble[0, ...])
        
        return scribbles


class CenterlineOnlyConfig(AdaptiveConfig):
    """Test-time ablation: force Centerline for all geometry types."""
    SIMPLE_SHAPE_WEIGHTS = {'centerline': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'centerline': 1.0}
    THIN_STRUCTURE_WEIGHTS = {'centerline': 1.0}
    SMALL_TARGET_WEIGHTS = {'centerline': 1.0}


class WaveOnlyConfig(AdaptiveConfig):
    """Test-time ablation: force WaveSkeleton for complex/simple shapes, keep centerline for thin/small."""
    SIMPLE_SHAPE_WEIGHTS = {'wave_skeleton': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'wave_skeleton': 1.0}


class ContourOnlyConfig(AdaptiveConfig):
    """Test-time ablation: force Contour for complex/simple shapes, keep centerline for thin/small."""
    SIMPLE_SHAPE_WEIGHTS = {'contour': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'contour': 1.0}


class CorrectionConfig(AdaptiveConfig):
    """
    修正 Scribble 专用配置。
    
    与初始 Scribble 的区别：
    - 关闭形态学预处理（error region 本身就很碎，不能再腐蚀）
    - 更低的面积阈值（修正区域通常很小）
    - 更细的粗细（修正要精准）
    """
    ENABLE_MASK_PREPROCESSING: bool = False   # 不做闭运算/开运算
    MIN_AREA_THRESHOLD: int = 10              # 与初始 scribble 对齐
    MIN_COMPONENT_AREA: int = 5               # 与初始 scribble 对齐
    MIN_THICKNESS: int = 2                    # 修正 scribble 粗细下限
    MAX_THICKNESS: int = 4                    # 修正 scribble 粗细上限（比初始更细）


class CenterlineOnlyCorrectionConfig(CorrectionConfig):
    """Correction config forcing Centerline only."""
    SIMPLE_SHAPE_WEIGHTS = {'centerline': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'centerline': 1.0}
    THIN_STRUCTURE_WEIGHTS = {'centerline': 1.0}
    SMALL_TARGET_WEIGHTS = {'centerline': 1.0}


class WaveOnlyCorrectionConfig(CorrectionConfig):
    """Correction config: WaveSkeleton for complex/simple, centerline for thin/small."""
    SIMPLE_SHAPE_WEIGHTS = {'wave_skeleton': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'wave_skeleton': 1.0}


class ContourOnlyCorrectionConfig(CorrectionConfig):
    """Correction config: Contour for complex/simple, centerline for thin/small."""
    SIMPLE_SHAPE_WEIGHTS = {'contour': 1.0}
    COMPLEX_SHAPE_WEIGHTS = {'contour': 1.0}


class CorrectionScribbleGenerator:
    """
    Generates correction scribbles for iterative refinement.
    
    Following ScribblePrompt paper:
    - Negative scribbles are used to correct FALSE POSITIVES (regions incorrectly selected)
    - Positive scribbles are used to correct FALSE NEGATIVES (regions incorrectly missed)
    
    This is NOT for initial annotation, but for subsequent correction iterations.
    
    Key differences from initial scribble:
    - No mask preprocessing (error regions are already small/fragmented)
    - Lower area thresholds (correction targets are small)
    - Thinner scribbles (corrections need precision)
    
    Usage in iterative training:
        1. Model predicts mask from initial scribble
        2. Compute error: error_region = gt - prediction
        3. False Positive (FP): error_region < 0 → generate negative scribble
        4. False Negative (FN): error_region > 0 → generate positive scribble
    """
    
    def __init__(self,
                 config: Optional[AdaptiveConfig] = None,
                 **kwargs):
        """
        Args:
            config: Adaptive configuration (defaults to CorrectionConfig)
            **kwargs: Passed to AdaptiveScribble
        """
        correction_config = config or CorrectionConfig()
        self.scribble_generator = AdaptiveScribble(config=correction_config, **kwargs)
    
    def __call__(self, 
                 error_region: torch.Tensor,
                 n_scribbles: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate correction scribbles from error region.
        
        Args:
            error_region: (b, 1, H, W) in [-1, 1] where:
                         +1 = False Negative (missed, need positive scribble)
                         -1 = False Positive (wrong, need negative scribble)
            n_scribbles: Number of scribbles per region
            
        Returns:
            pos_correction: (b, 1, H, W) positive scribbles for FN regions
            neg_correction: (b, 1, H, W) negative scribbles for FP regions
        """
        # Extract FN and FP regions
        false_neg = torch.clamp(error_region, min=0)   # FN: need positive scribble
        false_pos = -torch.clamp(error_region, max=0)  # FP: need negative scribble
        
        # Generate correction scribbles
        pos_correction = self.scribble_generator(false_neg, n_scribbles=n_scribbles)
        neg_correction = self.scribble_generator(false_pos, n_scribbles=n_scribbles)
        
        return pos_correction, neg_correction
    
    def from_prediction(self,
                        gt_mask: torch.Tensor,
                        pred_mask: torch.Tensor,
                        cutoff: float = 0.5,
                        n_scribbles: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate correction scribbles by comparing GT and prediction.
        
        Args:
            gt_mask: (b, 1, H, W) ground truth in [0, 1]
            pred_mask: (b, 1, H, W) prediction (logits or probabilities)
            cutoff: Threshold for binarizing prediction
            n_scribbles: Number of scribbles
            
        Returns:
            pos_correction: Positive scribbles for missed regions
            neg_correction: Negative scribbles for incorrectly selected regions
        """
        # Binarize
        binary_gt = (gt_mask > cutoff).int()
        binary_pred = (pred_mask > cutoff).int()
        
        # Compute error region
        error_region = (binary_gt - binary_pred).float()
        
        return self(error_region, n_scribbles=n_scribbles)


# Legacy alias for backward compatibility
NegativeAdaptiveScribble = CorrectionScribbleGenerator


# Convenience functions
def adaptive_scribble(mask: torch.Tensor, n_scribbles: int = 1, **kwargs) -> torch.Tensor:
    """
    Generate adaptive scribbles for a mask (POSITIVE scribbles only).
    
    This should be used for initial annotation. For correction scribbles
    during iterative refinement, use correction_scribble().
    
    Args:
        mask: (b, 1, H, W) or (1, H, W) mask in [0,1]
        n_scribbles: Number of scribbles
        **kwargs: Passed to AdaptiveScribble
        
    Returns:
        scribble: Same shape as mask
    """
    generator = AdaptiveScribble(**kwargs)
    return generator(mask, n_scribbles=n_scribbles)


def correction_scribble(gt_mask: torch.Tensor,
                        pred_mask: torch.Tensor,
                        cutoff: float = 0.5,
                        n_scribbles: int = 1,
                        **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate correction scribbles for iterative refinement.
    
    Following ScribblePrompt paper's iterative correction paradigm:
    - Compares GT with model prediction
    - Generates positive scribbles for False Negative regions
    - Generates negative scribbles for False Positive regions
    
    Args:
        gt_mask: (b, 1, H, W) ground truth in [0, 1]
        pred_mask: (b, 1, H, W) model prediction
        cutoff: Threshold for binarization
        n_scribbles: Number of scribbles per region
        **kwargs: Passed to CorrectionScribbleGenerator
        
    Returns:
        pos_correction: Positive scribbles for missed regions (FN)
        neg_correction: Negative scribbles for wrongly selected regions (FP)
    """
    generator = CorrectionScribbleGenerator(**kwargs)
    return generator.from_prediction(gt_mask, pred_mask, cutoff, n_scribbles)


# Legacy function (deprecated, use correction_scribble instead)
def negative_adaptive_scribble(mask: torch.Tensor, 
                                valid_mask: Optional[torch.Tensor] = None,
                                n_scribbles: int = 1, 
                                **kwargs) -> torch.Tensor:
    """
    DEPRECATED: Use correction_scribble() for iterative refinement.
    
    This function generates negative scribbles from background, which is not
    the typical usage in interactive segmentation.
    """
    import warnings
    warnings.warn(
        "negative_adaptive_scribble() is deprecated. "
        "For iterative correction, use correction_scribble(gt, pred) instead.",
        DeprecationWarning
    )
    # Generate scribble in background (1 - mask)
    generator = AdaptiveScribble(**kwargs)
    background = 1 - mask
    return generator(background, n_scribbles=n_scribbles)



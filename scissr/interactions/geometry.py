"""
Geometry Analyzer for Adaptive Scribble Selection

Analyzes mask geometric features to determine the optimal scribble strategy.

Author: ScribblePrompt Team
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import cv2


class GeometryAnalyzer:
    """
    几何特征分析器
    
    分析 mask 的几何特征，用于自适应策略选择：
    - area: 面积（像素数）
    - aspect_ratio: 长宽比（最小外接矩形）
    - compactness: 紧凑度（4πA/P², 圆形=1, 细长<1）
    - solidity: 实心度（面积/凸包面积）
    """
    
    @staticmethod
    def analyze_numpy(mask: np.ndarray) -> Dict:
        """
        分析 numpy mask 的几何特征
        
        Args:
            mask: (H, W) numpy array, 二值mask
            
        Returns:
            dict: 包含 area, aspect_ratio, compactness, solidity, centroid
        """
        area = np.sum(mask > 0)
        
        if area == 0:
            return {
                'area': 0,
                'aspect_ratio': 1.0,
                'compactness': 0.0,
                'solidity': 0.0,
                'centroid': (0, 0)
            }
        
        # 确保是 uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            coords = np.where(mask > 0)
            centroid = (int(np.mean(coords[1])), int(np.mean(coords[0])))
            return {
                'area': area,
                'aspect_ratio': 1.0,
                'compactness': 0.5,
                'solidity': 0.5,
                'centroid': centroid
            }
        
        # 使用最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        
        # 长宽比（最小外接矩形）
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # 紧凑度 (圆形=1, 细长/复杂<1)
        perimeter = cv2.arcLength(cnt, True)
        compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        
        # 实心度
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        
        # 质心
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            coords = np.where(mask > 0)
            cx, cy = int(np.mean(coords[1])), int(np.mean(coords[0]))
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'solidity': solidity,
            'centroid': (cx, cy)
        }
    
    @staticmethod
    def analyze_torch(mask: torch.Tensor) -> Dict:
        """
        分析 torch tensor mask 的几何特征
        
        Args:
            mask: (B, 1, H, W), (1, H, W) or (H, W) torch tensor
            
        Returns:
            dict: 包含 area, aspect_ratio, compactness, solidity, centroid
        """
        # Squeeze to 2D
        while len(mask.shape) > 2:
            mask = mask.squeeze(0)
        
        mask_np = mask.cpu().numpy()
        return GeometryAnalyzer.analyze_numpy(mask_np)
    
    @staticmethod
    def analyze(mask) -> Dict:
        """
        自动检测输入类型并分析几何特征
        
        Args:
            mask: numpy array 或 torch tensor
            
        Returns:
            dict: 几何特征
        """
        if isinstance(mask, torch.Tensor):
            return GeometryAnalyzer.analyze_torch(mask)
        else:
            return GeometryAnalyzer.analyze_numpy(mask)


# 便捷函数
def analyze_geometry(mask) -> Dict:
    """分析 mask 几何特征的便捷函数"""
    return GeometryAnalyzer.analyze(mask)


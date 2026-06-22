
from typing import Union, Tuple, List, Optional

import numpy as np
import torch

import kornia
import cv2
import voxynth

from .utils import _as_single_val

import os
# Prevent neurite from trying to load tensorflow
os.environ['NEURITE_BACKEND'] = 'pytorch' 


# -----------------------------------------------------------------------------
# Parent class
# -----------------------------------------------------------------------------

"""
WarpScribble 类是一个用于生成噪声遮罩和应用变形场的基类，主要用于处理和变形涂鸦（scribbles）。

该类提供了以下功能：
1. **初始化参数**：
   - `warp`：布尔值，决定是否应用变形。
   - `warp_smoothing`：变形平滑度设置，可以是整数或包含两个整数的元组/列表。
   - `warp_magnitude`：变形强度设置，可以是整数或包含两个整数的元组/列表。
   - `mask_smoothing`：噪声遮罩平滑度设置，可以是整数或包含两个整数的元组/列表。

2. **noise_mask 方法**：
   - 生成一个随机二值遮罩，通过阈值化平滑噪声来打破涂鸦。
   - 参数 `shape` 定义了噪声遮罩的形状，默认值为 (8,128,128)。
   - 返回值是一个形状为 (b, 1, H, W) 的噪声遮罩。

3. **apply_warp 方法**：
   - 使用随机变形场扭曲给定的遮罩 x。
   - 参数 `x` 是一个 torch.Tensor 类型的遮罩。
   - 返回值是经过变形处理后的归一化遮罩，形状与输入相同。

4. **batch_scribble 方法**：
   - 模拟一批示例（遮罩）的涂鸦。
   - 该方法为抽象方法，需要在子类中实现。

5. **__call__ 方法**：
   - 使类的实例可以像函数一样调用。
   - 参数 `mask` 是输入的遮罩，形状可以是 (b,1,H,W) 或 (1,H,W)。
   - 参数 `n_scribbles` 是生成的涂鸦数量，默认值为 1。
   - 返回值是生成的涂鸦遮罩，形状与输入相同。

该类主要用于图像处理中的涂鸦生成和变形，适用于需要随机性和自然效果的应用场景。
"""
class WarpScribble:
    """
    Parent scribble class with shared functions for generating noise masks (useful for breaking up scribbles) and applying deformation fields (to warp scribbles)
    """
    def __init__(self, 
                warp: bool = True, # 是否应用变形
                warp_smoothing: Union[int,Tuple[int],List[int]] = (4, 16), # 变形平滑度设置 
                warp_magnitude: Union[int,Tuple[int],List[int]] = (1, 6), # 变形强度设置
                mask_smoothing: Union[int,Tuple[int],List[int]] = (4, 16), # 噪声遮罩平滑度设置
                ):
        if isinstance(warp_smoothing, int):
            warp_smoothing = [warp_smoothing, warp_smoothing] # 保证是一个list
        if isinstance(warp_magnitude, int):
            warp_magnitude = [warp_magnitude, warp_magnitude] # 保证是一个list
        # Warp settings
        self.warp = warp
        self.warp_smoothing = list(warp_smoothing) 
        self.warp_magnitude = list(warp_magnitude)
        # Noise mask settings
        self.mask_smoothing = mask_smoothing 
        
    def noise_mask(self, shape: Union[Tuple[int],List[int]] = (8,128,128), device = None):
        """
        'shape' defines the shape of the noise mask to be generated
        Get a random binary mask by thresholding smoothed noise. The mask is used to break up the scribbles
        """
        if isinstance(self.mask_smoothing, tuple): # 检查 self.mask_smoothing 是否是一个元组
            get_smoothing = lambda: np.random.uniform(*self.mask_smoothing) # 创建一个匿名函数 get_smoothing，该函数从 self.mask_smoothing 元组中随机选择一个值。
        else:
            get_smoothing = lambda: self.mask_smoothing # 如果不是元组，get_smoothing 函数将直接返回 self.mask_smoothing 的值。

        noise = torch.stack([
            # 对于 shape[0] 次循环，每次生成一个二维 Perlin 噪声图
            voxynth.noise.perlin(shape=shape[-2:], # 设置噪声图的形状为输入 shape 的最后两个维度
            smoothing=get_smoothing(), # 设置为前面定义的get_smoothing函数
            magnitude=1, # 设置噪声的幅度为 1
            device=device) 
            for _ in range(shape[0])
        ]) # shape: b x H x W
        noise_mask = (noise > 0.0).int().unsqueeze(1)

        return noise_mask # shaoe: b x 1 x H x W
    
    def apply_warp(self, x: torch.Tensor):
        """
        Warp a given mask x using a random deformation field
        使用随机变形场扭曲给定 mask x
        deformation_field 是一个 位移场（displacement field），
        它用来描述如何在二维或三维空间中将输入图像中的每个像素 移动到新的位置，实现图像的随机形变。
        """
        if x.sum() > 0: # 避免在空白输入上执行不必要的形变运算，从而提高计算效率。
            # warp scribbles using a deformation field
            '''模拟手绘线条中的自然形变，使生成的 scribble 更具随机性和自然效果。'''
            deformation_field = voxynth.transform.random_transform(
                shape = x.shape[-2:],       # 输入的空间维度 (H, W)
                affine_probability = 0.0,   # 线性变换的概率设为 0（只执行非线性形变）
                warp_probability = 1.0,     # 确保始终执行非线性形变
                warp_integrations = 0,      # 控制形变场的平滑程度，0 表示单次变换
                warp_smoothing_range = self.warp_smoothing, # 随机采样形变场的平滑度
                warp_magnitude_range = self.warp_magnitude, # 随机采样形变场的强度
                voxsize = 1,                # 体素大小（一般设为 1）
                device = x.device,          # 保持与输入 `x` 相同的设备（CPU 或 GPU）
                isdisp = False              # 返回位移场而不是变换矩阵
                ) 

            warped = voxynth.transform.spatial_transform(x, trf = deformation_field, isdisp=False) # 对每个像素位置应用位移场，生成新的形变图像。
            if warped.sum() == 0:
                return x # 如果为空，说明形变导致所有像素都被移除（或移出图像范围），此时返回原始输入 x，避免无效输出。
            else:
                return (warped - warped.min()) / (warped.max() - warped.min()) # 对形变后的张量 warped 进行归一化操作，将像素值映射到 [0, 1] 区间，保持一致的输出范围。
        else:
            # Don't need to warp if mask is empty
            return x
    
    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1):
        """
        Simulate scribbles for a batch of examples (mask).
        """
        raise NotImplementedError # 这是一个抽象基类中的方法，它在父类中未实现具体逻辑，需要在子类中重载
    
    def __call__(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (b,1,H,W) or (1,H,W) mask in [0,1] to sample scribbles from
        Returns:
            scribble_mask: (b,1,H,W) or (1,H,W) mask(s) of scribbles on [0,1]
        """
        assert len(mask.shape) in [3,4], f"mask must be b x 1 x h x w or 1 x h x w. currently {mask.shape}"
        # 检查输入的 mask 是否是三维或四维张量，三维形状 (1, H, W) 或四维形状 (b, 1, H, W) 是合法的。
        if len(mask.shape)==3:
            # shape: 1 x h x w
            return self.batch_scribble(mask[None,...], n_scribbles=n_scribbles)[0,...]
            # 用 mask[None, ...] 在第 0 维扩展一个维度，变成 (1, 1, H, W)
        else:
            # shape: b x 1 x h x w
            return self.batch_scribble(mask, n_scribbles=n_scribbles)



# -----------------------------------------------------------------------------
# Line Scribbles
# -----------------------------------------------------------------------------

class LineScribble(WarpScribble):
    """
    Generates scribbles by 
        1) drawing lines connecting random points on the mask
        2) warping with a random deformation field
        3) then correcting any scribbles outside the mask
        5) optionally, limiting the max area of scribbles to k pixels
    """
    def __init__(self,
                 # Warp settings
                 warp: bool = True,
                 warp_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                 warp_magnitude: Union[int,Tuple[int],List[int]] = (1, 6),
                 mask_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                 # Line scribble settings
                 thickness: int = 1, 
                 preserve_scribble: bool = True, # if True, prevents empty scribble masks from being returned
                 max_pixels: Optional[int] = None, # per "scribble"
                 max_pixels_smooth: Optional[int] = 42,
                 # Viz             
                 show: bool = False
                 ):
        
        super().__init__(
            warp=warp, 
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        self.thickness = thickness
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from
            n_scribbles: number of line scribbles to sample initially
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        bs = mask.shape[0] # 获取批次大小 bs，即有多少张图像需要生成 scribble

        # Points to sample line endpoints from
        points = torch.nonzero(mask[:,0,...]) # 找到 mask 中所有非零像素点的位置，即可以作为 scribble 起点或终点的像素坐标。
        # mask[:,0,...] 提取单通道，即 (b,1,H,W)->(N,3) 3:(b,y,x)
        
        def sample_lines(indices): # indices表示某一张图像中的非零点的索引

            image = np.zeros(mask.shape[-2:]+(1,)) # (H,W,1)

            if len(indices) > 0:
                # Sample points for each example in the batch
                idx = np.random.randint(low=0, high=len(indices), size=2*n_scribbles)
                endpoints = points[indices,1:][idx,0,...]
                '''
                points[indices, 1:] 选择 points 中位于 indices 索引处的坐标，并只提取第 1 列和第 2 列的元素。
                [idx, 0, ...] idx 是随机生成的索引，表示从 points[indices, 1:] 中随机选择若干个坐标对。[0, ...] 表示从结果中提取二维坐标对。
                '''
                # Flip order of coordinates to be xy
                endpoints = torch.flip(endpoints, dims=(1,)).cpu().numpy() # 将 y, x 坐标顺序翻转为 x, y，并将结果转换为 NumPy 数组，以便使用 OpenCV 绘制线段。
                # Draw lines between the sample points
                for i in range(n_scribbles):
                    thickness = _as_single_val(self.thickness)
                    image = cv2.line(image, tuple(endpoints[i*2]), tuple(endpoints[i*2+1]), color=1, thickness=thickness)

            return torch.from_numpy(image) # shape: H x W x 1

        scribbles = torch.stack([
            sample_lines(torch.argwhere(points[:,0]==i)) for i in range(bs) # torch.argwhere(points[:, 0] == i)：提取属于第 i 张图像的候选点索引。
        ]).to(mask.device).moveaxis(-1,1).float() # shape: b x 1 x H x W
        # moveaxis(-1, 1)：将最后一个维度（H, W, 1）移动到通道位置，变成 (b, 1, H, W)。
        
        if self.warp:
            warped_scribbles = torch.stack([self.apply_warp(scribbles[b,...]) for b in range(bs)]) # shape: b x 1 x H x W
        else:
            warped_scribbles = scribbles

        # Remove lines outside the mask
        corrected_warped_scribbles = mask * warped_scribbles # 通过逐元素相乘，去除 mask 以外的线段，确保 scribble 只保留在 mask 内。

        if self.preserve_scribble:
            # If none of the scribble falls in the mask after warping, undo warping (如果形变后Scribble从有到无，则恢复成原始的scribble)
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1,2,3)) == 0)
            corrected_warped_scribbles[idx] = mask[idx] * scribbles[idx]

        if self.max_pixels is not None: # 限制 scribble 的最大像素数量，确保生成的线段不会覆盖过多区域
        
            noise = torch.stack([
                voxynth.noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, magnitude=1, device=mask.device) for _ in range(bs)
            ]).unsqueeze(1) # shape: b x 1 x H x W

            # Shift all noise to be positive
            if noise.min() < 0:
                noise = noise - noise.min()
            
            # Get the top k pixels (因为限制了最大像素数量，所以这里取 top k 像素）
            # 通过逐元素相乘，去除 mask 以外的线段，确保 scribble 只保留在 mask 内。)
            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels*n_scribbles), dim=1)

            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask)) # 只保留 top k 像素

            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles
        
        return corrected_warped_scribbles # b x 1 x H x W
    


# -----------------------------------------------------------------------------
# Median Axis Scribble
# -----------------------------------------------------------------------------

class CenterlineScribble(WarpScribble):
    """
    Generates scribbles by 
        1) skeletonizing the mask
        2) chopping up with a random noise mask 
        3) warping with a random deformation field
        4) then correcting any scribbles that fall outside the mask
        5) optionally, limiting the max area of scribbles to k pixels
    """
    def __init__(self, 
                # Warp settings
                warp: bool = True,
                warp_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                warp_magnitude: Union[int,Tuple[int],List[int]] = (1, 6),
                mask_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                # Thickness of skeleton
                dilate_kernel_size: Optional[int] = None,
                preserve_scribble: bool = True, # if True, prevents empty scribble masks from being returned
                max_pixels: Optional[int] = None, # per "scribble"
                max_pixels_smooth: int = 42,
                # Viz
                show : bool = False
                ):
        
        super().__init__(
            warp=warp, 
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        self.dilate_kernel_size = dilate_kernel_size
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show
    
    @staticmethod
    def _random_fragment(skel_np: np.ndarray, 
                         keep_ratio_range: Tuple[float, float] = (0.3, 0.6)) -> np.ndarray:
        """
        对骨架做随机像素丢弃，变成"虚线"。
        
        适用于 1px 骨架（腐蚀会整条消失），通过随机丢弃像素：
        - 避免模型死记"完美全长骨架"
        - 强迫模型学习连通性推断
        - 技术上极简，只需 numpy 索引
        
        Args:
            skel_np: (H, W) float numpy, 1px 骨架
            keep_ratio_range: 保留像素比例范围
            
        Returns:
            fragmented: (H, W) float numpy
        """
        pixels = np.argwhere(skel_np > 0)
        
        if len(pixels) < 10:
            return skel_np  # 太短不打散
        
        # 随机保留 50-80% 的像素
        keep_ratio = np.random.uniform(*keep_ratio_range)
        num_keep = max(5, int(len(pixels) * keep_ratio))
        
        keep_indices = np.random.choice(len(pixels), num_keep, replace=False)
        
        result = np.zeros_like(skel_np)
        selected = pixels[keep_indices]
        result[selected[:, 0], selected[:, 1]] = skel_np[selected[:, 0], selected[:, 1]]
        
        return result

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        """
        Simulate scribbles for a batch of examples.
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from. torch.int32 
            n_scribbles: (int) only used when max_pixels is set as a multiplier for total area of the scribbles
                currently, this argument does not control the number of components in the scribble mask 
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        assert len(mask.shape)==4, f"mask must be b x 1 x h x w. currently {mask.shape}"
        bs = mask.shape[0]

        mask_w_border = 255*mask.clone().moveaxis(1,-1) 
        #将通道维度从第 1 维移动到最后一维，形状从 (b, 1, H, W) 变为 (b, H, W, 1)。这是因为 OpenCV 需要处理形状为 (H, W, C) 的图像。
        mask_w_border[:,:,0,:] = 0
        mask_w_border[:,:,-1,:] = 0
        mask_w_border[:,0,:,:] = 0
        mask_w_border[:,-1,:,:] = 0
        # 将 mask 的四个边界区域设置为 0（黑色像素），以防止骨架化时线段连接到图像边缘。

        # Skeletonize the mask
        skeleton = torch.from_numpy(
            np.stack([
                cv2.ximgproc.thinning(mask_w_border[i,...].cpu().numpy().astype(np.uint8))/255 for i in range(bs)
            ])
        ).squeeze(-1).unsqueeze(1).to(mask.device).float() # shape: b x 1 x H x W

        # ===== 先截断再加粗（顺序很重要！） =====
        # 1. 在 1px 骨架上做腐蚀截断（只缩短两端，不影响宽度）
        # 2. 然后加粗到目标厚度
        # 如果反过来，腐蚀会把宽度也削回去
        
        fragmented_skeleton = torch.zeros_like(skeleton)
        for b in range(bs):
            skel_np = skeleton[b, 0, ...].cpu().numpy()
            fragmented = self._random_fragment(skel_np)
            fragmented_skeleton[b, 0, ...] = torch.from_numpy(fragmented).to(mask.device)
        
        if self.preserve_scribble:
            idx = torch.where(torch.sum(fragmented_skeleton, dim=(1,2,3)) == 0)
            fragmented_skeleton[idx] = skeleton[idx]
        
        # 加粗到目标厚度
        if self.dilate_kernel_size is not None:
            k = _as_single_val(self.dilate_kernel_size)
            if k > 1:
                iterations = max(1, (k - 1) // 2)
                kernel = torch.ones((3,3), device=mask.device)
                scribbles = fragmented_skeleton
                for _ in range(iterations):
                    scribbles = kornia.morphology.dilation(scribbles, kernel=kernel, engine='convolution')
            else:
                scribbles = fragmented_skeleton
        else:
            scribbles = fragmented_skeleton

        # Warp：只对宽目标 (max_radius >= 15) 做变形
        # 细/中目标 warp 容易把骨架推出 mask
        corrected_warped_scribbles = torch.zeros_like(scribbles)
        
        for b in range(bs):
            mask_np = (mask[b, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            dist_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
            max_radius = np.max(dist_map)
            
            if max_radius >= 15 and self.warp:
                # 宽目标：可以 warp（有足够缓冲区）
                warped = self.apply_warp(scribbles[b, ...])
                corrected_warped_scribbles[b, ...] = mask[b, ...] * warped
            else:
                # 细/中目标：不 warp，直接裁剪到 mask
                corrected_warped_scribbles[b, ...] = mask[b, ...] * scribbles[b, ...]
        
        if self.preserve_scribble:
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1,2,3)) == 0)
            corrected_warped_scribbles[idx] = mask[idx] * scribbles[idx]

        if self.max_pixels is not None:
        
            noise = torch.stack([
                voxynth.noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, magnitude=1, device=mask.device) for _ in range(bs)
            ]).unsqueeze(1) # shape: b x 1 x H x W

            # Shift all noise mask to be positive
            if noise.min() < 0:
                noise = noise - noise.min()
            
            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels*n_scribbles), dim=1)

            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))

            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles

        return corrected_warped_scribbles 
    


# -----------------------------------------------------------------------------
# Contour Scribbles (Improved: Inward Shrink + Continuous Contour)
# -----------------------------------------------------------------------------

class ContourScribble(WarpScribble):
    """
    Generates scribbles by simulating "circling" behavior:
        1) Erode the mask inward (safety margin, prevents overflow after warp)
        2) Extract boundary via morphological gradient
        3) Dilate to desired thickness
        4) Warp with a random deformation field (simulate hand tremor)
        5) Mild dropout: 1-2 small gaps (simulate pen lift), NOT Perlin noise
        6) Clip to original mask (safety check)
    
    Key differences from the old implementation:
    - No Perlin noise breaking (humans draw continuous circles, not dashed lines)
    - Inward shrink ensures warp doesn't push contour outside mask
    - For thin objects (max_dist < 3), falls back to skeleton
    """
    def __init__(self, 
                # Warp settings
                warp: bool = True,
                warp_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                warp_magnitude: Union[int,Tuple[int],List[int]] = (1, 6),
                mask_smoothing: Union[int,Tuple[int],List[int]] = (4, 16),
                # Contour settings
                dilate_kernel_size: Optional[Union[int, Tuple[int]]] = 3,
                shrink_ratio: Tuple[float, float] = (0.3, 0.6),  # 内缩比例范围
                max_shrink_pixels: int = 15,                       # 内缩上限
                # Mild dropout
                gap_probability: float = 0.7,       # 产生缺口的概率
                num_gaps: Tuple[int, int] = (1, 3),  # 缺口数量范围
                gap_size_ratio: float = 0.08,        # 每个缺口占轮廓长度的比例
                # Other settings
                preserve_scribble: bool = True,
                max_pixels: Optional[int] = None,
                max_pixels_smooth: Optional[int] = 42,
                show: bool = False
                ):
        
        super().__init__(
            warp=warp, 
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        
        self.dilate_kernel_size = dilate_kernel_size
        self.shrink_ratio = shrink_ratio
        self.max_shrink_pixels = max_shrink_pixels
        self.gap_probability = gap_probability
        self.num_gaps = num_gaps
        self.gap_size_ratio = gap_size_ratio
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show

    def _generate_single_contour(self, mask_np: np.ndarray) -> np.ndarray:
        """
        对单个 mask 生成内缩连续轮廓。
        
        Args:
            mask_np: (H, W) uint8 mask (0 or 255)
        Returns:
            scribble: (H, W) uint8 scribble (0 or 255)
        """
        H, W = mask_np.shape
        scribble = np.zeros((H, W), dtype=np.uint8)
        
        if np.sum(mask_np > 0) == 0:
            return scribble
        
        # Step 1: 计算距离变换，确定物体尺度
        dist_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        max_dist = np.max(dist_map)
        
        # 太细的物体 → fallback 到骨架
        if max_dist < 3:
            skeleton = cv2.ximgproc.thinning(mask_np)
            # 加粗骨架：1px 骨架用 3×3 核迭代膨胀
            k = _as_single_val(self.dilate_kernel_size) if self.dilate_kernel_size else 2
            if k > 1:
                iterations = max(1, (k - 1) // 2)
                dilate_kern = np.ones((3, 3), np.uint8)
                skeleton = cv2.dilate(skeleton, dilate_kern, iterations=iterations)
            return cv2.bitwise_and(skeleton, mask_np)
        
        # Step 2: 内缩（Inward Shrink）
        shrink_ratio = np.random.uniform(*self.shrink_ratio)
        shrink_amount = int(max_dist * shrink_ratio)
        shrink_amount = min(shrink_amount, self.max_shrink_pixels)
        shrink_amount = max(shrink_amount, 1)
        
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (shrink_amount * 2 + 1, shrink_amount * 2 + 1)
        )
        eroded = cv2.erode(mask_np, erode_kernel)
        
        # 如果腐蚀后为空，减小腐蚀量重试
        if np.sum(eroded) == 0:
            shrink_amount = max(1, shrink_amount // 2)
            erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (shrink_amount * 2 + 1, shrink_amount * 2 + 1)
            )
            eroded = cv2.erode(mask_np, erode_kernel)
        
        if np.sum(eroded) == 0:
            eroded = mask_np  # 最后兜底
        
        # Step 3: 提取外轮廓（用 findContours 而非形态学梯度，避免内部噪点）
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = np.zeros_like(eroded)
        if contours:
            cv2.drawContours(boundary, contours, -1, 255, thickness=1)
        
        # Step 4: 加粗到目标厚度
        # 形态学梯度（3×3 椭圆核）已经产生 ~2px 宽的边界
        # 用 3×3 核的迭代次数来精确控制额外增厚
        k = _as_single_val(self.dilate_kernel_size) if self.dilate_kernel_size else 0
        if k > 2:
            iterations = max(1, (k - 2) // 2)
            dilate_kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.dilate(boundary, dilate_kernel, iterations=iterations)
        
        # Step 5: 温和的缺口（Mild Dropout）
        if np.random.random() < self.gap_probability:
            boundary = self._apply_mild_gaps(boundary, mask_np)
        
        # 确保在原始 mask 内
        scribble = cv2.bitwise_and(boundary, mask_np)
        
        return scribble
    
    def _apply_mild_gaps(self, boundary: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """
        在轮廓上切 1-3 个小缺口，模拟画圈没封口。
        
        方法：用 cv2.findContours 获取轮廓周长，
        基于周长（而非像素总数）计算缺口大小。
        """
        # 用 findContours 获取轮廓的真实周长
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return boundary
        
        # 取最长的轮廓
        main_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=True))
        perimeter = cv2.arcLength(main_contour, closed=True)
        
        if perimeter < 30:
            return boundary  # 太短的轮廓不打断
        
        result = boundary.copy()
        num_gaps = np.random.randint(*self.num_gaps)
        
        # 缺口半径：基于周长的比例，限制在 [3, 20] 像素
        gap_radius = int(np.clip(perimeter * self.gap_size_ratio, 3, 20))
        
        # 从轮廓点上随机选位置
        contour_pts = main_contour.reshape(-1, 2)  # (N, 2) in (x, y)
        
        for _ in range(num_gaps):
            idx = np.random.randint(0, len(contour_pts))
            cx, cy = contour_pts[idx]
            cv2.circle(result, (int(cx), int(cy)), gap_radius, 0, -1)
        
        # 安全检查：确保不会擦除太多
        if np.sum(result) < np.sum(boundary) * 0.5:
            return boundary
        
        return result

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: Optional[int] = 1):
        """
        Args:
            mask: (b,1,H,W) mask in [0,1] to sample scribbles from
        Returns:
            scribble_mask: (b,1,H,W) mask(s) of scribbles in [0,1]
        """
        assert len(mask.shape) == 4, f"mask must be b x 1 x h x w. currently {mask.shape}"
        bs = mask.shape[0]
        device = mask.device
        
        scribbles = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(bs):
            mask_np = (mask[b, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            scribble_np = self._generate_single_contour(mask_np)
            scribbles[b, 0, ...] = torch.from_numpy(scribble_np).float() / 255.0
        
        scribbles = scribbles.to(device)
        
        # Warp
        if self.warp:
            warped_scribbles = torch.stack([self.apply_warp(scribbles[b, ...]) for b in range(bs)])
        else:
            warped_scribbles = scribbles
        
        # Clip to mask (safety check — 因为已经内缩，大部分都在 mask 内)
        corrected = mask * warped_scribbles
        
        if self.preserve_scribble:
            idx = torch.where(torch.sum(corrected, dim=(1, 2, 3)) == 0)[0]
            corrected[idx, ...] = mask[idx, ...] * scribbles[idx, ...]
        
        # Optional max_pixels
        if self.max_pixels is not None:
            noise = torch.stack([
                voxynth.noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth,
                                     magnitude=1, device=device) for _ in range(bs)
            ]).unsqueeze(1)
            
            if noise.min() < 0:
                noise = noise - noise.min()
            
            flat_mask = (noise * corrected).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels * n_scribbles), dim=1)
            
            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))
            corrected = binary_mask.view(*mask.shape) * corrected
        
        return corrected
    
    

# -----------------------------------------------------------------------------
# Wave Skeleton Scribbles (Our Innovation)
# -----------------------------------------------------------------------------

class WaveSkeletonScribble(WarpScribble):
    """
    Generates scribbles by:
        1) Extracting the skeleton of the mask
        2) Computing wave offsets based on spatial coordinates
        3) Applying endpoint retraction (simulating lazy annotation)
        4) Optionally dropping random segments (simulating discontinuous strokes)
        5) Warping with a random deformation field
        6) Correcting any scribbles outside the mask
    
    This strategy is best suited for large, compact shapes where:
    - Pure skeleton is too sparse
    - Wave oscillations have room to develop
    
    Key features:
    - Preserves geometric structure via skeleton guidance
    - Adds natural hand-drawn feel via sinusoidal waves
    - Simulates human behavior via endpoint retraction and segment dropout
    """
    def __init__(self,
                 # Warp settings
                 warp: bool = True,
                 warp_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 warp_magnitude: Union[int, Tuple[int], List[int]] = (1, 6),
                 mask_smoothing: Union[int, Tuple[int], List[int]] = (4, 16),
                 # Wave parameters
                 wavelength: Tuple[float, float] = (25, 45),      # Wave period in pixels
                 dist_ratio: float = 0.7,                          # 振幅 = dist × ratio（0.7 = 留30%安全余量）
                 tremor: float = 1.5,                              # Hand tremor noise std
                 # Human behavior simulation
                 endpoint_retract: Tuple[float, float] = (0.10, 0.20),  # Retract 10-20% from ends
                 segment_dropout_prob: float = 0.15,                     # Probability of dropping a segment
                 min_segment_length: int = 30,                           # Minimum segment length to keep
                 # Line settings
                 thickness: Tuple[int, int] = (3, 4),              # Random line thickness
                 # Other settings
                 preserve_scribble: bool = True,
                 max_pixels: Optional[int] = None,
                 max_pixels_smooth: int = 42,
                 show: bool = False
                 ):
        
        super().__init__(
            warp=warp,
            warp_smoothing=warp_smoothing,
            warp_magnitude=warp_magnitude,
            mask_smoothing=mask_smoothing,
        )
        
        # Wave parameters
        self.wavelength = wavelength
        self.dist_ratio = dist_ratio
        self.tremor = tremor
        
        # Human behavior
        self.endpoint_retract = endpoint_retract
        self.segment_dropout_prob = segment_dropout_prob
        self.min_segment_length = min_segment_length
        
        # Line settings
        self.thickness = thickness
        
        # Other
        self.preserve_scribble = preserve_scribble
        self.max_pixels = max_pixels
        self.max_pixels_smooth = max_pixels_smooth
        self.show = show

    def batch_scribble(self, mask: torch.Tensor, n_scribbles: int = 1) -> torch.Tensor:
        """
        Generate wave-skeleton scribbles for a batch of masks.
        
        Args:
            mask: (b, 1, H, W) mask in [0,1]
            n_scribbles: not used (kept for API compatibility)
            
        Returns:
            scribble_mask: (b, 1, H, W) mask of scribbles in [0,1]
        """
        assert len(mask.shape) == 4, f"mask must be b x 1 x h x w. currently {mask.shape}"
        bs = mask.shape[0]
        device = mask.device
        
        scribbles = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(bs):
            mask_np = (mask[b, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            scribble_np = self._generate_wave_skeleton(mask_np)
            scribbles[b, 0, ...] = torch.from_numpy(scribble_np).float() / 255.0
        
        scribbles = scribbles.to(device)
        
        # Apply warp
        if self.warp:
            warped_scribbles = torch.stack([self.apply_warp(scribbles[b, ...]) for b in range(bs)])
        else:
            warped_scribbles = scribbles
        
        # Remove scribbles outside the mask
        corrected_warped_scribbles = mask * warped_scribbles
        
        # Preserve scribbles if warping removed everything
        if self.preserve_scribble:
            idx = torch.where(torch.sum(corrected_warped_scribbles, dim=(1, 2, 3)) == 0)
            corrected_warped_scribbles[idx] = mask[idx] * scribbles[idx]
        
        # Limit max pixels if specified
        if self.max_pixels is not None:
            noise = torch.stack([
                voxynth.noise.perlin(shape=mask.shape[-2:], smoothing=self.max_pixels_smooth, 
                                     magnitude=1, device=device) for _ in range(bs)
            ]).unsqueeze(1)
            
            if noise.min() < 0:
                noise = noise - noise.min()
            
            flat_mask = (noise * corrected_warped_scribbles).view(bs, -1)
            vals, idx = flat_mask.topk(k=(self.max_pixels * n_scribbles), dim=1)
            
            binary_mask = torch.zeros_like(flat_mask)
            binary_mask.scatter_(dim=1, index=idx, src=torch.ones_like(flat_mask))
            corrected_warped_scribbles = binary_mask.view(*mask.shape) * corrected_warped_scribbles
        
        return corrected_warped_scribbles

    def _generate_wave_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Generate wave skeleton scribble for a single mask.
        
        Args:
            mask: (H, W) uint8 mask (0 or 255)
            
        Returns:
            scribble: (H, W) uint8 scribble (0 or 255)
        """
        from skimage.morphology import skeletonize
        
        H, W = mask.shape
        scribble = np.zeros((H, W), dtype=np.uint8)
        
        if np.sum(mask > 0) == 0:
            return scribble
        
        # Step 1: Extract skeleton
        skeleton = skeletonize(mask > 0)
        skel_points = np.array(list(zip(*np.where(skeleton))))  # (N, 2) - (y, x)
        
        if len(skel_points) == 0:
            return scribble
        
        # Step 2: Compute distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Step 3: Compute wave offsets
        wave_points_dict = self._compute_wave_offsets(skel_points, dist_transform, mask, H, W)
        
        if not wave_points_dict:
            # Fallback: just draw skeleton
            kernel = np.ones((3, 3), np.uint8)
            scribble = cv2.dilate((skeleton * 255).astype(np.uint8), kernel, iterations=1)
            return cv2.bitwise_and(scribble, mask)
        
        # Step 4: Trace paths and draw with human behavior simulation
        scribble = self._connect_wave_points_with_human_behavior(
            skeleton, wave_points_dict, mask, H, W
        )
        
        return scribble

    def _compute_wave_offsets(self, skel_points: np.ndarray, dist_transform: np.ndarray,
                               mask: np.ndarray, H: int, W: int) -> dict:
        """
        Compute wave offset for each skeleton point.
        
        核心设计：振幅 = 距离变换值 × 固定比例
        - 宽处振幅大（大波浪），窄处振幅小（近直线）
        - 自动退化：对极细目标(dist≈2)，振幅≈1，等效于 Centerline
        - 永远不会超出 mask 边界（振幅 < dist）
        """
        # 全局波浪方向（随机角度）
        wave_angle = np.random.uniform(0, 2 * np.pi)
        wave_dir = np.array([np.cos(wave_angle), np.sin(wave_angle)])
        perp_dir = np.array([-wave_dir[1], wave_dir[0]])
        
        # 波长自适应：基于骨架上的典型宽度
        # 振幅大时波长也大（舒缓大波浪），振幅小时波长也小（紧凑小抖动）
        skel_distances = dist_transform[skel_points[:, 0], skel_points[:, 1]]
        median_dist = np.median(skel_distances)
        typical_amplitude = median_dist * self.dist_ratio
        # 波长 ≈ 振幅 × 3~5 倍，保证波浪看起来自然
        wavelength_factor = np.random.uniform(3.0, 5.0)
        wavelength = max(typical_amplitude * wavelength_factor, 15)  # 最小 15px
        
        wave_points_dict = {}
        
        for y, x in skel_points:
            # 相位（基于空间坐标，保证连续性）
            pos = np.array([x, y])
            phase = np.dot(pos, wave_dir) / wavelength * 2 * np.pi
            
            # 振幅 = 距离变换值 × 安全系数
            # 宽处大振幅，窄处小振幅，永远不出界
            current_amplitude = dist_transform[y, x] * self.dist_ratio
            
            # 正弦偏移 + 微小手抖
            wave_value = np.sin(phase)
            tremor = np.random.normal(0, self.tremor)
            offset = wave_value * current_amplitude + tremor
            
            new_x = int(x + offset * perp_dir[0])
            new_y = int(y + offset * perp_dir[1])
            
            new_x = max(0, min(W - 1, new_x))
            new_y = max(0, min(H - 1, new_y))
            
            if mask[new_y, new_x] > 0:
                wave_points_dict[(y, x)] = (new_x, new_y)
        
        return wave_points_dict

    def _connect_wave_points_with_human_behavior(self, skeleton: np.ndarray, 
                                                   wave_points_dict: dict,
                                                   mask: np.ndarray,
                                                   H: int, W: int) -> np.ndarray:
        """
        Connect wave points following skeleton paths with human behavior simulation:
        1. Endpoint retraction: cut 10-20% from path ends
        2. Segment dropout: randomly drop some path segments
        """
        scribble = np.zeros((H, W), dtype=np.uint8)
        
        if not wave_points_dict:
            return scribble
        
        # 8-connectivity neighbors
        neighbors_offset = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1), (1, 0), (1, 1)]
        
        skel_img = skeleton.astype(np.uint8)
        
        def get_skel_neighbors(y, x):
            nbrs = []
            for dy, dx in neighbors_offset:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel_img[ny, nx] > 0:
                    nbrs.append((ny, nx))
            return nbrs
        
        # Find endpoints (degree 1 vertices)
        skel_points = set(wave_points_dict.keys())
        endpoints = []
        for (y, x) in skel_points:
            nbrs = get_skel_neighbors(y, x)
            if len(nbrs) == 1:
                endpoints.append((y, x))
        
        if not endpoints:
            endpoints = [list(skel_points)[0]]
        
        # Collect all paths via DFS
        all_paths = []
        visited = set()
        
        for start in endpoints:
            if start in visited:
                continue
            
            path_skel = [start]
            visited.add(start)
            current = start
            
            while True:
                nbrs = get_skel_neighbors(current[0], current[1])
                unvisited = [n for n in nbrs if n not in visited and n in skel_points]
                
                if not unvisited:
                    break
                
                next_pt = unvisited[0]
                path_skel.append(next_pt)
                visited.add(next_pt)
                current = next_pt
            
            if len(path_skel) > 5:
                all_paths.append(path_skel)
        
        # Handle remaining unvisited points (loops)
        remaining = skel_points - visited
        while remaining:
            start = remaining.pop()
            
            path_skel = [start]
            visited.add(start)
            current = start
            
            while True:
                nbrs = get_skel_neighbors(current[0], current[1])
                unvisited = [n for n in nbrs if n not in visited and n in remaining]
                
                if not unvisited:
                    break
                
                next_pt = unvisited[0]
                path_skel.append(next_pt)
                visited.add(next_pt)
                remaining.discard(next_pt)
                current = next_pt
            
            if len(path_skel) > 5:
                all_paths.append(path_skel)
        
        # Process each path with human behavior simulation
        for path_skel in all_paths:
            # Endpoint retraction
            retracted_path = self._apply_endpoint_retraction(path_skel)
            
            if len(retracted_path) < self.min_segment_length:
                continue
            
            # Segment dropout
            segments = self._apply_segment_dropout(retracted_path)
            
            # Draw each kept segment
            for segment in segments:
                if len(segment) < 10:
                    continue
                
                # Convert to wave points
                wave_path = []
                for skel_pt in segment:
                    if skel_pt in wave_points_dict:
                        wave_path.append(list(wave_points_dict[skel_pt]))
                
                if len(wave_path) > 10:
                    wave_path = self._smooth_points(wave_path, window_size=5)
                    thickness = np.random.randint(*self.thickness)
                    cv2.polylines(scribble, [np.array(wave_path)], False, 255, thickness)
        
        return scribble

    def _apply_endpoint_retraction(self, path: list) -> list:
        """
        Simulate lazy annotation: human doesn't draw all the way to boundaries.
        Cut 10-20% from both ends of the path.
        """
        if len(path) < 20:
            return path
        
        retract_ratio = np.random.uniform(*self.endpoint_retract)
        retract_count = int(len(path) * retract_ratio)
        
        # Ensure at least half the path remains
        retract_count = min(retract_count, len(path) // 4)
        
        if retract_count > 0:
            return path[retract_count:-retract_count]
        return path

    def _apply_segment_dropout(self, path: list) -> list:
        """
        Simulate discontinuous strokes: human might lift pen and skip sections.
        """
        if len(path) < 60:  # Short paths are not split
            return [path]
        
        # Split path into segments
        num_segments = np.random.randint(2, 4)
        segment_length = len(path) // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(path)
            segments.append(path[start_idx:end_idx])
        
        # Randomly keep/drop segments
        kept_segments = []
        for i, seg in enumerate(segments):
            # First and last segments have higher keep probability
            if i == 0 or i == len(segments) - 1:
                keep_prob = 0.95
            else:
                keep_prob = 1 - self.segment_dropout_prob
            
            if np.random.random() < keep_prob:
                kept_segments.append(seg)
        
        # Ensure at least one segment is kept
        if not kept_segments:
            kept_segments = [max(segments, key=len)]
        
        return kept_segments

    def _smooth_points(self, points: list, window_size: int = 5) -> list:
        """Moving average smoothing."""
        if len(points) < window_size * 2:
            return points
        
        smoothed = []
        half_win = window_size // 2
        
        for i in range(len(points)):
            start = max(0, i - half_win)
            end = min(len(points), i + half_win + 1)
            
            avg_x = sum(p[0] for p in points[start:end]) / (end - start)
            avg_y = sum(p[1] for p in points[start:end]) / (end - start)
            
            smoothed.append([int(avg_x), int(avg_y)])
        
        return smoothed

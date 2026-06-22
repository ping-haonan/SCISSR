import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ScribbleEncoderSimple(nn.Module):
    """
    简化版的ScribbleEncoder，用于实例分割，只处理单通道scribble输入
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        scribble_in_chans: int = 1,  # 单通道输入
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.scribble_in_chans = scribble_in_chans

        # 构建卷积网络
        self.conv1 = nn.Conv2d(scribble_in_chans, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, scribbles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scribbles: (B, 1, H, W) 单通道scribble输入
        Returns:
            embeddings: (B, embed_dim, H', W') scribble embeddings
        """
        # 将scribble resize到embedding size
        scribbles_resized = F.interpolate(
            scribbles, 
            size=self.image_embedding_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 通过卷积网络
        x = self.relu(self.conv1(scribbles_resized))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x

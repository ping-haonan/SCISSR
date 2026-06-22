"""
ScribbleEncoder V2: With Explicit Positive/Negative Embeddings

Key Improvements:
1. Separate embeddings for positive and negative scribbles (similar to SAM2's point embeddings)
2. Better handling of tri-valued scribble maps [-1, 0, +1]
3. More explicit semantic guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from sam2.modeling.sam2_utils import LayerNorm2d


class ScribbleEncoderV2(nn.Module):
    """
    Improved scribble encoder with explicit positive/negative embeddings.
    Similar to how SAM2's PromptEncoder handles point labels.
    """
    
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int], 
        input_image_size: Tuple[int, int], 
        scribble_in_chans: int = 16,
        activation = nn.GELU,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main scribble downscaling network (similar to original)
        self.scribble_downscaling = nn.Sequential(
            nn.Conv2d(1, scribble_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(scribble_in_chans // 4),
            activation(),
            nn.Conv2d(scribble_in_chans // 4, scribble_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(scribble_in_chans),
            activation(),
            nn.Conv2d(scribble_in_chans, embed_dim, kernel_size=1),
        ).to(device)
        
        # 🔑 Key addition: Learnable embeddings for positive and negative scribbles
        # Similar to SAM2's point_embeddings[0] (negative) and point_embeddings[1] (positive)
        self.positive_scribble_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, 1) * 0.01
        ).to(device)
        
        self.negative_scribble_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, 1) * 0.01
        ).to(device)
        
        # Default embedding if no scribble is provided
        self.no_scribble_embed = nn.Embedding(1, embed_dim).to(device)
        
        print(f"✅ ScribbleEncoderV2 initialized with explicit pos/neg embeddings")
    
    def forward(self, scribbles: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode scribble inputs with explicit positive/negative distinction.
        
        Args:
            scribbles: Tri-valued scribble map of shape (B, 1, H, W)
                      Values: +1 (positive), 0 (background), -1 (negative)
        
        Returns:
            Dense embeddings with shape (B, embed_dim, embed_H, embed_W)
        """
        if scribbles is None:
            # Return default embedding
            B = 1
            scribble_embedding = self.no_scribble_embed.weight.reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
            return scribble_embedding
        
        # Resize to target size
        target_size = (256, 256)
        if scribbles.shape[-2:] != target_size:
            scribbles = F.interpolate(
                scribbles,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        B = scribbles.shape[0]
        
        # Separate positive and negative scribbles
        # positive_mask: where scribbles > 0.5
        # negative_mask: where scribbles < -0.5
        positive_mask = (scribbles > 0.5).float()  # (B, 1, H, W)
        negative_mask = (scribbles < -0.5).float()  # (B, 1, H, W)
        
        # Process positive scribbles
        if positive_mask.sum() > 0:
            pos_embedding = self.scribble_downscaling(positive_mask)  # (B, embed_dim, H', W')
        else:
            # No positive scribbles
            embed_h, embed_w = self.image_embedding_size
            pos_embedding = torch.zeros(
                B, self.embed_dim, embed_h, embed_w,
                device=scribbles.device
            )
        
        # Process negative scribbles
        if negative_mask.sum() > 0:
            neg_embedding = self.scribble_downscaling(negative_mask)  # (B, embed_dim, H', W')
        else:
            # No negative scribbles
            embed_h, embed_w = self.image_embedding_size
            neg_embedding = torch.zeros(
                B, self.embed_dim, embed_h, embed_w,
                device=scribbles.device
            )
        
        # 🔑 Key: Add learnable embeddings to distinguish pos/neg semantics
        # Similar to: point_embedding[labels == 0] += point_embeddings[0].weight
        embed_h, embed_w = pos_embedding.shape[-2:]
        pos_semantic = self.positive_scribble_embed.expand(B, -1, embed_h, embed_w)
        neg_semantic = self.negative_scribble_embed.expand(B, -1, embed_h, embed_w)
        
        pos_embedding = pos_embedding + pos_semantic  # Add positive semantic
        neg_embedding = neg_embedding + neg_semantic  # Add negative semantic
        
        # Combine: positive adds, negative subtracts
        # This matches the intuition: pos pushes toward foreground, neg pushes toward background
        final_embedding = pos_embedding - neg_embedding
        
        return final_embedding


class ScribbleEncoderV2Alt(nn.Module):
    """
    Alternative V2: Use absolute value for magnitude, then add semantic embeddings.
    """
    
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int], 
        input_image_size: Tuple[int, int], 
        scribble_in_chans: int = 16,
        activation = nn.GELU,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Scribble downscaling (processes magnitude)
        self.scribble_downscaling = nn.Sequential(
            nn.Conv2d(1, scribble_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(scribble_in_chans // 4),
            activation(),
            nn.Conv2d(scribble_in_chans // 4, scribble_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(scribble_in_chans),
            activation(),
            nn.Conv2d(scribble_in_chans, embed_dim, kernel_size=1),
        ).to(device)
        
        # Learnable semantic embeddings
        self.positive_scribble_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, 1) * 0.02
        ).to(device)
        
        self.negative_scribble_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, 1) * 0.02
        ).to(device)
        
        self.no_scribble_embed = nn.Embedding(1, embed_dim).to(device)
        
        print(f"✅ ScribbleEncoderV2Alt initialized")
    
    def forward(self, scribbles: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Alternative approach: encode magnitude, then add directional semantic.
        """
        if scribbles is None:
            B = 1
            return self.no_scribble_embed.weight.reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        
        # Resize
        target_size = (256, 256)
        if scribbles.shape[-2:] != target_size:
            scribbles = F.interpolate(
                scribbles, size=target_size,
                mode='bilinear', align_corners=False
            )
        
        B = scribbles.shape[0]
        
        # Encode the magnitude (absolute value)
        scribble_magnitude = torch.abs(scribbles)  # (B, 1, H, W), values in [0, 1]
        magnitude_embedding = self.scribble_downscaling(scribble_magnitude)
        
        # Get sign masks
        positive_mask = (scribbles > 0.5).float()  # (B, 1, H, W)
        negative_mask = (scribbles < -0.5).float()  # (B, 1, H, W)
        
        # Downsample masks to match embedding size
        embed_size = magnitude_embedding.shape[-2:]
        pos_mask_down = F.interpolate(positive_mask, size=embed_size, mode='bilinear', align_corners=False)
        neg_mask_down = F.interpolate(negative_mask, size=embed_size, mode='bilinear', align_corners=False)
        
        # Add semantic embeddings based on sign
        embed_h, embed_w = embed_size
        semantic_embedding = torch.zeros_like(magnitude_embedding)
        
        # Where positive: add positive semantic
        semantic_embedding += pos_mask_down * self.positive_scribble_embed.expand(B, -1, embed_h, embed_w)
        
        # Where negative: add negative semantic
        semantic_embedding += neg_mask_down * self.negative_scribble_embed.expand(B, -1, embed_h, embed_w)
        
        # Combine magnitude and semantics
        final_embedding = magnitude_embedding + semantic_embedding
        
        return final_embedding


if __name__ == '__main__':
    # Test
    print("Testing ScribbleEncoderV2...")
    
    encoder_v2 = ScribbleEncoderV2(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        scribble_in_chans=16
    )
    
    # Test input
    B = 2
    scribbles = torch.randn(B, 1, 1024, 1024)
    scribbles[scribbles > 0.5] = 1.0   # Positive
    scribbles[scribbles < -0.5] = -1.0  # Negative
    scribbles[(scribbles >= -0.5) & (scribbles <= 0.5)] = 0.0  # Background
    
    print(f"Input shape: {scribbles.shape}")
    print(f"Positive pixels: {(scribbles > 0.5).sum()}")
    print(f"Negative pixels: {(scribbles < -0.5).sum()}")
    
    output = encoder_v2(scribbles)
    print(f"Output shape: {output.shape}")
    print(f"✅ Test passed!")


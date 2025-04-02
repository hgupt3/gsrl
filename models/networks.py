import torch
from torch import nn
import os
from torch.nn import functional as F
from torchvision import models
import random
from torchvision.transforms import v2
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

import albumentations as A
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block

class classification_net(nn.Module):
    """
    Classification model that uses a base model for feature extraction and ResNet18 for classification.
    Supports data augmentation during training.
    """
    def __init__(self, base_model, num_classes=16):
        super().__init__()
        self.base_model = base_model
        self.base_model.freeze_projection()

        # Initialize ResNet18 backbone for classification
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cls_fc = nn.Linear(in_features=128,out_features=16)
        self.fc1 = nn.Linear(in_features=self.backbone.fc.in_features+16,out_features=num_classes)   
        self.backbone.fc = nn.Identity()        
    
    def forward(self, x, c):
        # Extract features using base model
        batch = self.base_model(x, c)
        x = batch['proj']
        
        x = self.backbone(x)
        c = self.cls_fc(batch['cls_token'])
        x = torch.cat([x, c], dim=1)
        x = self.fc1(x)
        return x

class pose_estimation_net(nn.Module):
    """
    Pose estimation model that processes pairs of images to estimate relative pose.
    Uses a modified ResNet18 architecture with 6 input channels for paired images.
    """
    def __init__(self, base_model, augment=False):
        super().__init__()
        self.base_model = base_model
        self.base_model.freeze_projection()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(in_channels=6, 
            out_channels=self.backbone.conv1.out_channels,
            stride=self.backbone.conv1.stride,
            kernel_size=self.backbone.conv1.kernel_size,
            padding=self.backbone.conv1.padding,
        )
        self.backbone.fc = nn.Linear(in_features=self.backbone.fc.in_features, out_features=3)
    
    def forward(self, x0, x1, c):
        # Process both input images through base model
        x0 = self.base_model(x0, c)['proj']
        x1 = self.base_model(x1, c)['proj']
            
        # Concatenate features and predict pose
        x = torch.cat([x0, x1], dim=1)
        x = self.backbone(x)
        return x

class MLP_net(nn.Module):
    """
    MLP network for SITR
    """
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(in_features=self.base_model.cls_token.shape[-1], out_features=num_classes)
    
    def forward(self, x):
        # Process both input images through base model
        x = self.base_model(x)['latent'][:,0,:] # full cls token
        x = self.fc1(x)
        return x

class SITR(nn.Module):
    """
    Sensor-Invariant Tactile Representation (SITR) model.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 num_calibration=18):
        super().__init__()

        # Patch embedding and positional encoding setup
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Calibration-specific components
        self.num_calibration = num_calibration
        if self.num_calibration > 0:
            self.c_patch_embed = PatchEmbed(img_size, patch_size, num_calibration*3, embed_dim)
            self.c_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        # Token and positional embedding initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Decoder components
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True) 
        self.contrastive_head = nn.Linear(embed_dim, 128)

        self.initialize_weights()
        
    def freeze_projection(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        """Initialize model weights using various initialization strategies."""
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        if self.num_calibration > 0:
            c_pos_embed = get_2d_sincos_pos_embed(self.c_pos_embed.shape[-1], int(self.c_patch_embed.num_patches**.5))
            self.c_pos_embed.data.copy_(torch.from_numpy(c_pos_embed).float().unsqueeze(0))

        # Initialize patch embeddings
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        if self.num_calibration > 0:
            w = self.c_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize classification token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize linear layers and layer normalization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear layers and layer normalization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        Convert patch embeddings back to image format.
        Args:
            x: (N, L, patch_size**2 *3) tensor of patch embeddings
        Returns:
            imgs: (N, 3, H, W) tensor of reconstructed images
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def forward_encoder(self, x, c):
        """Forward pass through the encoder part of the model."""
        # Embed patches
        x = self.patch_embed(x)

        # Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        # Append classification token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Process calibration data if available
        if self.num_calibration > 0:
            c = self.c_patch_embed(c)
            c = c + self.c_pos_embed
            x = torch.cat((x, c), dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        if self.num_calibration > 0:
            x = x[:, :self.patch_embed.num_patches+1, :]

        return x

    def forward_decoder(self, x):
        """Forward pass through the decoder part of the model."""
        # Predict pixel values
        x = self.decoder_pred(x)

        # Remove classification token
        x = x[:, 1:, :]
        
        # Convert patches back to image format
        x = self.unpatchify(x)

        return x

    def forward(self, x, c):
        """
        Complete forward pass of the model.
        Returns reconstructed image, latent features, and contrastive embeddings.
        """
        latent = self.forward_encoder(x, c)
        cls_token = latent[:, 0, :]
        cls_token = self.contrastive_head(cls_token)
        cls_token = nn.functional.normalize(cls_token)
        
        proj = self.forward_decoder(latent) 
        return {'proj': proj, 'latent': latent, 'cls_token': cls_token}

def SITR_base(num_calibration=18, **kwargs):
    """Factory function to create a base SITR model with default parameters."""
    model = SITR(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, num_calibration=num_calibration, **kwargs)
    return model
    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    Args:
        embed_dim: embedding dimension
        grid_size: size of the grid
        cls_token: whether to include a classification token
    Returns:
        pos_embed: positional embedding tensor
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sinusoidal positional embeddings from a grid."""
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings.
    Args:
        embed_dim: embedding dimension
        pos: positions to encode
    Returns:
        emb: positional embedding tensor
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
    
if __name__ == '__main__':
    # Test model initialization and forward pass
    net = SITR_base()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")
    x = torch.rand([5,3,224,224])
    c = torch.rand([5,24,224,224])
    print(net(x, c).shape)
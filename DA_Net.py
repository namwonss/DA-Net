import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_
import math

# This is the official code of DA-Net for haze removal in remote sensing images (RSI).
# DA-Net: Dual Attention Network for Haze Removal in Remote Sensing Image
# IEEE Access
# 09/12/2024
# Namwon Kim (namwon@korea.ac.kr)

class ChannelBranch(nn.Module):
	#Channel Branch
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0),-1)
        channel_att_raw = self.fc( avg_pool )
        channel_att = torch.sigmoid( channel_att_raw ).unsqueeze(-1).unsqueeze(-1)
        return x*channel_att  
class SpatialBranch(nn.Module):
	# Spatial Branch
    def __init__(self, in_channels):
        super(SpatialBranch, self).__init__()
        self.spatial = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, padding_mode='reflect'),
                nn.Sigmoid()
                )
    def forward(self, x):
        scale = self.spatial(x)
        return x * scale
# Channel Spatial Attention Module
class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSpatialAttentionModule, self).__init__()
        self.channel_attention = ChannelBranch(in_channels)
        self.spatial_attention = SpatialBranch(in_channels)
    def forward(self, x):
        out = self.channel_attention(x) + self.spatial_attention(x)
        return out


class LocalChannelAttention(nn.Module):
    def __init__(self, dim):
        super(LocalChannelAttention, self).__init__()

        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, padding_mode='reflect')
        
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.Sigmoid()
        )
    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att = att.reshape(N, C, 1, 1)
        out = ((x * att) + x) + (self.local(x)*x)
        return out


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.Mish(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class DualAttentionBlock(nn.Module):
    def __init__(self, dim, network_depth):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.dim = dim

        # shallow feature extraction layer
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1) # main
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect') # main

        self.attn = ChannelSpatialAttentionModule(dim)

        # Local Channel Attention
        self.gp = LocalChannelAttention(dim)

        # Global Channel Attention
        self.cam = GlobalChannelAttention(dim)

        # Spatial Attention
        self.pam = SpatialAttention(dim)

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * 4.), out_features=dim)
        self.mlp2 = Mlp(network_depth, dim*3, hidden_features=int(dim * 4.), out_features=dim)

    def forward(self, x):
        # Channel Spatial Attention Module
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.mlp(x)
        x = identity + x

        # Parallel Attention Module
        identity = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.gp(x), self.cam(x), self.pam(x)], dim=1)
        x = self.mlp2(x)
        x = identity + x

        return x


# Global Channel Attention
class GlobalChannelAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super(GlobalChannelAttention, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.ca(x) * x

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super(SpatialAttention, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.spatial(x)*x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, network_depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [DualAttentionBlock(dim=dim, network_depth=network_depth) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')
    def forward(self, x):
        x = self.proj(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )
    def forward(self, x):
        x = self.proj(x)
        return x


class DA_Net_model(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1]):
        super(DA_Net_model, self).__init__()
        self.patch_size = 4

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], network_depth=sum(depths))

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)
        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], network_depth=sum(depths))

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], network_depth=sum(depths))

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3], network_depth=sum(depths))

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4], network_depth=sum(depths))
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=1)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
	
    def forward_features(self, x):

        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        feat = self.forward_features(x)
        
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x

def DA_Net_t():
    return DA_Net_model(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

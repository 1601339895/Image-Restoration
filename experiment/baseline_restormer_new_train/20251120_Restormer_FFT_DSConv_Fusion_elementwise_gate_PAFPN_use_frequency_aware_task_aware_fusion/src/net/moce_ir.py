# -*- coding: utf-8 -*-
# File  : FPN_Restormer_CA_CNN_Encoder.py
# Author: HeLei
# Date  : 2025/11/20

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torchsummary import summary  # 用于参数量统计
import cv2  # 用于图像读取和显示（测试用）
import numpy as np
import numbers


##########################################################################
## 基础工具模块（激活函数、归一化等）
##########################################################################
# GELU激活函数（独立实现，避免版本依赖）
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# LayerNorm（复用原Restormer代码，适配解码器）
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## 解码器依赖模块
##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias, gate_type=None):
        super(GatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gate_type = gate_type  # 门控类型
        self.bias = bias
        self.dim = dim

        # 1. 计算QKV+门控分数的总通道数
        if gate_type is None:
            # 无门控：保持原MDTA的QKV通道数（dim*3）
            self.qkv_out_channels = dim * 3
        elif gate_type == 'headwise':
            # Headwise门控：Q额外输出num_heads个通道（每个头1个标量门控）
            # 总通道数 = (Q+gate) + K + V = (dim + num_heads) + dim + dim = dim*3 + num_heads
            self.qkv_out_channels = dim * 3 + num_heads
        elif gate_type == 'elementwise':
            # Elementwise门控：Q额外输出dim个通道（与Q同维度，逐元素门控）
            # 总通道数 = (Q+gate) + K + V = (dim + dim) + dim + dim = dim*4
            self.qkv_out_channels = dim * 4
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}, choose from None, 'headwise', 'elementwise'")

        # 2. 初始化QKV投影（含门控分数）和Depthwise Conv（保持原MDTA结构）
        self.qkv = nn.Conv2d(dim, self.qkv_out_channels, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.qkv_out_channels, self.qkv_out_channels, kernel_size=3, stride=1, padding=1,
                                    groups=self.qkv_out_channels, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape  # 输入维度：(batch, dim, height, width)
        head_dim = c // self.num_heads  # 每个头的维度

        # 步骤1：QKV+门控分数的投影与Depthwise Conv（保持原MDTA流程）
        qkv_with_gate = self.qkv_dwconv(self.qkv(x))  # (b, qkv_out_channels, h, w)

        # 步骤2：拆分Q、K、V和门控分数
        if self.gate_type is None:
            # 无门控：直接拆分QKV
            q, k, v = qkv_with_gate.chunk(3, dim=1)  # q: (b, dim, h, w)
            gate_score = None
        elif self.gate_type == 'headwise':
            # Headwise门控：先拆分 (Q+gate)、K、V，再拆分Q和gate_score
            q_with_gate, k, v = qkv_with_gate.split([self.dim + self.num_heads, self.dim, self.dim],
                                                    dim=1)  # q_with_gate: (b, dim+num_heads, h, w)
            q, gate_score = q_with_gate.split([self.dim, self.num_heads],
                                              dim=1)  # q: (b, dim, h, w); gate_score: (b, num_heads, h, w)
        elif self.gate_type == 'elementwise':
            # Elementwise门控：先拆分 (Q+gate)、K、V，再拆分Q和gate_score
            q_with_gate, k, v = qkv_with_gate.split([self.dim * 2, self.dim, self.dim],
                                                    dim=1)  # q_with_gate: (b, 2*dim, h, w)
            q, gate_score = q_with_gate.chunk(2, dim=1)  # q: (b, dim, h, w); gate_score: (b, dim, h, w)

        # 步骤3：注意力计算（保持原MDTA的转置注意力逻辑）
        # 维度重排：(b, dim, h, w) → (b, num_heads, head_dim, h*w)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)

        # QK归一化与注意力权重计算
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, head, h*w, h*w)
        attn = attn.softmax(dim=-1)

        # SDPA输出：(b, head, c, h*w)
        out = attn @ v  # (b, num_heads, head_dim, h*w)

        # 步骤4：应用门控（论文核心：SDPA输出后施加sigmoid乘性门控）
        if self.gate_type is not None:
            # 调整门控分数维度，与out匹配
            if self.gate_type == 'headwise':
                # Headwise：gate_score → (b, num_heads, 1, h*w)（标量广播到整个头维度）
                gate_score = rearrange(gate_score, 'b head h w -> b head 1 (h w)', head=self.num_heads)
            elif self.gate_type == 'elementwise':
                # Elementwise：gate_score → (b, num_heads, head_dim, h*w)（与out完全同维度）
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)

            # 乘性sigmoid门控：动态过滤信息流
            out = out * torch.sigmoid(gate_score)

        # 步骤5：维度恢复与输出投影（保持原MDTA流程）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # (b, dim, h, w)
        out = self.project_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,gate_type=None):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        self.attn = GatedMDTA(dim, num_heads, bias, gate_type=gate_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## 编码器核心模块（轻量化FFT+深度可分离CNN）
##########################################################################
# ECA通道注意力（轻量，适配concat融合）
class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# 深度可分离卷积封装
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 核心：轻量化FFT-深度可分离CNN Block
class Light_FFT_DSConv_Block(nn.Module):
    def __init__(self, dim, bias, dilation_rate=1):
        super(Light_FFT_DSConv_Block, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2  # 空域/频域各占C/2通道

        # 空域分支
        self.spatial_branch = nn.Sequential(
            nn.BatchNorm2d(self.half_dim),
            GELU(),
            DepthwiseSeparableConv(
                in_channels=self.half_dim,
                out_channels=self.half_dim,
                kernel_size=3,
                padding=dilation_rate,
                dilation=dilation_rate,
                bias=bias
            ),
            nn.BatchNorm2d(self.half_dim)
        )

        # 频域分支
        self.fft_branch = nn.Sequential(
            nn.Conv2d(dim, self.half_dim, kernel_size=1, bias=bias),  # 实部+虚部（C通道）→ C/2
            nn.BatchNorm2d(self.half_dim),
            GELU(),
            DepthwiseSeparableConv(
                in_channels=self.half_dim,
                out_channels=self.half_dim,
                kernel_size=3,
                padding=dilation_rate,
                dilation=dilation_rate,
                bias=bias
            ),
            nn.BatchNorm2d(self.half_dim)
        )

        # 融合+通道注意力
        self.eca = ECA(channel=dim)
        self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        b, c, h, w = x.shape

        # 通道拆分
        spatial_x, fft_x = x.chunk(2, dim=1)

        # 空域分支前向
        spatial_out = self.spatial_branch(spatial_x)

        # 频域分支前向
        fft = torch.fft.fft2(fft_x, dim=(-2, -1))
        fft_feat = torch.cat([fft.real, fft.imag], dim=1)
        fft_out = self.fft_branch(fft_feat)

        # 融合+注意力+残差
        fusion_out = torch.cat([spatial_out, fft_out], dim=1)
        fusion_out = self.eca(fusion_out)
        return fusion_out + residual


##########################################################################
## 多尺度融合模块（FPN/PAFPN）
##########################################################################
# 下采样/上采样模块（复用原Restormer，确保一致性）
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# FPN/PAFPN专用下采样（保持通道数不变）
class DownsampleKeepChannels(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleKeepChannels, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # (n_feat//4) * 4 = n_feat (保持通道数)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# FPN/PAFPN专用上采样（保持通道数不变）
class UpsampleKeepChannels(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleKeepChannels, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 4*n_feat / 4 = n_feat (保持通道数)
        )

    def forward(self, x):
        return self.body(x)


# FPN融合模块（自上而下）
class FPN_Fusion(nn.Module):
    def __init__(self, dims=[48, 96, 192, 384], bias=False):
        super(FPN_Fusion, self).__init__()
        self.dims = dims  # [level1, level2, level3, latent]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dims[i], dims[1], kernel_size=1, bias=bias) for i in range(4)
        ])
        # 上采样层：使用保持通道数的上采样（因为lateral_convs已将所有特征统一到dims[1]=96）
        self.upsamples = nn.ModuleList([
            UpsampleKeepChannels(dims[1]),  # 输入96通道，输出96通道
            UpsampleKeepChannels(dims[1]),
            UpsampleKeepChannels(dims[1])
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(dims[1], dims[i], kernel_size=3, stride=1, padding=1, bias=bias) for i in range(4)
        ])

    def forward(self, features):
        # features: [level1, level2, level3, latent]
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 自上而下融合：latent -> level3 -> level2 -> level1
        fused_feats = [lateral_feats[3]]  # 从latent开始

        # latent -> level3
        up_feat = self.upsamples[0](fused_feats[0])  # 上采样latent
        fused = up_feat + lateral_feats[2]  # 与level3融合
        fused_feats.append(fused)

        # level3 -> level2
        up_feat = self.upsamples[1](fused_feats[1])  # 上采样level3
        fused = up_feat + lateral_feats[1]  # 与level2融合
        fused_feats.append(fused)

        # level2 -> level1
        up_feat = self.upsamples[2](fused_feats[2])  # 上采样level2
        fused = up_feat + lateral_feats[0]  # 与level1融合
        fused_feats.append(fused)

        fused_feats = fused_feats[::-1]  # 恢复为[level1, level2, level3, latent]
        output_feats = [conv(feat) for conv, feat in zip(self.output_convs, fused_feats)]
        return output_feats


# PAFPN融合模块（FPN+自下而上增强）
class PAFPN_Fusion(nn.Module):
    def __init__(self, dims=[48, 96, 192, 384], bias=False):
        super(PAFPN_Fusion, self).__init__()
        self.dims = dims
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dims[i], dims[1], kernel_size=1, bias=bias) for i in range(4)
        ])
        # 上采样层：使用保持通道数的上采样
        self.upsamples = nn.ModuleList([
            UpsampleKeepChannels(dims[1]),  # 输入96通道，输出96通道
            UpsampleKeepChannels(dims[1]),
            UpsampleKeepChannels(dims[1])
        ])
        # 下采样层：使用保持通道数的下采样
        self.downsamples = nn.ModuleList([
            DownsampleKeepChannels(dims[1]),  # 输入96通道，输出96通道
            DownsampleKeepChannels(dims[1])
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(dims[1], dims[i], kernel_size=3, stride=1, padding=1, bias=bias) for i in range(4)
        ])

    def forward(self, features):
        # 步骤1：侧向卷积统一通道到dims[1]
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 步骤2：自上而下融合（Top-Down）
        fused_feats = [lateral_feats[3]]  # 从latent开始

        # latent -> level3
        up_feat = self.upsamples[0](fused_feats[0])
        fused = up_feat + lateral_feats[2]
        fused_feats.append(fused)

        # level3 -> level2
        up_feat = self.upsamples[1](fused_feats[1])
        fused = up_feat + lateral_feats[1]
        fused_feats.append(fused)

        # level2 -> level1
        up_feat = self.upsamples[2](fused_feats[2])
        fused = up_feat + lateral_feats[0]
        fused_feats.append(fused)

        fused_feats = fused_feats[::-1]  # [level1, level2, level3, latent]

        # 步骤3：自下而上增强（Bottom-Up）
        enhanced_feats = [fused_feats[0]]  # 从level1开始

        # level1 -> level2
        down_feat = self.downsamples[0](enhanced_feats[0])
        enhanced = down_feat + fused_feats[1]
        enhanced_feats.append(enhanced)

        # level2 -> level3
        down_feat = self.downsamples[1](enhanced_feats[1])
        enhanced = down_feat + fused_feats[2]
        enhanced_feats.append(enhanced)

        enhanced_feats.append(fused_feats[3])  # 保持latent不变

        # 步骤4：恢复原始通道
        output_feats = [conv(feat) for conv, feat in zip(self.output_convs, enhanced_feats)]
        return output_feats


##########################################################################
## Patch Embedding（复用原Restormer）
##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


##########################################################################
## 最终模型：FFT-深度可分离CNN + FPN/PAFPN + Transformer解码器
##########################################################################
class Restormer_FFT_DSConv_Fusion(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],  # [level1, level2, level3, latent]的Block数
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],  # 解码器注意力头数
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 fusion_type='PAFPN',  # 可选：'None'（无融合）、'FPN'、'PAFPN'
                 gate_type = None,  ## 全局门控类型：None/'headwise'/'elementwise'
                 ):
        super(Restormer_FFT_DSConv_Fusion, self).__init__()

        self.gate_type = gate_type
        self.fusion_type = fusion_type
        self.dual_pixel_task = dual_pixel_task

        # 1. Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 2. 编码器（FFT-深度可分离CNN）
        self.encoder_level1 = nn.Sequential(*[
            Light_FFT_DSConv_Block(dim=dim, bias=bias, dilation_rate=1)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            Light_FFT_DSConv_Block(dim=int(dim * 2 ** 1), bias=bias, dilation_rate=2)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            Light_FFT_DSConv_Block(dim=int(dim * 2 ** 2), bias=bias, dilation_rate=4)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            Light_FFT_DSConv_Block(dim=int(dim * 2 ** 3), bias=bias, dilation_rate=8)
            for _ in range(num_blocks[3])
        ])

        # 3. 多尺度特征融合（FPN/PAFPN）
        if fusion_type in ['FPN', 'PAFPN']:
            dims = [dim, int(dim * 2 ** 1), int(dim * 2 ** 2), int(dim * 2 ** 3)]
            self.feature_fusion = FPN_Fusion(dims=dims, bias=bias) if fusion_type == 'FPN' else PAFPN_Fusion(dims=dims,
                                                                                                             bias=bias)
        else:
            self.feature_fusion = None  # 无融合

        # 4. 解码器（复用原Restormer）
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type)
            for _ in range(num_blocks[0])
        ])

        # 5. 细化模块
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type)
            for _ in range(num_refinement_blocks)
        ])

        # 6. 输出层
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # 1. 编码器前向
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # 2. 特征融合（可选）
        if self.fusion_type in ['FPN', 'PAFPN']:
            features = [out_enc_level1, out_enc_level2, out_enc_level3, latent]
            fused_level1, fused_level2, fused_level3, fused_latent = self.feature_fusion(features)
        else:
            fused_level1, fused_level2, fused_level3, fused_latent = out_enc_level1, out_enc_level2, out_enc_level3, latent

        # 3. 解码器前向
        inp_dec_level3 = self.up4_3(fused_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, fused_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, fused_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, fused_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        # 4. 输出
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out = self.output(out_dec_level1)
        else:
            out = self.output(out_dec_level1) + inp_img

        return out

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = Restormer_FFT_DSConv_Fusion(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],  # [level1, level2, level3, latent]的Block数
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],  # 解码器注意力头数
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
        fusion_type="PAFPN",  # 可选：'None'（无融合）、'FPN'、'PAFPN'
        gate_type = "elementwise",  # 可启用头级门控,elementwise\headwise

    ).cuda()
    inp = torch.randn(1, 3, 224, 224).cuda()  # (batch=1, channel=3, height=256, width=256)
    out = model(inp)
    print(out.shape)
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))
    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (inp))
    print(flop_count_table(flops))

    # model FPN parameters=11.403M  flops=94.504G   Max Memery : 7421.889 [M]  dim=48
    # model FPN elementwise parameters=11.775M  flops=0.1T    Max Memery : 8182.778 [M]  dim=48
    # model FPN headwise parameters=11.409M  flops=94.577G    Max Memery : 7621.647 [M]  dim=48

    # model PAFPN parameters=11.444M  flops=95.805G  Max Memery : 7426.856 [M]  dim=48
    # model PAFPN headwise parameters=11.451M  flops=95.878G    Max Memery : 7626.806 [M]  dim=48
    # model PAFPN headwise parameters=11.817M  flops=0.101T    Max Memery : 8188.808 [M]  dim=48

    # model None parameters=9.716M   flops=84.707G   Max Memery : 7372.983 [M]   dim =48
    # model None headwise parameters=9.723M   flops=84.78G   Max Memery : 7571.295 [M]  dim=48
    # model None elementwise parameters=10.089M  flops=90.238G    Max Memery : 8134.898 [M]  dim=48

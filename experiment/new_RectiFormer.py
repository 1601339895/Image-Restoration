# -*- coding: utf-8 -*-
# File  : dual_gate_restoration.py
# Author: HeLei
# Date  : 2025/12/17

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table

##########################################################################
# 复用Restormer的基础模块：LayerNorm、OverlapPatchEmbed、Downsample、Upsample
##########################################################################
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
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Restormer原始的OverlapPatchEmbed
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

# Restormer原始的Downsample
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 通道×4，尺寸÷2
        )

    def forward(self, x):
        return self.body(x)

# Restormer原始的Upsample
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 通道÷4，尺寸×2
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
# 你的创新模块：MSDRNet_NoModal（多尺度退化表征）
##########################################################################
class MSDRNet_NoModal(nn.Module):
    def __init__(self, context_dim=64, num_scales=3, dim_list=[48, 96, 192, 384]):
        super().__init__()
        self.context_dim = context_dim

        # 多尺度卷积：保持stride=1，匹配Restormer的PatchEmbed输出尺寸（H×W）
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=2 * s + 1, padding=s, stride=1)
            for s in range(num_scales)
        ])

        # 多尺度特征融合
        self.fusion = nn.Conv2d(16 * num_scales, context_dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.global_process = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim)
        )

        # 层级化Prompt生成（匹配Restormer的dim_list）
        self.layer_prompts = nn.ModuleList([
            nn.Linear(context_dim, dim) for dim in dim_list
        ])

    def forward(self, x):
        B = x.shape[0]

        # 1. 多尺度特征提取（输出尺寸：H×W）
        scale_feats = [conv(x) for conv in self.scale_convs]
        scale_feats = torch.cat(scale_feats, dim=1)
        feat = self.fusion(scale_feats)

        # 2. 全局特征池化
        global_feat = self.avg_pool(feat).squeeze(-1).squeeze(-1)

        # 3. 全局特征非线性变换
        global_feat = self.global_process(global_feat)

        # 4. 生成层级化Prompt（匹配Restormer的维度：48,96,192,384）
        layer_prompts = [fc(global_feat) for fc in self.layer_prompts]

        return layer_prompts, global_feat

##########################################################################
# 你的创新模块：MDPM（Prompt调制模块）
##########################################################################
class MDPM(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        # 1. 通道调制：仿射变换（gamma+beta）
        self.channel_proj = nn.Linear(context_dim, dim * 2)
        # 2. 空间调制：生成空间注意力图
        self.spatial_proj = nn.Linear(context_dim, dim)
        self.spatial_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        # 3. 注意力温度调制：生成动态温度因子
        self.temp_proj = nn.Linear(context_dim, 1)
        self.act = nn.GELU()

    def forward(self, x, context_emb):
        # x: [B, C, H, W]
        # context_emb: [B, dim]（层级化Prompt）
        B, C, H, W = x.shape

        # 1. 通道调制：仿射变换（gamma + beta）
        gamma, beta = self.channel_proj(context_emb).chunk(2, dim=-1)  # [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + gamma) + beta

        # 2. 空间调制：生成空间注意力图
        spatial_emb = self.spatial_proj(context_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        spatial_feat = x * spatial_emb
        spatial_attn = self.spatial_conv(self.act(spatial_feat))  # [B, 1, H, W]
        x = x * spatial_attn

        # 3. 生成动态温度因子（用于后续Attention的温度调整）
        dynamic_temp = self.temp_proj(context_emb).squeeze(-1)  # [B]
        # 归一化温度因子到合理范围（避免过大/过小）
        dynamic_temp = torch.sigmoid(dynamic_temp) * 2  # 映射到[0, 2]
        dynamic_temp = dynamic_temp.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1] 便于广播

        return x, dynamic_temp  # 返回调制后的特征 + 动态温度因子

##########################################################################
# 你的创新模块：DGSB（退化导向的频域选择块）
##########################################################################
def irfft2_safe(real_part, imag_part, s, norm='backward'):
    """安全执行 irfft2，避免 ComplexHalf 问题"""
    # 强制提升到 float32 构造复数
    complex_tensor = torch.complex(real_part.to(torch.float32), imag_part.to(torch.float32))
    out = torch.fft.irfft2(complex_tensor, s=s, norm=norm)
    # 转回原始 dtype（如 float16）
    return out.to(real_part.dtype)

class DGSB(nn.Module):
    def __init__(self, dim, context_dim=64):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim

        # 频域分支：高低频分别卷积
        self.high_freq_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.low_freq_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)

        # 空间分支：深度可分离卷积（轻量化）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        # 退化导向的融合权重（由全局退化特征控制）
        self.fusion_proj = nn.Linear(context_dim, 3)  # 低频、高频、空间的权重
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x, global_feat):
        B, C, H, W = x.shape
        original_dtype = x.dtype  # 记录原始类型（可能是 float16）

        # 1. 空间域分支（保持原 dtype）
        spatial_feat = self.spatial_conv(x)  # [B, C, H, W]

        # 2. 频率域分支：强制使用 float32 避免 ComplexHalf
        x_32 = x.to(torch.float32)  # 关键：避免 float16 → complex32
        x_fft = torch.fft.rfft2(x_32, norm='backward')  # 返回 complex64

        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag
        x_fft_cat = torch.cat([x_fft_real, x_fft_imag], dim=1)  # [B, 2C, H, W//2+1]

        # 生成高斯掩码（rfft2 的低频在 (0,0)）
        f_h, f_w = x_fft_real.shape[2], x_fft_real.shape[3]
        y, x_grid = torch.meshgrid(
            torch.arange(f_h, device=x.device),
            torch.arange(f_w, device=x.device),
            indexing='ij'
        )
        # 高斯中心为 (0, 0)，sigma匹配Restormer的空间维度
        dist = torch.sqrt(x_grid.float() ** 2 + y.float() ** 2)
        sigma = max(H, W) / 8.0
        low_freq_mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        high_freq_mask = 1 - low_freq_mask

        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0).expand_as(x_fft_cat)
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0).expand_as(x_fft_cat)

        low_fft = x_fft_cat * low_freq_mask
        high_fft = x_fft_cat * high_freq_mask

        low_fft_feat = self.low_freq_conv(low_fft)
        high_fft_feat = self.high_freq_conv(high_fft)

        # 拆分并逆变换
        low_real, low_imag = torch.chunk(low_fft_feat, 2, dim=1)
        high_real, high_imag = torch.chunk(high_fft_feat, 2, dim=1)

        low_ifft = irfft2_safe(low_real, low_imag, s=(H, W), norm='backward')
        high_ifft = irfft2_safe(high_real, high_imag, s=(H, W), norm='backward')

        # 3. 融合（保持原 dtype）
        fusion_weights = F.softmax(self.fusion_proj(global_feat), dim=-1)
        low_w, high_w, spatial_w = fusion_weights.unbind(-1)
        low_w = low_w.view(B, 1, 1, 1)
        high_w = high_w.view(B, 1, 1, 1)
        spatial_w = spatial_w.view(B, 1, 1, 1)

        fused = torch.cat([
            low_ifft * low_w,
            high_ifft * high_w,
            spatial_feat * spatial_w
        ], dim=1)

        out = self.fusion(fused)
        return out + x

##########################################################################
# 你的创新模块：GatedSpatialFFN（替换Restormer的FeedForward）
##########################################################################
class GatedSpatialFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GatedSpatialFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        # 创新：将输入投影为两份，一份用于内容，一份用于门控
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 门控分支的深度卷积：负责捕捉局部空间上下文
        self.dwconv_gate = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)

        # 内容分支的深度卷积：负责特征变换
        self.dwconv_content = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                        groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 1. 投影扩展
        x = self.project_in(x)
        x_content, x_gate = x.chunk(2, dim=1)

        # 2. 门控机制
        # Gate分支：先过DWConv感知邻域，再sigmoid生成门
        gate = torch.sigmoid(self.dwconv_gate(x_gate))

        # Content分支：标准DWConv + GELU
        content = F.gelu(self.dwconv_content(x_content))

        # 3. 门控应用 (Gating) - 空间上的选择性激活
        x = content * gate

        # 4. 投影回原维度
        x = self.project_out(x)
        return x

##########################################################################
# 你的创新模块：ElementwiseGatedAttention（替换Restormer的Attention）
##########################################################################
class ElementwiseGatedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ElementwiseGatedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim

        # Q, K, V 生成（匹配Restormer的MDTA结构）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)

        # 创新点：Gate 生成分支 - 深度可分离卷积感知局部上下文
        self.gate_generator = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),  # 局部感知
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 通道混合
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, dynamic_temp=None):
        b, c, h, w = x.shape

        # 1. 生成 Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 2. 生成 Elementwise Gate
        gate_score = self.gate_generator(x)

        # Reshape for MDTA（匹配Restormer的通道注意力计算）
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 动态温度调整（匹配Restormer的温度参数）
        if dynamic_temp is not None:
            temp = self.temperature.unsqueeze(0) * dynamic_temp
            attn = (q @ k.transpose(-2, -1)) * temp
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        # Reshape back
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 3. 应用 Elementwise Gating
        out = out * torch.sigmoid(gate_score)

        out = self.project_out(out)
        return out

##########################################################################
# 你的创新模块：DynamicTransformerBlock（替换Restormer的TransformerBlock）
# 对齐Restormer的流程，加入MDPM Prompt调制
##########################################################################
class DynamicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, context_dim):
        super(DynamicTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 替换为你的ElementwiseGatedAttention
        self.attn = ElementwiseGatedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 替换为你的GatedSpatialFFN
        self.ffn = GatedSpatialFFN(dim, ffn_expansion_factor, bias)
        # 加入MDPM Prompt调制
        self.prompt_fusion = MDPM(dim, context_dim)

    def forward(self, x, context_emb):
        # 对齐Restormer的流程：Norm → Attn → 残差
        residual = x
        x = self.norm1(x)
        # 加入Prompt调制
        x, dynamic_temp = self.prompt_fusion(x, context_emb)
        x = self.attn(x, dynamic_temp)
        x = residual + x

        # 对齐Restormer的流程：Norm → FFN → 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

##########################################################################
# 主模型：RectiFormer（完全对齐Restormer的结构和维度，修正Sequential问题）
##########################################################################
class RectiFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,  # Restormer基础维度
                 num_blocks=[4, 6, 6, 8],  # Restormer原始num_blocks
                 num_refinement_blocks=4,  # Restormer原始参数
                 heads=[1, 2, 4, 8],  # Restormer原始heads
                 ffn_expansion_factor=2.66,  # Restormer原始因子
                 bias=False,  # Restormer原始参数
                 LayerNorm_type='WithBias',  # Restormer默认值
                 context_dim=64,  # 全局退化特征维度
                 num_scales=3,  # 多尺度卷积数
                 dual_pixel_task=False  # Restormer原始参数
                 ):
        super().__init__()

        # Restormer的dim列表：[48, 96, 192, 384]
        self.dim_list = [int(dim * 2 ** i) for i in range(4)]
        self.dual_pixel_task = dual_pixel_task

        # 1. 多尺度退化表征网络（生成层级化Prompt和全局退化特征）
        self.context_net = MSDRNet_NoModal(
            context_dim=context_dim,
            num_scales=num_scales,
            dim_list=self.dim_list
        )

        # 2. Restormer原始的Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        # 3. 编码器（Encoder）：替换Sequential为ModuleList，解决多参数问题
        self.encoder_level1 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[0],
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[0]
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim_list[0])  # 48 → 96，尺寸H/2
        self.encoder_level2 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[1],
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(self.dim_list[1])  # 96 → 192，尺寸H/4
        self.encoder_level3 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[2],
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[2]
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(self.dim_list[2])  # 192 → 384，尺寸H/8
        self.latent = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[3],
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[3]
            ) for _ in range(num_blocks[3])
        ])

        # 瓶颈层：退化导向的频域-空间融合（DGSB）
        self.freq_fusion = DGSB(dim=self.dim_list[3], context_dim=context_dim)

        # 4. 解码器（Decoder）：替换Sequential为ModuleList
        self.up4_3 = Upsample(self.dim_list[3])  # 384 → 192，尺寸H/4
        self.reduce_chan_level3 = nn.Conv2d(self.dim_list[3], self.dim_list[2], kernel_size=1, bias=bias)  # 384→192
        self.decoder_level3 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[2],
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[2]
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(self.dim_list[2])  # 192 → 96，尺寸H/2
        self.reduce_chan_level2 = nn.Conv2d(self.dim_list[2], self.dim_list[1], kernel_size=1, bias=bias)  # 192→96
        self.decoder_level2 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[1],
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(self.dim_list[1])  # 96 → 48，尺寸H
        # Restormer的Decoder Level1拼接后无1×1降维（48+48=96）
        self.decoder_level1 = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[1],  # 96维（拼接后）
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_blocks[0])
        ])

        # 精细化模块：替换Sequential为ModuleList
        self.refinement = nn.ModuleList([  # 原Sequential → ModuleList
            DynamicTransformerBlock(
                dim=self.dim_list[1],  # 96维
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_refinement_blocks)
        ])

        # Restormer的dual_pixel_task相关
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, self.dim_list[1], kernel_size=1, bias=bias)

        # 输出层：对齐Restormer（96→3）
        self.output = nn.Conv2d(self.dim_list[1], out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape

        # 1. 获取层级化Prompt和全局退化特征（匹配Restormer的dim_list：48,96,192,384）
        layer_prompts, global_feat = self.context_net(inp_img)
        p1, p2, p3, p4 = layer_prompts  # p1:48, p2:96, p3:192, p4:384

        # 2. Restormer原始的Patch Embedding（3→48，尺寸H×W）
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, 48, H, W]

        # 3. 编码器（Encoder）：遍历ModuleList，传入x和context_emb
        out_enc_level1 = inp_enc_level1
        for block in self.encoder_level1:
            out_enc_level1 = block(out_enc_level1, p1)  # p1:48维，匹配维度

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 48→96，尺寸H/2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level2:
            out_enc_level2 = block(out_enc_level2, p2)  # p2:96维，匹配维度

        inp_enc_level3 = self.down2_3(out_enc_level2)  # 96→192，尺寸H/4
        out_enc_level3 = inp_enc_level3
        for block in self.encoder_level3:
            out_enc_level3 = block(out_enc_level3, p3)  # p3:192维，匹配维度

        inp_enc_level4 = self.down3_4(out_enc_level3)  # 192→384，尺寸H/8
        latent = inp_enc_level4
        for block in self.latent:
            latent = block(latent, p4)  # p4:384维，匹配维度

        # 瓶颈层：退化导向的频域-空间融合
        latent = self.freq_fusion(latent, global_feat)

        # 4. 解码器（Decoder）：遍历ModuleList，传入x和context_emb
        inp_dec_level3 = self.up4_3(latent)  # 384→192，尺寸H/4
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # 192+192=384
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # 384→192
        out_dec_level3 = inp_dec_level3
        for block in self.decoder_level3:
            out_dec_level3 = block(out_dec_level3, p3)  # p3:192维，匹配维度

        inp_dec_level2 = self.up3_2(out_dec_level3)  # 192→96，尺寸H/2
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # 96+96=192
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 192→96
        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level2:
            out_dec_level2 = block(out_dec_level2, p2)  # p2:96维，匹配维度

        inp_dec_level1 = self.up2_1(out_dec_level2)  # 96→48，尺寸H
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # 48+48=96
        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level1:
            out_dec_level1 = block(out_dec_level1, p2)  # p2:96维，匹配维度

        # 精细化模块：遍历ModuleList，传入x和context_emb
        for block in self.refinement:
            out_dec_level1 = block(out_dec_level1, p2)  # p2:96维，匹配维度

        # 5. 输出层：对齐Restormer的残差连接
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out = self.output(out_dec_level1)
        else:
            out = self.output(out_dec_level1) + inp_img

        return out

##########################################################################
# 测试代码（对齐Restormer的测试逻辑）
##########################################################################
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型（使用Restormer的原始配置）
    model = RectiFormer(
        dim=48,
        num_blocks=[4, 6, 6, 8],  # Restormer原始配置
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],  # Restormer原始配置
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',  # Restormer默认值
        context_dim=64,
        num_scales=3,
        dual_pixel_task=False
    ).to(device)

    # 测试输入（Restormer原始测试输入：224x224）
    batch_size = 1
    inp = torch.randn(batch_size, 3, 224, 224).to(device)

    # 前向推理
    out = model(inp)
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # 计算参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {model_params / 1e6:.3f} M")

    # 计算FLOPs（Restormer原始测试输入：224x224）
    flops = FlopCountAnalysis(model, inp)
    print("\nFLOPs Analysis:")
    print(flop_count_table(flops))

    # 测试显存占用（GPU）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\nMax GPU Memory Usage: {max_mem:.2f} MB")
        torch.cuda.empty_cache()

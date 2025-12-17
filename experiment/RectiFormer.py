# -*- coding: utf-8 -*-
# File  : dual_gate_restoration.py
# Author: HeLei
# Date  : 2025/12/17

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


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


class MSDRNet_NoModal(nn.Module):
    def __init__(self, context_dim=64, num_scales=3, dim_list=[48, 96, 192, 384]):
        super().__init__()
        self.context_dim = context_dim

        # 多尺度卷积：1x1, 3x3, 5x5...
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=2 * s + 1, padding=s, stride=2)
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

        # 层级化Prompt生成
        self.layer_prompts = nn.ModuleList([
            nn.Linear(context_dim, dim) for dim in dim_list
        ])

    def forward(self, x):
        B = x.shape[0]

        # 1. 多尺度特征提取
        scale_feats = [conv(x) for conv in self.scale_convs]
        scale_feats = torch.cat(scale_feats, dim=1)
        feat = self.fusion(scale_feats)

        # 2. 全局特征池化
        global_feat = self.avg_pool(feat).squeeze(-1).squeeze(-1)

        # 3. 全局特征非线性变换 (不再依赖模态权重)
        global_feat = self.global_process(global_feat)

        # 4. 生成层级化Prompt
        layer_prompts = [fc(global_feat) for fc in self.layer_prompts]

        return layer_prompts, global_feat


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
        # context_emb: [B, dim] (层级化Prompt) 或 [B, context_dim] (全局退化特征)
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
# 核心改进 3: 退化导向的频域选择块 (DGSB)
# 替代原FrequencySelectionBlock，实现退化导向的高低频选择与融合
##########################################################################

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

        # 1. 空间域分支
        spatial_feat = self.spatial_conv(x)  # [B, C, H, W]

        # 2. 频率域分支：FFT + 高低频分离
        x_fft = torch.fft.rfft2(x, norm='backward')  # [B, C, H, W//2+1] (complex)
        x_fft_real = x_fft.real  # [B, C, H, W//2+1]
        x_fft_imag = x_fft.imag  # [B, C, H, W//2+1]
        x_fft_cat = torch.cat([x_fft_real, x_fft_imag], dim=1)  # [B, 2C, H, W//2+1]

        # 生成高斯掩码实现高低频分离
        f_h, f_w = x_fft_real.shape[2], x_fft_real.shape[3]
        cx, cy = f_w // 2, f_h // 2  # 频域中心（低频）
        y, x_grid = torch.meshgrid(torch.arange(f_h), torch.arange(f_w), indexing='ij')
        y, x_grid = y.to(x.device), x_grid.to(x.device)
        dist = torch.sqrt((x_grid - cx) ** 2 + (y - cy) ** 2)  # 距离中心的距离
        low_freq_mask = torch.exp(-(dist ** 2) / (2 * (H // 8) ** 2))  # 低频掩码（高斯）
        high_freq_mask = 1 - low_freq_mask  # 高频掩码

        # 扩展掩码维度：[B, 2C, H, W//2+1]
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0).expand_as(x_fft_cat)
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0).expand_as(x_fft_cat)

        # 高低频分别卷积
        low_fft = x_fft_cat * low_freq_mask
        high_fft = x_fft_cat * high_freq_mask
        low_fft_feat = self.low_freq_conv(low_fft)
        high_fft_feat = self.high_freq_conv(high_fft)

        # 逆FFT转换回空间域
        low_real, low_imag = torch.chunk(low_fft_feat, 2, dim=1)  # [B, C, H, W//2+1]
        high_real, high_imag = torch.chunk(high_fft_feat, 2, dim=1)
        low_ifft = torch.fft.irfft2(torch.complex(low_real, low_imag), s=(H, W), norm='backward')  # [B, C, H, W]
        high_ifft = torch.fft.irfft2(torch.complex(high_real, high_imag), s=(H, W), norm='backward')

        # 3. 退化导向的特征融合
        fusion_weights = F.softmax(self.fusion_proj(global_feat), dim=-1)  # [B, 3] (low, high, spatial)
        low_weight, high_weight, spatial_weight = fusion_weights.unbind(-1)

        # 权重广播：[B, 1, 1, 1]
        low_weight = low_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        high_weight = high_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        spatial_weight = spatial_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 加权融合
        fused_feat = torch.cat([
            low_ifft * low_weight,
            high_ifft * high_weight,
            spatial_feat * spatial_weight
        ], dim=1)  # [B, 3C, H, W]
        out = self.fusion(fused_feat)  # [B, C, H, W]

        return out + x  # 残差连接


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

        # 3. 门控应用 (Gating) - 这里的乘法实现了空间上的选择性激活
        x = content * gate

        # 4. 投影回原维度
        x = self.project_out(x)
        return x



class ElementwiseGatedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ElementwiseGatedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim

        # Q, K, V 生成
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)

        # 创新点：Gate 生成分支 - 使用深度可分离卷积感知局部上下文，而非简单的1x1
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

        # 2. 生成 Elementwise Gate (利用独立分支，不占用Q的通道，更纯粹)
        gate_score = self.gate_generator(x)

        # Reshape for MDTA (计算通道间的协方差，而非空间像素)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)


        if dynamic_temp is not None:
            temp = self.temperature.unsqueeze(0) * dynamic_temp
            attn = (q @ k.transpose(-2, -1)) * temp
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        # Reshape back
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 3. 应用 Elementwise Gating (特征整流)
        out = out * torch.sigmoid(gate_score)  # 抑制 Attention 带来的全局噪声

        out = self.project_out(out)
        return out



class DynamicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, context_dim):
        super(DynamicTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # 使用新的 ElementwiseGatedAttention
        self.attn = ElementwiseGatedAttention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # 使用新的 GatedSpatialFFN
        self.ffn = GatedSpatialFFN(dim, ffn_expansion_factor, bias)

        self.prompt_fusion = MDPM(dim, context_dim)

    def forward(self, x, context_emb):
        # 1. Self-Attention 路径
        residual = x
        x = self.norm1(x)
        x, dynamic_temp = self.prompt_fusion(x, context_emb)  # 你的Prompt模块
        x = self.attn(x, dynamic_temp)
        x = residual + x

        # 2. FFN 路径
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 通道×4，尺寸÷2
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 通道÷2，尺寸×2
        )

    def forward(self, x):
        return self.body(x)



class RectiFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_encoder_blocks=[4, 6, 6, 8],  # 编码器块数
                 num_decoder_blocks=[2, 3, 3, 4],  # 解码器块数（非对称，减少计算）
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 context_dim=64,  # 全局退化特征维度
                 num_scales=3,  # 多尺度卷积数
                 ):

        super().__init__()
        self.dim = dim
        self.dim_list = [dim * (2 ** i) for i in range(4)]  # [48, 96, 192, 384]

        # 1. 多尺度退化表征网络（生成层级化Prompt和全局退化特征）
        self.context_net = MSDRNet_NoModal(
            context_dim=context_dim,
            num_scales=num_scales,
            dim_list=self.dim_list
        )

        # 2. Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 3. 编码器（Encoder）：层级化Prompt注入
        self.encoder_level1 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[0],
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[0]  # 层级化Prompt维度匹配
            ) for _ in range(num_encoder_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim_list[0])
        self.encoder_level2 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[1],
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_encoder_blocks[1])
        ])

        self.down2_3 = Downsample(self.dim_list[1])
        self.encoder_level3 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[2],
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[2]
            ) for _ in range(num_encoder_blocks[2])
        ])

        self.down3_4 = Downsample(self.dim_list[2])
        self.encoder_level4 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[3],
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[3]
            ) for _ in range(num_encoder_blocks[3])
        ])

        # 4. 瓶颈层：退化导向的频域-空间融合（DGSB）
        self.freq_fusion = DGSB(dim=self.dim_list[3], context_dim=context_dim)

        # 5. 解码器（Decoder）：非对称设计，层级化Prompt注入
        self.up4_3 = Upsample(self.dim_list[3])
        self.reduce_chan_level3 = nn.Conv2d(self.dim_list[2] * 2, self.dim_list[2], kernel_size=1, bias=bias)

        self.decoder_level3 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[2],
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[2]
            ) for _ in range(num_decoder_blocks[2])
        ])

        self.up3_2 = Upsample(self.dim_list[2])
        self.reduce_chan_level2 = nn.Conv2d(self.dim_list[1]*2, self.dim_list[1], kernel_size=1,bias=bias)
        self.decoder_level2 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[1],
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[1]
            ) for _ in range(num_decoder_blocks[1])
        ])

        self.up2_1 = Upsample(self.dim_list[1])
        self.reduce_chan_level1 = nn.Conv2d(self.dim_list[0]*2, self.dim_list[0], kernel_size=1,
                                            bias=bias)  # 输出48维
        self.decoder_level1 = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[0],
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[0]
            ) for _ in range(num_decoder_blocks[0])
        ])

        # 6. 精细化模块（Refinement）
        self.refinement = nn.ModuleList([
            DynamicTransformerBlock(
                dim=self.dim_list[0],
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                context_dim=self.dim_list[0]
            ) for _ in range(num_refinement_blocks)
        ])

        # 7. 输出层
        self.output = nn.Conv2d(self.dim_list[0], out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape

        # 1. 获取层级化Prompt和全局退化特征
        layer_prompts, global_feat = self.context_net(inp_img)  # layer_prompts: [P0, P1, P2, P3] (48,96,192,384)
        p1, p2, p3, p4 = layer_prompts  # 对应编码器/解码器的4个层级

        # 2. Embedding
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, 48, H, W]

        # 3. 编码器（Encoder）
        # Level 1 (48维)
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, p1)  # 注入P1

        # Level 2 (96维)
        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, 96, H/2, W/2]
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, p2)  # 注入P2

        # Level 3 (192维)
        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, 192, H/4, W/4]
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, p3)  # 注入P3

        # Level 4 (384维)
        inp_enc_level4 = self.down3_4(out_enc_level3)  # [B, 384, H/8, W/8]
        out_enc_level4 = inp_enc_level4
        for layer in self.encoder_level4:
            out_enc_level4 = layer(out_enc_level4, p4)  # 注入P4

        # 4. 瓶颈层：退化导向的频域-空间融合
        latent = self.freq_fusion(out_enc_level4, global_feat)  # 传入全局退化特征

        # 5. 解码器（Decoder）
        # Level 3
        inp_dec_level3 = self.up4_3(latent)  # [B, 192, H/4, W/4]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)  # [B, 192+192=384, H/4, W/4]
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # [B, 192, H/4, W/4]
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, p3)  # 注入P3

        # Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, 96, H/2, W/2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)  # [B, 96+96=192, H/2, W/2]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [B, 96, H/2, W/2]
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, p2)  # 注入P2

        # Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, 48, H, W]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)  # [B, 48+48=96, H, W]
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)  # [B, 96, H, W]
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, p1)  # 注入P1

        # 6. 精细化模块
        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, p1)  # 注入P1

        # 7. 输出与残差连接
        out = self.output(out_dec_level1) + inp_img  # 残差连接输入图像

        return out

##########################################################################
# 测试代码
##########################################################################

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import warnings

    warnings.filterwarnings('ignore')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = RectiFormer(
        dim=48,
        num_encoder_blocks=[4, 6, 6, 8],
        num_decoder_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        context_dim=64,
        num_scales=3,
    ).to(device)

    # 测试输入
    batch_size = 1
    inp = torch.randn(batch_size, 3, 128, 128).to(device)  # 128x128输入
    target = torch.randn(batch_size, 3, 128, 128).to(device)  # 目标图像

    # 前向推理
    out = model(inp)
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # 计算参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {model_params / 1e6:.3f} M")

    # 计算FLOPs（224x224输入）
    flops = FlopCountAnalysis(model, inp)
    print("\nFLOPs Analysis:")
    print(flop_count_table(flops))


    # 测试显存占用
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\nMax GPU Memory Usage: {max_mem:.2f} MB")
        torch.cuda.empty_cache()

# -*- coding: utf-8 -*-
# File  : 2_new_Rectiformer_CPM_DAGSC_new.py
# Author: HeLei
# Date  : 2025/12/23

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table


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
            nn.PixelUnshuffle(2)
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

class Adaptive_Gated_Fusion(nn.Module):
    """
    Adaptive_Gated_Fusion (AGF)
    改进点:
    1. 引入 3x3 DW-Conv 扩大感受野，利用邻域信息判断噪声。
    2. 引入 Channel Attention (SE机制)，抑制整体噪声严重的通道。
    3. 动态融合机制。
    """
    def __init__(self, in_dim, out_dim=None):
        super(Adaptive_Gated_Fusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim

        # -------------------------------------------------------
        # 1. 空间门控 (Spatial Gating) - 解决“看周围”的问题
        # -------------------------------------------------------
        # 输入是 [Enc, Dec]，维度是 2 * in_dim
        self.spatial_gate = nn.Sequential(
            # 降维
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim) if in_dim > 1 else nn.Identity(), # Optional: BN有助于收敛
            nn.ReLU(inplace=True),
            # 核心改进：使用 DW-Conv 3x3 获取局部上下文
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.ReLU(inplace=True),
            # 生成 0~1 的 Mask
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # -------------------------------------------------------
        # 2. 通道门控 (Channel Gating) - 解决“选通道”的问题
        # -------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, in_dim),
            nn.Sigmoid()
        )

        # -------------------------------------------------------
        # 3. 最终融合变换
        # -------------------------------------------------------
        # 输入是 [Filtered_Enc, Dec]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, self.out_dim, kernel_size=1),
            nn.GELU() # 激活函数换成 GELU 效果通常更好
        )

    def forward(self, f_enc, f_dec):
        """
        f_enc: Encoder 特征 (含有噪声/雨/雾)
        f_dec: Decoder 特征 (语义强，较干净)
        """
        # 1. 准备联合特征
        combined = torch.cat([f_enc, f_dec], dim=1)

        # 2. 计算空间权重 (H, W 维度上的筛选)
        # 哪个位置是雨/雾？(0代表噪声，1代表有效细节)
        spatial_mask = self.spatial_gate(combined)

        # 3. 计算通道权重 (C 维度上的筛选)
        # 哪个通道包含更多噪声？
        b, c, _, _ = combined.shape
        y = self.avg_pool(combined).view(b, c)
        channel_mask = self.channel_gate(y).view(b, self.in_dim, 1, 1)

        # 4. 双重加权筛选 Encoder 特征
        # 既考虑位置(Spatial)，也考虑通道(Channel)
        f_enc_filtered = f_enc * spatial_mask * channel_mask

        # 5. 融合 (保留 Decoder 的全部信息 + 筛选后的 Encoder 细节)
        out = torch.cat([f_enc_filtered, f_dec], dim=1)
        out = self.fusion_conv(out)

        return out


class Context_Aware_Module(nn.Module):
    def __init__(self, context_dim=64, num_scales=3, dim_list=[48, 96, 192, 384]):
        super().__init__()

        inter_dim = 48
        # 1. Stem: 浅层特征提取 (升维)
        self.stem = nn.Sequential(
            nn.Conv2d(3, inter_dim, kernel_size=3, padding=1, stride=1),
            nn.GELU()
        )

        # 2. 多尺度特征提取 (使用 Depthwise Convolution 节省参数)
        # 在特征空间上进行多尺度感知
        self.scale_branches = nn.ModuleList()
        for s in range(num_scales):
            # kernel sizes: 3x3, 5x5, 7x7...
            k_size = 2 * s + 3
            branch = nn.Sequential(
                # DW-Conv: 独立处理每个通道的空间信息，参数量极少
                nn.Conv2d(inter_dim, inter_dim, kernel_size=k_size, padding=k_size // 2, groups=inter_dim),
                # 1x1 Conv: 稍微进行一点通道交互 (Pointwise)
                nn.Conv2d(inter_dim, inter_dim, kernel_size=1)
            )
            self.scale_branches.append(branch)

        # 3. 特征融合
        # 输入维度: inter_dim * num_scales
        fusion_dim = inter_dim * num_scales
        self.fusion = nn.Conv2d(fusion_dim, context_dim, kernel_size=1)

        # 4. 空间门控 (Spatial Gating) - 之前建议的保留
        self.spatial_gate = nn.Conv2d(context_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 5. 全局统计处理 (Dual-Statistic) - 之前建议的保留
        self.global_process = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim)
        )

        self.layer_prompts = nn.ModuleList([
            nn.Linear(context_dim, dim) for dim in dim_list
        ])

    def forward(self, x):
        # 1. Stem: Pixel -> Feature
        x_feat = self.stem(x)  # [B, 32, H, W]

        # 2. Multi-scale Processing
        # 在特征图上并行跑不同尺度的卷积
        scale_feats = [branch(x_feat) for branch in self.scale_branches]
        scale_feats = torch.cat(scale_feats, dim=1)  # [B, 32*3, H, W]

        # 3. Fusion
        feat = self.fusion(scale_feats)

        # 4. Gating (聚焦有雨/雾的区域)
        gate_map = self.sigmoid(self.spatial_gate(feat))
        feat = feat * gate_map

        # 5. Statistical Pooling (Mean + Std)
        feat_avg = torch.mean(feat, dim=(2, 3))
        feat_std = torch.std(feat, dim=(2, 3))
        global_stat = torch.cat([feat_avg, feat_std], dim=1)

        # 6. Generate Context
        global_feat = self.global_process(global_stat)
        layer_prompts = [fc(global_feat) for fc in self.layer_prompts]

        return layer_prompts, global_feat


class CAFGB(nn.Module):
    """
    Context-Aware Frequency Gating (CAFG)
    改进点:
    1. 移除固定的高斯高低频分割，改为全频段处理。
    2. 引入 Context-Aware Frequency Gating (CAFG)，利用 context_emb 动态生成频域权值。
    3. 增强了频域和空域的交互机制。
    """

    def __init__(self, dim, context_dim=64):
        super().__init__()
        self.dim = dim

        # 1. 频域处理分支
        # 使用 1x1 卷积在复数域（连接 real/imag）进行特征变换
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        )

        # 2. 动态频域门控 (Context -> Frequency Mask)
        # 生成一个用于调制频域特征的权重 mask
        self.context_mapper = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2)  # 生成针对 real 和 imag 的权重
        )

        # 3. 空域处理分支 (保持局部细节)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # DW-Conv
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        # 4. 融合机制
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x, global_feat):
        B, C, H, W = x.shape

        # --- A. 空域分支 ---
        spatial_feat = self.spatial_conv(x)

        # --- B. 频域分支 ---
        x_32 = x.to(torch.float32)
        # 使用 rfft2 (实数转复数 FFT)
        x_fft = torch.fft.rfft2(x_32, norm='backward')
        # x_fft shape: [B, C, H, W/2 + 1]

        # 提取实部和虚部
        real = x_fft.real
        imag = x_fft.imag
        # [B, 2*C, H, W/2 + 1]
        f_cat = torch.cat([real, imag], dim=1)

        # B.1 频域卷积变换
        f_feat = self.freq_conv(f_cat)

        # B.2 动态门控 (Dynamic Gating)
        # 利用 context_emb (global_feat) 生成频域的注意力权重
        # global_feat: [B, context_dim] -> [B, 2*C]
        scale = self.context_mapper(global_feat)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)  # [B, 2*C, 1, 1]

        # 对频域特征进行加权 (相当于动态滤波器)
        f_weighted = f_feat * scale

        # B.3 逆变换回去
        w_real, w_imag = torch.chunk(f_weighted, 2, dim=1)
        # 注意: irfft2 需要复数输入
        w_complex = torch.complex(w_real, w_imag)
        freq_spatial = torch.fft.irfft2(w_complex, s=(H, W), norm='backward')

        # --- C. 融合 ---
        # 将空域特征和频域恢复特征拼接融合
        out = torch.cat([spatial_feat, freq_spatial], dim=1)
        out = self.fusion(out)

        return out + x


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

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


class Context_Gated_Attention(nn.Module):
    """
    Context_Gated_Attention (CGA)

    Refined for CVPR/ICCV:
    1. Intra-Attention Modulation: Instead of modulating the input, we modulate the internal components (Temperature & Value).
    2. Context-Gated Value: Dynamically suppresses noise-dominant channels in 'V' before aggregation.
    3. Adaptive Sharpness: Controls the entropy of the attention map via context-adaptive temperature.
    """

    def __init__(self, dim, num_heads, bias, context_dim):
        super(Context_Gated_Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # --- 1. Standard Attention Components ---
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Learnable base temperature (parameter)
        self.base_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # --- 2. Context Adapter (The "Reshaping" Part) ---
        # 生成动态温度缩放因子 (Scalar per head)
        self.temp_adapter = nn.Sequential(
            nn.Linear(context_dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_heads)
        )

        # 生成 Value 门控权重 (Channel-wise Vector)
        # 这是一个 "Feature Selection" 步骤
        self.value_gate_adapter = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.Sigmoid()  # 输出 0~1 的门控值
        )

        # --- 3. Local Branch ---
        # 保持并行局部路径，保留纹理细节
        self.local_mixer = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)

    def forward(self, x, context_emb):
        b, c, h, w = x.shape

        # -----------------------------------------------------------
        # Step 1: Generate Context Dynamics (Before expensive Attn)
        # -----------------------------------------------------------

        # A. Dynamic Temperature Factor
        # context_emb: [B, ctx_dim] -> temp_factor: [B, heads, 1, 1]
        temp_factor = self.temp_adapter(context_emb).view(b, self.num_heads, 1, 1)
        # 使用 Sigmoid*2 + 0.5 保证温度在 [0.5, 2.5] 之间浮动，避免极端值
        temp_factor = torch.sigmoid(temp_factor) * 2.0 + 0.5

        # B. Value Gating Mask
        # context_emb: [B, ctx_dim] -> v_gate: [B, 1, C, 1] (formatted for MDTA)
        # 我们需要将其 reshape 以匹配 MDTA 的 [B, Heads, C_head, N] 格式中的 C 维度
        v_gate = self.value_gate_adapter(context_emb)  # [B, C]
        v_gate = v_gate.view(b, self.num_heads, c // self.num_heads, 1)

        # -----------------------------------------------------------
        # Step 2: QKV Projection & Reshape
        # -----------------------------------------------------------
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for MDTA (Transposed Attention / Channel Attention)
        # Shape: [B, Heads, C_head, HW]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_unfold = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize Q, K for stable training
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # -----------------------------------------------------------
        # Step 3: Context-Guided Aggregation
        # -----------------------------------------------------------

        # --- A. Dynamic Temperature Attention Map ---
        # Attn = Softmax( (Q @ K.T) * Base_Temp * Context_Factor )
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 注入 Context 调节的温度
        # 这里的物理意义：对于浓雾（低频），Temperature 变小，Attention 变平滑（平均）；
        # 对于雨点（高频），Temperature 变大，Attention 变锐利（聚焦）。
        total_temp = self.base_temperature * temp_factor
        attn = (attn * total_temp).softmax(dim=-1)

        # --- B. Context-Gated Value Modulation (New!) ---
        # 在聚合之前，利用 Context "清洗" V
        # 物理意义：如果 Context 识别出当前通道主要是噪声，v_gate 会接近 0，
        # 从而防止噪声被 Global Attention 广播到全图。
        v_unfold = v_unfold * v_gate

        # Global Aggregation
        out_global = (attn @ v_unfold)
        out_global = rearrange(out_global, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # -----------------------------------------------------------
        # Step 4: Local Fusion
        # -----------------------------------------------------------
        out_local = self.local_mixer(v)

        # 最终融合
        out = self.project_out(out_global + out_local)

        return out


class Context_Gate_TransformerBlock(nn.Module):
    """
    CGTB:
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, context_dim):
        super(Context_Gate_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Context_Gated_Attention(dim, num_heads, bias, context_dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, context_emb):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, context_emb)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Context_Gated_IR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 context_dim=64,
                 num_scales=3,
                 ):
        super().__init__()

        # self.dim_list = [int(dim * 2 ** i) for i in range(4)]  # [48, 96, 192, 384]
        self.dim_list = [48, 96, 192, 384]

        # 1. 多尺度退化表征网络
        self.context_net = Context_Aware_Module(
            context_dim=context_dim,
            num_scales=num_scales,
            dim_list=self.dim_list
        )

        # 2. Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        # 3. Encoder
        self.encoder_level1 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[0], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[0])
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim_list[0])
        self.encoder_level2 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[1])
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(self.dim_list[1])
        self.encoder_level3 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[2])
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(self.dim_list[2])
        self.latent = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[3], heads[3], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[3])
            for _ in range(num_blocks[3])
        ])

        # 瓶颈层 Fusion
        self.freq_fusion = CAFGB(dim=self.dim_list[3], context_dim=context_dim)

        self.up4_3 = Upsample(self.dim_list[3])
        self.skip_fusion3 = Adaptive_Gated_Fusion(in_dim=self.dim_list[2])

        self.decoder_level3 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[2])
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(self.dim_list[2])
        self.skip_fusion2 = Adaptive_Gated_Fusion(in_dim=self.dim_list[1])

        self.decoder_level2 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[1])
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(self.dim_list[1])
        self.skip_fusion1 = Adaptive_Gated_Fusion(in_dim=self.dim_list[0], out_dim=self.dim_list[1])  # 48 -> 96

        self.decoder_level1 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[1])
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                    self.dim_list[1])
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(self.dim_list[1], out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape

        # 1. Context Gen
        layer_prompts, global_feat = self.context_net(inp_img)
        p1, p2, p3, p4 = layer_prompts

        # 2. Patch Embed
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, 48, H, W]

        # 3. Encoder
        out_enc_level1 = inp_enc_level1
        for block in self.encoder_level1:
            out_enc_level1 = block(out_enc_level1, p1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level2:
            out_enc_level2 = block(out_enc_level2, p2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = inp_enc_level3
        for block in self.encoder_level3:
            out_enc_level3 = block(out_enc_level3, p3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = inp_enc_level4
        for block in self.latent:
            latent = block(latent, p4)

        # Bottleneck Fusion
        latent = self.freq_fusion(latent, global_feat)

        # 4. Decoder with Gated Skip Connections

        # --- Level 3 ---
        inp_dec_level3 = self.up4_3(latent)  # [B, 192, H/4, W/4]
        # [改进点] 使用门控跳跃连接替代简单的Concat
        # 输入：Enc(192), Dec(192) -> 输出：Fused(192)
        inp_dec_level3 = self.skip_fusion3(out_enc_level3, inp_dec_level3)

        out_dec_level3 = inp_dec_level3
        for block in self.decoder_level3:
            out_dec_level3 = block(out_dec_level3, p3)

        # --- Level 2 ---
        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, 96, H/2, W/2]
        # [改进点] 使用门控跳跃连接
        # 输入：Enc(96), Dec(96) -> 输出：Fused(96)
        inp_dec_level2 = self.skip_fusion2(out_enc_level2, inp_dec_level2)

        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level2:
            out_dec_level2 = block(out_dec_level2, p2)

        # --- Level 1 ---
        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, 48, H, W]
        # [改进点] 使用门控跳跃连接，且自带维度提升
        # 输入：Enc(48), Dec(48) -> 输出：Fused(96)
        inp_dec_level1 = self.skip_fusion1(out_enc_level1, inp_dec_level1)

        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level1:
            out_dec_level1 = block(out_dec_level1, p2)

        # Refinement
        for block in self.refinement:
            out_dec_level1 = block(out_dec_level1, p2)

        out = self.output(out_dec_level1) + inp_img

        return out


##########################################################################
# 测试代码
##########################################################################
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Context_Gated_IR(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        context_dim=64,
    ).to(device)

    # 测试输入（确保是8的倍数，因为有3次下采样）
    inp = torch.randn(1, 3, 128, 128).to(device)

    # 运行推理
    out = model(inp)
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # 检查参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {model_params / 1e6:.3f} M")

    flops = FlopCountAnalysis(model, inp)
    print("\nFLOPs Analysis:")
    print(flop_count_table(flops))

    # 简单显存测试
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")

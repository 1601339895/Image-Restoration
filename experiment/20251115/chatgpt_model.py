import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


###############################################
# 基础组件：LayerNorm & ConvFFN
###############################################

class LayerNorm2d(nn.Module):
    """对 BCHW 进行 LN，专为图像特征设计"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class ConvFFN(nn.Module):
    """
    FFN with 1x1 + DWConv + 1x1
    用于增强局部结构，Restormer 默认结构
    """
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),  # DWConv
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.ffn(x)

###############################################
# 多头线性注意力（Restormer 的经典注意力）
###############################################

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.heads)
        k = rearrange(k, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.heads)
        v = rearrange(v, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum("b h c n, b h d n -> b h c d", k, v)
        out = torch.einsum("b h c d, b h c n -> b h d n", context, q)

        out = rearrange(out, "b h c (h0 w0) -> b (h c) h0 w0", h0=H, w0=W, h=self.heads)
        out = self.out(out)
        return out


###############################################
# Transformer Block（Restormer 标准结构）
###############################################

class TransformerBlock(nn.Module):
    """
    Restormer 基础单元：LinearAttention + FFN
    """
    def __init__(self, dim, num_heads=8, ffn_expansion=2):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = LinearAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim, expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


###############################################
# Down-Sample / Up-Sample
###############################################

class DownSample(nn.Module):
    """
    空间下采样：使用 PixelUnshuffle (分块下采样)
    通道C -> C*4，空间减半
    """
    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """
    上采样：PixelShuffle 上采样
    """
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        return self.up(x)
    

###############################################################
# Dual-Domain Prototype Bank + Degradation Residual Adapter
###############################################################

class DualDomainPrototypeBank(nn.Module):
    """
    双域退化原型：
    - Spatial prototype: 捕获空间纹理退化（雨、雾、噪声）
    - Frequency prototype: 捕获频域统计退化（模糊、噪声、条纹）
    """
    def __init__(self, num_prototypes=3, dim=64):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim

        # 空域原型 (K, C)
        self.spatial_proto = nn.Parameter(torch.randn(num_prototypes, dim))

        # 频域原型 (K, C)
        self.freq_proto = nn.Parameter(torch.randn(num_prototypes, dim))

        self.norm = nn.LayerNorm(dim)

    def forward(self, feat):
        """
        输入：
            feat: [B, C, H, W]
        输出：
            proto_embed: [B, K, C]  退化embedding
        """
        B, C, H, W = feat.shape

        # ---- 空域特征平均 ----
        spatial_feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # [B, C]

        # ---- 频域 FFT ----
        fft = torch.fft.rfft2(feat, norm="ortho")
        fft_amp = torch.abs(fft)  # 幅度谱
        freq_feat = F.adaptive_avg_pool2d(fft_amp, 1).squeeze(-1).squeeze(-1)  # [B, C]

        # L2-normalize
        sp_f = F.normalize(spatial_feat, dim=-1)       # [B, C]
        fr_f = F.normalize(freq_feat, dim=-1)          # [B, C]

        sp_p = F.normalize(self.spatial_proto, dim=-1) # [K, C]
        fr_p = F.normalize(self.freq_proto, dim=-1)    # [K, C]

        # 空域 similarity
        sp_sim = torch.einsum("bc,kc->bk", sp_f, sp_p)

        # 频域 similarity
        fr_sim = torch.einsum("bc,kc->bk", fr_f, fr_p)

        # 结合两种退化
        fusion_sim = (sp_sim + fr_sim) / 2.0

        weights = F.softmax(fusion_sim / 0.07, dim=-1)  # 温度系数

        # 融合原型作为退化 embedding
        proto_embed = torch.einsum("bk,kc->bkc", weights, self.spatial_proto)

        return proto_embed  # [B, K, C]


class DegradationResidualAdapter(nn.Module):
    """
    退化残差适配器 DRA：
    - 通过退化 embedding 生成 1x1 动态门控
    - 调整 Restormer 特征，增加退化感知能力
    """
    def __init__(self, dim, num_prototypes=3):
        super().__init__()
        self.dim = dim

        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim*2)
        )

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, feat, proto_embed):
        """
        feat: [B, C, H, W]
        proto_embed: [B, K, C]
        """
        B, C, H, W = feat.shape

        # 融合 K 个退化 embedding
        embed = proto_embed.mean(dim=1)  # [B, C]

        gate = self.fc(embed)  # [B, 2C]
        gate1, gate2 = gate.chunk(2, dim=-1)

        gate1 = gate1.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        gate2 = gate2.unsqueeze(-1).unsqueeze(-1)

        # 动态门控调节特征
        out = feat * torch.sigmoid(gate1) + self.conv(feat) * torch.sigmoid(gate2)
        return out


#################################################################
# Difficulty-Aware Dual-Domain Routing Mixture of Experts (D2R-MoE)
#################################################################

class Expert(nn.Module):
    """
    每个专家是不同复杂度的 Transformer Block 堆叠
    """
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, num_heads=8) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class DifficultyEstimator(nn.Module):
    """
    根据空间/频域复杂度计算 difficulty score
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim*2, 1)

    def forward(self, spatial_feat, freq_feat):
        x = torch.cat([spatial_feat, freq_feat], dim=-1)
        return torch.sigmoid(self.fc(x))  # [B, 1]


class D2R_MoE(nn.Module):
    """
    结合 Difficulty-aware routing + dual-domain difficulty + cross-expert attention
    """
    def __init__(self, dim, expert_depths=[1, 2, 4]):
        """
        expert_depths:
            轻量专家(1层)
            中等专家(2层)
            重型专家(4层)
        """
        super().__init__()

        self.experts = nn.ModuleList([Expert(dim, d) for d in expert_depths])

        self.diff_estimator = DifficultyEstimator(dim)

        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Spatial difficulty feature
        sp = F.adaptive_avg_pool2d(x, 1).flatten(1)       # [B, C]

        # Frequency difficulty feature
        fft = torch.fft.rfft2(x, norm="ortho")
        fft_amp = torch.abs(fft)
        fr = F.adaptive_avg_pool2d(fft_amp, 1).flatten(1) # [B, C]

        difficulty = self.diff_estimator(sp, fr)  # [B,1]

        # 计算专家权重：difficulty 越大越偏向复杂专家
        alpha = torch.cat([
            1 - difficulty,                 # 简单专家
            difficulty * (1 - difficulty),  # 中专家
            difficulty                      # 重专家
        ], dim=-1)  # [B, 3]

        # 归一化
        alpha = F.softmax(alpha, dim=-1)

        # 每个专家输出
        outputs = [exp(x) for exp in self.experts]  # 3 个 [B,C,H,W]

        # 融合
        fused = 0
        for i in range(3):
            fused += outputs[i] * alpha[:, i].view(B, 1, 1, 1)

        # cross-attention 增强
        seq = fused.flatten(2).transpose(1, 2)  # [B, HW, C]
        seq2, _ = self.cross_attn(seq, seq, seq)
        seq2 = seq2.transpose(1, 2).view(B, C, H, W)

        return x + seq2   # residual



#################################################################
# Final Full Network: DDP + DRA + D2R-MoE + 4/3 Stage Restormer
#################################################################

class Stage(nn.Module):
    """
    多个 Transformer Blocks 构成的 Stage
    """
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class CompleteRestormer(nn.Module):
    def __init__(self, in_ch=3, base=48, num_prototypes=3):
        super().__init__()

        # ----------------------------------------------------------
        # 1. 浅层特征
        # ----------------------------------------------------------
        self.shallow = nn.Conv2d(in_ch, base, 3, padding=1)

        # ----------------------------------------------------------
        # 2. Degradation modules
        # ----------------------------------------------------------
        self.proto_bank = DualDomainPrototypeBank(num_prototypes, base)
        self.dra1 = DegradationResidualAdapter(base)
        self.dra2 = DegradationResidualAdapter(base*2)
        self.dra3 = DegradationResidualAdapter(base*4)
        self.dra4 = DegradationResidualAdapter(base*8)

        # ----------------------------------------------------------
        # 3. Encoder (4 levels)
        # ----------------------------------------------------------
        self.enc1 = Stage(base,      4)
        self.down1 = DownSample(base)        # → C*4
        self.enc2 = Stage(base*4,    6)
        self.down2 = DownSample(base*4)
        self.enc3 = Stage(base*16,   6)
        self.down3 = DownSample(base*16)
        self.enc4 = Stage(base*64,   8)

        # Bottleneck
        self.bottleneck = Stage(base*64, 4)

        # ----------------------------------------------------------
        # 4. Decoder (3 levels)
        # ----------------------------------------------------------
        self.up3 = UpSample(base*64)
        self.dec3 = D2R_MoE(base*16, expert_depths=[1,2,4])

        self.up2 = UpSample(base*16)
        self.dec2 = D2R_MoE(base*4, expert_depths=[1,2,4])

        self.up1 = UpSample(base*4)
        self.dec1 = D2R_MoE(base, expert_depths=[1,2,4])

        # ----------------------------------------------------------
        # 5. Refinement
        # ----------------------------------------------------------
        self.refine = Stage(base, 4)

        # Output
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # ----------------------------------------------------------
        # Shallow feature
        # ----------------------------------------------------------
        s = self.shallow(x)

        # 退化 embedding
        proto_embed = self.proto_bank(s)

        # ----------------------------------------------------------
        # Encoder path
        # ----------------------------------------------------------
        e1 = self.enc1(s)
        e1 = self.dra1(e1, proto_embed)

        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        e2 = self.dra2(e2, proto_embed)

        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        e3 = self.dra3(e3, proto_embed)

        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        e4 = self.dra4(e4, proto_embed)

        b = self.bottleneck(e4)

        # ----------------------------------------------------------
        # Decoder path
        # ----------------------------------------------------------
        u3 = self.up3(b)
        u3 = u3 + e3
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = u2 + e2
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = u1 + e1
        d1 = self.dec1(u1)

        # ----------------------------------------------------------
        # Refinement
        # ----------------------------------------------------------
        r = self.refine(d1)

        out = self.out(r)

        return x + out  # residual learning



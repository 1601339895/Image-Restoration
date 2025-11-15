import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from fvcore.nn import FlopCountAnalysis, flop_count_table


#####################################################################
# ========== 1. 基础组件：LayerNorm, FFN, Attention, Transformer =====
#####################################################################
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    """Restormer FFN (Memory-Optimized)"""
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        # Reduce expansion factor internally to save memory
        hidden = int(dim * min(expansion * 0.5, 1.5))  # Cap at 1.5x for memory efficiency

        self.project_in = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, padding=1, groups=hidden*2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Restormer Linear Attention"""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.num_heads)
        k = rearrange(k, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.num_heads)
        v = rearrange(v, "b (h c) h0 w0 -> b h c (h0 w0)", h=self.num_heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum("b h c n, b h d n -> b h c d", k, v)
        out = torch.einsum("b h c d, b h c n -> b h d n", context, q)

        out = rearrange(out, "b h c (h0 w0) -> b (h c) h0 w0", h0=H, w0=W)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Restormer Transformer block"""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type)
        ])

        self.attn = Attention(dim, num_heads, bias=bias)
        self.ffn = FeedForward(dim, expansion=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x


#####################################################################
# ================= 2. DDP 双域退化建模 + DRA =========================
#####################################################################

class DualDomainPrototypeBank(nn.Module):
    """Learn K spatial + K frequency degradation prototypes"""
    def __init__(self, num_prototypes, dim):
        super().__init__()

        self.spatial = nn.Parameter(torch.randn(num_prototypes, dim))
        self.freq = nn.Parameter(torch.randn(num_prototypes, dim))

    def forward(self, feat):
        B, C, H, W = feat.shape

        sp = F.adaptive_avg_pool2d(feat, 1).flatten(1)     # [B, C]

        fft = torch.fft.rfft2(feat)
        freq = torch.abs(fft)
        fr = F.adaptive_avg_pool2d(freq, 1).flatten(1)     # [B, C]

        sp_n = F.normalize(sp, dim=-1)
        fr_n = F.normalize(fr, dim=-1)

        sp_p = F.normalize(self.spatial, dim=-1)
        fr_p = F.normalize(self.freq, dim=-1)

        sim = (
            torch.mm(sp_n, sp_p.t()) +
            torch.mm(fr_n, fr_p.t())
        ) / 2

        w = F.softmax(sim / 0.07, dim=-1)   # [B, K]
        proto = torch.einsum("bk,kc->bkc", w, self.spatial)

        return proto


class DegradationResidualAdapter(nn.Module):
    """Use degradation embedding to modulate feature maps (Memory-Optimized)"""
    def __init__(self, dim, proto_dim):
        super().__init__()
        # Simplify MLP: reduce hidden layer size
        hidden_dim = max(dim // 2, 32)  # Use smaller hidden dimension
        self.fc = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim*2)
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, feat, proto):
        B, C, H, W = feat.shape
        emb = proto.mean(dim=1)  # [B, proto_dim]

        g = self.fc(emb)
        g1, g2 = g.chunk(2, dim=-1)

        g1 = g1.view(B, C, 1, 1)
        g2 = g2.view(B, C, 1, 1)

        return feat * torch.sigmoid(g1) + self.conv(feat) * torch.sigmoid(g2)


#####################################################################
# ================ 3. D2R-MoE：难度感知专家系统 =========================
#####################################################################

class Expert(nn.Module):
    """专家由多个 transformer 组成"""
    def __init__(self, dim, depth, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(depth)
            ]
        )
    def forward(self, x):
        return self.blocks(x)


class DifficultyEstimator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim*2, 1)

    def forward(self, sp, fr):
        x = torch.cat([sp, fr], dim=-1)
        return torch.sigmoid(self.fc(x))


class D2R_MoE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        # Reduce expert depth to save memory
        self.experts = nn.ModuleList([
            Expert(dim, 1, num_heads, ffn_expansion_factor, bias, LayerNorm_type),  # Easy
            Expert(dim, 1, num_heads, ffn_expansion_factor, bias, LayerNorm_type),  # Medium (reduced from 2)
            Expert(dim, 2, num_heads, ffn_expansion_factor, bias, LayerNorm_type),  # Hard (reduced from 4)
        ])

        self.diff_est = DifficultyEstimator(dim)
        # Use num_heads that divides dim evenly
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        sp = F.adaptive_avg_pool2d(x, 1).flatten(1)
        fft = torch.fft.rfft2(x)
        fr = F.adaptive_avg_pool2d(torch.abs(fft), 1).flatten(1)

        difficulty = self.diff_est(sp, fr)  # [B,1]

        weight = torch.cat([
            1 - difficulty,
            difficulty * (1 - difficulty),
            difficulty
        ], dim=1)

        weight = F.softmax(weight, dim=1)

        out = 0
        for i in range(3):
            out += self.experts[i](x) * weight[:, i].view(B, 1, 1, 1)

        seq = out.flatten(2).transpose(1, 2)
        seq2, _ = self.cross_attn(seq, seq, seq)
        seq2 = seq2.transpose(1, 2).view(B, C, H, W)

        return x + seq2


#####################################################################
# ==================== 4. 下采样、上采样等 =============================
#####################################################################

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # PixelUnshuffle(2) increases channels by 4x, so n_feat//2 -> n_feat*2
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # PixelShuffle(2) decreases channels by 4x, so n_feat*2 -> n_feat//2
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#####################################################################
# ===================== 5. Final Full Network ========================
#####################################################################

class DDP_D2R_Restormer(nn.Module):
    """
    使用 Restormer 官方 API 风格初始化方式
    """
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
                 num_prototypes=3):

        super().__init__()

        # ----------------------------------------------------
        # Shallow feature extraction
        # ----------------------------------------------------
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)

        # ----------------------------------------------------
        # Degradation modeling
        # ----------------------------------------------------
        self.bank = DualDomainPrototypeBank(num_prototypes, dim)
        self.dra = nn.ModuleList([
            DegradationResidualAdapter(dim, dim),        # proto_dim = dim
            DegradationResidualAdapter(dim*2, dim),      # proto_dim = dim
            DegradationResidualAdapter(dim*4, dim),      # proto_dim = dim
            DegradationResidualAdapter(dim*8, dim),      # proto_dim = dim
        ])

        # ----------------------------------------------------
        # Encoder Stages (4 levels)
        # dim -> dim*2 -> dim*4 -> dim*8
        # ----------------------------------------------------
        self.encoder = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[0])
            ]),

            nn.Sequential(*[
                TransformerBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[1])
            ]),

            nn.Sequential(*[
                TransformerBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[2])
            ]),

            nn.Sequential(*[
                TransformerBlock(dim*8, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[3])
            ])
        ])

        self.down = nn.ModuleList([
            Downsample(dim),      # dim -> dim*2
            Downsample(dim*2),    # dim*2 -> dim*4
            Downsample(dim*4)     # dim*4 -> dim*8
        ])

        # ----------------------------------------------------
        # Bottleneck (Memory-Optimized: reduced from 2 blocks to 1)
        # ----------------------------------------------------
        self.bottleneck = nn.Sequential(
            TransformerBlock(dim*8, heads[3], ffn_expansion_factor, bias, LayerNorm_type),
        )

        # ----------------------------------------------------
        # Decoder (3 levels)
        # dim*8 -> dim*4 -> dim*2 -> dim
        # ----------------------------------------------------
        self.up = nn.ModuleList([
            Upsample(dim*8),   # dim*8 -> dim*4
            Upsample(dim*4),   # dim*4 -> dim*2
            Upsample(dim*2)    # dim*2 -> dim
        ])

        self.decoder = nn.ModuleList([
            D2R_MoE(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type),
            D2R_MoE(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type),
            D2R_MoE(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type),
        ])

        # ----------------------------------------------------
        # Refinement stage
        # ----------------------------------------------------
        self.refine = nn.Sequential(
            *[
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    ################################################################
    # Forward
    ################################################################
    def forward(self, x):
        B, C, H, W = x.shape

        s = self.patch_embed(x)

        proto = self.bank(s)

        # ------------- Encoder -------------
        e1 = self.encoder[0](s)
        e1 = self.dra[0](e1, proto)

        d1 = self.down[0](e1)

        e2 = self.encoder[1](d1)
        e2 = self.dra[1](e2, proto)

        d2 = self.down[1](e2)

        e3 = self.encoder[2](d2)
        e3 = self.dra[2](e3, proto)

        d3 = self.down[2](e3)

        e4 = self.encoder[3](d3)
        e4 = self.dra[3](e4, proto)

        b = self.bottleneck(e4)

        # ------------- Decoder -------------
        u3 = self.up[0](b)
        u3 = u3 + e3
        d3 = self.decoder[0](u3)

        u2 = self.up[1](d3)
        u2 = u2 + e2
        d2 = self.decoder[1](u2)

        u1 = self.up[2](d2)
        u1 = u1 + e1
        d1 = self.decoder[2](u1)

        # ------------- Refinement -------------
        r = self.refine(d1)

        out = self.output(r)

        return x + out

if __name__ == "__main__":
    # test
    model = DDP_D2R_Restormer(inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 1, 2, 2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 num_prototypes=3).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    out = model(x)
    print(out.shape)
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
  
    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
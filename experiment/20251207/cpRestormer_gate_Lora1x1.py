import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from fvcore.nn import FlopCountAnalysis, flop_count_table


# ===================== 新增 LoRaConv 类 =====================
class LoRaConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank, bias=True):
        super(LoRaConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        # 1x1卷积实现低秩分解：Linear1 -> Linear2 等价于 Conv1 -> Conv2
        self.conv1 = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x


# 保留原LayerNorm定义
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


# ===================== 改造 FeedForward：替换1×1卷积为LoRaConv =====================
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, lora_rank=8):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        # 替换1×1卷积：nn.Conv2d → LoRaConv
        self.project_in = LoRaConv(dim, hidden_features * 2, rank=lora_rank, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 替换1×1卷积：nn.Conv2d → LoRaConv
        self.project_out = LoRaConv(hidden_features, dim, rank=lora_rank, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ===================== 改造 GatedMDTA：替换1×1卷积为LoRaConv =====================
class GatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias, gate_type=None, lora_rank=8):
        super(GatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gate_type = gate_type  # 门控类型
        self.bias = bias
        self.dim = dim

        # 1. 计算QKV+门控分数的总通道数
        if gate_type is None:
            self.qkv_out_channels = dim * 3
        elif gate_type == 'headwise':
            self.qkv_out_channels = dim * 3 + num_heads
        elif gate_type == 'elementwise':
            self.qkv_out_channels = dim * 4
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}, choose from None, 'headwise', 'elementwise'")

        # 2. 替换1×1卷积：nn.Conv2d → LoRaConv
        self.qkv = LoRaConv(dim, self.qkv_out_channels, rank=lora_rank, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.qkv_out_channels, self.qkv_out_channels, kernel_size=3, stride=1, padding=1,
                                    groups=self.qkv_out_channels, bias=bias)
        # 替换1×1卷积：nn.Conv2d → LoRaConv
        self.project_out = LoRaConv(dim, dim, rank=lora_rank, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape  # 输入维度：(batch, dim, height, width)
        head_dim = c // self.num_heads  # 每个头的维度

        # 步骤1：QKV+门控分数的投影与Depthwise Conv
        qkv_with_gate = self.qkv_dwconv(self.qkv(x))  # (b, qkv_out_channels, h, w)

        # 步骤2：拆分Q、K、V和门控分数
        if self.gate_type is None:
            q, k, v = qkv_with_gate.chunk(3, dim=1)  # q: (b, dim, h, w)
            gate_score = None
        elif self.gate_type == 'headwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim + self.num_heads, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.split([self.dim, self.num_heads], dim=1)
        elif self.gate_type == 'elementwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim * 2, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.chunk(2, dim=1)

        # 步骤3：注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, head, h*w, h*w)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (b, num_heads, head_dim, h*w)

        # 步骤4：应用门控
        if self.gate_type is not None:
            if self.gate_type == 'headwise':
                gate_score = rearrange(gate_score, 'b head h w -> b head 1 (h w)', head=self.num_heads)
            elif self.gate_type == 'elementwise':
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
            out = out * torch.sigmoid(gate_score)

        # 步骤5：维度恢复与输出投影
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


# ===================== 改造 GatedTransformerBlock：传递lora_rank参数 =====================
class GatedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, gate_type=None, lora_rank=8):
        super(GatedTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 传递lora_rank给门控注意力
        self.attn = GatedMDTA(dim, num_heads, bias, gate_type=gate_type, lora_rank=lora_rank)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 传递lora_rank给FeedForward
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, lora_rank=lora_rank)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# 保留原PatchEmbed、Downsample、Upsample定义
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        # 3×3卷积，非1×1，不替换
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            # 3×3卷积，非1×1，不替换
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            # 3×3卷积，非1×1，不替换
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# ===================== 改造 MoCEIR：传递lora_rank并替换剩余1×1卷积 =====================
class LoRa_Gate_Restormer(nn.Module):
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
                 dual_pixel_task=False,
                 gate_type=None,
                 lora_rank=8):  # 新增lora_rank参数，暴露给外部配置
        super(LoRa_Gate_Restormer, self).__init__()
        self.gate_type = gate_type
        self.lora_rank = lora_rank  # 保存LoRa秩参数
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 1. 编码器：传递lora_rank参数
        self.encoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[3])
        ])

        # 2. 解码器：替换1×1卷积+传递lora_rank
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        # 替换1×1卷积：nn.Conv2d → LoRaConv
        self.reduce_chan_level3 = LoRaConv(int(dim * 2 ** 3), int(dim * 2 ** 2), rank=lora_rank, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        # 替换1×1卷积：nn.Conv2d → LoRaConv
        self.reduce_chan_level2 = LoRaConv(int(dim * 2 ** 2), int(dim * 2 ** 1), rank=lora_rank, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_blocks[0])
        ])

        # 3. 精修模块：传递lora_rank
        self.refinement = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, lora_rank=lora_rank
            ) for _ in range(num_refinement_blocks)
        ])

        # 双像素任务专属模块：替换1×1卷积
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            # 替换1×1卷积：nn.Conv2d → LoRaConv
            self.skip_conv = LoRaConv(dim, int(dim * 2 ** 1), rank=lora_rank, bias=bias)

        # 输出层是3×3卷积，非1×1，不替换
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, labels=None):
        # 前向流程与原代码完全一致
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    # 自定义LoRa秩（lora_rank），在参数中体现
    model = LoRa_Gate_Restormer(
        dim=32,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        gate_type="headwise",  # 可选：None/headwise/elementwise
        LayerNorm_type='WithBias',
        bias=False,
        lora_rank=8  # 自定义LoRa秩，核心参数（可设为4/8/16等，需小于对应层的输入通道数）
    ).cuda()
    # 前向推理
    inp = torch.randn(1, 3, 224, 224).cuda()
    out = model(inp)
    print(f"Output shape: {out.shape}")

    # 显存统计
    if torch.cuda.is_available():
        print('{:>16s} : {:<.3f} [M]'.format('Max Memory',
                                             torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))
    else:
        print("CUDA not available, skip memory stats")

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (inp))
    print(flop_count_table(flops))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math


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


# 保留原FeedForward定义（无需修改）
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


##########################################################################
## 改造后的门控注意力：支持Headwise/Elementwise两种模式
## gate_type: None（无门控）、'headwise'（头级门控）、'elementwise'（元素级门控）
##########################################################################
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
                print(f"Gate score mean: {torch.sigmoid(gate_score).mean().item()}")
            elif self.gate_type == 'elementwise':
                # Elementwise：gate_score → (b, num_heads, head_dim, h*w)（与out完全同维度）
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
                print(f"Gate score mean: {torch.sigmoid(gate_score).mean().item()}")

            # 乘性sigmoid门控：动态过滤信息流
            out = out * torch.sigmoid(gate_score)

        # 步骤5：维度恢复与输出投影（保持原MDTA流程）
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # (b, dim, h, w)
        out = self.project_out(out)

        return out



##########################################################################
## 改造后的TransformerBlock：支持传递门控类型
##########################################################################
class GatedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, gate_type=None):
        super(GatedTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 替换为门控注意力
        self.attn = GatedMDTA(dim, num_heads, bias, gate_type=gate_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 残差连接逻辑不变
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# 保留原PatchEmbed、Downsample、Upsample定义（无需修改）
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


##########################################################################
## 改造后的Restormer：支持全局配置门控类型
##########################################################################
class GatedRestormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## 可选 'BiasFree'
                 dual_pixel_task=False,  ## 仅用于双像素散焦去模糊，需设inp_channels=6
                 gate_type=None  ## 全局门控类型：None/'headwise'/'elementwise'
                 ):
        super(GatedRestormer, self).__init__()
        self.gate_type = gate_type  # 全局门控配置
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 1. 编码器：Level 1~4，传递门控类型
        self.encoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[3])
        ])

        # 2. 解码器：Level 3~1，传递门控类型
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_blocks[0])
        ])

        # 3. 精修模块：传递门控类型
        self.refinement = nn.Sequential(*[
            GatedTransformerBlock(
                dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type
            ) for _ in range(num_refinement_blocks)
        ])

        # 双像素任务专属模块（保持不变）
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # 前向流程与原Restormer完全一致，仅注意力模块替换为门控版本
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

        # 双像素任务专属逻辑（保持不变）
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    # 头级门控：gate_type='headwise'
    headwise_restormer = GatedRestormer(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        gate_type="headwise",  # 启用头级门控,elementwise\headwise
        LayerNorm_type='WithBias',
        bias=False
    )
    # 前向推理
    inp = torch.randn(1, 3, 256, 256)  # (batch=1, channel=3, height=256, width=256)
    out = headwise_restormer(inp)
    print(out.shape)
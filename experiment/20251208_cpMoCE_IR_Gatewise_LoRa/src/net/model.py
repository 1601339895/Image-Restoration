import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table


# ==============================================================================
# 1. 改进版 LoRaConv (方案一：引入 Depthwise Conv 和 GELU)
# ==============================================================================
class BetterLoRaConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank, bias=True):
        super(BetterLoRaConv, self).__init__()

        # 1. 降维映射 (Squeeze)
        self.conv1 = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)

        # 2. 深度卷积 (Depthwise Conv) - 核心改进
        # groups=rank 保证参数量极低 (K*K*C)，用于捕获空间上下文信息
        self.dwconv = nn.Conv2d(rank, rank, kernel_size=3, stride=1,
                                padding=1, groups=rank, bias=False)

        # 3. 非线性激活 - 核心改进
        # 增加网络的非线性拟合能力，弥补低秩分解带来的表达力损失
        self.act = nn.GELU()

        # 4. 升维映射 (Expand)
        self.conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


# ==============================================================================
# 2. 基础模块定义 (保持原逻辑不变)
# ==============================================================================
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


class GatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias, gate_type=None):
        super(GatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gate_type = gate_type
        self.dim = dim
        self.bias = bias

        if gate_type is None:
            self.qkv_out_channels = dim * 3
        elif gate_type == 'headwise':
            self.qkv_out_channels = dim * 3 + num_heads
        elif gate_type == 'elementwise':
            self.qkv_out_channels = dim * 4

        self.qkv = nn.Conv2d(dim, self.qkv_out_channels, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.qkv_out_channels, self.qkv_out_channels, kernel_size=3, stride=1, padding=1,
                                    groups=self.qkv_out_channels, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads
        qkv_with_gate = self.qkv_dwconv(self.qkv(x))

        if self.gate_type is None:
            q, k, v = qkv_with_gate.chunk(3, dim=1)
            gate_score = None
        elif self.gate_type == 'headwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim + self.num_heads, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.split([self.dim, self.num_heads], dim=1)
        elif self.gate_type == 'elementwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim * 2, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        if self.gate_type is not None:
            if self.gate_type == 'headwise':
                gate_score = rearrange(gate_score, 'b head h w -> b head 1 (h w)', head=self.num_heads)
            elif self.gate_type == 'elementwise':
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
            out = out * torch.sigmoid(gate_score)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GatedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, gate_type=None):
        super(GatedTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = GatedMDTA(dim, num_heads, bias, gate_type=gate_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x): return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x): return self.body(x)


# ==============================================================================
# 3. 主模型架构
# ==============================================================================
class LoRa_Gate_Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 dual_pixel_task=False, gate_type=None):
        super(LoRa_Gate_Restormer, self).__init__()
        self.gate_type = gate_type
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                  LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            GatedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type) for _ in
            range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, labels=None):
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


# ==============================================================================
# 4. 混合策略替换函数 (方案二：按层级动态调整 Ratio)
# ==============================================================================
def apply_mixed_strategy_lora(model, Lora_ffn_ratio=0.25, Lora_attn_ratio=0.5):
    """
    智能替换函数：
    1. FFN层 (FeedForward): 使用 ffn_ratio (低, e.g., 0.25)，因为参数冗余大。
    2. Attention层 (GatedMDTA): 使用 attn_ratio (中, e.g., 0.5)，保护特征提取能力。
    3. 关键层 (Reduce/Head/Skip): 保持原生卷积，防止瓶颈效应。
    """

    replaced_count = 0
    ignored_count = 0

    # 定义递归函数处理子模块
    def replace_recursive(module, parent_name_hint=""):
        nonlocal replaced_count, ignored_count

        for name, child in module.named_children():
            # 更新上下文名称，用于判断当前处于什么模块中
            current_context = f"{parent_name_hint}.{name}" if parent_name_hint else name

            # 1. 识别 1x1 卷积
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):

                # --- 策略核心逻辑 ---
                target_ratio = None

                # (A) 忽略名单：Reduce通道层、输入输出头、Skip连接 -> 保持原生
                if any(x in current_context for x in ['reduce', 'head', 'skip', 'down', 'up', 'patch']):
                    ignored_count += 1
                    continue  # 跳过替换

                # (B) FFN 层：激进压缩
                elif 'ffn' in current_context or 'project' in name:
                    # 注意：GatedMDTA 也有 project，但在 ffn 上下文中优先匹配
                    # 这里可以通过父类类型做更精确判断，但在命名规范下字符串匹配足够
                    target_ratio = Lora_ffn_ratio

                # (C) Attention 层：保守压缩 (QKV, Attention output)
                elif 'attn' in current_context or 'qkv' in name:
                    target_ratio = Lora_attn_ratio

                # (D) 其他未知层：默认使用保守策略
                else:
                    target_ratio = Lora_attn_ratio

                # --- 执行替换 ---
                if target_ratio is not None:
                    in_c = child.in_channels
                    out_c = child.out_channels

                    # 只有当压缩能真正减少参数时才替换 (rank < min_dim)
                    calc_rank = int(min(in_c, out_c) * target_ratio)
                    rank = max(2, calc_rank)

                    # 如果 Rank 并没有显著小于原始维度（比如 ratio=1），则不替换
                    if rank >= min(in_c, out_c):
                        ignored_count += 1
                        continue

                    bias = child.bias is not None

                    # 使用 BetterLoRaConv 替换
                    new_layer = BetterLoRaConv(in_c, out_c, rank, bias)
                    setattr(module, name, new_layer)
                    replaced_count += 1

            else:
                # 递归处理
                replace_recursive(child, current_context)

    replace_recursive(model)
    print(f"✅ Replacement Done: {replaced_count} layers replaced, {ignored_count} layers kept original.")
    return model


# ==============================================================================
# 5. 测试与验证
# ==============================================================================
if __name__ == "__main__":
    # 配置
    print("=== Creating Model ===")
    inp = torch.randn(1, 3, 224, 224).cuda()

    # 1. 实例化模型
    model = LoRa_Gate_Restormer(
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        ffn_expansion_factor=2.66,
        heads=[1,2,4,8],
        LayerNorm_type='WithBias',
        gate_type=None, # 可选：None/headwise/elementwise
        ).cuda()

    # 计算原始参数量和FLOPs
    print("\n[Original Model]")
    flops_orig = FlopCountAnalysis(model, inp)
    print(f"FLOPs: {flops_orig.total() / 1e9:.2f} G")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 2. 应用混合策略 + BetterLoRa
    # ffn_ratio=0.25 (激进压缩大参数层)
    # attn_ratio=0.5 (适度压缩敏感层)
    print("\n[Applying Mixed Strategy & BetterLoRaConv]")
    model = apply_mixed_strategy_lora(model, ffn_ratio=0.6, attn_ratio=0.6)
    model.cuda()
    # 3. 计算新模型指标
    print("\n[Optimized Model]")
    flops_new = FlopCountAnalysis(model, inp)
    print(f"FLOPs: {flops_new.total() / 1e9:.2f} G")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 4. 验证输出形状
    out = model(inp)
    print(f"\nOutput Shape Check: {out.shape} (Should be [1, 3, 224, 224])")

    # 5. 显存检查
    print(f"Max Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
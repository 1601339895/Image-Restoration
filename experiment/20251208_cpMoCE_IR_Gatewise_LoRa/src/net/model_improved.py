import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table


# ==============================================================================
# 1. Dynamic Feature Refinement Gate (DFRG) - Core Innovation
# ==============================================================================
class DFRG(nn.Module):
    """
    Dynamic Feature Refinement Gate - Adaptively refines features based on degradation type.
    This allows task-specific feature modulation without significantly increasing parameters.
    """
    def __init__(self, channels, reduction=4, num_experts=5):
        super(DFRG, self).__init__()
        self.channels = channels
        self.num_experts = num_experts

        # Global context extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Squeeze to low-dimensional space
        self.fc1 = nn.Conv2d(channels * 2, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Task-aware modulation weights
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

        # Expert selection mechanism (lightweight)
        self.expert_selector = nn.Sequential(
            nn.Conv2d(channels // reduction, num_experts, 1, bias=True),
            nn.Softmax(dim=1)
        )

        # Spatial attention branch
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, task_id=None):
        b, c, h, w = x.shape

        # Channel attention
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        pool_cat = torch.cat([avg_pool, max_pool], dim=1)

        squeeze = self.relu(self.fc1(pool_cat))
        channel_gate = torch.sigmoid(self.fc2(squeeze))

        # Task-aware expert selection (optional, used when task_id is needed)
        expert_weights = self.expert_selector(squeeze)  # [B, num_experts, 1, 1]

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_gate = self.spatial_gate(spatial_cat)

        # Combined gating
        out = x * channel_gate * spatial_gate

        return out, expert_weights


# ==============================================================================
# 2. Enhanced LoRa with Spatial Awareness
# ==============================================================================
class SpatialAwareLoRaConv(nn.Module):
    """
    Enhanced LoRa with spatial context modeling.
    Key improvement: Adds conditional convolution based on feature statistics.
    """
    def __init__(self, in_channels, out_channels, rank, bias=True):
        super(SpatialAwareLoRaConv, self).__init__()

        # Down-projection
        self.conv_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)

        # Spatial context module - Innovation: Dynamic kernel selection
        self.spatial_context = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False),
            nn.BatchNorm2d(rank),
            nn.GELU(),
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False),
        )

        # Feature statistics modulation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rank, rank // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rank // 4, rank, 1),
            nn.Sigmoid()
        )

        # Up-projection
        self.conv_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # Down-projection
        x_down = self.conv_down(x)

        # Spatial processing
        spatial_feat = self.spatial_context(x_down)

        # Global context modulation
        global_weights = self.global_context(spatial_feat)
        modulated_feat = spatial_feat * global_weights

        # Residual connection in latent space
        x_latent = x_down + modulated_feat

        # Up-projection
        out = self.conv_up(x_latent)

        return out


# ==============================================================================
# 3. Cross-Scale Feature Fusion (CSFF)
# ==============================================================================
class CSFF(nn.Module):
    """
    Cross-Scale Feature Fusion module for better information flow across scales.
    Handles different channel sizes between encoder and decoder features.
    """
    def __init__(self, enc_channels, dec_channels):
        super(CSFF, self).__init__()

        # Use the sum of both channels for concatenation
        self.conv_merge = nn.Conv2d(enc_channels + dec_channels, dec_channels, 1, bias=False)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(dec_channels, dec_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dec_channels, dec_channels, 3, padding=1, bias=False),
        )

        # Attention for feature selection
        self.attention = nn.Sequential(
            nn.Conv2d(dec_channels, dec_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels // 4, dec_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, enc_feat, dec_feat):
        # Ensure spatial dimensions match
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:],
                                    mode='bilinear', align_corners=False)

        # Concatenate and merge
        cat_feat = torch.cat([enc_feat, dec_feat], dim=1)
        merged = self.conv_merge(cat_feat)
        refined = self.conv_refine(merged)

        # Attention-weighted fusion
        att_weights = self.attention(refined)
        out = refined * att_weights + merged

        return out


# ==============================================================================
# 4. Basic Building Blocks (Keep existing)
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
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                               stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ==============================================================================
# 5. Enhanced Attention with DFRG
# ==============================================================================
class EnhancedGatedMDTA(nn.Module):
    """Enhanced Multi-Dconv Head Transposed Attention with DFRG integration."""
    def __init__(self, dim, num_heads, bias, gate_type=None, use_dfrg=True):
        super(EnhancedGatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gate_type = gate_type
        self.dim = dim
        self.bias = bias
        self.use_dfrg = use_dfrg

        if gate_type is None:
            self.qkv_out_channels = dim * 3
        elif gate_type == 'headwise':
            self.qkv_out_channels = dim * 3 + num_heads
        elif gate_type == 'elementwise':
            self.qkv_out_channels = dim * 4

        self.qkv = nn.Conv2d(dim, self.qkv_out_channels, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.qkv_out_channels, self.qkv_out_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     groups=self.qkv_out_channels, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Add DFRG for dynamic refinement
        if use_dfrg:
            self.dfrg = DFRG(dim, reduction=4)

    def forward(self, x, task_id=None):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        # Apply DFRG before attention
        if self.use_dfrg:
            x, expert_weights = self.dfrg(x, task_id)

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
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)',
                                      head=self.num_heads, c=head_dim)
            out = out * torch.sigmoid(gate_score)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class EnhancedTransformerBlock(nn.Module):
    """Enhanced Transformer Block with improved residual connections."""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 gate_type=None, use_dfrg=True):
        super(EnhancedTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = EnhancedGatedMDTA(dim, num_heads, bias, gate_type=gate_type, use_dfrg=use_dfrg)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # Learnable scaling factors for residual connections
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

    def forward(self, x, task_id=None):
        x = x + self.gamma1 * self.attn(self.norm1(x), task_id)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
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


# ==============================================================================
# 6. Main Enhanced Model Architecture
# ==============================================================================
class ImprovedLoRaGateRestormer(nn.Module):
    """
    Improved All-in-One Image Restoration Model with:
    1. Dynamic Feature Refinement Gates (DFRG)
    2. Spatial-Aware LoRa convolutions
    3. Cross-Scale Feature Fusion (CSFF)
    4. Enhanced residual connections
    """
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias',
                 dual_pixel_task=False, gate_type=None, use_dfrg=True):
        super(ImprovedLoRaGateRestormer, self).__init__()

        self.gate_type = gate_type
        self.use_dfrg = use_dfrg
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=dim, num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                   LayerNorm_type=LayerNorm_type, gate_type=gate_type,
                                   use_dfrg=use_dfrg)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)

        # Encoder Level 2
        self.encoder_level2 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        # Encoder Level 3
        self.encoder_level3 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # Latent (Bottleneck)
        self.latent = nn.Sequential(*[
            EnhancedTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[3])
        ])

        # Decoder with CSFF
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.csff3 = CSFF(int(dim * 2 ** 2), int(dim * 2 ** 2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2),
                                           kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.csff2 = CSFF(int(dim * 2 ** 1), int(dim * 2 ** 1))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1),
                                           kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        # CSFF1: accepts dim (48) from encoder_level1 and dim (48) from up2_1 (upsample halves channels)
        self.csff1 = CSFF(dim, dim)
        self.reduce_chan_level1 = nn.Conv2d(dim, dim,
                                           kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            EnhancedTransformerBlock(dim=dim, num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            EnhancedTransformerBlock(dim=dim, num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                   gate_type=gate_type, use_dfrg=use_dfrg)
            for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=bias)

    def forward(self, inp_img, task_id=None):
        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Decoder with CSFF
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self.csff3(out_enc_level3, inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.csff2(out_enc_level2, inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.csff1(out_enc_level1, inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


# ==============================================================================
# 7. Enhanced LoRa Replacement Strategy
# ==============================================================================
def apply_enhanced_lora_strategy(model, lora_ffn_ratio=0.5, lora_attn_ratio=0.8):
    """
    Enhanced LoRa replacement using SpatialAwareLoRaConv.
    More aggressive compression with better performance preservation.
    """
    replaced_count = 0
    ignored_count = 0

    def replace_recursive(module, parent_name_hint=""):
        nonlocal replaced_count, ignored_count

        for name, child in module.named_children():
            current_context = f"{parent_name_hint}.{name}" if parent_name_hint else name

            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                target_ratio = None

                # Skip critical layers
                if any(x in current_context for x in ['reduce', 'output', 'skip', 'patch', 'csff']):
                    ignored_count += 1
                    continue

                # FFN layers: aggressive compression
                elif 'ffn' in current_context or 'project' in name:
                    target_ratio = lora_ffn_ratio

                # Attention layers: moderate compression
                elif 'attn' in current_context or 'qkv' in name:
                    target_ratio = lora_attn_ratio

                # DFRG layers: conservative compression
                elif 'dfrg' in current_context:
                    target_ratio = 0.75  # Keep more capacity in gates

                else:
                    target_ratio = lora_attn_ratio

                if target_ratio is not None:
                    in_c = child.in_channels
                    out_c = child.out_channels
                    calc_rank = int(min(in_c, out_c) * target_ratio)
                    rank = max(4, calc_rank)  # Minimum rank of 4 for stability

                    if rank >= min(in_c, out_c):
                        ignored_count += 1
                        continue

                    bias = child.bias is not None
                    new_layer = SpatialAwareLoRaConv(in_c, out_c, rank, bias)
                    setattr(module, name, new_layer)
                    replaced_count += 1
            else:
                replace_recursive(child, current_context)

    replace_recursive(model)
    print(f"✅ Enhanced LoRa Replacement: {replaced_count} layers replaced, {ignored_count} kept original")
    return model


# ==============================================================================
# 8. Model Testing
# ==============================================================================
if __name__ == "__main__":
    print("=== Testing Improved Model ===")
    inp = torch.randn(1, 3, 224, 224).cuda()

    # Original model
    print("\n[Creating Enhanced Model]")
    model = ImprovedLoRaGateRestormer(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        ffn_expansion_factor=2.66,
        heads=[1, 2, 4, 8],
        LayerNorm_type='WithBias',
        gate_type='elementwise',
        use_dfrg=True
    ).cuda()

    print("\n[Original Model Stats]")
    flops_orig = FlopCountAnalysis(model, inp)
    params_orig = sum(p.numel() for p in model.parameters())
    print(f"FLOPs: {flops_orig.total() / 1e9:.2f} G")
    print(f"Params: {params_orig / 1e6:.2f} M")

    # Apply enhanced LoRa
    print("\n[Applying Enhanced LoRa Strategy]")
    model = apply_enhanced_lora_strategy(model, lora_ffn_ratio=0.5, lora_attn_ratio=0.8)
    model.cuda()

    print("\n[Optimized Model Stats]")
    flops_new = FlopCountAnalysis(model, inp)
    params_new = sum(p.numel() for p in model.parameters())
    print(f"FLOPs: {flops_new.total() / 1e9:.2f} G")
    print(f"Params: {params_new / 1e6:.2f} M")
    print(f"Param Reduction: {(1 - params_new/params_orig) * 100:.2f}%")
    print(f"FLOPs Reduction: {(1 - flops_new.total()/flops_orig.total()) * 100:.2f}%")

    # Test forward pass
    with torch.no_grad():
        out = model(inp)
    print(f"\n✅ Output Shape: {out.shape}")
    print(f"Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

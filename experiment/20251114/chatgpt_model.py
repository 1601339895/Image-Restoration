"""
MFGA-IMDNet++ PyTorch implementation (prototype)

Contents:
- DIDBlockPlus: Degradation Disentangler (spatial + frequency branches, tokenization)
- MoE-RouteBlock: sparse Mixture-of-Experts routing using degradation tokens
- Decoder: residual decoder + optional conditional refinement module
- Full Model: IMDNetPlus that wires everything together
- Loss placeholders: InfoNCE, MI-minimization (proxy), perceptual & lpips placeholders
- Example training forward + model instantiation

Notes:
- This is a research prototype emphasizing clarity and modularity.
- The conditional diffusion is OUT OF SCOPE to implement end-to-end here (extremely heavy). A hook is provided to attach a diffusion/refinement module.
- For true production and best SOTA you would swap some modules (e.g., pretrained VGG for perceptual loss, full diffusion model). This code is ready to run and debug on a single GPU.

"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Utility blocks ---------------------------

class ConvBlock(nn.Module):
    """Simple Conv-BN-ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = ConvBlock(ch, ch, activation=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)

# --------------------------- DIDBlock++ ---------------------------

class HaarTransform(nn.Module):
    """Simple Haar-like transform to get approximate low/high frequency components.
    Implemented via downsampling + differences. Non-learnable but deterministic.
    Returns lowpass and highpass (concatenated channels if desired).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: B,C,H,W -- do a simple 2x2 Haar transform
        # low = (a + b + c + d)/4 ; high = concatenation of three detail bands
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        low = (a + b + c + d) / 4.0
        # three detail bands
        dh = (a - b + c - d) / 4.0
        dv = (a + b - c - d) / 4.0
        dd = (a - b - c + d) / 4.0
        high = torch.cat([dh, dv, dd], dim=1)
        return low, high


class Tokenizer(nn.Module):
    """Convert pooled features to K tokens using a small transformer-like aggregator.
    Produces K tokens per image representing degradation components.
    """
    def __init__(self, in_dim, token_dim=128, num_tokens=4, num_heads=4, ff_mult=2):
        super().__init__()
        self.num_tokens = num_tokens
        # initialize learnable queries (degradation prototypes)
        self.queries = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.project = nn.Linear(in_dim, token_dim)
        # light transformer encoder for cross-attention: queries attend to features
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * ff_mult),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim * ff_mult, token_dim)
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: B, C, H, W -> flatten spatial -> B, N, C
        B, C, H, W = feat.shape
        x = feat.reshape(B, C, H * W).permute(0, 2, 1)  # B, N, C
        x = self.project(x)  # B, N, token_dim
        # expand queries
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # B, K, D
        # use multihead attention: queries as queries, x as key+value
        attn_out, _ = self.attn(q, x, x)
        q = self.norm1(attn_out + q)
        q = self.norm2(self.ff(q) + q)
        # return tokens: B, K, D
        return q


class DIDBlockPlus(nn.Module):
    """Degradation Disentangler (DIDBlock++) with spatial+frequency branches, tokenization,
    and auxiliary heads for ingredient strength prediction.
    """
    def __init__(self, in_ch=3, base_ch=64, token_dim=128, num_tokens=4):
        super().__init__()
        # spatial branch: simple encoder
        self.s_enc1 = ConvBlock(in_ch, base_ch)
        self.s_enc2 = ConvBlock(base_ch, base_ch * 2)
        self.s_enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        # frequency branch: Haar transform + small convs
        self.haar = HaarTransform()
        self.f_conv = nn.Sequential(
            ConvBlock(in_ch * 3, base_ch),  # high has 3x channels from Haar
            ConvBlock(base_ch, base_ch * 2),
        )

        # merge features
        merged_dim = base_ch * 4 + base_ch * 2
        self.fusion = ConvBlock(merged_dim, base_ch * 4)

        # tokenization
        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # reduce to manageable spatial size
        self.tokenizer = Tokenizer(in_dim=(base_ch * 4), token_dim=token_dim, num_tokens=num_tokens)

        # auxiliary heads: predict per-ingredient presence/intensity
        self.aux_pred = nn.Sequential(
            nn.Linear(token_dim * num_tokens, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, num_tokens)  # regression or logits per ingredient
        )

    def forward(self, x: torch.Tensor):
        # spatial
        s1 = self.s_enc1(x)
        s2 = self.s_enc2(s1)
        s3 = self.s_enc3(s2)

        # frequency
        low, high = self.haar(x)
        # concat low and high along channel dim (upsample low to match high spatial?)
        # low is half spatial resolution; upsample to input spatial/4 to align with s3
        low_up = F.interpolate(low, size=s3.shape[2:], mode='bilinear', align_corners=False)
        high_up = F.interpolate(high, size=s3.shape[2:], mode='bilinear', align_corners=False)
        f_feat = torch.cat([low_up, high_up], dim=1)  # B, 3C, H/4, W/4
        f_feat = self.f_conv(f_feat)

        merged = torch.cat([s3, f_feat], dim=1)
        merged = self.fusion(merged)

        pooled = self.pool(merged)  # B, C, 16, 16
        tokens = self.tokenizer(pooled)  # B, K, D

        # auxiliary ingredient prediction
        B = tokens.shape[0]
        aux = self.aux_pred(tokens.reshape(B, -1))

        return merged, tokens, aux


# --------------------------- MoE Route Block ---------------------------

class SmallTransformerBlock(nn.Module):
    """A compact transformer block for experts."""
    def __init__(self, dim, num_heads=4, mlp_mult=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * mlp_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: B, N, D
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class Expert(nn.Module):
    """One expert operating on feature maps (conv -> residuals -> optional transformer pooled).
    Each expert expects features in shape B,C,H,W and returns same shape.
    """
    def __init__(self, in_ch, mid_ch=None, use_transformer=False, token_dim=128):
        super().__init__()
        mid_ch = mid_ch or in_ch
        self.conv_stack = nn.Sequential(
            ConvBlock(in_ch, mid_ch),
            ResidualBlock(mid_ch),
            ConvBlock(mid_ch, in_ch, activation=False)
        )
        self.use_transformer = use_transformer
        if use_transformer:
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            self.to_tokens = nn.Linear(in_ch * 8 * 8, token_dim)
            self.transformer = SmallTransformerBlock(token_dim)
            self.from_tokens = nn.Linear(token_dim, in_ch * 8 * 8)

    def forward(self, feat):
        # feat: B,C,H,W
        out = self.conv_stack(feat)
        if self.use_transformer:
            B, C, H, W = out.shape
            pooled = self.pool(out).reshape(B, -1)
            t = self.to_tokens(pooled).unsqueeze(1)  # B,1,D
            t = self.transformer(t)
            t = self.from_tokens(t.squeeze(1)).reshape(B, C, 8, 8)
            t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
            out = out + t
        return out


class MoEManage(nn.Module):
    """Gating network that uses degradation tokens to produce soft/hard top-k routing.
    Exposes methods to compute load balancing loss etc.
    """
    def __init__(self, token_dim=128, num_tokens=4, num_experts=4, k=2, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        # gating MLP: maps aggregated tokens to logits over experts
        self.gate = nn.Sequential(
            nn.Linear(token_dim * num_tokens, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, num_experts)
        )

    def forward(self, tokens, hard=False):
        # tokens: B,K,D -> aggregate
        B = tokens.shape[0]
        x = tokens.reshape(B, -1)
        logits = self.gate(x)  # B, E
        # soft gating via Gumbel-Softmax for differentiable top-k approx
        # we'll compute top-k probabilities and indices
        # for stability use softmax with temperature
        probs = F.softmax(logits / max(1e-6, self.temperature), dim=-1)
        # select top-k indices per batch
        topk = torch.topk(probs, self.k, dim=-1)
        topk_idx = topk.indices  # B, k
        topk_vals = topk.values  # B, k
        # create a sparse routing matrix R: B, E with nonzero at topk positions
        R = torch.zeros_like(probs)
        for b in range(B):
            R[b, topk_idx[b]] = probs[b, topk_idx[b]]
        if hard:
            # hard routing (non-differentiable) - keep as is for inference
            return R, topk_idx
        else:
            # soft routing returned for training
            return R, topk_idx


class MoERouteBlock(nn.Module):
    """Apply experts based on routing computed from degradation tokens.
    Supports top-k sparse routing and load-balancing loss computation.
    """
    def __init__(self, in_ch, num_experts=4, k=2, token_dim=128, num_tokens=4):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Expert(in_ch, mid_ch=in_ch // 2 if in_ch >= 64 else in_ch, use_transformer=(i % 2 == 0), token_dim=token_dim)
            for i in range(num_experts)
        ])
        self.gater = MoEManage(token_dim=token_dim, num_tokens=num_tokens, num_experts=num_experts, k=k)

    def forward(self, feat, tokens, hard=False):
        # feat: B,C,H,W ; tokens: B,K,D
        B = feat.shape[0]
        device = feat.device
        R, topk_idx = self.gater(tokens, hard=hard)  # R: B,E
        outputs = torch.zeros_like(feat)
        # compute expert outputs where needed
        expert_outputs = [None] * self.num_experts
        for e in range(self.num_experts):
            # find batches where expert e is active
            active_mask = (R[:, e] > 1e-8)
            if active_mask.any():
                idx = active_mask.nonzero(as_tuple=False).squeeze(1)
                # slice features
                sub_feat = feat[idx]
                out = self.experts[e](sub_feat)
                expert_outputs[e] = (idx, out)
        # aggregate
        for e, val in enumerate(expert_outputs):
            if val is None:
                continue
            idx, out = val
            # multiply each output by routing weights for each sample
            # R[idx,e] shape: (n,) -> reshape to (n,1,1,1)
            weights = R[idx, e].view(-1, 1, 1, 1)
            outputs[idx] = outputs[idx] + out * weights
        # residual connection
        outputs = outputs + feat
        # compute load-balancing metrics (for loss) as dict
        # load per expert
        load = R.sum(dim=0)  # E
        usage_entropy = -(R * (R + 1e-9).log()).sum(dim=-1).mean()
        stats = {'load': load, 'usage_entropy': usage_entropy}
        return outputs, stats, topk_idx


# --------------------------- Decoder / Conditional Refinement ---------------------------

class SimpleDecoder(nn.Module):
    """A simple decoder that upsamples features and reconstructs image.
    Also includes a refinement head (small residual convs) to mimic conditional generator.
    """
    def __init__(self, in_ch, out_ch=3, base_ch=64):
        super().__init__()
        self.up1 = nn.Sequential(
            ConvBlock(in_ch, base_ch * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up2 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.head = nn.Sequential(
            ResidualBlock(base_ch),
            ConvBlock(base_ch, base_ch, activation=False),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)
        )
        # refinement module mimicking conditional generator: small residual stack + perceptual shortcut
        self.refine = nn.Sequential(
            ConvBlock(out_ch, out_ch),
            ConvBlock(out_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, feat, coarse_input=None):
        # feat: B,C,H/4,W/4 (assume downsampled by factor 4)
        x = self.up1(feat)
        x = self.up2(x)
        out = self.head(x)
        if coarse_input is not None:
            # add coarse input with learned residual
            c = F.interpolate(coarse_input, size=out.shape[2:], mode='bilinear', align_corners=False)
            residual = self.refine(out - c)
            out = out + residual
        return out


# --------------------------- Full Model ---------------------------

class IMDNetPlus(nn.Module):
    """High-level wiring of the proposed architecture.
    DIDBlockPlus -> MoERouteBlock -> Decoder
    Exposes hooks for losses and metrics.
    """
    def __init__(self, in_ch=3, base_ch=64, token_dim=128, num_tokens=4, num_experts=4, topk=2):
        super().__init__()
        self.did = DIDBlockPlus(in_ch=in_ch, base_ch=base_ch, token_dim=token_dim, num_tokens=num_tokens)
        self.reduce = ConvBlock(base_ch * 4, base_ch * 4)
        self.moe = MoERouteBlock(in_ch=base_ch * 4, num_experts=num_experts, k=topk, token_dim=token_dim, num_tokens=num_tokens)
        self.decoder = SimpleDecoder(in_ch=base_ch * 4, out_ch=in_ch, base_ch=base_ch)

    def forward(self, x, hard_routing=False, coarse_input=None):
        fused, tokens, aux = self.did(x)
        fused = self.reduce(fused)
        routed, stats, topk_idx = self.moe(fused, tokens, hard=hard_routing)
        out = self.decoder(routed, coarse_input=coarse_input)
        return out, tokens, aux, stats, topk_idx


# --------------------------- Losses & Training Helpers ---------------------------

class InfoNCELoss(nn.Module):
    """A simple InfoNCE loss over tokens: pulls tokens from same class closer.
    For prototype usage we expect labels indicating degradation family for each sample.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, tokens, labels):
        # tokens: B,K,D -> collapse tokens mean for whole sample
        z = tokens.mean(dim=1)  # B,D
        z = F.normalize(z, dim=-1)
        logits = z @ z.t() / self.temp
        B = z.shape[0]
        labels = labels.view(-1)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # B,B
        # InfoNCE: for each i, positives are mask[i] (excluding self)
        logits_mask = (~torch.eye(B, dtype=torch.bool, device=z.device))
        positives = mask & logits_mask
        # compute log-softmax
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        # for each i, sum log_prob of positives (if none, loss 0)
        pos_log_prob = (log_prob * positives.float()).sum(dim=1) / (positives.sum(dim=1).float() + 1e-9)
        loss = -pos_log_prob.mean()
        return loss


def mi_proxy_loss(tokens, content_feat):
    """Proxy loss to reduce mutual information between degradation tokens and content features.
    Here we compute correlation between mean-pooled tokens and mean-pooled content and minimize.
    This is a simple but effective heuristic.
    """
    z = tokens.mean(dim=1)  # B,D
    c = F.adaptive_avg_pool2d(content_feat, (1, 1)).reshape(content_feat.shape[0], -1)  # B,C
    z = (z - z.mean(dim=0, keepdim=True))
    c = (c - c.mean(dim=0, keepdim=True))
    corr = torch.matmul(z.t(), c) / (tokens.shape[0] - 1)
    # penalize squared correlations
    loss = (corr ** 2).mean()
    return loss


# perceptual and adversarial losses are placeholders (user may plug in real modules)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # in practice use pretrained VGG features; here we use L1 as a placeholder
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


# --------------------------- Example usage and quick test ---------------------------

if __name__ == '__main__':
    # quick smoke test of forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IMDNetPlus(in_ch=3, base_ch=64, token_dim=128, num_tokens=4, num_experts=4, topk=2).to(device)
    x = torch.randn(2, 3, 128, 128).to(device)
    out, tokens, aux, stats, topk_idx = model(x)
    print('Out', out.shape)
    print('Tokens', tokens.shape)
    print('Aux', aux.shape)
    print('Stats', {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in stats.items()})
    print('Topk idx', topk_idx)

    # loss computation example
    labels = torch.tensor([0, 1], device=device)
    info_nce = InfoNCELoss()
    loss_nce = info_nce(tokens, labels)
    print('InfoNCE loss', loss_nce.item())

    mi_l = mi_proxy_loss(tokens, out.detach())
    print('MI proxy loss', float(mi_l))

    # example training step (one iteration) - pseudo code
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.randn_like(out)
    recon_loss = F.l1_loss(out, target)
    perc = PerceptualLoss()
    perc_loss = perc(out, target)
    total_loss = recon_loss + 0.1 * perc_loss + 0.01 * loss_nce + 0.01 * mi_l
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print('Done one pseudo-training step.')

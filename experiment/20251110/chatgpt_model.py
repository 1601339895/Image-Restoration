import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------
# 1️⃣ 退化估计器 (DE)
# ------------------------------------------------------------
class DegradationEstimator(nn.Module):
    """Estimate degradation vector from image input"""
    def __init__(self, in_channels=3, d_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, d_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        d = self.fc(feat)
        return d  # (B, d_dim)


# ------------------------------------------------------------
# 2️⃣ 退化嵌入 (z)
# ------------------------------------------------------------
class DegradationEmbedding(nn.Module):
    """Map degradation vector d -> embedding z"""
    def __init__(self, d_dim=8, z_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, d):
        return self.mlp(d)  # (B, z_dim)


# ------------------------------------------------------------
# 3️⃣ 退化感知特征扰动器 (DAFP)
#     结合空间域和频域的双重扰动
# ------------------------------------------------------------
class DAFP(nn.Module):
    def __init__(self, in_dim, z_dim, eps=0.1):
        super().__init__()
        self.spatial_gen = nn.Sequential(
            nn.Conv2d(in_dim + z_dim, in_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        )
        self.freq_gen = nn.Sequential(
            nn.Conv2d(in_dim + z_dim, in_dim, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        self.eps = eps

    def forward(self, feat, z):
        B, C, H, W = feat.shape
        z_expand = z.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Spatial perturbation
        s_in = torch.cat([feat, z_expand], dim=1)
        delta_s = self.spatial_gen(s_in)

        # Frequency perturbation
        f_feat = torch.fft.fft2(feat)
        f_real = f_feat.real
        f_imag = f_feat.imag
        f_mag = torch.sqrt(f_real ** 2 + f_imag ** 2)
        f_in = torch.cat([f_mag, z_expand], dim=1)
        delta_f = self.freq_gen(f_in)

        delta = delta_s + delta_f
        norm = torch.norm(delta.view(B, -1), dim=1, keepdim=True) + 1e-8
        delta = delta / norm.view(B, 1, 1, 1) * self.eps
        feat_prime = feat + delta
        return feat_prime, delta


# ------------------------------------------------------------
# 4️⃣ FiLM 条件归一化
# ------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, feat_dim, z_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(z_dim, feat_dim)
        self.beta_fc = nn.Linear(z_dim, feat_dim)
        self.norm = nn.BatchNorm2d(feat_dim)

    def forward(self, feat, z):
        gamma = self.gamma_fc(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(z).unsqueeze(-1).unsqueeze(-1)
        feat_norm = self.norm(feat)
        return feat_norm * (1 + gamma) + beta


# ------------------------------------------------------------
# 5️⃣ Feature Alignment Loss
# ------------------------------------------------------------
def feature_alignment_loss(feat, feat_prime):
    return F.mse_loss(F.normalize(feat, dim=1), F.normalize(feat_prime, dim=1))


# ------------------------------------------------------------
# 6️⃣ Gradient Alignment Regularizer
# ------------------------------------------------------------
def gradient_alignment_regularizer(losses, params):
    grads = []
    for L in losses:
        g = torch.autograd.grad(L, params, retain_graph=True, create_graph=True)
        grads.append(torch.cat([gi.flatten() for gi in g if gi is not None]))
    reg = 0
    count = 0
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            cos_sim = F.cosine_similarity(grads[i], grads[j], dim=0)
            reg += (1 - cos_sim)
            count += 1
    return reg / count if count > 0 else 0


# ------------------------------------------------------------
# 7️⃣ Restormer-like Block (简化版)
# ------------------------------------------------------------
class RestormerBlock(nn.Module):
    """Simplified Transformer block for Restormer"""
    def __init__(self, dim, z_dim):
        super().__init__()
        self.norm1 = FiLM(dim, z_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = FiLM(dim, z_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, z):
        B, C, H, W = x.shape
        feat = x.flatten(2).permute(0, 2, 1)
        x1 = self.norm1(x, z)
        q = k = v = x1.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        x2 = x + attn_out
        x3 = self.norm2(x2, z)
        ffn_out = self.ffn(x3.flatten(2).permute(0, 2, 1))
        ffn_out = ffn_out.permute(0, 2, 1).view(B, C, H, W)
        return x2 + ffn_out


# ------------------------------------------------------------
# 8️⃣ 完整网络集成
# ------------------------------------------------------------
class DegradationAwareRestormer(nn.Module):
    def __init__(self, in_channels=3, dim=64, d_dim=8, z_dim=64):
        super().__init__()
        self.de = DegradationEstimator(in_channels, d_dim)
        self.embed = DegradationEmbedding(d_dim, z_dim)
        self.encoder = nn.Conv2d(in_channels, dim, 3, 1, 1)
        self.rest_block = RestormerBlock(dim, z_dim)
        self.dafp = DAFP(dim, z_dim)
        self.decoder = nn.Conv2d(dim, in_channels, 3, 1, 1)

    def forward(self, x):
        d = self.de(x)
        z = self.embed(d)
        feat = self.encoder(x)
        feat_prime, delta = self.dafp(feat, z)
        feat_mod = self.rest_block(feat_prime, z)
        out = self.decoder(feat_mod)
        return out, feat, feat_prime, delta, z


# ------------------------------------------------------------
# ✅ 测试代码
# ------------------------------------------------------------
if __name__ == "__main__":
    model = DegradationAwareRestormer(in_channels=3, dim=64)
    x = torch.randn(2, 3, 128, 128).requires_grad_()

    out, feat, feat_prime, delta, z = model(x)
    L_align = feature_alignment_loss(feat, feat_prime)
    L_ga = gradient_alignment_regularizer([L_align, out.mean()], model.parameters())

    print("Output shape:", out.shape)
    print("Degradation embedding z:", z.shape)
    print("Perturbation norm:", torch.norm(delta, dim=(1,2,3)))
    print("L_align:", L_align.item())
    print("L_ga:", L_ga.item())

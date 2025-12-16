# -*- coding: utf-8 -*-
# File  : dare_net.py
# Date  : 2025/12/10
# Context : CVPR 2025 - Asymmetric Dynamic Receptive Field Network for All-in-One IR

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## 1. 动态提示生成器 (Degradation Prompt Generator)
## 创新：使用对比学习思想，不仅提取特征，还计算“退化置信度”
##########################################################################

class PromptGen(nn.Module):
    def __init__(self, in_c=3, embed_dim=64):
        super().__init__()
        # 提取多尺度特征以感知不同类型的退化
        self.fe_small = nn.Sequential(nn.Conv2d(in_c, embed_dim // 4, 3, 1, 1), nn.LeakyReLU())
        self.fe_medium = nn.Sequential(nn.Conv2d(in_c, embed_dim // 4, 5, 1, 2), nn.LeakyReLU())  # 感受野大一点
        self.fe_large = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(in_c, embed_dim // 2, 3, 1, 1), nn.LeakyReLU())  # 全局信息

        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        # 输出两个向量：
        # 1. content_prompt: 指导特征融合
        # 2. scale_weights: 指导感受野选择 (Kernel Selection)
        self.to_prompt = nn.Linear(embed_dim, embed_dim)
        self.to_scale = nn.Linear(embed_dim, 3)  # 假设有3种尺度的感受野

    def forward(self, x):
        x1 = self.fe_small(x)
        x2 = self.fe_medium(x)
        x3 = F.interpolate(self.fe_large(x), size=x1.shape[2:], mode='bilinear')

        feat = torch.cat([x1, x2, x3], dim=1)
        global_feat = self.fusion(feat)

        prompt = self.to_prompt(global_feat)
        scale_weights = F.softmax(self.to_scale(global_feat), dim=-1)

        return prompt, scale_weights


##########################################################################
## 2. Encoder 核心：动态感受野专家 (Dynamic Receptive Field MoE)
## 动机：不同退化需要不同的感受野。
## 机制：SK-Net (Selective Kernel) 的 MoE 升级版。
##########################################################################

class DynamicReceptiveExpert(nn.Module):
    def __init__(self, dim, scale_idx):
        super().__init__()
        # 定义三种不同的感受野专家
        if scale_idx == 0:
            # 专家0：专注微小细节 (如噪点) -> 1x1 + 3x3
            self.op = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
            )
        elif scale_idx == 1:
            # 专家1：专注中等纹理 (如雨纹) -> 3x3 Dilated 2
            self.op = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim)
        elif scale_idx == 2:
            # 专家2：专注全局雾霾/模糊 -> 5x5 Dilated 3 or 7x7
            self.op = nn.Conv2d(dim, dim, 5, padding=6, dilation=3, groups=dim)

    def forward(self, x):
        return self.op(x)


class D_MoE_Block(nn.Module):
    """
    Encoder Block: 侧重于“自适应提取特征”
    根据 Prompt 的 scale_weights 动态决定使用哪个感受野
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 定义3个不同感受野的专家
        self.experts = nn.ModuleList([
            DynamicReceptiveExpert(dim, 0),
            DynamicReceptiveExpert(dim, 1),
            DynamicReceptiveExpert(dim, 2)
        ])

        self.proj = nn.Conv2d(dim, dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x, prompt, scale_weights):
        # x: [B, C, H, W]
        # scale_weights: [B, 3] -> 由 PromptNet 生成，决定每个样本更倾向于哪种感受野

        b, c, h, w = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # --- 动态感受野路由 ---
        # 这是一个 Soft Routing，为了梯度平滑和特征丰富度，我们计算加权和
        # 类似 SK-Net，但是由 Prompt 显式指导

        feat_experts = []
        for expert in self.experts:
            feat_experts.append(expert(x))
        feat_experts = torch.stack(feat_experts, dim=1)  # [B, 3, C, H, W]

        # 调整权重维度以进行广播 [B, 3] -> [B, 3, 1, 1, 1]
        weights = scale_weights.view(b, 3, 1, 1, 1)

        # 加权融合：网络根据退化类型，自动“聚焦”到合适的感受野
        x_adaptive = (feat_experts * weights).sum(dim=1)

        # Prompt 调制 (Feature Modulation)
        x_adaptive = x_adaptive * (1 + prompt.view(b, c, 1, 1))

        x = shortcut + self.proj(x_adaptive)
        x = x + self.ffn(x)

        return x


##########################################################################
## 3. Decoder 核心：内容恢复专家 (Restoration MoE)
## 动机：解码器不需要再搜索感受野，而是需要高频/低频内容的精准重建。
## 机制：Top-K 路由，选择最合适的重建算子。
##########################################################################

class RestorationExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用Gated Conv与Simple Attention结合，专注于内容重建
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return self.body(x)


class F_MoE_Block(nn.Module):
    """
    Decoder Block: 侧重于“选择最合适的专家恢复图像”
    使用 Top-K 路由机制，基于 Feature 内容 + Prompt 进行选择
    """

    def __init__(self, dim, num_experts=4, topk=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.topk = topk

        self.experts = nn.ModuleList([RestorationExpert(dim) for _ in range(num_experts)])

        # 路由门控：输入 Feature + Prompt
        self.router = nn.Linear(dim * 2, num_experts)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, prompt, scale_weights=None):
        # Decoder 忽略 scale_weights，只关注 prompt
        b, c, h, w = x.shape
        shortcut = x

        # Norm
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_norm)
        x_in = x_norm.permute(0, 3, 1, 2)

        # --- 路由决策 ---
        # 结合当前像素特征均值 和 全局Prompt
        feat_avg = x_in.mean(dim=[2, 3])  # [B, C]
        router_in = torch.cat([feat_avg, prompt], dim=1)  # [B, 2C]

        logits = self.router(router_in)  # [B, num_experts]
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.topk, dim=-1)

        # --- 稀疏执行 ---
        out = torch.zeros_like(x_in)
        for k in range(self.topk):
            idx = indices[:, k]  # [B]
            w = weights[:, k].view(b, 1, 1, 1)

            for i, expert in enumerate(self.experts):
                # 创建 mask
                mask = (idx == i).float().view(b, 1, 1, 1)
                if mask.sum() > 0:
                    out += mask * w * expert(x_in)

        x = shortcut + self.proj(out)
        return x


##########################################################################
## 4. 主网络架构 (DaReNeT)
##########################################################################

class DaReNeT(nn.Module):
    def __init__(self,
                 in_ch=3,
                 out_ch=3,
                 dim=48,
                 num_blocks=[2, 3, 3, 4],  # Encoder 深度
                 num_dec_blocks=[2, 2, 2],  # Decoder 深度 (通常可以比Encoder浅一点)
                 heads=[1, 2, 4, 8]):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_ch, dim, 3, 1, 1)

        # 1. 核心指挥官：Prompt Generator
        self.prompt_net = PromptGen(in_ch, dim)

        # 2. Encoder (D-MoE: Dynamic Receptive Field)
        self.encoder_layers = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])

        curr_dim = dim
        for i in range(4):  # 4 stages
            stage_blocks = nn.ModuleList([
                D_MoE_Block(curr_dim) for _ in range(num_blocks[i])
            ])
            self.encoder_layers.append(stage_blocks)

            if i < 3:
                self.downsamples.append(nn.Conv2d(curr_dim, curr_dim * 2, 2, 2))
                curr_dim *= 2

        # 3. Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1),
            nn.GELU()
        )

        # 4. Decoder (F-MoE: Feature Restoration)
        self.decoder_layers = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.skips = nn.ModuleList([])

        for i in range(3):
            self.upsamples.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 2, 2))
            curr_dim //= 2

            self.skips.append(nn.Conv2d(curr_dim * 2, curr_dim, 1))

            stage_blocks = nn.ModuleList([
                F_MoE_Block(curr_dim) for _ in range(num_dec_blocks[i])
            ])
            self.decoder_layers.append(stage_blocks)

        self.tail = nn.Conv2d(dim, out_ch, 3, 1, 1)

    def forward(self, x):
        # 1. Generate Prompt & Dynamic Scale Weights
        # prompt: [B, C], 用于调制特征
        # scale_weights: [B, 3], 用于Encoder指导感受野选择
        prompt_base, scale_weights = self.prompt_net(x)

        feats = self.patch_embed(x)
        skip_connections = []

        # 2. Encoder: 带着“寻找病灶”的任务去提取特征
        curr_prompt = prompt_base
        for i, layer in enumerate(self.encoder_layers):
            # 调整Prompt维度以匹配当前层通道数
            if curr_prompt.shape[1] != feats.shape[1]:
                curr_prompt = prompt_base.repeat(1, feats.shape[1] // prompt_base.shape[1])

            for block in layer:
                # 传入 scale_weights 动态控制感受野
                feats = block(feats, curr_prompt, scale_weights)

            skip_connections.append(feats)

            if i < len(self.downsamples):
                feats = self.downsamples[i](feats)

        # 3. Bottleneck
        feats = self.bottleneck(feats)

        # 4. Decoder: 带着“专家会诊”的逻辑去恢复
        for i, layer in enumerate(self.decoder_layers):
            # Upsample
            feats = self.upsamples[i](feats)

            # Skip Fusion
            skip = skip_connections.pop()
            feats = torch.cat([feats, skip], dim=1)
            feats = self.skips[i](feats)

            # 调整Prompt维度
            if curr_prompt.shape[1] != feats.shape[1]:
                curr_prompt = prompt_base.repeat(1, feats.shape[1] // prompt_base.shape[1])

            for block in layer:
                # Decoder 不需要 scale_weights，而是自己路由
                feats = block(feats, curr_prompt)

        out = self.tail(feats) + x
        return out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DaReNeT(dim=32).to(device)
    inp = torch.randn(1, 3, 256, 256).to(device)

    print("Testing Forward Pass...")
    out = model(inp)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
# -*- coding: utf-8 -*-
# Enhanced Encoder Design for All-in-One Image Restoration
# Author: Analysis and Suggestions
# Date: 2025/01/20

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


##########################################################################
## 改进方案1：增强频域建模 - 频率感知门控卷积
##########################################################################
class FrequencyAwareBlock(nn.Module):
    """频率感知块：显式建模低频/中频/高频成分"""
    def __init__(self, dim, bias=False):
        super(FrequencyAwareBlock, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2

        # 1. 空域分支（保持原设计）
        self.spatial_branch = nn.Sequential(
            nn.BatchNorm2d(self.half_dim),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1, groups=self.half_dim, bias=bias),
            nn.Conv2d(self.half_dim, self.half_dim, 1, bias=bias),
            nn.BatchNorm2d(self.half_dim)
        )

        # 2. 频域分支 - 频率感知设计
        # 2.1 实部+虚部 → 频率幅度+相位提取
        self.fft_mag_phase = nn.Sequential(
            nn.Conv2d(dim, self.half_dim, 1, bias=bias),  # 实+虚 → C/2
            nn.BatchNorm2d(self.half_dim),
            nn.GELU()
        )

        # 2.2 频率成分分解（低频/高频门控）
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化捕获频率分布
            nn.Conv2d(self.half_dim, self.half_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(self.half_dim * 2, self.half_dim * 2, 1),  # 输出低频/高频权重
            nn.Sigmoid()
        )

        # 2.3 频域深度卷积处理
        self.fft_conv = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1, groups=self.half_dim, bias=bias),
            nn.Conv2d(self.half_dim, self.half_dim, 1, bias=bias),
            nn.BatchNorm2d(self.half_dim)
        )

        # 3. 跨域交互注意力
        self.cross_domain_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        # 通道拆分
        spatial_x, fft_x = x.chunk(2, dim=1)

        # === 空域分支 ===
        spatial_out = self.spatial_branch(spatial_x)

        # === 频域分支 ===
        # 1. FFT变换
        fft = torch.fft.fft2(fft_x, dim=(-2, -1))
        fft_real_imag = torch.cat([fft.real, fft.imag], dim=1)  # (B, C, H, W)

        # 2. 幅度+相位特征提取
        fft_feat = self.fft_mag_phase(fft_real_imag)  # (B, C/2, H, W)

        # 3. 频率感知门控（自适应调整低频/高频权重）
        freq_weights = self.freq_gate(fft_feat)  # (B, C, 1, 1)
        low_freq_weight, high_freq_weight = freq_weights.chunk(2, dim=1)

        # 4. 低频/高频分离处理（简化版：用平滑滤波近似）
        fft_smooth = F.avg_pool2d(fft_feat, 3, 1, 1)  # 低频近似
        fft_detail = fft_feat - fft_smooth  # 高频近似
        fft_weighted = low_freq_weight * fft_smooth + high_freq_weight * fft_detail

        # 5. 频域卷积处理
        fft_out = self.fft_conv(fft_weighted)

        # === 跨域融合 ===
        fusion = torch.cat([spatial_out, fft_out], dim=1)
        cross_attn = self.cross_domain_attn(fusion)
        fusion_out = fusion * cross_attn

        return fusion_out + residual


##########################################################################
## 改进方案2：任务感知编码器 - 动态特征路由
##########################################################################
class TaskAwareEncoder(nn.Module):
    """任务感知编码器：根据降质类型动态调整特征提取"""
    def __init__(self, dim, num_tasks=5, bias=False):
        """
        Args:
            num_tasks: 任务数量（如：去噪/去模糊/去雨/超分/压缩伪影去除）
        """
        super(TaskAwareEncoder, self).__init__()
        self.dim = dim
        self.num_tasks = num_tasks

        # 1. 任务感知模块：轻量级任务识别
        self.task_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_tasks, 1, bias=bias)
        )

        # 2. 任务特定特征提取器（多专家架构）
        self.task_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias),
                nn.Conv2d(dim, dim, 1, bias=bias),
                nn.BatchNorm2d(dim),
                nn.GELU()
            ) for _ in range(num_tasks)
        ])

        # 3. 共享骨干（通用特征提取）
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 4. 任务融合
        self.task_fusion = nn.Conv2d(dim * 2, dim, 1, bias=bias)

    def forward(self, x):
        residual = x

        # 1. 任务感知权重预测（软路由）
        task_weights = self.task_predictor(x)  # (B, num_tasks, 1, 1)
        task_weights = F.softmax(task_weights, dim=1)

        # 2. 多专家特征提取
        expert_feats = torch.stack([expert(x) for expert in self.task_experts], dim=1)  # (B, num_tasks, C, H, W)

        # 3. 动态特征聚合（软路由加权）
        task_weights = task_weights.unsqueeze(2)  # (B, num_tasks, 1, 1, 1)
        task_specific_feat = (expert_feats * task_weights).sum(dim=1)  # (B, C, H, W)

        # 4. 共享特征 + 任务特定特征
        shared_feat = self.shared_backbone(x)
        fusion_feat = self.task_fusion(torch.cat([shared_feat, task_specific_feat], dim=1))

        return fusion_feat + residual


##########################################################################
## 改进方案3：自适应空域-频域交互
##########################################################################
class AdaptiveSpatialFreqInteraction(nn.Module):
    """自适应空域-频域交互：根据输入内容动态调整双域权重"""
    def __init__(self, dim, bias=False):
        super(AdaptiveSpatialFreqInteraction, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2

        # 空域分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1, groups=self.half_dim, bias=bias),
            nn.Conv2d(self.half_dim, self.half_dim, 1, bias=bias),
            nn.BatchNorm2d(self.half_dim),
            nn.GELU()
        )

        # 频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv2d(dim, self.half_dim, 1, bias=bias),  # 实+虚 → C/2
            nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1, groups=self.half_dim, bias=bias),
            nn.Conv2d(self.half_dim, self.half_dim, 1, bias=bias),
            nn.BatchNorm2d(self.half_dim),
            nn.GELU()
        )

        # 双域交互注意力
        self.interaction_attn = nn.Sequential(
            # 输入：spatial(C/2) + freq(C/2) = C
            nn.Conv2d(dim, dim // 4, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1, bias=bias),  # 输出：spatial_weight(C/2) + freq_weight(C/2)
        )

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        # 通道拆分
        spatial_x, freq_x = x.chunk(2, dim=1)

        # === 空域处理 ===
        spatial_feat = self.spatial_branch(spatial_x)

        # === 频域处理 ===
        fft = torch.fft.fft2(freq_x, dim=(-2, -1))
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)
        freq_feat = self.freq_branch(fft_concat)

        # === 自适应交互 ===
        # 1. 拼接空域+频域特征
        concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)

        # 2. 生成双域交互权重（Sigmoid门控）
        interaction_weights = torch.sigmoid(self.interaction_attn(concat_feat))
        spatial_weight, freq_weight = interaction_weights.chunk(2, dim=1)

        # 3. 加权融合（自适应平衡空域/频域重要性）
        adaptive_spatial = spatial_feat * spatial_weight
        adaptive_freq = freq_feat * freq_weight
        fusion = torch.cat([adaptive_spatial, adaptive_freq], dim=1)

        return fusion + residual


##########################################################################
## 集成建议：分阶段替换编码器
##########################################################################
class ImprovedEncoder(nn.Module):
    """集成改进的编码器设计"""
    def __init__(self, dim, num_blocks=4, num_tasks=5, bias=False):
        super(ImprovedEncoder, self).__init__()

        # 阶段1：浅层使用频率感知（捕获降质特征）
        self.stage1 = nn.Sequential(*[
            FrequencyAwareBlock(dim, bias) for _ in range(num_blocks // 2)
        ])

        # 阶段2：深层使用任务感知（任务特定建模）
        self.stage2 = nn.Sequential(*[
            TaskAwareEncoder(dim, num_tasks, bias) for _ in range(num_blocks // 2)
        ])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x


if __name__ == '__main__':
    # 测试模块
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 频率感知块
    print("=" * 50)
    print("Testing FrequencyAwareBlock...")
    model1 = FrequencyAwareBlock(dim=96).to(device)
    x = torch.randn(2, 96, 64, 64).to(device)
    out1 = model1(x)
    print(f"Input: {x.shape}, Output: {out1.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()) / 1e6:.2f}M")

    # 2. 任务感知编码器
    print("\n" + "=" * 50)
    print("Testing TaskAwareEncoder...")
    model2 = TaskAwareEncoder(dim=96, num_tasks=5).to(device)
    out2 = model2(x)
    print(f"Input: {x.shape}, Output: {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M")

    # 3. 自适应交互
    print("\n" + "=" * 50)
    print("Testing AdaptiveSpatialFreqInteraction...")
    model3 = AdaptiveSpatialFreqInteraction(dim=96).to(device)
    out3 = model3(x)
    print(f"Input: {x.shape}, Output: {out3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()) / 1e6:.2f}M")

    # 4. 集成编码器
    print("\n" + "=" * 50)
    print("Testing ImprovedEncoder...")
    model4 = ImprovedEncoder(dim=96, num_blocks=4, num_tasks=5).to(device)
    out4 = model4(x)
    print(f"Input: {x.shape}, Output: {out4.shape}")
    print(f"Parameters: {sum(p.numel() for p in model4.parameters()) / 1e6:.2f}M")

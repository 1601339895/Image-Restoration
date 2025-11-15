import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torchvision



class DepthwiseSeparableConv(nn.Module):
    """轻量化改进：深度可分离卷积替换标准卷积，降低计算量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷积：逐通道卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # 点卷积：通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DepthwiseSeparableNAFBlock(nn.Module):
    """改进版NAFBlock：用深度可分离卷积替换原卷积，保留通道注意力"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        # 深度可分离卷积替换原空间卷积
        self.conv1 = DepthwiseSeparableConv(c, dw_channel, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(dw_channel, dw_channel, kernel_size=3, padding=1)
        self.conv3 = DepthwiseSeparableConv(dw_channel, c, kernel_size=3, padding=1)
        
        # 通道注意力（保留原NAFBlock核心机制）
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        # FFN轻量化
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * FFN_Expand, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Conv2d(c * FFN_Expand, c, kernel_size=1, padding=0),
            nn.Dropout(drop_out_rate)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        # 通道注意力加权
        sca_weight = self.sca(x)
        x = x * sca_weight
        x = self.conv3(x)
        # 残差连接+FFN
        x = x + residual
        x = x + self.ffn(x)
        return x
    
class DegradationInteractionGraph(nn.Module):
    """创新点1：退化交互图（DIG）- 建模不同退化成分的耦合关系"""
    def __init__(self, deg_dim=64, num_deg_types=3):
        super().__init__()
        self.num_deg_types = num_deg_types  # 退化类型数：雾/雨/噪声
        self.deg_dim = deg_dim  # 每个退化成分的特征维度
        
        # 图注意力层：学习退化间的交互权重（边权重）
        self.graph_att = nn.Sequential(
            nn.Linear(deg_dim * 2, deg_dim),
            nn.GELU(),
            nn.Linear(deg_dim, 1),
            nn.Sigmoid()
        )
        
        # 交互特征融合
        self.fusion = nn.Conv2d(num_deg_types * deg_dim, deg_dim, kernel_size=1, padding=0)

    def forward(self, deg_features):
        """
        Args:
            deg_features: 各退化成分特征 (B, num_deg_types*deg_dim, H, W)
        Returns:
            fused_deg: 融合交互信息的退化特征 (B, deg_dim, H, W)
        """
        # 1. 拆分各退化成分特征（B, deg_dim, H, W）× num_deg_types
        deg_list = torch.split(deg_features, self.deg_dim, dim=1)  # 列表长度=num_deg_types
        
        # 2. 计算退化间的交互权重（边权重）
        att_weights = []
        for i in range(self.num_deg_types):
            for j in range(self.num_deg_types):
                if i == j:
                    # 自交互权重固定为1
                    weight = torch.ones_like(deg_list[i][:, 0:1, :, :])
                else:
                    # 跨退化交互：拼接两个退化特征后计算注意力
                    i_feat = deg_list[i].permute(0, 2, 3, 1)  # (B, H, W, deg_dim)
                    j_feat = deg_list[j].permute(0, 2, 3, 1)
                    concat_feat = torch.cat([i_feat, j_feat], dim=-1)  # (B, H, W, 2*deg_dim)
                    weight = self.graph_att(concat_feat).permute(0, 3, 1, 2)  # (B, 1, H, W)
                att_weights.append(weight)
        
        # 3. 加权融合各退化特征（含交互信息）
        weighted_deg = []
        for i in range(self.num_deg_types):
            # 第i个退化的加权特征 = 自身特征 × 自权重 + 其他退化特征 × 交互权重
            base = deg_list[i] * att_weights[i*self.num_deg_types + i]  # 自交互
            for j in range(self.num_deg_types):
                if i != j:
                    base += deg_list[j] * att_weights[i*self.num_deg_types + j]  # 跨交互
            weighted_deg.append(base)
        
        # 4. 整合所有退化的交互特征
        fused_deg = torch.cat(weighted_deg, dim=1)  # (B, num_deg_types*deg_dim, H, W)
        fused_deg = self.fusion(fused_deg)  # (B, deg_dim, H, W)
        return fused_deg

class DynamicFrequencyAttention(nn.Module):
    """创新点2：动态频域注意力（DFA）- 适配不同退化的频域特性"""
    def __init__(self, in_channels, num_deg_types=3):
        super().__init__()
        self.num_deg_types = num_deg_types
        self.in_channels = in_channels
        
        # 退化类型分类器：预测输入图像的退化组合（如雾+噪声）
        self.deg_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, num_deg_types),  # 输出各退化的存在概率
            nn.Sigmoid()
        )
        
        # 频域掩码生成器：根据退化类型生成频域权重
        self.mask_generator = nn.Sequential(
            nn.Linear(num_deg_types, 64),
            nn.GELU(),
            nn.Linear(64, in_channels),  # 每个通道对应一个频域掩码权重
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: 空间域特征 (B, C, H, W)
        Returns:
            x_freq: 动态加权后的频域特征 (B, C, H, W)
            deg_prob: 退化类型概率 (B, num_deg_types)
        """
        B, C, H, W = x.shape
        
        # 1. 预测退化类型概率
        deg_prob = self.deg_classifier(x)  # (B, num_deg_types)
        
        # 2. 生成频域掩码权重（每个通道对应一个权重）
        mask_weights = self.mask_generator(deg_prob).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # 3. 空间域转频域（FFT）
        x_fft = fft.fft2(x, dim=(-2, -1))  # (B, C, H, W) 复数特征
        x_fft_abs = torch.abs(x_fft)  # 取幅度谱
        x_fft_phase = torch.angle(x_fft)  # 取相位谱
        
        # 4. 动态频域加权（根据退化类型调整高低频权重）
        # 例：噪声→增强高频权重，雾→增强低频权重
        x_fft_weighted = x_fft_abs * mask_weights  # 幅度谱加权
        # 重构复数频域特征
        x_fft_weighted = x_fft_weighted * torch.exp(1j * x_fft_phase)
        
        # 5. 频域转空间域（IFFT）
        x_freq = torch.real(fft.ifft2(x_fft_weighted, dim=(-2, -1)))  # 实数值空间特征
        return x_freq, deg_prob

class ImprovedDIDBlock(nn.Module):
    """创新版DIDBlock：整合DFA（动态频域）和DIG（退化交互）"""
    def __init__(self, in_channels, deg_dim=64, num_deg_types=3):
        super().__init__()
        self.in_channels = in_channels
        self.deg_dim = deg_dim
        
        # 1. 空间特征提取（轻量化NAFBlock）
        self.spatial_feat = DepthwiseSeparableNAFBlock(in_channels)
        
        # 2. 动态频域注意力（DFA）
        self.dfa = DynamicFrequencyAttention(in_channels, num_deg_types)
        
        # 3. 退化交互图（DIG）
        self.dig = DegradationInteractionGraph(deg_dim, num_deg_types)
        
        # 4. 统计系数（SC）：空间-频域特征解耦
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.std_pool = lambda x: torch.std(x, dim=(-2, -1), keepdim=True)  # 标准差池化
        self.sg = lambda x: x.chunk(2, dim=1)[0] * torch.sigmoid(x.chunk(2, dim=1)[1])  # 简单门
        self.sc_conv = nn.Conv2d(in_channels * 2, deg_dim * num_deg_types, kernel_size=1, padding=0)
        
        # 5. 干净特征生成（CF = E - DI）
        self.di_conv = nn.Conv2d(deg_dim, in_channels, kernel_size=1, padding=0)
        self.cf_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x, low_res_x):
        """
        Args:
            x: 上一级编码器特征 (B, C, H, W)
            low_res_x: 低分辨率退化图像特征 (B, C, H, W)
        Returns:
            E: 编码特征（供下一级）(B, C, H, W)
            DI: 解耦退化信息 (B, deg_dim, H, W)
            CF: 干净跳跃特征 (B, C, H, W)
        """
        # 1. 特征拼接与空间特征提取
        x_cat = torch.cat([x, low_res_x], dim=1)  # (B, 2C, H, W)
        x_cat = nn.Conv2d(x_cat.shape[1], self.in_channels, 1, 1, 0)(x_cat)  # 通道调整
        E = self.spatial_feat(x_cat)  # 空间编码特征 (B, C, H, W)
        
        # 2. 动态频域特征提取（DFA）
        freq_feat, deg_prob = self.dfa(E)  # (B, C, H, W), (B, num_deg_types)
        
        # 3. 统计系数（SC）计算：空间+频域特征融合
        gap_feat = self.global_pool(E)  # (B, C, 1, 1)
        std_feat = self.std_pool(freq_feat)  # (B, C, 1, 1)
        sc_feat = torch.cat([gap_feat, std_feat], dim=1)  # (B, 2C, 1, 1)
        sc_feat = self.sg(sc_feat)  # 简单门激活
        sc_feat = self.sc_conv(sc_feat)  # (B, num_deg_types*deg_dim, 1, 1)
        sc_feat = F.interpolate(sc_feat, size=E.shape[-2:], mode='bilinear', align_corners=True)
        
        # 4. 退化交互建模（DIG）
        DI_raw = self.dig(sc_feat)  # (B, deg_dim, H, W)
        DI = self.di_conv(DI_raw)  # 调整通道数与E一致 (B, C, H, W)
        
        # 5. 干净特征生成（CF = E - DI）
        CF = E - DI
        CF = self.cf_conv(CF)  # 特征优化
        return E, DI_raw, CF, deg_prob

class FusionBlock(nn.Module):
    """改进版FBlock：用可学习矩阵聚合各级退化信息"""
    def __init__(self, deg_dim=64, num_levels=4):
        super().__init__()
        self.num_levels = num_levels  # 编码器级数
        # 可学习聚合矩阵（每级退化信息对应一个权重）
        self.learnable_weights = nn.Parameter(torch.ones(num_levels, 1, deg_dim, 1, 1))
        nn.init.constant_(self.learnable_weights, 1/num_levels)  # 初始均匀权重
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(deg_dim, deg_dim, kernel_size=1, padding=0)

    def forward(self, di_list):
        """
        Args:
            di_list: 各级编码器的DI特征列表 (B, deg_dim, H_i, W_i) × num_levels
        Returns:
            global_di: 全局融合退化信息 (B, deg_dim, H_0, W_0)
        """
        # 1. 统一分辨率（上采样到最高分辨率）
        target_size = di_list[0].shape[-2:]
        di_upsampled = []
        for i in range(self.num_levels):
            di = di_list[i]
            if di.shape[-2:] != target_size:
                di = F.interpolate(di, size=target_size, mode='bilinear', align_corners=True)
            di_upsampled.append(di * self.learnable_weights[i])  # 加权
        
        # 2. 求和融合
        global_di = torch.sum(torch.stack(di_upsampled, dim=1), dim=1)  # (B, deg_dim, H0, W0)
        global_di = self.fusion_conv(global_di)
        return global_di

class AdaptiveThresholdGenerator(nn.Module):
    """创新点3：自适应阈值生成器 - 根据退化强度动态调整τ"""
    def __init__(self, deg_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(deg_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出τ ∈ [0,1]
        )
        # 限制τ范围：弱退化→τ大（0.2-0.3），强退化→τ小（0.1-0.2）
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.shift = nn.Parameter(torch.tensor(0.1))

    def forward(self, global_di):
        """
        Args:
            global_di: 全局退化特征 (B, deg_dim, H, W)
        Returns:
            tau: 自适应阈值 (B, 1, 1, 1)
        """
        # 1. 全局退化强度计算（平均池化）
        deg_intensity = torch.mean(global_di, dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        # 2. MLP生成基础阈值
        tau_base = self.mlp(deg_intensity.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, 1, 1, 1)
        # 3. 调整阈值范围：τ = 0.1 + 0.2*(1-退化强度)（强度越高，τ越小）
        tau = self.shift + self.scale * (1 - deg_intensity)
        tau = torch.clamp(tau, 0.1, 0.3)  # 限制在合理范围
        return tau


class MultiScaleBranch(nn.Module):
    """创新点4：多尺度分支 - 细/中/粗尺度处理不同退化细节"""
    def __init__(self, in_channels, scale_ratios=[1, 0.5, 0.25]):
        super().__init__()
        self.scale_ratios = scale_ratios  # 细/中/粗尺度比例
        self.branches = nn.ModuleList()
        
        for ratio in scale_ratios:
            if ratio == 1:  # 细尺度：处理局部细节（雨滴/噪声）
                branch = DepthwiseSeparableNAFBlock(in_channels)
            elif ratio == 0.5:  # 中尺度：处理区域退化（局部雾浓）
                branch = nn.Sequential(
                    nn.AvgPool2d(2, 2),
                    DepthwiseSeparableNAFBlock(in_channels),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            else:  # 粗尺度：处理全局退化（全图雾）
                branch = nn.Sequential(
                    nn.AvgPool2d(4, 4),
                    DepthwiseSeparableNAFBlock(in_channels),
                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                )
            self.branches.append(branch)
        
        # 多尺度注意力融合
        self.att_fusion = nn.Conv2d(in_channels * len(scale_ratios), in_channels, kernel_size=1, padding=0)

    def forward(self, x, global_di):
        """
        Args:
            x: 解码器输入特征 (B, C, H, W)
            global_di: 全局退化特征（用于注意力权重）(B, deg_dim, H, W)
        Returns:
            fused_feat: 多尺度融合特征 (B, C, H, W)
        """
        # 1. 各尺度分支前向
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 2. 基于退化特征的注意力权重
        deg_att = F.adaptive_avg_pool2d(global_di, 1)  # (B, deg_dim, 1, 1)
        deg_att = nn.Conv2d(global_di.shape[1], len(self.branches), 1, 1, 0)(deg_att)  # (B, 3, 1, 1)
        deg_att = F.softmax(deg_att, dim=1)
        
        # 3. 加权融合
        weighted_outputs = []
        for i, (out, att) in enumerate(zip(branch_outputs, deg_att.unbind(1))):
            weighted_outputs.append(out * att.unsqueeze(1))
        
        # 4. 特征整合
        fused_feat = torch.cat(weighted_outputs, dim=1)
        fused_feat = self.att_fusion(fused_feat)
        return fused_feat


class ImprovedTABlock(nn.Module):
    """改进版TABlock：整合自适应阈值+多尺度融合"""
    def __init__(self, in_channels, deg_dim=64):
        super().__init__()
        self.in_channels = in_channels
        
        # 1. 上下文特征提取
        self.context_conv = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(in_channels)
        
        # 2. 自适应阈值生成器
        self.tau_generator = AdaptiveThresholdGenerator(deg_dim)
        
        # 3. 多尺度分支
        self.multi_scale_branch = MultiScaleBranch(in_channels)
        
        # 4. 退化信息注入
        self.di_inject = nn.Conv2d(deg_dim, in_channels, kernel_size=1, padding=0)

    def forward(self, x, skip_cf, global_di):
        """
        Args:
            x: 解码器上一级特征 (B, C, H, W)
            skip_cf: 编码器跳跃连接干净特征 (B, C, H, W)
            global_di: 全局退化特征 (B, deg_dim, H, W)
        Returns:
            out: TABlock输出特征 (B, C, H, W)
        """
        # 1. 跳跃连接融合
        x = x + skip_cf
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 2, 1)  # LayerNorm
        x = self.context_conv(x)
        
        # 2. 退化信息注入
        di_injected = self.di_inject(global_di)
        x = x + di_injected
        
        # 3. 生成自适应阈值
        tau = self.tau_generator(global_di)  # (B, 1, 1, 1)
        
        # 4. 多尺度分支前向
        multi_scale_out = self.multi_scale_branch(x, global_di)
        
        # 5. 稀疏激活（仅保留权重≥tau的分支输出）
        # 计算分支权重（基于退化特征的相关性）
        branch_weight = F.cosine_similarity(x, multi_scale_out, dim=1, eps=1e-6).unsqueeze(1)  # (B, 1, H, W)
        branch_weight = torch.where(branch_weight >= tau, branch_weight, torch.tensor(0., device=x.device))
        
        # 加权输出
        out = multi_scale_out * branch_weight
        return out


class ImprovedIMDNet(nn.Module):
    """完整改进版IMDNet：整合所有创新模块"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 num_encoder_levels=4, num_didblock_per_level=[4,4,4,8], 
                 num_tablock_per_level=[2,2,2,2], deg_dim=64, num_deg_types=3,):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_encoder_levels = num_encoder_levels

        
        # 1. 输入卷积
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        
        # 2. 编码器（含ImprovedDIDBlock）
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()  # 编码器下采样
        for level in range(num_encoder_levels):
            # 下采样（除第一级外）
            if level > 0:
                self.downsample.append(nn.Conv2d(base_channels * (2**(level-1)), 
                                                base_channels * (2**level), 
                                                kernel_size=2, stride=2, padding=0))
            # 该级DIDBlock
            didblocks = nn.ModuleList()
            for _ in range(num_didblock_per_level[level]):
                didblock = ImprovedDIDBlock(
                    in_channels=base_channels * (2**level),
                    deg_dim=deg_dim,
                    num_deg_types=num_deg_types
                )
                didblocks.append(didblock)
            self.encoder.append(didblocks)
        
        # 3. 中间块（额外DIDBlock）
        self.middle_block = nn.ModuleList()
        for _ in range(8):  # 8个DIDBlock
            self.middle_block.append(ImprovedDIDBlock(
                in_channels=base_channels * (2**(num_encoder_levels-1)),
                deg_dim=deg_dim,
                num_deg_types=num_deg_types
            ))
        
        # 4. 融合块FBlock
        self.fblock = FusionBlock(deg_dim=deg_dim, num_levels=num_encoder_levels + 1)  # +1是中间块
        
        # 5. 解码器（含ImprovedTABlock）
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()  # 解码器上采样
        for level in range(num_encoder_levels-1, -1, -1):
            # 上采样
            self.upsample.append(nn.ConvTranspose2d(
                base_channels * (2**(level+1)) if level < num_encoder_levels-1 else base_channels * (2**level),
                base_channels * (2**level),
                kernel_size=2, stride=2, padding=0
            ))
            # 该级TABlock
            tablocks = nn.ModuleList()
            for _ in range(num_tablock_per_level[level]):
                tablock = ImprovedTABlock(
                    in_channels=base_channels * (2**level),
                    deg_dim=deg_dim
                )
                tablocks.append(tablock)
            self.decoder.append(tablocks)
        
        # 6. 输出卷积（生成残差图）
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, low_res_x_list=None, ir_x=None, frame_feats=None):
        """
        Args:
            x: 输入图像 (B, 3, H, W)；视频模式下为单帧 (B, 3, H, W)
            low_res_x_list: 低分辨率输入列表 (B, 3, H/2^i, W/2^i) × num_encoder_levels
            ir_x: 红外图像（跨模态模式）(B, 3, H, W)
            frame_feats: 多帧特征（视频模式）(B, 3, 3, H, W) → (B, num_frames, C, H, W)
        Returns:
            restored: 恢复图像 (B, 3, H, W)
            global_di: 全局退化特征（用于损失计算）(B, deg_dim, H, W)
        """
        # 1. 输入处理
        x = self.input_conv(x)  # (B, C, H, W)
        
        
        # 2. 编码器前向（记录各级DI和CF）
        encoder_E = []  # 编码特征
        encoder_DI = []  # 退化信息
        encoder_CF = []  # 干净跳跃特征
        current_E = x
        
        for level in range(self.num_encoder_levels):
            # 下采样（除第一级）
            if level > 0:
                current_E = self.downsample[level-1](current_E)
            
            # 该级DIDBlock
            for didblock in self.encoder[level]:
                low_res_x = low_res_x_list[level] if low_res_x_list is not None else current_E
                current_E, current_DI, current_CF, _ = didblock(current_E, low_res_x)
            
            # 保存结果
            encoder_E.append(current_E)
            encoder_DI.append(current_DI)
            encoder_CF.append(current_CF)
        
        # 3. 中间块前向
        middle_E = current_E
        middle_DI = []
        for didblock in self.middle_block:
            low_res_x = F.interpolate(low_res_x_list[-1], size=middle_E.shape[-2:], mode='bilinear', align_corners=True)
            middle_E, current_DI, _, _ = didblock(middle_E, low_res_x)
        middle_DI.append(current_DI)
        
        # 4. FBlock融合退化信息
        all_DI = encoder_DI + middle_DI  # 各级DI + 中间块DI
        global_di = self.fblock(all_DI)  # (B, deg_dim, H, W)
        
        # 5. 解码器前向
        current_D = middle_E
        for level_idx in range(self.num_encoder_levels):
            # 对应编码器的反向级别
            encoder_level = self.num_encoder_levels - 1 - level_idx
            
            # 上采样
            if level_idx > 0:
                current_D = self.upsample[level_idx](current_D)
            
            # 该级TABlock（结合跳跃连接CF）
            skip_CF = encoder_CF[encoder_level]
            for tablock in self.decoder[level_idx]:
                current_D = tablock(current_D, skip_CF, global_di)
        
        # 6. 生成残差图并输出
        residual = self.output_conv(current_D)  # (B, 3, H, W)
      
        restored = x[:, :3, :, :] + residual  # 原始输入+残差
        
        restored = torch.clamp(restored, 0., 1.)  # 像素值限制
        return restored, global_di


class DomainAdaptiveLoss(nn.Module):
    """创新点7：域自适应损失 - 合成-真实域对齐+循环一致性"""
    def __init__(self, lambda_char=1.0, lambda_edge=0.05, lambda_freq=0.1, lambda_decouple=0.001, lambda_domain=0.5, lambda_cycle=0.3):
        super().__init__()
        self.lambda_char = lambda_char  # Charbonnier损失权重
        self.lambda_edge = lambda_edge  # 边缘损失权重
        self.lambda_freq = lambda_freq  # 频域损失权重
        self.lambda_decouple = lambda_decouple  # 解耦损失权重
        self.lambda_domain = lambda_domain  # 域对齐损失权重
        self.lambda_cycle = lambda_cycle  # 循环一致性损失权重
        
        # 边缘检测（Laplacian）
        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.laplacian.weight.data = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]] * 3)
        
        # 域分类器（区分合成/真实域）
        self.domain_clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def charbonnier_loss(self, pred, target, eps=1e-3):
        """Charbonnier损失（鲁棒L1损失）"""
        return torch.sqrt(torch.square(pred - target) + eps**2).mean()

    def edge_loss(self, pred, target):
        """边缘损失"""
        pred_edge = self.laplacian(pred)
        target_edge = self.laplacian(target)
        return F.l1_loss(pred_edge, target_edge)

    def frequency_loss(self, pred, target):
        """频域损失（FFT幅度谱损失）"""
        pred_fft = fft.fft2(pred, dim=(-2, -1))
        target_fft = fft.fft2(target, dim=(-2, -1))
        pred_abs = torch.abs(pred_fft)
        target_abs = torch.abs(target_fft)
        return F.l1_loss(pred_abs, target_abs)

    def decouple_loss(self, cf, di):
        """解耦损失（最大化CF和DI的余弦距离）"""
        cf_flat = cf.view(cf.shape[0], -1)
        di_flat = di.view(di.shape[0], -1)
        cos_sim = F.cosine_similarity(cf_flat, di_flat, dim=1)
        return (1 - cos_sim).mean()  # 目标：cos_sim→0，损失→1

    def domain_alignment_loss(self, synth_feat, real_feat):
        """域对齐损失（对比学习缩小合成-真实域差距）"""
        # 1. 域分类损失（让分类器无法区分域）
        synth_domain = self.domain_clf(synth_feat)
        real_domain = self.domain_clf(real_feat)
        domain_loss = F.binary_cross_entropy(synth_domain, torch.zeros_like(synth_domain)) + \
                      F.binary_cross_entropy(real_domain, torch.ones_like(real_domain))
        
        # 2. 特征距离损失（最小化合成-真实特征距离）
        synth_norm = F.normalize(synth_feat.view(synth_feat.shape[0], -1), dim=1)
        real_norm = F.normalize(real_feat.view(real_feat.shape[0], -1), dim=1)
        dist_loss = torch.mean(torch.sqrt(torch.sum(torch.square(synth_norm - real_norm), dim=1)))
        
        return domain_loss + dist_loss

    def cycle_consistency_loss(self, x, restored_x, deg_model):
        """循环一致性损失（x→恢复→再退化→x）"""
        # deg_model：退化模型（模拟真实退化）
        re_degraded_x = deg_model(restored_x)
        return F.l1_loss(x, re_degraded_x)

    def forward(self, pred, target, cf, di, synth_feat=None, real_feat=None, x=None, deg_model=None):
        """
        Args:
            pred: 模型输出 (B, 3, H, W)
            target: 干净标签 (B, 3, H, W)
            cf: 干净特征 (B, C, H, W)
            di: 退化信息 (B, deg_dim, H, W)
            synth_feat: 合成域特征 (B, C, H, W)
            real_feat: 真实域特征 (B, C, H, W)
            x: 原始退化图像（循环损失）(B, 3, H, W)
            deg_model: 退化模型（循环损失）
        Returns:
            total_loss: 总损失
        """
        # 基础损失
        loss_char = self.charbonnier_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)
        loss_freq = self.frequency_loss(pred, target)
        loss_decouple = self.decouple_loss(cf, di)
        
        # 域自适应损失（仅训练时）
        loss_domain = torch.tensor(0., device=pred.device)
        if synth_feat is not None and real_feat is not None:
            loss_domain = self.domain_alignment_loss(synth_feat, real_feat)
        
        # 循环一致性损失（仅真实数据自监督训练）
        loss_cycle = torch.tensor(0., device=pred.device)
        if x is not None and deg_model is not None and self.training:
            loss_cycle = self.cycle_consistency_loss(x, pred, deg_model)
        
        # 总损失
        total_loss = self.lambda_char * loss_char + \
                     self.lambda_edge * loss_edge + \
                     self.lambda_freq * loss_freq + \
                     self.lambda_decouple * loss_decouple + \
                     self.lambda_domain * loss_domain + \
                     self.lambda_cycle * loss_cycle
        return total_loss


if __name__ == 'main':
    base_channels = 48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedIMDNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        is_video=False,
        is_crossmodal=False
    ).to(device)



def train_improved_imdnet():
    """训练流程：含合成-真实域自适应训练"""
    # 1. 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 16
    lr = 2e-4
    base_channels = 64
    
    # 3. 退化模型（用于循环一致性损失）
    class RealDegradationModel(nn.Module):
        """模拟真实退化：雾+雨+噪声"""
        def forward(self, x):
            # 简化实现：实际需根据真实数据分布设计
            x = x + 0.05 * torch.randn_like(x)  # 噪声
            x = x * (0.7 + 0.3 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))  # 雾（亮度衰减）
            return x
    deg_model = RealDegradationModel().to(device)
    
    # 4. 模型初始化
    model = ImprovedIMDNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        is_video=False,
        is_crossmodal=False
    ).to(device)
    
    # 5. 优化器与损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DomainAdaptiveLoss()
    loader = torchvision.datasets()
    epochs_phase = 100
        
    for epoch in range(epochs_phase):
        total_loss = 0.
        for batch_idx, data in enumerate(loader):
            if phase == 'synth_pretrain':
                degraded_x, clean_x, low_res_x_list = data  # 合成数据：退化图+干净图+低分辨率图
                degraded_x, clean_x = degraded_x.to(device), clean_x.to(device)
                low_res_x_list = [x.to(device) for x in low_res_x_list]
                
                # 前向
                pred, global_di = model(x=degraded_x, low_res_x_list=low_res_x_list)
                # 取第一个编码器的CF用于解耦损失
                cf = model.encoder[0][-1].cf_conv(model.encoder[0][-1].E - model.encoder[0][-1].DI)
                
                # 计算损失（无域对齐损失）
                loss = criterion(
                    pred=pred,
                    target=clean_x,
                    cf=cf,
                    di=global_di,
                    synth_feat=None,
                    real_feat=None,
                    x=None,
                    deg_model=None
                )
            else:
                degraded_x, clean_x, low_res_x_list, synth_x = data  # 真实数据：含对应合成数据
                degraded_x, clean_x, synth_x = degraded_x.to(device), clean_x.to(device), synth_x.to(device)
                low_res_x_list = [x.to(device) for x in low_res_x_list]
                
                # 前向（真实数据+合成数据）
                pred_real, di_real = model(x=degraded_x, low_res_x_list=low_res_x_list)
                pred_synth, di_synth = model(x=synth_x, low_res_x_list=[F.interpolate(synth_x, size=x.shape[-2:]) for x in low_res_x_list])
                
                # 取CF
                cf_real = model.encoder[0][-1].cf_conv(model.encoder[0][-1].E - model.encoder[0][-1].DI)
                cf_synth = model.encoder[0][-1].cf_conv(model.encoder[0][-1].E - model.encoder[0][-1].DI)
                
                # 计算损失（含域对齐+循环损失）
                loss = criterion(
                    pred=pred_real,
                    target=clean_x,
                    cf=cf_real,
                    di=di_real,
                    synth_feat=di_synth,
                    real_feat=di_real,
                    x=degraded_x,
                    deg_model=deg_model
                )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * degraded_x.shape[0]
        
        # 日志
        avg_loss = total_loss / len(loader.dataset)
        print(f"Phase: {phase}, Epoch: {epoch+1}/{epochs_phase}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
    
    # 保存阶段模型
    torch.save(model.state_dict(), f'imdnet_{phase}.pth')
    
    # 7. 推理示例
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(1, 3, 256, 256).to(device)  # 测试输入
        low_res_x_list = [F.interpolate(test_x, size=(256//(2**i), 256//(2**i))) for i in range(4)]
        pred, _ = model(x=test_x, low_res_x_list=low_res_x_list)
        print(f"Inference done, output shape: {pred.shape}")



# 启动训练
if __name__ == "__main__":
    train_improved_imdnet()

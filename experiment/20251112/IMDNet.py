import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SimpleGate(nn.Module):
    """
    论文3.2节：SimpleGate（SG），替代非线性激活函数
    功能：将输入特征沿通道维度分割为两部分，通过线性门控机制控制特征流动
    公式参考：论文公式(3)中SG(·)定义
    """
    def forward(self, x):
        # 沿通道维度分割为两部分：x1（前C/2通道）、x2（后C/2通道）
        x1, x2 = x.chunk(2, dim=1)
        # 门控机制：x1 * sigmoid(x2)（论文中"线性门"的简化实现，符合统计特性要求）
        return x1 * torch.sigmoid(x2)


class SimplifiedChannelAttention(nn.Module):
    """
    论文3.3节：Simplified Channel Attention（SCA）
    功能：轻量化通道注意力，增强关键通道特征（用于TABlock）
    结构参考：论文图3(c)中SCA模块
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP（全局平均池化）
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()  # 注意力权重生成
        )

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn  # 通道注意力加权

"""
class NAFBlock(nn.Module):
    论文3.2节：NAFBlock（基础空间特征提取模块，来自参考论文[3]）
    功能：作为DIDBlock的空间域特征提取核心，捕捉图像局部与全局依赖
    结构参考：论文图3(b)中NAFBlock模块

    def __init__(self, channel, reduction=4, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=bias)
        self.norm1 = nn.LayerNorm(channel)  # 层归一化（沿通道维度）
        self.sg1 = SimpleGate()  # 替换ReLU的门控激活
        
        self.conv2 = nn.Conv2d(channel//2, channel//2, kernel_size=3, padding=1, bias=bias)
        self.norm2 = nn.LayerNorm(channel//2)
        self.sg2 = SimpleGate()
        
        # 通道注意力（SCA简化版）
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel//2, channel//reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel//2, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv2d(channel//2, channel, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(channel)

    def forward(self, x):
        residual = x  # 残差连接
        
        # 第一分支：卷积+归一化+门控
        x = self.conv1(x)
        x = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)  # LayerNorm需调整维度
        x = self.sg1(x)
        
        # 第二分支：卷积+归一化+门控+注意力
        x = self.conv2(x)
        x = self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.sg2(x)
        attn = self.attn(x)
        x = x * attn
        
        # 输出分支：卷积+归一化+残差
        x = self.conv3(x)
        x = self.norm3(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x + residual  # 残差融合
"""

class NAFBlock(nn.Module):
    """
    修复通道数不匹配：注意力模块输入通道适配SimpleGate后的减半通道
    """
    def __init__(self, channel, reduction=4, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=bias)
        self.norm1 = nn.LayerNorm(channel)  # 沿通道维度
        self.sg1 = SimpleGate()  # 通道数减半：channel → channel//2
        
        self.conv2 = nn.Conv2d(channel//2, channel//2, kernel_size=3, padding=1, bias=bias)
        self.norm2 = nn.LayerNorm(channel//2)
        self.sg2 = SimpleGate()  # 通道数再次减半？不！sg2输入已为channel//2，输出为(channel//2)//2=channel//4
        
        # 关键修复：注意力模块输入通道改为 channel//4（sg2后的通道数）
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 输入通道：sg2输出为 (channel//2)//2 = channel//4
            nn.Conv2d(channel//4, (channel//4)//reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # 输出通道：与sg2输出通道一致（channel//4）
            nn.Conv2d((channel//4)//reduction, channel//4, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 关键修复：conv3输入通道改为 channel//4（sg2+注意力后的通道数）
        self.conv3 = nn.Conv2d(channel//4, channel, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(channel)

    def forward(self, x):
        residual = x  # 残差连接 torch.Size([2, 128, 256, 256])
        
        # 第一分支：conv1→norm1→sg1（通道数：channel → channel//2）
        x = self.conv1(x)
        # LayerNorm需调整维度：(B,C,H,W) → (B,H,W,C) → 归一化后恢复
        x = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)  # torch.Size([2, 128, 256, 256])
        x = self.sg1(x)  # 此时x通道数：channel//2
        
        # 第二分支：conv2→norm2→sg2（通道数：channel//2 → channel//4）
        x = self.conv2(x)
        x = self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.sg2(x)  # 此时x通道数：channel//4
        # 注意力加权（输入通道与x一致：channel//4）
        attn = self.attn(x)
        x = x * attn
        
        # 输出分支：conv3→norm3→残差融合（通道数：channel//4 → channel）
        x = self.conv3(x)
        x = self.norm3(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x + residual  # 残差融合：x（channel） + residual（channel）


class DynamicFilter(nn.Module):
    """
    论文3.2节：Dynamic Filter（DF），动态频域分解模块
    功能：将空间特征E_l分解为高频（F_H）和低频（F_L）成分
    公式参考：论文公式(2)：F_H, F_L = DF(E_l)
    设计思路：通过可学习动态权重实现频域分离，高频捕捉边缘/细节，低频捕捉平滑区域
    """
    def __init__(self, channel):
        super().__init__()
        # 动态权重生成（基于通道注意力）
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.Sigmoid()  # 权重范围[0,1]，控制频域分离比例
        )
        # 高频提取（3x3卷积增强边缘）
        self.high_freq_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        # 低频提取（平均池化平滑）
        self.low_freq_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 生成动态权重
        weight = self.weight_gen(x)
        # 高频特征：权重引导的边缘增强
        F_H = self.high_freq_conv(x) * weight
        # 低频特征：权重引导的平滑
        F_L = self.low_freq_pool(x) * (1 - weight)
        return F_H, F_L


class StatisticalCoefficient(nn.Module):
    """
    论文3.2节：Statistical Coefficient（SC），统计系数计算模块
    功能：对空间/频域特征进行统计分析，生成退化信息表征
    公式参考：论文公式(3)中SC(·)定义：结合GAP、STD、SG和1x1卷积
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.channel = channel
        # GAP（全局平均池化）和STD（标准差池化）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.std = lambda x: torch.sqrt(torch.var(x, dim=(2,3), keepdim=True) + 1e-6)  # 避免方差为0
        
        # 1x1卷积+SG处理分支（GAP和STD各一个分支）
        self.process_gap = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, padding=0),
            SimpleGate(),
            nn.Conv2d((channel//reduction)//2, channel, kernel_size=1, padding=0)  # SG后通道减半，需恢复
        )
        self.process_std = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, padding=0),
            SimpleGate(),
            nn.Conv2d((channel//reduction)//2, channel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # 计算GAP和STD特征
        gap_feat = self.gap(x)
        std_feat = self.std(x)
        
        # 分别处理GAP和STD特征
        gap_sc = self.process_gap(gap_feat)
        std_sc = self.process_std(std_feat)
        
        # 拼接得到最终统计系数（论文公式(3)中⊕表示拼接）
        sc = torch.cat([gap_sc, std_sc], dim=1)
        # 调整通道数与输入一致（拼接后通道为2*channel，需压缩）
        sc = nn.Conv2d(2*self.channel, self.channel, kernel_size=1, padding=0).to(x.device)(sc)
        return sc

class DIDBlock(nn.Module):
    """
    论文3.2节：退化成分解耦块（DIDBlock）
    核心功能：分离退化特征（DI）与清晰图像特征（CF），输出用于下一级的编码特征（E）
    公式参考：
        1. E_l = NAFBlock(f_1x1^c([E_{l-1}, Convs(I_l)]))  # 空间特征提取
        2. F_H, F_L = DF(E_l)  # 频域分解
        3. DI_l = [SC(F_H) ⊕ SC(E_l) ⊕ SC(E_l)]/6 ⊗ E_l  # 退化信息计算
        4. CF_l = E_l ⊖ DI_l  # 清晰特征生成
    结构参考：论文图3(b)
    """
    def __init__(self, in_channel, out_channel, reduction=4):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # 1x1卷积：调整拼接后的通道数（E_prev + Convs(I_l) 的通道数 → out_channel）
        self.conv1x1 = nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, padding=0)
        
        # 空间特征提取：NAFBlock
        self.nafblock = NAFBlock(channel=out_channel, reduction=reduction)
        
        # 频域分解：DynamicFilter
        self.df = DynamicFilter(channel=out_channel)
        
        # 统计系数计算：SC（分别处理F_H、E_l）
        self.sc = StatisticalCoefficient(channel=out_channel, reduction=reduction)
        
        # 退化信息加权：1x1卷积调整维度（SC输出 → 与E_l通道匹配）
        self.di_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, E_prev, I_l_convs):
        """
        输入：
            E_prev: 上一级编码器输出特征（E_{l-1}），shape=(B, C, H, W)
            I_l_convs: 低分辨率退化图像经过Convs处理后的特征，shape=(B, C, H, W)
        输出：
            E_l: 下一级编码器输入特征，shape=(B, out_channel, H, W)
            DI_l: 解耦的退化信息，shape=(B, out_channel, H, W)
            CF_l: 清晰图像特征（用于跳跃连接），shape=(B, out_channel, H, W)
        """
        # Step 1: 拼接E_prev和I_l_convs，调整通道数（论文公式1前半部分）
        concat_feat = torch.cat([E_prev, I_l_convs], dim=1)  # (B, 2*in_channel, H, W)
        concat_feat = self.conv1x1(concat_feat)  # (B, out_channel, H, W)
        
        # Step 2: NAFBlock提取空间特征E_l（论文公式1）
        E_l = self.nafblock(concat_feat)  # (B, out_channel, H, W)
        
        # Step 3: DynamicFilter分解高频（F_H）和低频（F_L）（论文公式2）
        F_H, _ = self.df(E_l)  # 仅使用F_H参与退化信息计算（低频对退化不敏感）
        
        # Step 4: 计算统计系数SC（论文公式3中SC(·)）
        sc_FH = self.sc(F_H)    # SC(F_H)
        sc_El = self.sc(E_l)    # SC(E_l)
        
        # Step 5: 计算退化信息DI_l（论文公式3）
        # [SC(F_H) ⊕ SC(E_l) ⊕ SC(E_l)] → 拼接后平均（除以6是论文经验值）
        sc_concat = torch.cat([sc_FH, sc_El, sc_El], dim=1)  # (B, 3*out_channel, H, W)
        sc_avg = sc_concat.mean(dim=1, keepdim=True).repeat(1, self.out_channel, 1, 1)  # (B, out_channel, H, W)
        DI_l = self.di_conv(sc_avg) * E_l  # ⊗ 表示逐元素乘法
        
        # Step 6: 生成清晰特征CF_l（论文公式4：E_l - DI_l）
        CF_l = E_l - DI_l
        
        return E_l, DI_l, CF_l
    

class FBlock(nn.Module):
    """
    论文3.1/3.4节：融合块（FBlock）
    核心功能：通过可学习矩阵整合所有层级的退化信息（DI），生成全局退化表征（DI_hat）
    公式参考：论文公式(6)：DI_hat_{l-1} = DI_l ⊕ W * DI_hat_l
    结构参考：论文图3(a)中FBlock模块
    设计思路：层级间递归融合，每级DI与下一级融合后的DI_hat加权拼接
    """
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        # 可学习权重W（论文中W通过反向传播优化，初始为1）
        self.W = nn.Parameter(torch.ones(1, channel, 1, 1), requires_grad=True)
        # 1x1卷积：调整拼接后的通道数（DI_l + W*DI_hat_l → channel）
        self.conv_fuse = nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0)

    def forward(self, di_list):
        """
        输入：
            di_list: 所有编码器层级的DI特征列表，顺序为[DI_1, DI_2, DI_3, DI_4, DI_mid]
        输出：
            di_hat_list: 每级对应的融合后退化表征列表，与di_list顺序一致
        """
        di_hat_list = []
        # 从最深层（中间块）开始递归融合（论文中层级从深到浅）
        di_hat_prev = di_list[-1]  # 初始值：中间块的DI
        di_hat_list.append(di_hat_prev)
        
        # 反向遍历编码器层级（从第4级→第1级）
        for di in reversed(di_list[:-1]):
            # 论文公式(6)：DI_hat_{l-1} = DI_l ⊕ W * DI_hat_l
            weighted_di_hat = di_hat_prev * self.W  # W * DI_hat_l
            fused = torch.cat([di, weighted_di_hat], dim=1)  # DI_l ⊕ 加权DI_hat
            di_hat_curr = self.conv_fuse(fused)  # 融合后调整通道数
            di_hat_list.append(di_hat_curr)
            di_hat_prev = di_hat_curr
        
        # 调整顺序为[DI_hat_1, DI_hat_2, DI_hat_3, DI_hat_4, DI_hat_mid]
        di_hat_list = di_hat_list[::-1]
        return di_hat_list

class TABlock(nn.Module):
    """
    论文3.3节：任务自适应块（TABlock）
    核心功能：基于融合后的退化表征（DI_hat）动态激活/融合功能分支，选择最优恢复路径
    公式参考：
        1. DC_{l-1} = f_1x1^c(SCA(SG(f_3x3^dwc(f_1x1^c(LN(D_hat_{l-1}))) ⊕ DI_hat_{l-1})))
        2. DX_0 = Block_0(DC_{l-1} ⊕ D_hat_{l-1})  # 通用分支
        3. W_n = Sigmoid(f_1x1^c(GAP(DC_{l-1})))  # 分支激活权重
        4. DX_n = W_n*Block_n(DX_{n-1}) (W_n≥τ) else DX_{n-1}  # 稀疏激活
    结构参考：论文图3(c)
    设计：1个通用分支 + 3个功能分支（模拟不同恢复任务），阈值τ=0.2（论文经验值）
    """
    def __init__(self, in_channel, out_channel, num_branches=4, tau=0.2):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_branches = num_branches  # 1（通用）+3（功能）=4个分支
        self.tau = tau  # 稀疏激活阈值
        
        # Step 1: 预处理模块（LN + 1x1卷积 + 深度卷积 + SG）
        self.preprocess = nn.Sequential(
            nn.LayerNorm(in_channel),  # LN(D_hat_{l-1})
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),  # f_1x1^c
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),  # f_3x3^dwc（深度卷积）
            SimpleGate()  # SG(·)
        )
        
        # Step 2: 拼接DI_hat后调整通道数（匹配SCA输入）
        self.conv_after_concat = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, padding=0)
        
        # Step 3: 简化通道注意力（SCA）
        self.sca = SimplifiedChannelAttention(channel=in_channel)
        
        # Step 4: 生成DC_{l-1}的1x1卷积
        self.conv_dc = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0)
        
        # Step 5: 分支模块（Block_0~Block_3）：通用分支+功能分支
        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, padding=0),  # f_1x1^c([DC, D_hat])
                    SimpleGate(),  # SG(·)
                    nn.Conv2d(in_channel//2, out_channel, kernel_size=3, padding=1)  # 输出通道调整
                )
            )
        
        # Step 6: 分支激活权重生成（W_n = Sigmoid(f_1x1^c(GAP(DC_{l-1})))）
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP(DC_{l-1})
            nn.Conv2d(in_channel, num_branches, kernel_size=1, padding=0),  # f_1x1^c
            nn.Sigmoid()  # Sigmoid(·)
        )

    def forward(self, D_hat, DI_hat):
        """
        输入：
            D_hat: 解码器上一级输出与跳跃连接CF的融合特征，shape=(B, in_channel, H, W)
            DI_hat: FBlock融合后的退化表征，shape=(B, in_channel, H, W)
        输出：
            DX_final: 动态融合后的恢复特征，shape=(B, out_channel, H, W)
        """
        # Step 1: 预处理D_hat（LN + 1x1 + 深度卷积 + SG）
        D_hat_pre = self.preprocess(D_hat.permute(0,2,3,1)).permute(0,3,1,2)  # 适配LayerNorm维度
        
        # Step 2: 拼接D_hat_pre与DI_hat，调整通道数
        concat_dc = torch.cat([D_hat_pre, DI_hat], dim=1)  # D_hat_pre ⊕ DI_hat
        concat_dc = self.conv_after_concat(concat_dc)
        
        # Step 3: SCA注意力加权，生成DC_{l-1}（论文公式5）
        DC = self.sca(concat_dc)
        DC = self.conv_dc(DC)
        
        # Step 4: 生成分支激活权重W_n（论文公式8）
        W = self.weight_gen(DC)  # (B, num_branches, 1, 1)
        W = W.unsqueeze(-1)  # 适配特征维度：(B, num_branches, 1, 1, 1)
        
        # Step 5: 分支计算与稀疏激活（论文公式7、9）
        DX_prev = torch.zeros_like(D_hat[:, :self.out_channel, :, :])  # 初始值
        for i in range(self.num_branches):
            # 分支输入：DC ⊕ D_hat（论文公式7中DC_{l-1} ⊕ D_hat_{l-1}）
            branch_in = torch.cat([DC, D_hat], dim=1)
            # 分支输出：Block_i(branch_in)
            branch_out = self.branches[i](branch_in)
            
            # 稀疏激活：W_n ≥ τ则使用当前分支，否则沿用前一分支
            if i == 0:
                # 第0个分支：通用分支，强制激活（保证基线性能）
                DX_curr = branch_out
            else:
                # 其他分支：根据权重激活
                mask = (W[:, i, :, :, :] >= self.tau).float()  # 激活掩码
                DX_curr = mask * branch_out + (1 - mask) * DX_prev
            
            DX_prev = DX_curr
        
        return DX_prev


class IMDNet(nn.Module):
    """
    论文3.1节：IMDNet整体架构（Encoder-Decoder + FBlock）
    核心流程：
        1. 浅层特征提取：输入退化图像I → F0
        2. 编码器：4级编码器 + 1个中间块（均含DIDBlock），输出E、DI、CF
        3. 低分辨率图像处理：Convs模块生成I_l_convs，输入编码器各级
        4. FBlock：整合所有DI → DI_hat
        5. 解码器：4级解码器（均含TABlock），结合CF和DI_hat生成残差I_R
        6. 输出：Î = I + I_R（论文公式：恢复图像=输入+残差）
    结构参考：论文图3(a)、表1（编码器/解码器模块数量）
    """
    def __init__(self, in_channels=3, out_channels=3, 
                 encoder_did_nums=[4,4,4,8], mid_did_num=8, decoder_tab_nums=[2,2,2,2],
                 base_channel=64):
        """
        参数说明：
            encoder_did_nums: 编码器4级的DIDBlock数量，对应论文3.1节：[4,4,4,8]
            mid_did_num: 中间块的DIDBlock数量，对应论文3.1节：8
            decoder_tab_nums: 解码器4级的TABlock数量，对应论文3.1节：[2,2,2,2]
            base_channel: 基础通道数，编码器通道数按base_channel×2^l递增
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channel = base_channel
        
        # -------------------------- 1. 浅层特征提取 --------------------------
        self.shallow_conv = nn.Conv2d(in_channels, base_channel, kernel_size=3, padding=1)
        
        # -------------------------- 2. 低分辨率图像Convs模块 --------------------------
        # 功能：对原图像下采样4个尺度，生成I_l_convs（对应编码器4级）
        self.lr_convs = nn.ModuleList()
        for i in range(4):  # 4个尺度（1/1, 1/2, 1/4, 1/8）
            down_scale = 2 ** i
            self.lr_convs.append(
                nn.Sequential(
                    # 下采样：双线性插值
                    nn.Upsample(scale_factor=1/down_scale, mode='bilinear', align_corners=True),
                    # Convs模块：2层3x3卷积 + 1x1通道调整（匹配编码器输入通道）
                    nn.Conv2d(in_channels, base_channel * (2 ** i), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_channel * (2 ** i), base_channel * (2 ** i), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_channel * (2 ** i), base_channel * (2 ** i), kernel_size=1, padding=0)
                )
            )
        
        # -------------------------- 3. 编码器（Encoder） --------------------------
        self.encoder = nn.ModuleList()
        self.encoder_did_nums = encoder_did_nums
        # 4级编码器，通道数：base_channel → 2*base_channel → 4*base_channel → 8*base_channel
        for l in range(4):
            in_ch = base_channel * (2 ** l) if l == 0 else base_channel * (2 ** (l))
            out_ch = base_channel * (2 ** (l + 1))
            # 每级包含encoder_did_nums[l]个DIDBlock
            did_blocks = nn.ModuleList()
            for _ in range(encoder_did_nums[l]):
                did_blocks.append(DIDBlock(in_channel=in_ch, out_channel=out_ch))
            # 下采样模块（每级编码器末尾，降低分辨率）
            downsample = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)  # stride=2下采样
            self.encoder.append(nn.ModuleDict({'did_blocks': did_blocks, 'downsample': downsample}))
        
        # -------------------------- 4. 中间块（Middle Block） --------------------------
        self.mid_block = nn.ModuleList()
        mid_in_ch = base_channel * (2 ** 4)  # 编码器最后一级输出通道：8*base_channel
        for _ in range(mid_did_num):
            self.mid_block.append(DIDBlock(in_channel=mid_in_ch, out_channel=mid_in_ch))
        
        # -------------------------- 5. 融合块（FBlock） --------------------------
        self.fblock = FBlock(channel=mid_in_ch)  # 输入DI的通道数=中间块通道数
        
        # -------------------------- 6. 解码器（Decoder） --------------------------
        self.decoder = nn.ModuleList()
        self.decoder_tab_nums = decoder_tab_nums
        # 4级解码器，通道数：8*base_channel → 4*base_channel → 2*base_channel → base_channel
        for l in range(4):
            in_ch = base_channel * (2 ** (4 - l))
            out_ch = base_channel * (2 ** (3 - l))
            # 上采样模块（每级解码器开头，恢复分辨率，匹配跳跃连接CF）
            upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # stride=2上采样
            # 每级包含decoder_tab_nums[l]个TABlock
            tab_blocks = nn.ModuleList()
            for _ in range(decoder_tab_nums[l]):
                tab_blocks.append(TABlock(in_channel=out_ch * 2, out_channel=out_ch))  # in_ch=out_ch(CF)+out_ch(上采样)
            self.decoder.append(nn.ModuleDict({'upsample': upsample, 'tab_blocks': tab_blocks}))
        
        # -------------------------- 7. 残差生成与输出 --------------------------
        self.residual_conv = nn.Conv2d(base_channel, out_channels, kernel_size=3, padding=1)  # 生成残差I_R

    def forward(self, I):
        """
        输入：
            I: 退化图像，shape=(B, 3, H, W)
        输出：
            I_hat: 恢复图像，shape=(B, 3, H, W)
            I_R: 残差图像，shape=(B, 3, H, W)
        """
        B, C, H, W = I.shape  # torch.Size([2, 3, 256, 256])
        
        # -------------------------- Step 1: 浅层特征提取 --------------------------
        F0 = self.shallow_conv(I)  # (B, base_channel, H, W)  torch.Size([2, 64, 256, 256])
        
        # -------------------------- Step 2: 处理低分辨率图像（生成I_l_convs） --------------------------
        lr_convs_feats = []
        for l in range(4):
            # 生成第l级编码器对应的低分辨率特征I_l_convs
            lr_feat = self.lr_convs[l](I)
            # 调整尺寸与当前编码器特征匹配（避免下采样后尺寸不匹配）
            target_hw = (H // (2 ** l), W // (2 ** l))
            lr_feat = F.interpolate(lr_feat, size=target_hw, mode='bilinear', align_corners=True)
            lr_convs_feats.append(lr_feat) 
        # lr_convs_feats[0].shape=torch.Size([2, 64, 256, 256]), lr_convs_feats[1].shape=torch.Size([2, 128, 128, 128])
        # lr_convs_feats[2].shape=torch.Size([2, 256, 64, 64]),  lr_convs_feats[3].shape=torch.Size([2, 512, 32, 32])
        
        # -------------------------- Step 3: 编码器前向（输出E、DI、CF列表） --------------------------
        encoder_E = [F0]  # 编码器输入特征列表：E_0=F0, E_1, E_2, E_3, E_4(中间块输出)
        encoder_DI = []    # 编码器DI列表：DI_1, DI_2, DI_3, DI_4
        encoder_CF = []    # 编码器CF列表：CF_1, CF_2, CF_3, CF_4
        
        # 4级编码器前向
        for l in range(4):
            E_prev = encoder_E[-1]
            I_l_convs = lr_convs_feats[l]
            did_blocks = self.encoder[l]['did_blocks']
            downsample = self.encoder[l]['downsample']
            
            # 该级所有DIDBlock前向（取最后一个DIDBlock的输出作为该级结果）
            E_l, DI_l, CF_l = None, None, None
            for did_block in did_blocks:
                E_l, DI_l, CF_l = did_block(E_prev, I_l_convs)
                E_prev = E_l  # 更新E_prev为当前DIDBlock输出
            
            # 下采样（用于下一级编码器输入）
            E_down = downsample(E_l)
            
            # 保存该级结果
            encoder_E.append(E_down)
            encoder_DI.append(DI_l)
            encoder_CF.append(CF_l)
        
        # 中间块前向（8个DIDBlock）
        mid_E = encoder_E[-1]
        mid_DI, mid_CF = None, None
        for did_block in self.mid_block:
            mid_E, mid_DI, mid_CF = did_block(mid_E, I_l_convs=lr_convs_feats[-1])  # 用最后一级低分辨率特征
        
        # 补充中间块结果到列表
        encoder_E.append(mid_E)
        encoder_DI.append(mid_DI)
        encoder_CF.append(mid_CF)
        
        # -------------------------- Step 4: FBlock融合退化信息（DI → DI_hat） --------------------------
        di_hat_list = self.fblock(encoder_DI)  # di_hat_list: [DI_hat_1, DI_hat_2, DI_hat_3, DI_hat_4, DI_hat_mid]
        
        # -------------------------- Step 5: 解码器前向（结合CF和DI_hat） --------------------------
        decoder_D = [encoder_E[-1]]  # 解码器输入特征：初始为中间块输出E_mid
        
        # 4级解码器前向（反向遍历：从第4级→第1级）
        for l in range(4):
            D_prev = decoder_D[-1]
            upsample = self.decoder[l]['upsample']
            tab_blocks = self.decoder[l]['tab_blocks']
            # 对应级的CF和DI_hat
            CF_l = encoder_CF[3 - l]  # encoder_CF顺序：1→4，解码器顺序：4→1，需反向
            DI_hat_l = di_hat_list[3 - l]
            
            # Step 5.1: 上采样（恢复分辨率，匹配CF_l）
            D_up = upsample(D_prev)
            # 调整尺寸与CF_l完全一致
            D_up = F.interpolate(D_up, size=CF_l.shape[2:], mode='bilinear', align_corners=True)
            
            # Step 5.2: 融合上采样特征与CF（跳跃连接）
            D_fuse = torch.cat([D_up, CF_l], dim=1)  # (B, 2*out_ch, H_l, W_l)
            
            # Step 5.3: 该级所有TABlock前向
            D_curr = D_fuse
            for tab_block in tab_blocks:
                D_curr = tab_block(D_hat=D_curr, DI_hat=DI_hat_l)
            
            # 保存该级结果
            decoder_D.append(D_curr)
        
        # -------------------------- Step 6: 生成残差与恢复图像 --------------------------
        # 解码器最后一级输出 → 残差图像I_R
        I_R = self.residual_conv(decoder_D[-1])
        # 调整残差尺寸与输入一致
        I_R = F.interpolate(I_R, size=(H, W), mode='bilinear', align_corners=True)
        # 恢复图像：Î = I + I_R（论文公式）
        I_hat = I + I_R
        
        return I_hat, I_R
    
# -------------------------- 初始化IMDNet --------------------------
if __name__ == "__main__":
    # 配置参数（完全遵循论文）
    imdnet = IMDNet(
        in_channels=3,
        out_channels=3,
        encoder_did_nums=[4,4,4,8],  # 论文3.1节：编码器每级DIDBlock数量
        mid_did_num=8,                # 论文3.1节：中间块DIDBlock数量
        decoder_tab_nums=[2,2,2,2],  # 论文3.1节：解码器每级TABlock数量
        base_channel=64               # 基础通道数（论文未明确，取常用值64）
    )
    
    # 初始化权重（论文未指定，采用默认初始化）
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            init.constant_(m, 1.0)  # FBlock的W初始化为1（论文3.3节）
    
    imdnet.apply(init_weights)
    
    # -------------------------- 测试前向流程 --------------------------
    # 模拟退化图像输入（Batch=2, Channel=3, Height=256, Width=256）
    degraded_image = torch.randn(2, 3, 256, 256)
    # 前向传播
    restored_image, residual = imdnet(degraded_image)
    
    # 输出尺寸验证
    print(f"退化图像尺寸: {degraded_image.shape}")
    print(f"恢复图像尺寸: {restored_image.shape}")
    print(f"残差图像尺寸: {residual.shape}")
    print("前向流程验证通过！")

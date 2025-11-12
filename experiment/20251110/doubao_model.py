import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.autograd import grad


class LightweightDegradationEstimator(nn.Module):
    """
    轻量退化估计分支（LDEB）：
    - 输入：Restormer 浅层特征 (B, C_in, H, W)，C_in 通常为 64（Restormer 初始通道）
    - 输出：
      1. theta_hat: 退化强度 (B, 3) → [fog_intensity, rain_intensity, lowlight_intensity]
      2. omega_hat: 退化类型权重 (B, 3) → 经 Softmax 归一化，满足 sum(omega) = 1
    """
    def __init__(self, C_in=64, hidden_dim=128, seq_len=16, num_heads=8):
        super().__init__()
        self.C_in = C_in
        self.seq_len = seq_len  # 特征图下采样后的空间尺寸（16×16）
        self.d_model = hidden_dim  # Transformer 输入维度
        
        # 1. 局部退化特征提取（轻量 CNN）
        self.cnn_encoder = nn.Sequential(
            # 卷积层1：64→128，3×3卷积+BN+ReLU
            nn.Conv2d(C_in, hidden_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            # 卷积层2：128→256
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 卷积层3：256→256（保持通道数）
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 2. 全局退化分布捕捉（微型 Transformer）
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*2,  # FFN 隐藏层维度
                dropout=0.1,
                activation='gelu',
                batch_first=True  # 输入格式 (B, seq_len, d_model)
            ),
            num_layers=1  # 仅1层，保证轻量
        )
        
        # 3. 输出头：分别预测 theta（回归）和 omega（归一化权重）
        self.theta_head = nn.Linear(hidden_dim, 3)  # 3类退化强度
        self.omega_head = nn.Linear(hidden_dim, 3)  # 3类退化权重
    
    def forward(self, x):
        """
        x: 输入浅层特征 (B, C_in, H, W)，H/W 通常为 512/256（Restormer 输入下采样前）
        """
        # Step 1: CNN 提取局部特征 → (B, hidden_dim, H/4, W/4)
        cnn_feat = self.cnn_encoder(x)  # 经2次stride=2下采样，尺寸变为原来的1/4
        B, C, H, W = cnn_feat.shape
        
        # Step 2: 调整为 Transformer 输入格式 (B, seq_len, d_model)
        # 下采样到 seq_len×seq_len（如16×16），确保序列长度统一
        cnn_feat_down = F.adaptive_avg_pool2d(cnn_feat, (self.seq_len, self.seq_len))  # (B, C, 16, 16)
        transformer_input = cnn_feat_down.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, 256, 256)
        
        # Step 3: Transformer 编码全局特征
        transformer_feat = self.transformer_encoder(transformer_input)  # (B, 256, 256)
        global_feat = transformer_feat.mean(dim=1)  # 全局平均池化 → (B, 256)
        
        # Step 4: 预测退化参数
        theta_hat = self.theta_head(global_feat)  # (B, 3) → 强度值（无约束，训练时用MSE监督）
        omega_logits = self.omega_head(global_feat)  # (B, 3) → 未归一化权重
        omega_hat = F.softmax(omega_logits, dim=1)  # 归一化 → (B, 3)，sum=1
        
        return theta_hat, omega_hat

class DegradationAwarePerturbationModule(nn.Module):
    """
    退化感知特征扰动模块（DAPM）：
    - 输入：当前阶段特征图 (B, C, H, W) + 退化参数 (theta, omega)
    - 输出：扰动后的特征图 (B, C, H, W) + 扰动正则化项（用于损失约束）
    """
    def __init__(self, C_feat, epsilon=0.1):
        super().__init__()
        self.C_feat = C_feat  # 当前阶段特征通道数
        self.epsilon = epsilon  # 扰动强度上限（L2约束）
        
        # 扰动生成器：2层全连接，输入 theta(3) + omega(3) = 6维，输出3个基础扰动
        self.perturb_generator = nn.Sequential(
            nn.Linear(6, 256),  # 隐藏层维度256
            nn.ReLU(inplace=True),
            nn.Linear(256, C_feat * 3)  # 输出3个扰动向量（每个维度C_feat）
        )
        
        # 空间调制：1×1卷积将扰动向量映射为空间注意力图
        self.spatial_modulator = nn.Conv2d(C_feat, 1, kernel_size=1, stride=1, padding=0)
        
        # 初始化：确保初始扰动较小
        nn.init.xavier_uniform_(self.perturb_generator[0].weight)
        nn.init.xavier_uniform_(self.perturb_generator[2].weight)
        nn.init.zeros_(self.perturb_generator[0].bias)
        nn.init.zeros_(self.perturb_generator[2].bias)
    
    def forward(self, feat, theta, omega):
        """
        feat: 当前阶段特征图 (B, C_feat, H, W)
        theta: 退化强度 (B, 3)
        omega: 退化权重 (B, 3)
        """
        B, C, H, W = feat.shape
        
        # Step 1: 生成3个基础扰动向量
        # 拼接 theta 和 omega → (B, 6)
        deg_params = torch.cat([theta, omega], dim=1)  # (B, 6)
        # 生成扰动向量 → (B, C_feat×3) → 拆分3个 (B, C_feat)
        perturb_raw = self.perturb_generator(deg_params)  # (B, 3*C)
        P_fog, P_rain, P_low = torch.chunk(perturb_raw, chunks=3, dim=1)  # 各(B, C)
        
        # Step 2: 基于 omega 加权融合扰动（退化权重引导）
        P = omega[:, 0:1] * P_fog + omega[:, 1:2] * P_rain + omega[:, 2:3] * P_low  # (B, C)
        
        # Step 3: 扰动正则化（L2约束，避免过度扰动）
        perturb_reg = torch.norm(P, p=2, dim=1).mean()  # 全局平均L2范数
        perturb_reg = torch.clamp(perturb_reg, max=self.epsilon)  # 限制上限
        
        # Step 4: 通道调制（逐通道权重调整）
        P_chan = torch.sigmoid(P)  # 归一化到[0,1] → (B, C)
        P_chan = P_chan.unsqueeze(-1).unsqueeze(-1)  # 扩展为 (B, C, 1, 1)
        feat_chan = feat * P_chan  # 通道加权 → (B, C, H, W)
        
        # Step 5: 空间调制（逐像素增强/抑制）
        P_spat = P.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        P_spat = P_spat.expand(-1, -1, H, W)  # (B, C, H, W)
        spatial_attn = torch.sigmoid(self.spatial_modulator(P_spat))  # (B, 1, H, W)
        feat_spat = feat_chan + feat_chan * spatial_attn  # 残差式空间调制
        
        # Step 6: 最终扰动特征（避免梯度消失，加原始特征残差）
        feat_perturbed = feat + 0.1 * feat_spat  # 0.1为扰动强度系数（可微调）
        
        return feat_perturbed, perturb_reg
    

class DegradationAwareReconstructionHead(nn.Module):
    """
    退化感知重建头（DARH）：
    - 输入：多尺度融合特征 (B, C_fuse, H, W) + 退化权重 omega
    - 输出：重建干净图像 (B, 3, H, W)
    """
    def __init__(self, C_fuse=256, out_channels=3, kernel_size=3):
        super().__init__()
        self.C_fuse = C_fuse  # 融合特征通道数（Restormer 最后阶段通道数）
        self.out_channels = out_channels  # RGB图像为3
        self.kernel_size = kernel_size
        
        # 3个子任务基础重建核（雾去除/雨去除/低光增强）
        self.kernel_fog = nn.Parameter(torch.randn(out_channels, C_fuse, kernel_size, kernel_size))
        self.kernel_rain = nn.Parameter(torch.randn(out_channels, C_fuse, kernel_size, kernel_size))
        self.kernel_low = nn.Parameter(torch.randn(out_channels, C_fuse, kernel_size, kernel_size))
        
        # 基础偏置项
        self.bias_fog = nn.Parameter(torch.randn(out_channels))
        self.bias_rain = nn.Parameter(torch.randn(out_channels))
        self.bias_low = nn.Parameter(torch.randn(out_channels))
        
        # 1×1卷积调整融合特征通道（可选，增强适配性）
        self.adjust_conv = nn.Conv2d(C_fuse, C_fuse, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化重建核（用Xavier初始化保证数值稳定）
        nn.init.xavier_uniform_(self.kernel_fog)
        nn.init.xavier_uniform_(self.kernel_rain)
        nn.init.xavier_uniform_(self.kernel_low)
        nn.init.zeros_(self.bias_fog)
        nn.init.zeros_(self.bias_rain)
        nn.init.zeros_(self.bias_low)
    
    def forward(self, fuse_feat, omega):
        """
        fuse_feat: 多尺度融合特征 (B, C_fuse, H, W)
        omega: 退化权重 (B, 3) → [omega_fog, omega_rain, omega_low]
        """
        # Step 1: 调整融合特征（可选，提升表达能力）
        fuse_feat = self.relu(self.adjust_conv(fuse_feat))  # (B, C_fuse, H, W)
        B, C, H, W = fuse_feat.shape
        
        # Step 2: 基于 omega 动态加权重建核和偏置
        # 权重扩展为 (B, 1, 1, 1) 适配卷积核维度
        omega_fog = omega[:, 0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        omega_rain = omega[:, 1:2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega_low = omega[:, 2:3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 动态核：omega_fog*kernel_fog + ... → (B, out_channels, C_fuse, kernel_size, kernel_size)
        dynamic_kernel = (
            omega_fog * self.kernel_fog.unsqueeze(0) +
            omega_rain * self.kernel_rain.unsqueeze(0) +
            omega_low * self.kernel_low.unsqueeze(0)
        )
        
        # 动态偏置：omega_fog*bias_fog + ... → (B, out_channels)
        dynamic_bias = (
            omega[:, 0:1] * self.bias_fog.unsqueeze(0) +
            omega[:, 1:2] * self.bias_rain.unsqueeze(0) +
            omega[:, 2:3] * self.bias_low.unsqueeze(0)
        )
        
        # Step 3: 动态卷积重建（batch-wise 卷积）
        # 方法：用 torch.nn.functional.conv2d 的 groups=1，遍历每个batch
        recon_imgs = []
        for b in range(B):
            # 单个batch的特征和核 → (1, C_fuse, H, W) 和 (out_channels, C_fuse, k, k)
            feat_b = fuse_feat[b:b+1]
            kernel_b = dynamic_kernel[b]
            bias_b = dynamic_bias[b]
            # 卷积重建 → (1, 3, H, W)
            recon_b = F.conv2d(feat_b, kernel_b, bias=bias_b, padding=self.kernel_size//2)
            recon_imgs.append(recon_b)
        
        # 拼接所有batch → (B, 3, H, W)
        recon_img = torch.cat(recon_imgs, dim=0)
        # 归一化到[0,1]（适配图像输出）
        recon_img = torch.clamp(recon_img, 0.0, 1.0)
        
        return recon_img


class DARestormer(nn.Module):
    """Degradation-Aware Restormer 完整模型"""
    def __init__(self, restormer_backbone, C_in=64, C_fuse=256):
        super().__init__()
        self.restormer_backbone = restormer_backbone  # 加载预训练 Restormer 主干
        self.ldeb = LightweightDegradationEstimator(C_in=C_in)  # LDEB 模块
        self.dapm_list = nn.ModuleList([
            DegradationAwarePerturbationModule(C_feat=64),  # 阶段1特征通道64
            DegradationAwarePerturbationModule(C_feat=128), # 阶段2特征通道128
            DegradationAwarePerturbationModule(C_feat=256)  # 阶段3特征通道256
        ])
        self.darh = DegradationAwareReconstructionHead(C_fuse=C_fuse)  # 重建头
    
    def forward(self, x):
        """
        x: 输入退化图像 (B, 3, H, W) → 范围[0,1]
        """
        # Step 1: Restormer 浅层特征提取（假设主干返回各阶段特征）
        shallow_feat, stage_feats = self.restormer_backbone.extract_feats(x)  # stage_feats: 3个阶段的特征
        
        # Step 2: LDEB 估计退化参数
        theta_hat, omega_hat = self.ldeb(shallow_feat)
        
        # Step 3: 各阶段特征经 DAPM 扰动
        perturbed_feats = []
        perturb_reg_total = 0.0
        for i, (feat, dapm) in enumerate(zip(stage_feats, self.dapm_list)):
            feat_perturbed, perturb_reg = dapm(feat, theta_hat, omega_hat)
            perturbed_feats.append(feat_perturbed)
            perturb_reg_total += perturb_reg
        
        # Step 4: Restormer 主干后续处理（特征融合）
        fuse_feat = self.restormer_backbone.fuse_feats(perturbed_feats)
        
        # Step 5: DARH 重建干净图像
        pred_img = self.darh(fuse_feat, omega_hat)
        
        # 返回预测图像、退化参数、扰动正则项
        return pred_img, theta_hat, omega_hat, perturb_reg_total

# 使用示例
if __name__ == "__main__":
    # 1. 加载 Restormer 主干（需替换为实际 Restormer 实现）
    import Restormer.Restormer # 假设已实现 Restormer
    restormer_backbone = Restormer()
    
    # 2. 初始化 DA-Restormer
    model = DARestormer(restormer_backbone=restormer_backbone)
    
    # 3. 初始化损失函数
    criterion = TotalLoss(model=model)
    
    # 4. 模拟输入数据
    batch_size, H, W = 2, 256, 256
    x = torch.randn(batch_size, 3, H, W)  # 输入退化图像
    gt = torch.randn(batch_size, 3, H, W)  # 真实干净图像
    theta_gt = torch.randn(batch_size, 3)   # 真实退化强度
    omega_gt = F.softmax(torch.randn(batch_size, 3), dim=1)  # 真实退化权重
    
    # 5. 前向传播
    pred_img, theta_hat, omega_hat, perturb_reg = model(x)
    
    # 6. 计算损失
    loss_dict = criterion(pred_img, gt, theta_hat, omega_hat, theta_gt, omega_gt)
    
    # 7. 反向传播（训练时）
    loss_dict['total'].backward()
    
    # 打印损失
    print("Total Loss:", loss_dict['total'].item())
    print("GDL Loss:", loss_dict['gdl'].item())



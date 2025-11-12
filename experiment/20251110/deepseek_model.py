import torch
import torch.nn as nn
import torch.nn.functional as F

class DegradationPriorEncoder(nn.Module):
    """
    退化先验编码器
    输入: 退化图像 + 手工先验图
    输出: 紧凑的退化表征向量d
    """
    
    def __init__(self, in_channels=6, feature_dim=128, hidden_dim=256, output_dim=512):
        """
        Args:
            in_channels: 输入通道数 (RGB + 手工先验图)
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出退化表征维度
        """
        super().__init__()
        
        # 浅层CNN编码器
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二层
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第三层
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )
        
        # 输出映射层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def compute_handcrafted_priors(self, x):
        """
        计算手工先验图
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            priors: 手工先验图 [B, 3, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 暗通道先验 (对雾敏感)
        dark_channel, _ = torch.min(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 2. 亮度图 (对低光敏感)
        brightness = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 3. 局部对比度图 (对雨纹和细节丢失敏感)
        # 使用拉普拉斯算子计算局部对比度
        laplacian_kernel = torch.tensor([[0, 1, 0], 
                                       [1, -4, 1], 
                                       [0, 1, 0]], dtype=torch.float32, device=x.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        contrast = F.conv2d(brightness, laplacian_kernel, padding=1)
        contrast = torch.abs(contrast)
        
        # 归一化到[0, 1]
        dark_channel = (dark_channel - dark_channel.min()) / (dark_channel.max() - dark_channel.min() + 1e-8)
        brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-8)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
        
        priors = torch.cat([dark_channel, brightness, contrast], dim=1)
        return priors
    
    def forward(self, x):
        """
        Args:
            x: 输入退化图像 [B, 3, H, W]
        Returns:
            d: 退化表征向量 [B, output_dim]
        """
        # 计算手工先验图
        handcrafted_priors = self.compute_handcrafted_priors(x)  # [B, 3, H, W]
        
        # 拼接原始图像和手工先验图
        x_combined = torch.cat([x, handcrafted_priors], dim=1)  # [B, 6, H, W]
        
        # 通过编码器
        features = self.encoder(x_combined)  # [B, hidden_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, hidden_dim]
        
        # 输出映射
        d = self.output_proj(features)  # [B, output_dim]
        
        return d

class ParameterGenerationNetwork(nn.Module):
    """
    参数生成网络
    输入: 退化表征向量d
    输出: 所有Transformer块所需的调制参数 {γ_i, β_i, α_i}
    """
    
    def __init__(self, 
                 degradation_dim=512, 
                 hidden_dim=1024, 
                 num_blocks=8, 
                 num_groups=4, 
                 feature_dim=64):
        """
        Args:
            degradation_dim: 退化表征维度
            hidden_dim: 隐藏层维度
            num_blocks: Transformer块总数
            num_groups: 参数分组数 (编码器/瓶颈/解码器等)
            feature_dim: 特征通道数 (用于γ, β的维度)
        """
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_groups = num_groups
        self.feature_dim = feature_dim
        
        # 主MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(degradation_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_groups * feature_dim * 3)  # γ, β, α 各 feature_dim 维
        )
        
        # 参数分组映射: 将num_groups的参数分配到num_blocks个块
        self.group_assignment = self._create_group_assignment(num_blocks, num_groups)
        
    def _create_group_assignment(self, num_blocks, num_groups):
        """
        创建块到参数组的映射
        例如: [0, 0, 1, 1, 2, 2, 3, 3] 表示前2个块用第0组参数，接着2个用第1组等
        """
        assert num_blocks % num_groups == 0, "num_blocks must be divisible by num_groups"
        blocks_per_group = num_blocks // num_groups
        assignment = []
        for group_idx in range(num_groups):
            assignment.extend([group_idx] * blocks_per_group)
        return assignment
    
    def forward(self, d):
        """
        Args:
            d: 退化表征向量 [B, degradation_dim]
        Returns:
            params_dict: 包含所有块参数的字典
                {
                    'gammas': List[Tensor], 每个元素 [B, feature_dim, 1, 1]
                    'betas': List[Tensor], 每个元素 [B, feature_dim, 1, 1]  
                    'alphas': List[Tensor], 每个元素 [B, feature_dim, 1, 1]
                }
        """
        B = d.shape[0]
        
        # 通过MLP生成原始参数
        raw_params = self.mlp(d)  # [B, num_groups * feature_dim * 3]
        
        # 重塑参数
        raw_params = raw_params.view(B, self.num_groups, self.feature_dim * 3)  # [B, num_groups, feature_dim*3]
        
        # 分割γ, β, α
        gammas_group = raw_params[:, :, :self.feature_dim]  # [B, num_groups, feature_dim]
        betas_group = raw_params[:, :, self.feature_dim:2*self.feature_dim]  # [B, num_groups, feature_dim]
        alphas_group = raw_params[:, :, 2*self.feature_dim:]  # [B, num_groups, feature_dim]
        
        # 为每个块分配参数
        gammas, betas, alphas = [], [], []
        for block_idx in range(self.num_blocks):
            group_idx = self.group_assignment[block_idx]
            
            gamma = gammas_group[:, group_idx, :].unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
            beta = betas_group[:, group_idx, :].unsqueeze(-1).unsqueeze(-1)    # [B, feature_dim, 1, 1]
            alpha = alphas_group[:, group_idx, :].unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
            
            # 对α应用sigmoid确保在[0,1]范围内
            alpha = torch.sigmoid(alpha)
            
            gammas.append(gamma)
            betas.append(beta)
            alphas.append(alpha)
        
        return {
            'gammas': gammas,
            'betas': betas, 
            'alphas': alphas
        }

class DAPerturbationModule(nn.Module):
    """
    退化感知特征扰动模块
    集成到Restormer的Transformer块中
    """
    
    def __init__(self, feature_dim):
        """
        Args:
            feature_dim: 特征通道数
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 层归一化 (与特征维度解耦)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, gamma, beta, alpha):
        """
        Args:
            x: 输入特征 [B, H*W, C] 或 [B, C, H, W] (取决于Restormer的实现)
            gamma: 缩放参数 [B, C, 1, 1] 或 [B, C]
            beta: 偏置参数 [B, C, 1, 1] 或 [B, C]  
            alpha: 门控参数 [B, C, 1, 1] 或 [B, C]
        Returns:
            x_perturbed: 扰动后特征
        """
        B, C, H, W = x.shape
        
        # 确保参数形状匹配
        if gamma.dim() == 4:
            gamma = gamma.squeeze(-1).squeeze(-1)  # [B, C]
        if beta.dim() == 4:
            beta = beta.squeeze(-1).squeeze(-1)    # [B, C]  
        if alpha.dim() == 4:
            alpha = alpha.squeeze(-1).squeeze(-1)  # [B, C]
        
        # 重塑特征为 [B, C, H*W] 用于LayerNorm
        x_reshaped = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 应用层归一化
        x_norm = self.norm(x_reshaped)  # [B, H*W, C]
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # 恢复形状 [B, C, H, W]
        
        # 扩展参数到特征图大小
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 应用仿射变换
        x_transformed = gamma * x_norm + beta
        
        # 渐进式扰动: 结合原始特征和变换特征
        x_perturbed = alpha * x_transformed + (1 - alpha) * x
        
        return x_perturbed


# Restormer Transformer块集成示例
class DARestormerBlock(nn.Module):
    """
    集成了退化感知特征扰动的Restormer块
    """
    
    def __init__(self, dim, num_heads, ffw_expansion=2.66):
        super().__init__()
        
        # Restormer原有组件
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiDconvHeadTransposedAttention(dim, num_heads)  # 假设这是Restormer的注意力
        
        self.norm2 = nn.LayerNorm(dim) 
        self.ffn = FeedForward(dim, ffw_expansion)  # Restormer的前馈网络
        
        # 新增: 特征扰动模块 (在FFN之后应用)
        self.perturbation = DAPerturbationModule(dim)
        
    def forward(self, x, gamma, beta, alpha):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            gamma, beta, alpha: 调制参数
        """
        # Restormer原有前向传播
        # 第一个注意力部分
        x = x + self.attn(self.norm1(x))
        
        # 前馈网络部分  
        x = x + self.ffn(self.norm2(x))
        
        # 新增: 在FFN之后应用退化感知特征扰动
        x = self.perturbation(x, gamma, beta, alpha)
        
        return x

class AdaptiveGatingPerturbation(nn.Module):
    """
    自适应门控扰动策略
    增强版的扰动模块，包含更精细的门控机制
    """
    
    def __init__(self, feature_dim, use_channel_wise_gating=True, use_spatial_wise_gating=False):
        """
        Args:
            feature_dim: 特征通道数
            use_channel_wise_gating: 是否使用通道级门控
            use_spatial_wise_gating: 是否使用空间级门控 (计算量较大)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_channel_wise_gating = use_channel_wise_gating
        self.use_spatial_wise_gating = use_spatial_wise_gating
        
        # 层归一化
        self.norm = nn.LayerNorm(feature_dim)
        
        # 通道级门控网络 (可选)
        if use_channel_wise_gating:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feature_dim, feature_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 4, feature_dim, 1),
                nn.Sigmoid()
            )
        
        # 空间级门控网络 (可选，计算量较大)
        if use_spatial_wise_gating:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 4, 1, 3, padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, x, gamma, beta, alpha):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            gamma, beta, alpha: 基础调制参数
        Returns:
            x_perturbed: 扰动后特征
        """
        B, C, H, W = x.shape
        
        # 基础扰动计算 (与之前相同)
        x_reshaped = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_reshaped)  # [B, H*W, C]
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        # 扩展参数
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        alpha_base = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 基础变换
        x_transformed = gamma * x_norm + beta
        
        # 自适应门控
        if self.use_channel_wise_gating and self.use_spatial_wise_gating:
            # 通道级门控
            channel_weights = self.channel_gate(x)  # [B, C, 1, 1]
            # 空间级门控  
            spatial_weights = self.spatial_gate(x)  # [B, 1, H, W]
            # 结合通道和空间门控
            adaptive_alpha = alpha_base * channel_weights * spatial_weights
            
        elif self.use_channel_wise_gating:
            # 仅通道级门控
            channel_weights = self.channel_gate(x)  # [B, C, 1, 1]
            adaptive_alpha = alpha_base * channel_weights
            
        elif self.use_spatial_wise_gating:
            # 仅空间级门控
            spatial_weights = self.spatial_gate(x)  # [B, 1, H, W]
            adaptive_alpha = alpha_base * spatial_weights
            
        else:
            # 不使用自适应门控
            adaptive_alpha = alpha_base
        
        # 渐进式扰动
        x_perturbed = adaptive_alpha * x_transformed + (1 - adaptive_alpha) * x
        
        return x_perturbed


# 完整的DA-Restormer集成示例
class DARestormer(nn.Module):
    """
    完整的退化感知Restormer模型
    """
    
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 num_blocks=8,
                 feature_dim=64,
                 degradation_dim=512):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        # 退化先验编码器
        self.degradation_encoder = DegradationPriorEncoder(
            in_channels=3, output_dim=degradation_dim
        )
        
        # 参数生成网络
        self.param_generator = ParameterGenerationNetwork(
            degradation_dim=degradation_dim,
            num_blocks=num_blocks,
            feature_dim=feature_dim
        )
        
        # Restormer骨干网络 (这里简化表示)
        self.restormer_blocks = nn.ModuleList([
            DARestormerBlock(feature_dim, num_heads=8) for _ in range(num_blocks)
        ])
        
        # 输入输出投影层
        self.input_proj = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        self.output_proj = nn.Conv2d(feature_dim, out_channels, 3, padding=1)
    
    def forward(self, x):
        # 提取退化表征
        degradation_repr = self.degradation_encoder(x)  # [B, degradation_dim]
        
        # 生成调制参数
        params = self.param_generator(degradation_repr)
        gammas = params['gammas']
        betas = params['betas'] 
        alphas = params['alphas']
        
        # 输入投影
        x = self.input_proj(x)
        
        # 通过Restormer块
        for i, block in enumerate(self.restormer_blocks):
            x = block(x, gammas[i], betas[i], alphas[i])
        
        # 输出投影
        x = self.output_proj(x)
        
        return x   



if __name__ == '__main__':
    # 模型初始化
    model = DARestormer(
        in_channels=3,
        out_channels=3, 
        num_blocks=8,
        feature_dim=64,
        degradation_dim=512
    )

    # 前向传播示例
    input_image = torch.randn(4, 3, 256, 256)  # [B, C, H, W]
    output_image = model(input_image)

    print(f"输入形状: {input_image.shape}")
    print(f"输出形状: {output_image.shape}")



class ProgressiveDAPerturbation(nn.Module):
    """
    集成渐进式扰动策略的退化感知特征扰动模块
    将基础扰动、通道门控、空间门控和渐进式融合集成在一个模块中
    """
    
    def __init__(self, feature_dim, use_channel_gating=True, use_spatial_gating=True):
        """
        Args:
            feature_dim: 特征通道数
            use_channel_gating: 是否使用通道级自适应门控
            use_spatial_gating: 是否使用空间级自适应门控
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_channel_gating = use_channel_gating
        self.use_spatial_gating = use_spatial_gating
        
        # 层归一化 (与特征统计解耦)
        self.norm = nn.LayerNorm(feature_dim)
        
        # === 渐进式扰动策略的核心组件 ===
        
        # 1. 通道级自适应门控网络
        if use_channel_gating:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化 [B, C, 1, 1]
                nn.Conv2d(feature_dim, max(4, feature_dim // 8), 1),  # 压缩
                nn.ReLU(inplace=True),
                nn.Conv2d(max(4, feature_dim // 8), feature_dim, 1),  # 恢复
                nn.Sigmoid()  # 输出 [0, 1]
            )
        
        # 2. 空间级自适应门控网络 (轻量级设计)
        if use_spatial_gating:
            self.spatial_gate = nn.Sequential(
                # 使用深度可分离卷积减少计算量
                nn.Conv2d(feature_dim, max(4, feature_dim // 8), 3, padding=1, groups=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(4, feature_dim // 8), 1, 3, padding=1),  # 输出单通道空间权重
                nn.Sigmoid()  # 输出 [0, 1]
            )
        
        # 3. 残差连接权重学习 (可选)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))  # 可学习的残差权重初始值
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        if self.use_channel_gating:
            # 通道门控最后一层初始化为接近0.5，让网络平稳启动
            nn.init.constant_(self.channel_gate[-2].weight, 0)
            nn.init.constant_(self.channel_gate[-2].bias, 0)
        
        if self.use_spatial_gating:
            # 空间门控最后一层初始化为接近0.5
            nn.init.constant_(self.spatial_gate[-2].weight, 0)
            nn.init.constant_(self.spatial_gate[-2].bias, 0)
    
    def compute_adaptive_alpha(self, x, base_alpha):
        """
        计算自适应门控系数
        Args:
            x: 输入特征 [B, C, H, W]
            base_alpha: 基础门控参数 [B, C, 1, 1]
        Returns:
            adaptive_alpha: 自适应门控系数 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 基础alpha扩展
        alpha = base_alpha.expand(B, C, H, W)  # [B, C, H, W]
        
        # 通道级自适应
        if self.use_channel_gating:
            channel_weights = self.channel_gate(x)  # [B, C, 1, 1]
            alpha = alpha * channel_weights.expand(B, C, H, W)
        
        # 空间级自适应
        if self.use_spatial_gating:
            spatial_weights = self.spatial_gate(x)  # [B, 1, H, W]
            alpha = alpha * spatial_weights.expand(B, C, H, W)
        
        return alpha
    
    def progressive_fusion(self, original, transformed, alpha):
        """
        渐进式特征融合
        Args:
            original: 原始特征 [B, C, H, W]
            transformed: 变换后特征 [B, C, H, W]
            alpha: 自适应门控系数 [B, C, H, W]
        Returns:
            fused: 融合后特征
        """
        # 方法1: 简单的线性插值
        # fused = alpha * transformed + (1 - alpha) * original
        
        # 方法2: 带可学习权重的残差连接 (更稳定)
        residual_weight = torch.sigmoid(self.residual_weight)  # 约束在[0,1]
        fused = alpha * transformed + residual_weight * (1 - alpha) * original
        
        return fused
    
    def forward(self, x, gamma, beta, alpha_base):
        """
        前向传播: 集成所有渐进式扰动策略
        Args:
            x: 输入特征 [B, C, H, W]
            gamma: 缩放参数 [B, C, 1, 1]
            beta: 偏置参数 [B, C, 1, 1]
            alpha_base: 基础门控参数 [B, C, 1, 1]
        Returns:
            x_perturbed: 扰动后特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # === 步骤1: 特征归一化 ===
        # 重塑为LayerNorm需要的形状 [B, H*W, C]
        x_reshaped = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_reshaped)  # [B, H*W, C]
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # 恢复形状 [B, C, H, W]
        
        # === 步骤2: 基础仿射变换 ===
        # 确保参数形状正确
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        if beta.dim() == 2:
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        if alpha_base.dim() == 2:
            alpha_base = alpha_base.unsqueeze(-1).unsqueeze(-1)
        
        # 应用仿射变换
        x_transformed = gamma * x_norm + beta  # [B, C, H, W]
        
        # === 步骤3: 计算自适应门控系数 ===
        adaptive_alpha = self.compute_adaptive_alpha(x, alpha_base)  # [B, C, H, W]
        
        # === 步骤4: 渐进式特征融合 ===
        x_perturbed = self.progressive_fusion(x, x_transformed, adaptive_alpha)
        
        return x_perturbed


# 集成到Restormer块的完整示例
class DARestormerBlockWithProgressivePerturbation(nn.Module):
    """
    集成渐进式扰动策略的Restormer块
    """
    
    def __init__(self, dim, num_heads, ffw_expansion=2.66, 
                 use_channel_gating=True, use_spatial_gating=True):
        super().__init__()
        
        # Restormer原有组件
        self.norm1 = nn.LayerNorm(dim)
        # 注意: 这里使用简化的注意力实现，实际应使用Restormer的MDTA
        self.attn = self._create_attention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = self._create_ffn(dim, ffw_expansion)
        
        # 集成渐进式扰动模块 (关键位置: 在FFN之后)
        self.progressive_perturbation = ProgressiveDAPerturbation(
            feature_dim=dim,
            use_channel_gating=use_channel_gating,
            use_spatial_gating=use_spatial_gating
        )
        
    def _create_attention(self, dim, num_heads):
        """创建注意力机制 (简化版)"""
        return nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def _create_ffn(self, dim, expansion):
        """创建前馈网络 (简化版)"""
        hidden_dim = int(dim * expansion)
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, gamma, beta, alpha_base):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            gamma, beta, alpha_base: 从参数生成网络得到的调制参数
        """
        B, C, H, W = x.shape
        
        # === Restormer原有流程 ===
        
        # 1. 注意力部分
        # 重塑为序列形式 [B, H*W, C]
        x_reshaped = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 层归一化 + 注意力 + 残差
        x_attn, _ = self.attn(self.norm1(x_reshaped), self.norm1(x_reshaped), self.norm1(x_reshaped))
        x_reshaped = x_reshaped + x_attn
        
        # 恢复空间格式 [B, C, H, W]
        x = x_reshaped.transpose(1, 2).view(B, C, H, W)
        
        # 2. 前馈网络部分
        x_reshaped = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x_ffn = self.ffn(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + x_ffn
        x = x_reshaped.transpose(1, 2).view(B, C, H, W)
        
        # === 关键集成: 在FFN之后应用渐进式扰动 ===
        x = self.progressive_perturbation(x, gamma, beta, alpha_base)
        
        return x


# 完整的训练配置示例
class ProgressivePerturbationConfig:
    """渐进式扰动策略的配置类"""
    
    def __init__(self):
        # 基础配置
        self.use_channel_gating = True
        self.use_spatial_gating = True
        
        # 训练策略
        self.warmup_epochs = 10  # 预热阶段，让网络先学习基础特征
        
        # 损失权重
        self.perturbation_loss_weight = 0.1  # 扰动相关损失的权重
    
    def get_perturbation_strength(self, current_epoch, total_epochs):
        """
        根据训练进度调整扰动强度
        早期弱扰动，后期强扰动
        """
        if current_epoch < self.warmup_epochs:
            # 预热阶段，弱扰动
            return 0.1
        else:
            # 线性增加扰动强度
            progress = (current_epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            return 0.1 + 0.9 * progress


# 使用示例和测试
def test_progressive_perturbation():
    """测试渐进式扰动模块"""
    
    # 初始化模块
    feature_dim = 64
    perturbation_module = ProgressiveDAPerturbation(
        feature_dim=feature_dim,
        use_channel_gating=True,
        use_spatial_gating=True
    )
    
    # 测试输入
    B, C, H, W = 2, feature_dim, 32, 32
    x = torch.randn(B, C, H, W)
    
    # 模拟参数生成网络的输出
    gamma = torch.ones(B, C, 1, 1) * 0.8  # 轻微缩放
    beta = torch.zeros(B, C, 1, 1)        # 无偏置
    alpha_base = torch.ones(B, C, 1, 1) * 0.3  # 中等扰动强度
    
    # 前向传播
    output = perturbation_module(x, gamma, beta, alpha_base)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入输出差异: {torch.mean(torch.abs(output - x)):.4f}")
    
    # 验证输出范围
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    return output

if __name__ == "__main__":
    # 运行测试
    test_progressive_perturbation()

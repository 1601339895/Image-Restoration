import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWeightingReorder(nn.Module):
    """
    通道权重学习与重组模块：
    1. 学习每个通道的重要性权重
    2. 对通道进行加权
    3. 根据权重对通道重排序（可选，默认启用）
    """
    def __init__(self, in_channels, reduction_ratio=16, reorder=True):
        """
        Args:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道注意力中全连接层的缩减率（控制参数量）
            reorder: 是否根据权重对通道重排序（True/False）
        """
        super(ChannelWeightingReorder, self).__init__()
        self.in_channels = in_channels
        self.reorder = reorder
        
        # 1. 通道权重学习模块（类似SE模块的挤压-激励结构）
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化：(b, c, h, w) -> (b, c, 1, 1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),  # 升维
            nn.Sigmoid()  # 输出权重（0~1之间）
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状为 (batch_size, in_channels, height, width)
        Returns:
            out: 加权并（可选）重组后的特征图，形状为 (batch_size, in_channels, height, width)
            weights: 学习到的通道权重，形状为 (batch_size, in_channels)
        """
        batch_size, c, h, w = x.shape
        
        # --------------------------
        # 步骤1：学习通道权重
        # --------------------------
        # 挤压：全局平均池化 + 展平
        squeeze = self.squeeze(x).view(batch_size, c)  # (b, c, 1, 1) -> (b, c)
        # 激励：学习权重
        weights = self.excitation(squeeze)  # (b, c)，每个元素是对应通道的权重（0~1）
        
        # --------------------------
        # 步骤2：通道加权
        # --------------------------
        # 权重扩展为 (b, c, 1, 1)，与输入特征图逐通道相乘
        weighted_x = x * weights.view(batch_size, c, 1, 1)  # (b, c, h, w)
        
        # --------------------------
        # 步骤3：通道重组（根据权重排序）
        # --------------------------
        if self.reorder:
            # 对每个样本的通道权重排序，获取降序索引（权重高的通道排在前面）
            # argsort返回的是升序索引，[..., ::-1]反转得到降序
            sorted_indices = torch.argsort(weights, dim=1, descending=True)  # (b, c)
            
            # 根据索引重组通道
            # 为了适配gather，需要将索引扩展为 (b, c, 1, 1)，并与特征图维度对齐
            sorted_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1)  # (b, c, 1, 1)
            # gather：按通道维度（dim=1）根据索引重组
            out = torch.gather(weighted_x, dim=1, index=sorted_indices.expand(-1, -1, h, w))
        else:
            out = weighted_x  # 不重组，仅返回加权结果
        
        return out, weights


# --------------------------
# 测试代码
# --------------------------
if __name__ == "__main__":
    # 模拟输入：batch_size=2，通道数=8，高=32，宽=32
    x = torch.randn(2, 8, 32, 32)
    print("输入形状：", x.shape)  # torch.Size([2, 8, 32, 32])
    
    # 实例化模块（启用通道重组）
    model = ChannelWeightingReorder(in_channels=8, reduction_ratio=4, reorder=True)
    
    # 前向传播
    out, weights = model(x)
    
    # 输出验证
    print("输出形状：", out.shape)  # 应与输入相同：torch.Size([2, 8, 32, 32])
    print("权重形状：", weights.shape)  # torch.Size([2, 8])
    
    # 查看第一个样本的通道权重及排序后索引（验证重组逻辑）
    sample_idx = 0
    print(f"\n样本{sample_idx}的通道权重：", weights[sample_idx].detach().numpy())
    sorted_indices = torch.argsort(weights[sample_idx], descending=True)
    print(f"样本{sample_idx}的通道排序索引（降序）：", sorted_indices.numpy())
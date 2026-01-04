import math
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import pytorch_msssim

# 定义支持的损失聚合模式
_reduction_modes = ['none', 'mean', 'sum']

# -----------------------------------------------------------------------------
# PSNR 损失（含权重支持、自定义数据范围）
# -----------------------------------------------------------------------------
def psnr_loss(pred, target, weight=None, data_range=1.0, reduction='mean'):
    """
    计算PSNR值（转为损失形式时返回-PSNR）
    Args:
        pred: 预测张量 (N,C,H,W)
        target: 目标张量 (N,C,H,W)
        weight: 元素权重 (N,C,H,W)，可选
        data_range: 像素值范围（如[0,255]则设为255.0）
        reduction: 损失聚合方式 ('none'/'mean'/'sum')
    Returns:
        psnr_val: PSNR值（张量）
    """
    # 计算逐元素MSE
    mse = F.mse_loss(pred, target, reduction='none')
    # 应用元素权重（若有）
    if weight is not None:
        mse = mse * weight
    # 每个样本的MSE（沿通道、高度、宽度维度平均）
    mse = mse.mean((1, 2, 3))  # 形状：(N,)
    
    # 防止MSE为0导致log10出错
    mse = torch.clamp(mse, min=1e-10)
    # PSNR公式：10 * log10((data_range^2) / MSE)
    psnr_val = 10 * torch.log10((data_range ** 2) / mse)
    
    # 聚合结果
    if reduction == 'mean':
        return psnr_val.mean()
    elif reduction == 'sum':
        return psnr_val.sum()
    elif reduction == 'none':
        return psnr_val
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Supported: {_reduction_modes}")

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range  # 像素值范围
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W) 预测张量
            target (Tensor): (N, C, H, W) 目标张量
            weight (Tensor, optional): (N, C, H, W) 元素权重
        Returns:
            loss: 加权后的负PSNR损失
        """
        psnr_val = psnr_loss(pred, target, weight, self.data_range, self.reduction)
        return self.loss_weight * (-psnr_val)  # 负PSNR作为损失（越小越好）

# -----------------------------------------------------------------------------
# SSIM 相关损失
# -----------------------------------------------------------------------------
def SSIM_loss(pred_img, real_img, data_range):
    """计算SSIM值（pytorch_msssim实现）"""
    return pytorch_msssim.ssim(pred_img, real_img, data_range=data_range)

class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)

class SSIMloss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.):
        super(SSIMloss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        # 1 - SSIM 作为损失（越小越好）
        return self.loss_weight * (1 - SSIM_loss(pred, target, self.data_range))

# -----------------------------------------------------------------------------
# GAN 损失（修复设备不匹配问题）
# -----------------------------------------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        # 选择损失函数
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        """根据输入张量的设备和形状生成目标标签"""
        target_tensor = None
        device = input.device
        dtype = input.dtype
        if target_is_real:
            # 检查是否需要重新创建标签
            create_label = ((self.real_label_var is None) or 
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.full(input.size(), self.real_label, device=device, dtype=dtype)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or 
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.full(input.size(), self.fake_label, device=device, dtype=dtype)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# -----------------------------------------------------------------------------
# Focal L1 损失（添加alpha除零保护）
# -----------------------------------------------------------------------------
class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, epsilon=1e-6, alpha=0.1):
        """
        Focal L1 Loss with adjusted weighting for output values in [0, 1].
        Args:
            gamma (float): Focusing parameter. Larger gamma focuses more on hard examples.
            epsilon (float): Small constant to prevent weights from being zero.
            alpha (float): Scaling factor to normalize error values.
        """
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha  # 缩放因子，防止误差值过小

    def forward(self, pred, target):
        """
        Compute the Focal L1 Loss between the predicted and target images.
        Args:
            pred (torch.Tensor): Predicted image [b, c, h, w].
            target (torch.Tensor): Ground truth image [b, c, h, w].
        Returns:
            torch.Tensor: Scalar Focal L1 Loss.
        """
        # 计算绝对误差，添加alpha除零保护
        abs_err = torch.abs(pred - target) / max(self.alpha, 1e-6)
        # 计算焦点权重（对数变换防止权重为零）
        focal_weight = (torch.log(1 + abs_err + self.epsilon)) ** self.gamma
        # 加权损失
        focal_l1_loss = focal_weight * abs_err
        # 返回平均损失
        return focal_l1_loss.mean()

# -----------------------------------------------------------------------------
# FFT 损失（显式指定rfft2维度）
# -----------------------------------------------------------------------------
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        # 计算二维实部FFT（显式指定维度，提升可读性）
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        # 分离实部和虚部，堆叠为张量
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)

# -----------------------------------------------------------------------------
# Edge 损失（修复通道数硬编码问题）
# -----------------------------------------------------------------------------
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l2', reduction='mean'):
        super(EdgeLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # 选择损失函数
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        # 初始化高斯核（先设为1通道，forward时动态匹配输入通道）
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0)  # 形状：(1, 5, 5)
        self.weight = loss_weight

    def conv_gauss(self, img):
        """高斯卷积"""
        n_channels = img.size(1)  # 获取输入通道数
        # 动态匹配通道数
        kernel = self.kernel.repeat(n_channels, 1, 1, 1).to(img.device)
        # 填充（复制模式）
        pad = kernel.size(2) // 2
        img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        """拉普拉斯核计算边缘"""
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, pred, target):
        loss = self.criterion(self.laplacian_kernel(pred), self.laplacian_kernel(target))
        return loss * self.weight

# -----------------------------------------------------------------------------
# VGG19 感知损失（修复.cuda()硬编码，添加eval模式）
# -----------------------------------------------------------------------------
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        # 加载预训练VGG19权重
        vgg_pretrained_features = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        # 分割VGG19的特征层
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # 固定参数（若不需要梯度）
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l1', reduction='mean'):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19()  # 移除.cuda()硬编码
        self.vgg.eval()  # 设置为评估模式（重要，防止BatchNorm等层变化）
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):
        # 动态匹配设备
        self.vgg = self.vgg.to(x.device)
        # 前向传播获取VGG特征
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # 目标特征 detach（不计算梯度）
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss

# -----------------------------------------------------------------------------
# 温度调度器（修正公式和注释）
# -----------------------------------------------------------------------------
class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        """
        Scheduler for Gumbel-Softmax temperature that decreases using a cosine annealing schedule.
        Args:
            start_temp (float): Initial temperature (e.g., 5.0).
            end_temp (float): Final temperature (e.g., 0.01).
            total_steps (int): Total number of steps/epochs to anneal over.
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):
        """
        Get the temperature value for the current step, following a cosine annealing schedule.
        Args:
            step (int): Current step or epoch.
        Returns:
            temperature (float): The temperature for the Gumbel-Softmax at this step.
        """
        if step >= self.total_steps:
            return self.end_temp

        # 余弦退火公式：从start_temp平滑下降到end_temp
        cos_inner = math.pi * step / self.total_steps
        temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + math.cos(cos_inner))

        return temp
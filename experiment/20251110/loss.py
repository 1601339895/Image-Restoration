
class PerceptualLoss(nn.Module):
    """感知损失：基于VGG19的relu5_4层特征"""
    def __init__(self):
        super().__init__()
        # 加载预训练VGG19，冻结参数
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg_feat = nn.Sequential(*list(vgg19.children())[:35])  # relu5_4层（第35层）
        for param in self.vgg_feat.parameters():
            param.requires_grad = False
        
        # 图像归一化（适配VGG输入）
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, pred, gt):
        """
        pred: 预测图像 (B, 3, H, W) → 范围[0,1]
        gt: 真实图像 (B, 3, H, W) → 范围[0,1]
        """
        # 归一化
        pred_norm = self.normalize(pred)
        gt_norm = self.normalize(gt)
        
        # 提取特征
        pred_feat = self.vgg_feat(pred_norm)
        gt_feat = self.vgg_feat(gt_norm)
        
        # L2损失
        return F.mse_loss(pred_feat, gt_feat)

class FrequencyLoss(nn.Module):
    """频域损失：基于FFT的实部+虚部误差，优化图像结构"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        """
        pred/gt: (B, 3, H, W) → 范围[0,1]
        """
        # 转换到频域（FFT）
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))  # (B, 3, H, W) 复数张量
        gt_fft = torch.fft.fft2(gt, dim=(-2, -1))
        
        # 分离实部和虚部，计算L2损失
        pred_real, pred_imag = pred_fft.real, pred_fft.imag
        gt_real, gt_imag = gt_fft.real, gt_fft.imag
        
        loss_real = F.mse_loss(pred_real, gt_real)
        loss_imag = F.mse_loss(pred_imag, gt_imag)
        
        return (loss_real + loss_imag) / 2  # 平均实虚部损失

class GradientDivergenceLoss(nn.Module):
    """
    梯度分歧损失（GDL）：直接优化不同子任务对共享参数的梯度一致性
    输入：模型共享参数 + 3个子任务损失 + 模型（用于获取参数）
    输出：梯度分歧损失值
    """
    def __init__(self, model):
        super().__init__()
        self.model = model  # DA-Restormer 模型实例
        # 定义共享参数（Restormer 主干+三个创新模块的参数，排除退化估计分支的专属参数）
        self.shared_params = [p for name, p in model.named_parameters() if 'ldeb' not in name]
    
    def _compute_gradient(self, loss, params):
        """计算单个损失对参数的梯度，返回flatten后的梯度向量"""
        grads = grad(loss, params, retain_graph=True, create_graph=True)  # 保留计算图用于反向传播
        grad_flat = torch.cat([g.flatten() for g in grads])  # 拼接所有参数的梯度 → (D,)
        return grad_flat
    
    def forward(self, pred, gt, theta_gt, omega_gt):
        """
        pred: 预测图像 (B, 3, H, W)
        gt: 真实图像 (B, 3, H, W)
        theta_gt: 真实退化强度 (B, 3)
        omega_gt: 真实退化权重 (B, 3)
        """
        # Step 1: 计算3个子任务损失（基于退化权重 omega_gt 区分主导任务）
        # 雾去除损失：omega_gt[:,0] 为雾的权重，加权L1损失
        loss_fog = (omega_gt[:, 0:1] * F.l1_loss(pred, gt, reduction='none')).mean()
        # 雨去除损失
        loss_rain = (omega_gt[:, 1:2] * F.l1_loss(pred, gt, reduction='none')).mean()
        # 低光增强损失
        loss_low = (omega_gt[:, 2:3] * F.l1_loss(pred, gt, reduction='none')).mean()
        
        # Step 2: 计算每个子任务损失对共享参数的梯度
        g_fog = self._compute_gradient(loss_fog, self.shared_params)  # (D,)
        g_rain = self._compute_gradient(loss_rain, self.shared_params)  # (D,)
        g_low = self._compute_gradient(loss_low, self.shared_params)  # (D,)
        
        # Step 3: 计算梯度向量间的余弦相似度
        def cos_sim(a, b):
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)[0]
        
        cos_fog_rain = cos_sim(g_fog, g_rain)
        cos_fog_low = cos_sim(g_fog, g_low)
        cos_rain_low = cos_sim(g_rain, g_low)
        
        # Step 4: 梯度分歧损失（1 - 平均余弦相似度，最小化即梯度对齐）
        gdl_loss = 1 - (cos_fog_rain + cos_fog_low + cos_rain_low) / 3
        
        return gdl_loss, loss_fog, loss_rain, loss_low

class TotalLoss(nn.Module):
    """总损失函数：整合所有基础损失和核心损失"""
    def __init__(self, model):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.percep_loss = PerceptualLoss()
        self.freq_loss = FrequencyLoss()
        self.gdl_loss = GradientDivergenceLoss(model)
        
        # 损失权重（可通过交叉验证调整）
        self.w_percep = 0.1
        self.w_freq = 0.05
        self.w_est = 0.2
        self.w_gdl = 0.3
    
    def forward(self, pred, gt, theta_hat, omega_hat, theta_gt, omega_gt):
        """
        pred: 预测图像 (B, 3, H, W)
        gt: 真实图像 (B, 3, H, W)
        theta_hat/omega_hat: LDEB 预测的退化参数
        theta_gt/omega_gt: 真实退化参数
        """
        # 1. 基础损失
        loss_l1 = self.l1_loss(pred, gt)
        loss_percep = self.percep_loss(pred, gt)
        loss_freq = self.freq_loss(pred, gt)
        
        # 2. 退化估计损失（MSE监督）
        loss_theta = F.mse_loss(theta_hat, theta_gt)
        loss_omega = F.mse_loss(omega_hat, omega_gt)
        loss_est = loss_theta + loss_omega
        
        # 3. 梯度分歧损失
        loss_gdl, loss_fog, loss_rain, loss_low = self.gdl_loss(pred, gt, theta_gt, omega_gt)
        
        # 4. 总损失
        total_loss = (
            loss_l1 +
            self.w_percep * loss_percep +
            self.w_freq * loss_freq +
            self.w_est * loss_est +
            self.w_gdl * loss_gdl
        )
        
        # 返回所有损失（用于日志记录）
        loss_dict = {
            'total': total_loss,
            'l1': loss_l1,
            'percep': loss_percep,
            'freq': loss_freq,
            'est': loss_est,
            'gdl': loss_gdl,
            'fog': loss_fog,
            'rain': loss_rain,
            'low': loss_low
        }
        
        return loss_dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from test2 import Restormer, DDFE
import os

class DDFEVisualizer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = Restormer().to(device)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # 注册hook来捕获DDFE的频域特征
        self.freq_features = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                x, global_feat = input
                B, C, H, W = x.shape

                # 执行FFT
                x_32 = x.to(torch.float32)
                x_fft = torch.fft.rfft2(x_32, norm='ortho')

                # 提取实部和虚部
                real = x_fft.real
                imag = x_fft.imag

                # 计算振幅和相位
                amplitude = torch.sqrt(real**2 + imag**2)
                phase = torch.atan2(imag, real)

                self.freq_features[name] = {
                    'amplitude': amplitude.detach().cpu(),
                    'phase': phase.detach().cpu(),
                    'real': real.detach().cpu(),
                    'imag': imag.detach().cpu()
                }
            return hook

        # 为DDFE模块注册hook
        self.model.freq_fusion.register_forward_hook(hook_fn('ddfe'))

    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img

    def visualize(self, image_path, save_path=None):
        # 加载图像
        img_tensor, img_pil = self.load_image(image_path)

        # 前向传播
        with torch.no_grad():
            output = self.model(img_tensor)

        # 获取频域特征
        amplitude = self.freq_features['ddfe']['amplitude'][0]  # [C, H, W/2+1]
        phase = self.freq_features['ddfe']['phase'][0]

        # 对所有通道取平均
        amplitude_mean = amplitude.mean(dim=0).numpy()
        phase_mean = phase.mean(dim=0).numpy()

        # 对数尺度显示振幅
        amplitude_log = np.log(amplitude_mean + 1e-8)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 输入图像
        axes[0].imshow(img_pil)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')

        # 振幅图
        im1 = axes[1].imshow(amplitude_log, cmap='jet')
        axes[1].set_title('Frequency Amplitude (log scale)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # 相位图
        im2 = axes[2].imshow(phase_mean, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[2].set_title('Frequency Phase', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

    def visualize_multi_channel(self, image_path, num_channels=6, save_path=None):
        """可视化多个通道的频域特征"""
        img_tensor, img_pil = self.load_image(image_path)

        with torch.no_grad():
            output = self.model(img_tensor)

        amplitude = self.freq_features['ddfe']['amplitude'][0]
        phase = self.freq_features['ddfe']['phase'][0]

        total_channels = amplitude.shape[0]
        num_channels = min(num_channels, total_channels)

        fig, axes = plt.subplots(3, num_channels, figsize=(3*num_channels, 9))

        # 第一行：输入图像（只在第一列显示）
        axes[0, 0].imshow(img_pil)
        axes[0, 0].set_title('Input Image', fontsize=10)
        axes[0, 0].axis('off')
        for i in range(1, num_channels):
            axes[0, i].axis('off')

        # 第二行和第三行：振幅和相位
        for i in range(num_channels):
            amp = amplitude[i].numpy()
            phs = phase[i].numpy()

            # 振幅（对数尺度）
            amp_log = np.log(amp + 1e-8)
            im1 = axes[1, i].imshow(amp_log, cmap='jet')
            axes[1, i].set_title(f'Ch {i} Amplitude', fontsize=10)
            axes[1, i].axis('off')

            # 相位
            im2 = axes[2, i].imshow(phs, cmap='hsv', vmin=-np.pi, vmax=np.pi)
            axes[2, i].set_title(f'Ch {i} Phase', fontsize=10)
            axes[2, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-channel visualization to {save_path}")

        plt.show()


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "path/to/your/model_weights.pth"  # 修改为你的模型权重路径
    IMAGE_PATH = "path/to/your/test_image.png"     # 修改为你的测试图像路径
    SAVE_DIR = "visualizations"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 创建可视化器
    visualizer = DDFEVisualizer(MODEL_PATH, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 单张图像可视化（平均所有通道）
    save_path = os.path.join(SAVE_DIR, "ddfe_freq_visualization.png")
    visualizer.visualize(IMAGE_PATH, save_path=save_path)

    # 多通道可视化
    save_path_multi = os.path.join(SAVE_DIR, "ddfe_freq_multi_channel.png")
    visualizer.visualize_multi_channel(IMAGE_PATH, num_channels=6, save_path=save_path_multi)

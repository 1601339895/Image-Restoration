# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
from pathlib import Path

class CalibDataset(Dataset):
    """校准数据集"""
    def __init__(self, data_dir, max_samples=1000):
        self.data_dir = data_dir
        self.image_files = []
        self.max_samples = max_samples
        
        # 支持的图片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 收集所有图片文件
        self._collect_image_files()
        
        # 如果图片数量超过最大样本数，随机采样
        if len(self.image_files) > max_samples:
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_samples]
        
        print(f"校准数据集: 从 {data_dir} 加载了 {len(self.image_files)} 张图片")
        
        # 数据预处理管道
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),          # 调整到目标尺寸
            transforms.ToTensor(),                  # 转换为tensor，范围[0,1]
        ])
    
    def _collect_image_files(self):
        """收集所有图片文件"""
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            print(f"警告: 校准数据目录不存在 {self.data_dir}")
            return
        
        # 递归搜索所有图片文件
        for img_path in data_path.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in self.supported_formats:
                self.image_files.append(str(img_path))
        
        # 如果没有找到图片，尝试直接在根目录查找
        if not self.image_files:
            for file_path in data_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    self.image_files.append(str(file_path))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # 尝试用PIL加载图片
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
            
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
            # 如果加载失败，返回随机噪声
            return torch.randn(3, 128, 128)

class FallbackCalibDataset(Dataset):
    """备用校准数据集 - 使用随机数据"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        print(f"使用随机数据作为校准数据集: {num_samples} 个样本")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成符合人脸数据分布的随机数据
        # 使用正态分布，均值0.5，标准差0.2
        image = torch.normal(0.5, 0.2, (3, 128, 128))
        image = torch.clamp(image, 0.0, 1.0)  # 限制在[0,1]范围
        return image

def build_calib_loader(data_dir, batch=1, max_samples=1000, num_workers=2):
    """
    构建校准数据加载器
    
    Args:
        data_dir (str): 校准数据目录
        batch (int): 批次大小
        max_samples (int): 最大样本数
        num_workers (int): 数据加载线程数
    
    Returns:
        DataLoader: 校准数据加载器
    """
    
    # 首先尝试从指定目录加载真实数据
    if os.path.exists(data_dir):
        dataset = CalibDataset(data_dir, max_samples=max_samples)
        
        # 如果没有找到图片，使用备用数据集
        if len(dataset) == 0:
            print(f"在 {data_dir} 中未找到有效图片，使用随机数据")
            dataset = FallbackCalibDataset(num_samples=max_samples)
    else:
        print(f"校准数据目录不存在 {data_dir}，使用随机数据")
        dataset = FallbackCalibDataset(num_samples=max_samples)
    
    # 创建数据加载器
    loader = DataLoader(
        dataset, 
        batch_size=batch, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"校准数据加载器创建成功: batch_size={batch}, 总样本数={len(dataset)}")
    return loader

def test_calib_loader():
    """测试校准数据加载器"""
    print("=== 测试校准数据加载器 ===")
    
    # 测试路径（您可以修改为实际路径）
    test_dir = "/data/jg-face-unlock-ppl_txy/base/fea/calibration/face_300"
    
    try:
        loader = build_calib_loader(test_dir, batch=4, max_samples=20)
        
        print(f"数据加载器创建成功")
        print(f"数据集大小: {len(loader.dataset)}")
        print(f"批次数量: {len(loader)}")
        
        # 测试加载几个批次
        for i, batch in enumerate(loader):
            print(f"批次 {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            print(f"  数值范围: [{batch.min():.3f}, {batch.max():.3f}]")
            print(f"  均值: {batch.mean():.3f}, 标准差: {batch.std():.3f}")
            
            if i >= 2:  # 只测试前3个批次
                break
                
        print("✓ 数据加载器测试通过")
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行测试
    test_calib_loader()

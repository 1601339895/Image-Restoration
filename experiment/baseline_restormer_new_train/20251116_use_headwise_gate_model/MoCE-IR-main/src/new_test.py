import os
import re
import glob
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from lightning.pytorch import Trainer

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from net.moce_ir import MoCEIR
from options import train_options
from utils.test_utils import save_img
from data.dataset_utils import IRBenchmarks, CDD11


####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
    err = np.sum((image_true - image_test) **2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range=1.0,
                                               full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    return np.mean(ssim)


def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range=1.0, full=False)


####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.net = MoCEIR(
            dim=32,
            num_blocks=[4, 6, 6, 8],
            heads=[1, 2, 4, 8],
            gate_type="headwise",  
            LayerNorm_type='WithBias',
            bias=False
        )

    def forward(self, x):
        return self.net(x)


####################################################################################################
"""
def run_test(opts, net, dataset, subset_name):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=4)

    if opts.save_results:
        save_dir = os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{subset_name}")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr_list, ssim_list, lpips_list = [], [], []
    
    with torch.no_grad():
        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader, desc=f"Testing {subset_name}"):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            # 模型推理
            restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # 计算LPIPS
            restored_clamped = torch.clamp(restored, 0, 1)
            lpips_val = calc_lpips(clean_patch, restored_clamped).cpu().item()
            lpips_list.append(lpips_val)

            # 转换为numpy格式计算PSNR和SSIM
            restored_np = restored_clamped.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean_np = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # 计算指标
            ssim_val = calc_ssim(clean_np, restored_np)
            ssim_list.append(ssim_val)
            psnr_val = peak_signal_noise_ratio(clean_np, restored_np, data_range=1)
            psnr_list.append(psnr_val)

            # 保存结果图像
            if opts.save_results:
                base_name = os.path.splitext(os.path.basename(clean_name[0]))[0]
                save_name = f"{base_name}_psnr{psnr_val:.2f}_ssim{ssim_val:.4f}.png"
                save_img(os.path.join(save_dir, save_name), img_as_ubyte(restored_np))

    # 返回平均指标
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)
"""

# 修改run_test函数，移除手动的cuda()调用，由Trainer管理设备
def run_test(opts, net, dataset, subset_name):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=8)  # 增加num_workers加速数据加载

    if opts.save_results:
        save_dir = os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{subset_name}")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").to(net.device)  # 自动适配设备
    psnr_list, ssim_list, lpips_list = [], [], []
    
    with torch.no_grad():
        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader, desc=f"Testing {subset_name}"):
            degrad_patch, clean_patch = degrad_patch.to(net.device), clean_patch.to(net.device)  # 自动适配设备

            # 模型推理
            restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # 计算LPIPS
            restored_clamped = torch.clamp(restored, 0, 1)
            lpips_val = calc_lpips(clean_patch, restored_clamped).item()
            lpips_list.append(lpips_val)

            # 转换为numpy格式计算PSNR和SSIM
            restored_np = restored_clamped.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean_np = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # 计算指标
            ssim_val = calc_ssim(clean_np, restored_np)
            ssim_list.append(ssim_val)
            psnr_val = peak_signal_noise_ratio(clean_np, restored_np, data_range=1)
            psnr_list.append(psnr_val)

            # 保存结果图像
            if opts.save_results:
                base_name = os.path.splitext(os.path.basename(clean_name[0]))[0]
                save_name = f"{base_name}_psnr{psnr_val:.2f}_ssim{ssim_val:.4f}.png"
                save_img(os.path.join(save_dir, save_name), img_as_ubyte(restored_np))

    # 返回平均指标
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)

####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # 定义所有11个测试集
    test_subsets = [
        "low", "haze", "rain", "snow",
        "low_haze", "low_rain", "low_snow",
        "haze_rain", "haze_snow",
        "low_haze_rain", "low_haze_snow"
    ]

    # 获取所有模型文件并按epoch排序
    ckpt_files = glob.glob(os.path.join(opt.checkpoint_id, "epoch=*.ckpt"))
    ckpt_info = []
    for ckpt in ckpt_files:
        match = re.search(r"epoch=(\d+)", ckpt)
        if match:
            epoch = int(match.group(1))
            ckpt_info.append((epoch, ckpt))
    ckpt_info.sort(key=lambda x: x[0])  # 按epoch升序排列
    print(f"发现 {len(ckpt_info)} 个模型文件，已按epoch排序")

    # 创建结果表格（多级表头）
    columns = [("epoch", "")]
    for subset in test_subsets:
        columns.extend([(subset, "PSNR"), (subset, "SSIM"), (subset, "LPIPS")])
    results_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))

    # 初始化Trainer（管理4块GPU并行）
    trainer = Trainer(
        accelerator="gpu",
        devices=4,  # 使用4块GPU
        strategy="ddp",  # 分布式数据并行
        precision="16-mixed"  # 可选：混合精度加速
    )

    # 遍历所有模型
    for epoch, ckpt_path in ckpt_info:
        print(f"\n===== 处理模型: epoch={epoch} =====")
        try:
            # 加载模型
            net = PLTestModel.load_from_checkpoint(ckpt_path, opt=opt)
            net.eval()
        except Exception as e:
            print(f"模型加载失败: {e}，跳过该模型")
            continue

        # 存储当前epoch的结果
        row_data = {"epoch": epoch}
        
        # 遍历所有测试集
        for subset in test_subsets:
            print(f"\n----- 测试集: CDD11_{subset} -----")
            try:
                # 创建数据集
                dataset = CDD11(opt, split="test", subset=subset)
                # 运行测试（Trainer会自动分发到4块GPU）
                psnr, ssim, lpips = run_test(opt, net, dataset, subset)
                # 记录结果
                row_data[(subset, "PSNR")] = round(psnr, 4)
                row_data[(subset, "SSIM")] = round(ssim, 4)
                row_data[(subset, "LPIPS")] = round(lpips, 4)
                print(f"结果: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}")
            except Exception as e:
                print(f"测试集 {subset} 失败: {e}，结果记为NaN")
                row_data[(subset, "PSNR")] = None
                row_data[(subset, "SSIM")] = None
                row_data[(subset, "LPIPS")] = None

        # 添加到表格
        results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)

    # 保存结果到Excel
    output_path = os.path.join(opt.checkpoint_id, "测试结果汇总.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"\n所有结果已保存至: {output_path}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要布尔值参数')


def train_options():
    parser = argparse.ArgumentParser()
    # 设置默认参数，用户无需手动输入
    parser.add_argument('--model', type=str, default='MoCE_IR_S', help='模型名称')
    parser.add_argument('--checkpoint_id', type=str, 
                        default='/home/aiswjtu/hl/new_image_restoration/20251116_use_headwise_gate_model/MoCE-IR-main/checkpoints/2025_11_17_10_47_26',
                        help='模型 checkpoint 目录')
    parser.add_argument('--data_file_dir', type=str, 
                        default='/home/aiswjtu/hl/img_data/Data/',
                        help='数据集根目录')
    parser.add_argument('--save_results', type=str2bool, default=False, 
                        help='是否保存修复后的图像')
    parser.add_argument('--de_type', nargs='+', 
                        default=['denoise_15', 'denoise_25', 'denoise_50', 'dehaze', 'derain', 'deblur', 'synllie'],
                        help='退化类型列表')
    parser.add_argument('--trainset', type=str, default='CDD11_low', 
                        help='训练集类型（如CDD11_low、CDD11_haze等）')
    parser.add_argument('--benchmarks', nargs='+', default=['cdd11'], 
                        help='基准测试集列表（如cdd11）')

    return parser.parse_args()


if __name__ == '__main__':
    opt = train_options()
    main(opt)
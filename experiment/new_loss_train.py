# -*- coding: utf-8 -*-
# File  : train.py
# Author: HeLei
# Date  : 2025/12/20

import sys
import os
import time
import pathlib
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch import seed_everything

# 引入你的网络和配置
from net.model import RectiFormer
from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11

from losses import FocalL1Loss, FFTLoss, SSIMloss, PSNRLoss


# ==============================================================================
# 1. 辅助回调函数 (保持清洁的日志输出)
# ==============================================================================

class TeeLogger(object):
    """同时输出到终端和日志文件"""

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class PrintEpochCallback(Callback):
    """每个Epoch结束时打印清晰的指标"""

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("loss/total_epoch")
        psnr = trainer.callback_metrics.get("metric/psnr_epoch")
        lr = trainer.callback_metrics.get("LR")

        loss_val = f"{loss.item():.6f}" if loss is not None else "N/A"
        psnr_val = f"{psnr.item():.2f}" if psnr is not None else "N/A"
        lr_val = f"{lr.item():.2e}" if lr is not None else "N/A"

        rank_zero_info(f"-" * 60)
        rank_zero_info(f"Epoch [{trainer.current_epoch:03d}/{trainer.max_epochs}] | "
                       f"Loss: {loss_val} | PSNR: {psnr_val} dB | LR: {lr_val}")
        rank_zero_info(f"-" * 60)


# ==============================================================================
# 2. Lightning Module (训练核心)
# ==============================================================================

class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.save_hyperparameters(vars(opt))

        # 1. 初始化模型
        self.net = RectiFormer(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_refinement_blocks=opt.num_refinement_blocks,
            heads=opt.heads,
            # 如果你的模型里加了context_dim等参数，请在这里补充
        )


        # (A) L1 Loss (主要): 使用 FocalL1Loss 挖掘难样本
        # 如果你只想用纯 L1，改用 nn.L1Loss() 即可，但 FocalL1 通常 PSNR 更高
        self.criterion_pixel = FocalL1Loss(gamma=0.5, alpha=1.0)

        # (B) FFT Loss (频域): 恢复纹理，提升 SSIM
        self.criterion_fft = FFTLoss(loss_weight=1.0)

        # (C) SSIM Loss (结构): 显式优化 SSIM 指标
        self.criterion_ssim = SSIMloss(loss_weight=1.0, data_range=1.0)

        # 监控指标 (PSNR)
        self.metric_psnr = PSNRLoss()

        # 3. 权重 (从 options 获取)
        self.w_l1 = opt.lambda_l1
        self.w_fft = opt.lambda_fft
        self.w_ssim = opt.lambda_ssim

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        # 前向传播
        restored = self.net(degrad_patch)

        # 计算 Loss (加权求和)
        l_pixel = self.criterion_pixel(restored, clean_patch)
        l_fft = self.criterion_fft(restored, clean_patch)
        l_ssim = self.criterion_ssim(restored, clean_patch)

        # 总损失
        loss = (self.w_l1 * l_pixel) + \
               (self.w_fft * l_fft) + \
               (self.w_ssim * l_ssim)

        # 打印部分 Step 日志 (仿照你提供的 MoCEIR 风格)
        if self.trainer.is_global_zero and batch_idx % 100 == 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            print(f"[Epoch {self.current_epoch:03d}, Step {batch_idx:05d}] "
                  f"Loss: {loss.item():.6f} (L1:{l_pixel.item():.4f} FFT:{l_fft.item():.4f}), LR: {lr:.2e}")

        # 记录日志 (Sync_dist=True 确保多卡训练准确)
        self.log("loss/total", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/l1", l_pixel, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/fft", l_fft, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/ssim", l_ssim, on_step=False, on_epoch=True, sync_dist=True)

        # 记录 LR
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 记录 PSNR (仅供参考)
        with torch.no_grad():
            cur_psnr = -1 * self.metric_psnr(restored, clean_patch)
            self.log("metric/psnr", cur_psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr, weight_decay=1e-4)

        # All-in-One 任务建议 Warmup 稍微长一点 (10~15 epochs)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15 if not self.opt.fine_tune_from else 1,
            max_epochs=self.opt.epochs
        )
        return [optimizer], [scheduler]




def main(opt):
    # 时间戳与日志目录
    time_stamp = opt.time_stamp  # 从外部传入或在此生成
    if not time_stamp:
        time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        opt.time_stamp = time_stamp

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 1. 终端输出重定向 (保留文件日志)
    log_file_path = os.path.join(log_dir, f"train_log_{time_stamp}.txt")
    sys.stdout = TeeLogger(log_file_path)

    print("=" * 50)
    print(f"Training Strategy: L1({opt.lambda_l1}) + FFT({opt.lambda_fft}) + SSIM({opt.lambda_ssim})")
    print("=" * 50)

    # 2. Logger
    if opt.wblogger:
        name = f"{opt.model}_{time_stamp}"
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt, log_model="all")
    else:
        logger = TensorBoardLogger(save_dir=log_dir, default_hp_metric=False)

    # 3. 模型
    if opt.fine_tune_from:
        print(f"Fine-tuning from: {opt.fine_tune_from}")
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"),
            opt=opt
        )
    else:
        model = PLTrainModel(opt)

    # 4. Checkpoint 回调
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_epochs=5,
        save_top_k=3,
        save_last=True,
        monitor="metric/psnr_epoch",
        mode="max",
        filename='{epoch}-{step}-psnr{metric/psnr_epoch:.2f}'
    )

    # 5. 数据集
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    # 6. Trainer (重点：无梯度裁剪)
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true" if opt.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback, PrintEpochCallback()],  #
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        log_every_n_steps=50,
        precision="16-mixed",  # 混合精度省显存，提速度
    )

    # 7. 恢复训练
    resume_ckpt_path = None
    if opt.resume_from:
        resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        print(f"Resuming from: {resume_ckpt_path}")

    # 8. 开始训练
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        ckpt_path=resume_ckpt_path
    )


if __name__ == '__main__':
    train_opt = train_options()
    # 保证可复现性
    seed_everything(42, workers=True)
    main(train_opt)

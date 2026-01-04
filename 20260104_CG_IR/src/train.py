import sys 
import os
import time
import pathlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch import Trainer, seed_everything

from net.model_no_use_CAFGB import Context_Gated_IR
from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FocalL1Loss, FFTLoss


# ========== 新增：双重输出记录器 (屏幕 + 文件) ==========
class TeeLogger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入，防止缓冲区滞后

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ========== 新增：每 Epoch 打印 Loss 和 LR 回调 ==========
class PrintEpochCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("loss/total_epoch")
        lr = trainer.callback_metrics.get("LR Schedule")

        loss_val = f"{loss.item():.6f}" if loss is not None else "N/A"
        lr_val = f"{lr.item():.8f}" if lr is not None else "N/A"
        
        rank_zero_info(f"\n" + "-"*40)
        rank_zero_info(f"Epoch [{trainer.current_epoch}/{trainer.max_epochs}] Completed")
        rank_zero_info(f"Avg Loss: {loss_val} | Learning Rate: {lr_val}")
        rank_zero_info("-" * 40 + "\n")

# 训练时间统计回调
class TrainingTimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    def on_train_start(self, trainer, pl_module):
        """训练开始时记录时间"""
        self.start_time = time.time()
        rank_zero_info(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_end(self, trainer, pl_module):
        """训练结束时计算并输出总时间"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # 转换为小时:分钟:秒格式
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        rank_zero_info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # 将所有参数保存到hparams
        self.save_hyperparameters(self._convert_opt_to_dict(opt))
        
        # 初始化模型
        self.net = Context_Gated_IR(
                dim=opt.dim,
                num_blocks=opt.num_blocks,
                num_refinement_blocks=opt.num_refinement_blocks,
                heads=opt.heads,
                context_dim=opt.context_dim,
                num_scales=opt.num_scales,
        )
        
        # self.loss_fn = nn.L1Loss()

        # 2. 损失函数初始化
        # (A) Focal L1 Loss（挖掘难样本，若需纯L1可改用nn.L1Loss()）
        self.criterion_pixel = FocalL1Loss(gamma=0.5, alpha=1.0)
        # (B) FFT Loss（频域损失，恢复纹理）
        self.criterion_fft = FFTLoss(loss_weight=0.1)  # 原来是1.0

    def _convert_opt_to_dict(self, opt):
        """将argparse.Namespace转换为字典，处理特殊类型"""
        opt_dict = vars(opt)
        for key, value in opt_dict.items():
            if isinstance(value, (np.ndarray, np.generic)):
                opt_dict[key] = value.tolist()
            elif isinstance(value, pathlib.Path):
                opt_dict[key] = str(value)
        return opt_dict
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
    
        # 计算各损失项
        l_pixel = self.criterion_pixel(restored, clean_patch)
        l_fft = self.criterion_fft(restored, clean_patch)

        # 总损失（加权求和）
        loss = l_pixel + l_fft
            
        # 记录日志（sync_dist=True 确保多卡训练指标准确）
        self.log("loss/total", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/l1", l_pixel, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/fft", l_fft, on_step=False, on_epoch=True, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # LR 只需要 epoch 级别记录即可
        self.log("LR Schedule", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
        
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        
        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.opt.epochs
            )      
        return [optimizer], [scheduler]


def main(opt):
    # ========== 核心复用唯一的time_stamp ==========
    time_stamp = opt.time_stamp
    
    # 日志文件夹：基于唯一time_stamp创建
    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ========== 新增：设置双重日志记录 (Console + File) ==========
    # 将标准输出重定向到 log 文件，同时保留屏幕输出
    log_file_path = os.path.join(log_dir, f"train_log_{time_stamp}.txt")
    sys.stdout = TeeLogger(log_file_path)
    # 如果你也想捕获报错信息到文件，取消下面这行的注释
    # sys.stderr = TeeLogger(log_file_path)

    # 只在主进程打印参数
    rank_zero_info("="*50)
    rank_zero_info("Training Options")
    rank_zero_info("="*50)
    for key, value in sorted(vars(opt).items()):
        rank_zero_info(f"{key:<30}: {value}")
    rank_zero_info("="*50)
    rank_zero_info(f"Logging to file: {log_file_path}")
    
    # 初始化logger
    if opt.wblogger:
        name = f"{opt.model}_{time_stamp}"
        logger = WandbLogger(
            name=name, 
            save_dir=log_dir, 
            config=opt,
            log_model="all"
        )
    else:
        logger = TensorBoardLogger(
            save_dir=log_dir,
            default_hp_metric=False
        )

    # 创建模型
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), 
            opt=opt
        )
    else:
        model = PLTrainModel(opt)
    print(model)
    # 检查点文件夹
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, 
        every_n_epochs=5, 
        save_top_k=-1, 
        save_last=True
    )
    
    # 实例化回调
    time_callback = TrainingTimeCallback()
    print_callback = PrintEpochCallback()  # <--- 新增
    
    # 创建数据集和数据加载器
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
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, time_callback, print_callback], 
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        log_every_n_steps=10,
        # precision="16-mixed"
    )
    
    # 恢复训练的检查点路径
    resume_ckpt_path = None
    if opt.resume_from:
        resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        rank_zero_info(f"\nResuming training from checkpoint: {resume_ckpt_path}")

    # 开始训练
    rank_zero_info(f"\nStarting training...")
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        ckpt_path=resume_ckpt_path
    )
    
    # 训练总结
    if hasattr(time_callback, 'end_time'):
        total_time = time_callback.end_time - time_callback.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        rank_zero_info(f"\n" + "="*50)
        rank_zero_info("Training Summary")
        rank_zero_info("="*50)
        rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s")
        rank_zero_info(f"Epochs completed: {trainer.current_epoch}")
        rank_zero_info(f"Model checkpoints saved to: {checkpoint_path}")
        rank_zero_info(f"Logs saved to: {log_dir}")
        rank_zero_info("="*50)


if __name__ == '__main__':
    train_opt = train_options()
    
    # 生成唯一的时间戳
    unique_time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    train_opt.time_stamp = unique_time_stamp
    
    seed_everything(42, workers=True)

    main(train_opt)
# from typing import List
# import os
# import time
# import pathlib
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import torch.nn as nn
# import torch.optim as optim
# import lightning.pytorch as pl
# from torch.utils.data import DataLoader
# from lightning.pytorch.callbacks import ModelCheckpoint, Callback
# from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
# from lightning.pytorch.utilities.rank_zero import rank_zero_info
# from lightning.pytorch import Trainer, seed_everything


# from net.model import LoRa_Gate_Restormer,apply_mixed_strategy_lora
# from options import train_options
# from utils.schedulers import LinearWarmupCosineAnnealingLR
# from data.dataset_utils import AIOTrainDataset, CDD11


# # 训练时间统计回调
# class TrainingTimeCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.start_time = None
#         self.end_time = None

#     def on_train_start(self, trainer, pl_module):
#         """训练开始时记录时间"""
#         self.start_time = time.time()
#         rank_zero_info(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     def on_train_end(self, trainer, pl_module):
#         """训练结束时计算并输出总时间"""
#         self.end_time = time.time()
#         total_time = self.end_time - self.start_time
        
#         # 转换为小时:分钟:秒格式
#         hours = int(total_time // 3600)
#         minutes = int((total_time % 3600) // 60)
#         seconds = int(total_time % 60)
        
#         rank_zero_info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")


# class PLTrainModel(pl.LightningModule):
#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt
        
#         # 将所有参数保存到hparams（关键：Lightning会自动将hparams存入检查点）
#         self.save_hyperparameters(self._convert_opt_to_dict(opt))
        
#         # 初始化模型
#         self.net = LoRa_Gate_Restormer(
#             dim=opt.dim,
#             num_blocks=opt.num_blocks,
#             num_refinement_blocks=opt.num_refinement_blocks,
#             heads=opt.heads,
#             gate_type=opt.gate_type
#         )
#         self.net = apply_mixed_strategy_lora(self.net, 
#                                              Lora_ffn_ratio=opt.LoRa_ffn_ratio, 
#                                              Lora_attn_ratio=opt.LoRa_attn_ratio,
#                                              )
        
#         self.loss_fn = nn.L1Loss()
    
#     def _convert_opt_to_dict(self, opt):
#         """将argparse.Namespace转换为字典，处理特殊类型"""
#         opt_dict = vars(opt)
#         # 处理numpy类型/列表等不能序列化的类型
#         for key, value in opt_dict.items():
#             if isinstance(value, (np.ndarray, np.generic)):
#                 opt_dict[key] = value.tolist()
#             elif isinstance(value, pathlib.Path):
#                 opt_dict[key] = str(value)
#         return opt_dict
    
#     def forward(self, x):
#         return self.net(x)
    
#     def training_step(self, batch, batch_idx):
#         ([clean_name, de_id], degrad_patch, clean_patch) = batch
#         restored = self.net(degrad_patch, de_id)
    
#         loss = self.loss_fn(restored, clean_patch)
            
#         self.log("Train_Loss", loss, sync_dist=True)
#         lr = self.trainer.optimizers[0].param_groups[0]["lr"]
#         self.log("LR Schedule", lr, sync_dist=True)

#         return loss
        
#     def lr_scheduler_step(self, scheduler, metric):
#         scheduler.step()
    
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)  # 使用opt中的lr而非硬编码
#         scheduler = LinearWarmupCosineAnnealingLR(
#             optimizer=optimizer,
#             warmup_epochs=15,
#             max_epochs=150
#         )
        
#         if self.opt.fine_tune_from:
#             scheduler = LinearWarmupCosineAnnealingLR(
#                 optimizer=optimizer,
#                 warmup_epochs=1,
#                 max_epochs=self.opt.epochs
#             )      
#         return [optimizer], [scheduler]


# def main(opt):
#     # ========== 核心修复：全程复用唯一的time_stamp ==========
#     time_stamp = opt.time_stamp  # 直接使用opt中预先生成的唯一的time_stamp
    
#     # 只在主进程打印参数
#     rank_zero_info("="*50)
#     rank_zero_info("Training Options")
#     rank_zero_info("="*50)
#     for key, value in sorted(vars(opt).items()):
#         rank_zero_info(f"{key:<30}: {value}")
#     rank_zero_info("="*50)
        
#     # 日志文件夹：基于唯一time_stamp创建
#     log_dir = os.path.join("logs/", time_stamp)
#     pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
#     # 初始化logger（确保TensorBoardLogger也保存hparams）
#     if opt.wblogger:
#         name = f"{opt.model}_{time_stamp}"
#         logger = WandbLogger(
#             name=name, 
#             save_dir=log_dir, 
#             config=opt,
#             log_model="all"
#         )
#     else:
#         logger = TensorBoardLogger(
#             save_dir=log_dir,
#             default_hp_metric=False  # 禁用默认的hp_metric
#         )

#     # 创建模型
#     if opt.fine_tune_from:
#         model = PLTrainModel.load_from_checkpoint(
#             os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), 
#             opt=opt
#         )
#     else:
#         model = PLTrainModel(opt)

#     # 检查点文件夹：基于唯一time_stamp创建
#     checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
#     pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
#     # 修复：移除错误的save_hyperparameters参数
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=checkpoint_path, 
#         every_n_epochs=5, 
#         save_top_k=-1, 
#         save_last=True  # 仅保留有效参数
#     )
    
#     # 训练时间回调
#     time_callback = TrainingTimeCallback()
    
#     # 创建数据集和数据加载器
#     if "CDD11" in opt.trainset:
#         _, subset = opt.trainset.split("_")
#         trainset = CDD11(opt, split="train", subset=subset)
#     else:
#         trainset = AIOTrainDataset(opt)
        
#     trainloader = DataLoader(
#         trainset, 
#         batch_size=opt.batch_size, 
#         pin_memory=True, 
#         shuffle=True, 
#         drop_last=True, 
#         num_workers=opt.num_workers
#     )
    
#     # 创建训练器 - 添加混合精度设置
#     trainer = pl.Trainer(
#         max_epochs=opt.epochs,
#         accelerator="gpu",
#         devices=opt.num_gpus,
#         strategy="ddp_find_unused_parameters_true",
#         logger=logger,
#         callbacks=[checkpoint_callback, time_callback],  # 添加时间回调
#         accumulate_grad_batches=opt.accum_grad,
#         deterministic=True,
#         log_every_n_steps=10,  # 可选：控制日志打印频率
#         precision="16-mixed"  # 启用混合精度训练，适用于V100
#     )
    
#     # 恢复训练的检查点路径
#     resume_ckpt_path = None
#     if opt.resume_from:
#         resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
#         rank_zero_info(f"\nResuming training from checkpoint: {resume_ckpt_path}")

#     # 开始训练
#     rank_zero_info(f"\nStarting training...")
#     trainer.fit(
#         model=model, 
#         train_dataloaders=trainloader, 
#         ckpt_path=resume_ckpt_path
#     )
    
#     # 训练总结
#     if hasattr(time_callback, 'end_time'):
#         total_time = time_callback.end_time - time_callback.start_time
#         hours = int(total_time // 3600)
#         minutes = int((total_time % 3600) // 60)
#         seconds = int(total_time % 60)
#         rank_zero_info(f"\n" + "="*50)
#         rank_zero_info("Training Summary")
#         rank_zero_info("="*50)
#         rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s")
#         rank_zero_info(f"Epochs completed: {trainer.current_epoch}")
#         rank_zero_info(f"Model checkpoints saved to: {checkpoint_path}")
#         rank_zero_info(f"Logs saved to: {log_dir}")
#         rank_zero_info("="*50)


# if __name__ == '__main__':
#     # ========== 关键：在解析参数后、主进程中唯一生成时间戳 ==========
#     train_opt = train_options()
    
#     # 生成唯一的时间戳（只生成一次！）
#     unique_time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
#     # 将时间戳添加到opt中，全程复用
#     train_opt.time_stamp = unique_time_stamp
    
#     # 运行主训练函数
#     main(train_opt)



import sys  # <--- 新增：用于控制标准输出
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

from net.model import LoRa_Gate_Restormer, apply_mixed_strategy_lora
from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11


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
        # 从 callback_metrics 中获取 epoch 级别的平均指标
        # 注意：这里对应的 key 需要与 training_step 中 self.log 的 name 一致
        # 如果 training_step 中设置了 on_step=True, on_epoch=True，通常 key 会变为 name_epoch
        loss = trainer.callback_metrics.get("Train_Loss_epoch")
        if loss is None:
            loss = trainer.callback_metrics.get("Train_Loss") # 尝试获取原始名称
            
        lr = trainer.callback_metrics.get("LR Schedule")

        # 格式化数值
        loss_val = f"{loss.item():.6f}" if loss is not None else "N/A"
        lr_val = f"{lr.item():.8f}" if lr is not None else "N/A"
        
        # 使用 rank_zero_info 确保只在主进程打印
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
        self.net = LoRa_Gate_Restormer(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_refinement_blocks=opt.num_refinement_blocks,
            heads=opt.heads,
            gate_type=opt.gate_type
        )
        self.net = apply_mixed_strategy_lora(self.net, 
                                             Lora_ffn_ratio=opt.LoRa_ffn_ratio, 
                                             Lora_attn_ratio=opt.LoRa_attn_ratio,
                                             )
        
        self.loss_fn = nn.L1Loss()
    
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
        restored = self.net(degrad_patch, de_id)
    
        loss = self.loss_fn(restored, clean_patch)
            
        # ========== 修改：确保记录 Epoch 级别的 Loss ==========
        # on_step=True: 在每一步记录（用于平滑曲线）
        # on_epoch=True: 计算整个 Epoch 的平均值（用于回调打印）
        # prog_bar=True: 显示在进度条上
        self.log("Train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
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
        # ========== 修改：加入 print_callback 到回调列表 ==========
        callbacks=[checkpoint_callback, time_callback, print_callback], 
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        log_every_n_steps=10,
        precision="16-mixed"
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
    
    main(train_opt)
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

# ========== 导入自定义模块 ==========
from net.new_mode import SALoRA_GFS_AllInOne, apply_mixed_strategy_lora
from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
# 数据供给代码（无需修改，直接导入）
from data.dataset_utils import AIOTrainDataset, CDD11

# ========== 日志重定向类 ==========
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

# ========== 自定义回调 ==========
class PrintEpochCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 获取Epoch级别的平均指标
        loss = trainer.callback_metrics.get("Train_Loss_epoch") or trainer.callback_metrics.get("Train_Loss")
        lr = trainer.callback_metrics.get("LR Schedule")

        # 格式化输出
        loss_val = f"{loss.item():.6f}" if loss is not None else "N/A"
        lr_val = f"{lr.item():.8f}" if lr is not None else "N/A"
        
        # 只在主进程打印
        rank_zero_info(f"\n" + "-"*40)
        rank_zero_info(f"Epoch [{trainer.current_epoch}/{trainer.max_epochs}] Completed")
        rank_zero_info(f"Avg Loss: {loss_val} | Learning Rate: {lr_val}")
        rank_zero_info("-" * 40 + "\n")

class TrainingTimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        rank_zero_info(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # CDD-11 数据集信息打印（匹配论文）
        if trainer.model.opt.train_scene == 'cdd11':
            rank_zero_info(f"CDD-11 Config: {trainer.model.opt.trainset} | Tasks: {trainer.model.opt.task_num} | Epochs: {trainer.model.opt.epochs}")
            rank_zero_info(f"CDD-11 Dataset Path: {os.path.join(trainer.model.opt.data_file_dir, 'cdd11')}")

    def on_train_end(self, trainer, pl_module):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        rank_zero_info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")

# ========== Lightning训练模型 ==========
class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # 保存超参数（含CDD-11配置）
        self.save_hyperparameters(self._convert_opt_to_dict(opt))

        # 初始化ALL-in-One网络（适配CDD-11的11类任务）
        self.net = SALoRA_GFS_AllInOne(
            in_channels=3,
            out_channels=3,
            base_dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_refine_blocks=opt.num_refinement_blocks,
            heads=opt.heads,
            base_rank=4,
            norm_type='WithBias',
            task_num=opt.task_num,       # CDD-11: 2/4/5/11
            task_emb_dim=opt.task_emb_dim  # 16维适配11类任务
        )

        # 应用混合LoRA策略（CDD-11复合退化推荐保留）
        self.net = apply_mixed_strategy_lora(
            self.net,
            ffn_ratio=opt.LoRa_ffn_ratio,
            attn_ratio=opt.LoRa_attn_ratio
        )
        
        # 损失函数（CDD-11复合退化用L1Loss更稳定）
        self.loss_fn = nn.L1Loss()

        # 任务类型映射（适配CDD-11的11类退化名称）
        self.de2task_map = {de: idx for idx, de in enumerate(opt.de_type)}
        # CDD-11别名映射（兼容数据代码的文件夹命名）
        self.cdd11_alias_map = {
            'low-light': 'low', 'snow': 'snow',  # 兼容数据代码可能的别名
            'low_haze': 'low+haze', 'low_rain': 'low+rain', 'low_snow': 'low+snow',
            'haze_rain': 'haze+rain', 'haze_snow': 'haze+snow',
            'low_haze_rain': 'low+haze+rain', 'low_haze_snow': 'low+haze+snow'
        }
    
    def _convert_opt_to_dict(self, opt):
        """将argparse.Namespace转换为可保存的字典"""
        opt_dict = vars(opt)
        for key, value in opt_dict.items():
            if isinstance(value, (np.ndarray, np.generic)):
                opt_dict[key] = value.tolist()
            elif isinstance(value, pathlib.Path):
                opt_dict[key] = str(value)
            elif isinstance(value, list) and all(isinstance(x, int) for x in value):
                opt_dict[key] = value  # 保留num_blocks/heads等列表
        return opt_dict
    
    def forward(self, x, task_id=0):
        return self.net(x, task_id=task_id)
    
    def training_step(self, batch, batch_idx):
        # 数据代码返回格式：([clean_name, de_id/de_type], degrad_patch, clean_patch)
        ([clean_name, de_info], degrad_patch, clean_patch) = batch
        
        # 适配CDD-11的退化名称（兼容下划线/加号、别名）
        if isinstance(de_info, str):
            # 替换下划线为加号（兼容数据代码文件夹命名）
            de_info = de_info.replace('_', '+')
            # 映射别名（如low-light→low）
            de_info = self.cdd11_alias_map.get(de_info, de_info)
            # 获取task_id（CDD-11 11类的ID）
            task_id = self.de2task_map.get(de_info, 0)
        else:
            # 数值型de_id（直接映射）
            task_id = de_info

        # 网络前向（传入CDD-11的task_id）
        restored = self.net(degrad_patch, task_id=task_id)
    
        # 计算损失
        loss = self.loss_fn(restored, clean_patch)
            
        # 日志记录（CDD-11场景增加任务ID监控）
        self.log("Train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("Task_ID", torch.tensor(task_id).to(self.device), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
        
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def configure_optimizers(self):
        # 优化器（CDD-11复合退化增加权重衰减）
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr, weight_decay=2e-4)
        
        # 学习率调度器（适配CDD-11 200 epoch）
        warmup_epochs = 20 if self.opt.train_scene == 'cdd11' else 15  # CDD-11更长预热
        warmup_epochs = 1 if self.opt.fine_tune_from else warmup_epochs
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=self.opt.epochs
        )
        
        return [optimizer], [scheduler]

# ========== 主训练函数 ==========
def main(opt):
    # 1. 初始化日志目录（含CDD-11细分类型）
    log_dir = os.path.join("logs/", opt.time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 2. 日志重定向（控制台+文件）
    log_file_path = os.path.join(log_dir, f"train_log_{opt.time_stamp}.txt")
    sys.stdout = TeeLogger(log_file_path)
    rank_zero_info("="*60)
    rank_zero_info(f"Training Config - Scene: {opt.train_scene} | Subset: {opt.trainset} | Model: {opt.model}")
    rank_zero_info("="*60)
    for key, value in sorted(vars(opt).items()):
        rank_zero_info(f"{key:<30}: {value}")
    rank_zero_info("="*60)

    # 3. 初始化Logger（WandB/TensorBoard）
    if opt.wblogger:
        # CDD-11场景增加标签
        logger_tags = [opt.train_scene, opt.trainset, opt.model, "CDD-11"] if opt.train_scene == 'cdd11' else [opt.train_scene, opt.model]
        logger = WandbLogger(
            name=opt.time_stamp, 
            save_dir=log_dir, 
            config=opt,
            tags=logger_tags,
            log_model="all"
        )
    else:
        logger = TensorBoardLogger(
            save_dir=log_dir,
            default_hp_metric=False
        )

    # 4. 初始化模型（恢复/微调）
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), 
            opt=opt
        )
        rank_zero_info(f"Fine-tuning from checkpoint: {opt.fine_tune_from}")
    else:
        model = PLTrainModel(opt)

    # 5. Checkpoint配置（CDD-11增加subset标识）
    checkpoint_path = os.path.join(opt.ckpt_dir, opt.time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, 
        every_n_epochs=10,  # CDD-11 200 epoch，每10 epoch保存一次
        save_top_k=-1, 
        save_last=True,
        filename="{epoch:03d}-{Train_Loss_epoch:.4f}-{opt.trainset}"
    )

    # 6. 初始化回调
    time_callback = TrainingTimeCallback()
    print_callback = PrintEpochCallback()

    # 7. 加载数据集（不修改数据代码，适配CDD-11）
    rank_zero_info(f"\nLoading Dataset - Type: {opt.trainset} | Scene: {opt.train_scene}")
    if opt.trainset == "standard":
        # Three/Five场景：加载标准数据集（AIOTrainDataset）
        trainset = AIOTrainDataset(opt)
    else:
        # CDD-11场景：加载复合退化数据集（CDD11）
        # 提取subset（all/single/double/triple）
        subset = opt.trainset.replace("CDD11_", "")  
        trainset = CDD11(opt, split="train", subset=subset)
        # CDD-11数据集规模打印（匹配论文：13013训练对）
        rank_zero_info(f"CDD-11 Training Samples: {len(trainset)} (Paper: 13,013 for training)")
    
    # 数据加载器（CDD-11更大patch，调整num_workers）
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.num_workers if opt.train_scene != 'cdd11' else 16  # CDD-11更大分辨率，更多workers
    )
    rank_zero_info(f"Dataset loaded - Total samples: {len(trainset)} | Batch size: {opt.batch_size}")

    # 8. 初始化Trainer（CDD-11适配更大分辨率）
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true" if opt.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback, time_callback, print_callback], 
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        log_every_n_steps=20,  # CDD-11样本更多，减少日志频率
        precision="16-mixed",  # 混合精度训练适配1080×720分辨率
        gradient_clip_val=0.5  # CDD-11复合退化增加梯度裁剪，防止爆炸
    )

    # 9. 恢复训练（可选）
    resume_ckpt_path = None
    if opt.resume_from:
        resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        rank_zero_info(f"Resuming training from: {resume_ckpt_path}")

    # 10. 开始训练
    rank_zero_info(f"\nStarting training for {opt.epochs} epochs...")
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        ckpt_path=resume_ckpt_path
    )

    # 11. 训练总结（CDD-11专属）
    rank_zero_info(f"\n" + "="*60)
    rank_zero_info("Training Summary")
    rank_zero_info("="*60)
    rank_zero_info(f"Scene: {opt.train_scene} | Subset: {opt.trainset} | Model: {opt.model}")
    rank_zero_info(f"Epochs completed: {trainer.current_epoch} | Tasks: {opt.task_num}")
    rank_zero_info(f"Checkpoints saved to: {checkpoint_path}")
    rank_zero_info(f"Logs saved to: {log_dir}")
    if opt.train_scene == 'cdd11':
        rank_zero_info(f"CDD-11 Test Samples (Paper): 2,200 | Test Path: {os.path.join(opt.data_file_dir, 'cdd11/test')}")
    if hasattr(time_callback, 'end_time'):
        total_time = time_callback.end_time - time_callback.start_time
        rank_zero_info(f"Total training time: {total_time/3600:.2f} hours")
    rank_zero_info("="*60)

if __name__ == '__main__':
    # 固定随机种子（CDD-11复现性）
    seed_everything(42, workers=True)
    # 加载训练配置
    train_opt = train_options()
    # 启动训练
    main(train_opt)
from typing import List

import os
import pathlib
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint,Callback
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss

import time


class TimeLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"[Training Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_epoch_start(self, trainer, pl_module):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[Epoch {trainer.current_epoch}] Start Time: {current_time}")

    def on_train_end(self, trainer, pl_module):
        end_time = time.time()
        total_hours = (end_time - self.start_time) / 3600.0
        print(f"[Training End] Total training time: {total_hours:.2f} hours")


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt

        self.net = MoCEIR(
            dim=32,
            num_blocks=[4, 6, 6, 8],
            heads=[1, 2, 4, 8],
            gate_type="elementwise",  # 启用头级门控,elementwise\headwise
            LayerNorm_type='WithBias',
            bias=False
            )
        
    
        self.loss_fn = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)

        loss = self.loss_fn(restored,clean_patch)

        if self.trainer.is_global_zero and batch_idx % 50 == 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            print(f"[Epoch {self.current_epoch:03d}, Step {batch_idx:05d}] Loss: {loss.item():.6f}, LR: {lr:.2e}")
            

        self.log("Train_Loss", loss, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss
        
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        
        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=1,max_epochs=self.opt.epochs)      
        return [optimizer],[scheduler]
                        


def main(opt):
    print("Options")
    print(opt)

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    time_logging_callback = TimeLoggingCallback()
        
    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger  = WandbLogger(name=name, save_dir=log_dir, config=opt) 
        
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=2, save_top_k=-1, save_last=True)
    
    # Create datasets and dataloaders
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
        
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    
    
    # Create trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true",
                         logger=logger,
                         callbacks=[checkpoint_callback,time_logging_callback],
                         accumulate_grad_batches=opt.accum_grad,
                         deterministic=True)
    
    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # Train model
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        ckpt_path=checkpoint_path  # Specify the checkpoint path to resume from
    )
    


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)




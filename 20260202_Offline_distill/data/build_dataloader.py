# build_dataloader.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .sampler_dataset import SamplerDataset
from .sampler_core import build_sample_datasets
from .preprocess_core import TeacherManager
from .ddp_split_dataset import DDPSplitDataset


def build_train_dataloader(config):
    datasets_cfg = build_sample_datasets(config)
    teacher_mgr = TeacherManager(config)

    sampler_dataset = SamplerDataset(
        datasets_cfg=datasets_cfg,
        batch_size=config["sampler_batch_size"],
        config=config,
        teacher_mgr=teacher_mgr,
    )

    imgs_per_dataset = sum(d["imgs"] for d in config["datasets"]["datasets"])

    ddp_dataset = DDPSplitDataset(
        base_dataset=sampler_dataset,
        batch_size=config["batch_size"],
        imgs_per_dataset=imgs_per_dataset,
        ddp_real=config.get("ddp_real", True),
    )

    loader = DataLoader(
        ddp_dataset,
        batch_size=None,
        num_workers=0,
    )
    return loader

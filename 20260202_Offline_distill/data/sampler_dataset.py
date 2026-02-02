# sampler_dataset.py
import torch
from torch.utils.data import IterableDataset

from .sampler_core import sample_one_batch
from .preprocess_core import preprocess_batch


class SamplerDataset(IterableDataset):
    def __init__(self, datasets_cfg, batch_size, config, teacher_mgr):
        self.datasets_cfg = datasets_cfg
        self.batch_size = batch_size
        self.config = config
        self.teacher_mgr = teacher_mgr

    def __iter__(self):
        while True:
            raw_batch = {}
            for dataset in self.datasets_cfg:
                raw_batch[dataset["name"]] = sample_one_batch(
                    dataset, self.batch_size
                )

            batch = preprocess_batch(
                raw_batch, self.config, self.teacher_mgr
            )
            yield batch

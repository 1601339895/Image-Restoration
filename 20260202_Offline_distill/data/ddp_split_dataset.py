# ddp_split_dataset.py
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist


class DDPSplitDataset(IterableDataset):
    """
    等价 dp_split.py
    """

    def __init__(
        self,
        base_dataset,
        batch_size,
        imgs_per_dataset,
        ddp_real=True,
    ):
        """
        Args:
            base_dataset: SamplerDataset
            batch_size: 全局 batch_size
            imgs_per_dataset: sum(dataset["imgs"])
            ddp_real: 是否真实 DDP
        """
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.imgs_per_dataset = imgs_per_dataset
        self.ddp_real = ddp_real

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.new_batch_size = batch_size // self.world_size
        if not ddp_real:
            self.new_batch_size = self.new_batch_size * self.world_size

    def __iter__(self):
        base_iter = iter(self.base_dataset)

        for data in base_iter:
            # data: dict[dataset_name -> batch_dict]
            split_data = {}

            for dname, batch in data.items():
                split_batch = {}

                for k, v in batch.items():
                    if not hasattr(v, "shape"):
                        split_batch[k] = v
                        continue

                    # 原 dp_split reshape 逻辑
                    B = v.shape[0]
                    v = v.reshape(
                        -1,
                        self.new_batch_size * self.imgs_per_dataset,
                        *v.shape[1:]
                    )

                    if self.ddp_real:
                        split_batch[k] = v[self.rank]
                    else:
                        split_batch[k] = v[0]

                split_data[dname] = split_batch

            yield split_data

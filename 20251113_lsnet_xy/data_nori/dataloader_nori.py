import torch
import numpy as np
from typing import List, Tuple, Iterator, Optional
from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
import nori2 as nori  # 假设使用nori2库
from .common import config
import cv2
import refile
import json


class BalancedSampler(Sampler):
    """确保每个batch中正负样本数量各占一半的采样器"""
    def __init__(
        self,
        dataset: List[Tuple[bool, str]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.half_batch = batch_size // 2  # 每个类别在batch中的数量

        # 分离正负样本的索引
        self.positive_indices = [i for i, (label, _) in enumerate(dataset) if label]
        self.negative_indices = [i for i, (label, _) in enumerate(dataset) if not label]

        self.num_pos = len(self.positive_indices)
        self.num_neg = len(self.negative_indices)

        # 校验：batch_size必须为偶数
        if batch_size % 2 != 0:
            raise ValueError("batch_size必须为偶数（正负样本各占一半）")
        # 校验：必须同时存在正负样本
        if self.num_pos == 0 or self.num_neg == 0:
            raise ValueError("数据集必须同时包含正负样本才能使用平衡采样")

    def __iter__(self) -> Iterator[int]:
        # 复制索引列表（避免修改原始列表）
        pos_indices = self.positive_indices.copy()
        neg_indices = self.negative_indices.copy()

        # 打乱索引（若需要）
        if self.shuffle:
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)

        # 计算总批次数量
        total_batches = min(self.num_pos, self.num_neg) // self.half_batch
        if not self.drop_last and (min(self.num_pos, self.num_neg) % self.half_batch != 0):
            total_batches += 1  # 保留最后一个不完整但平衡的批次

        # 生成每个batch的索引
        for b in range(total_batches):
            # 从正负样本中各取half_batch个（支持循环采样）
            pos_start = (b * self.half_batch) % self.num_pos
            neg_start = (b * self.half_batch) % self.num_neg

            # 处理索引越界（循环取数）
            pos_end = pos_start + self.half_batch
            if pos_end <= self.num_pos:
                batch_pos = pos_indices[pos_start:pos_end]
            else:
                batch_pos = pos_indices[pos_start:] + pos_indices[:pos_end % self.num_pos]

            neg_end = neg_start + self.half_batch
            if neg_end <= self.num_neg:
                batch_neg = neg_indices[neg_start:neg_end]
            else:
                batch_neg = neg_indices[neg_start:] + neg_indices[:neg_end % self.num_neg]

            # 合并并打乱批次内的顺序
            batch_indices = batch_pos + batch_neg
            if self.shuffle:
                np.random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return 2 * (min(self.num_pos, self.num_neg) // self.half_batch) * self.half_batch
        else:
            return 2 * ((min(self.num_pos, self.num_neg) + self.half_batch - 1) // self.half_batch) * self.half_batch

class DataloaderNori:
    def __init__(
        self,
        dataset: List[Tuple[bool, str]],  # 输入数据格式：[(label1, noriid1), (label2, noriid2), ...]
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        collate_fn=default_collate,
        drop_last: bool = False,
        resize_size: Tuple[int, int] = (224, 224)  # 统一resize尺寸
    ):
        """
        支持noriid加载和自动resize的自定义数据加载器
        
        Args:
            dataset: 数据列表，每个元素为 (label: bool, noriid: str)
            batch_size: 批次大小
            shuffle: 是否打乱数据顺序
            sampler: 采样器（若指定则忽略shuffle）
            collate_fn: 批次拼接函数
            drop_last: 是否丢弃最后一个不完整批次
            resize_size: 图像统一调整后的尺寸，默认(224, 224)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.resize_size = resize_size
        self.fetcher = nori.Fetcher()  # 初始化nori fetcher
        
        if sampler is not None:
            # 若用户传入sampler，仍优先使用（但建议使用默认平衡采样）
            self.sampler = sampler
        else:
            # 强制使用平衡采样器
            self.sampler = BalancedSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )

    def _load_nori(self, nori_id):
        str_b = self.fetcher.get(nori_id)
        img_raw = cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_UNCHANGED)
        return img_raw


    def _load_and_preprocess(self, noriid: str) -> torch.Tensor:
        """加载nori图像并预处理（resize + 格式转换）"""    
        # 2. 转换为PIL图像（自动处理不同格式）
        img = self._load_nori(noriid)  # 确保为RGB格式
        
        # 3. 统一resize到目标尺寸
        img_resized = cv2.resize(img, self.resize_size)  # 双线性插值
        
        # 4. 转为numpy数组并转为CHW格式（PyTorch要求）
        img_np = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)
        
        # 5. 转为Tensor并归一化（可选，根据需求调整）
        return torch.tensor(img_np) / 255.0  # 归一化到[0,1]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        batch_samples = []
        batch_targets = []

        # 遍历平衡采样器生成的索引
        for idx in self.sampler:
            label, noriid = self.dataset[idx]
            img_tensor = self._load_and_preprocess(noriid)
            batch_samples.append(img_tensor)
            batch_targets.append(torch.tensor(int(label), dtype=torch.long))

            # 达到batch_size则输出
            if len(batch_samples) == self.batch_size:
                # 可根据需要选择使用collate_fn或直接stack
                yield self.collate_fn(batch_samples), self.collate_fn(batch_targets)
                # 若无需collate_fn，也可直接用torch.stack：
                # yield torch.stack(batch_samples), torch.stack(batch_targets)
                batch_samples = []
                batch_targets = []

    def __len__(self) -> int:
        """计算批次数量"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _get_indices(self) -> Iterator[int]:
        """获取数据索引（支持打乱）"""
        if self.shuffle and not hasattr(self, 'sampler'):
            yield from self.indices
        else:
            yield from self.sampler

def get_dirt_dataloader():
    false_data_count = 0
    train_data_dirt = []

    for train_meat in config.train_file_0428_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name:
                false_data_count+=1
                continue
            label = info_dict['level'] != '0'
            train_data_dirt.append((label, dataid))
    print(f'false_data_count:{false_data_count}')
    print(f'use_data_count:')
    count_1 = sum(1 for label, _ in train_data_dirt if label)
    count_0 = len(train_data_dirt) - count_1
    print('True:', count_1, "False:", count_0)

    for train_meat in config.train_file_glass_0527_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name_0527:
                false_data_count+=1
                continue
            label = info_dict['level'] != '0'
            train_data_dirt.append((label, dataid))
    print(f'false_data_count add0527:{false_data_count}')
    print(f'use_data_count add0527:')
    count_1 = sum(1 for label, _ in train_data_dirt if label)
    count_0 = len(train_data_dirt) - count_1
    print('True:', count_1, "False:", count_0)


    # 初始化数据加载器
    data_loader = DataloaderNori(
        dataset=train_data_dirt,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        resize_size=(224, 224)
    )

    return data_loader

# 使用示例
if __name__ == "__main__":
    false_data_count = 0
    train_data_dirt = []

    for train_meat in config.train_file_0428_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name:
                false_data_count+=1
                continue
            label = info_dict['level'] != '0'
            train_data_dirt.append((label, dataid))
    print(f'false_data_count:{false_data_count}')
    print(f'use_data_count:')
    count_1 = sum(1 for label, _ in train_data_dirt if label)
    count_0 = len(train_data_dirt) - count_1
    print('True:', count_1, "False:", count_0)

    for train_meat in config.train_file_glass_0527_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name_0527:
                false_data_count+=1
                continue
            label = info_dict['level'] != '0'
            train_data_dirt.append((label, dataid))
    print(f'false_data_count add0527:{false_data_count}')
    print(f'use_data_count add0527:')
    count_1 = sum(1 for label, _ in train_data_dirt if label)
    count_0 = len(train_data_dirt) - count_1
    print('True:', count_1, "False:", count_0)


    # 初始化数据加载器
    data_loader = DataloaderNori(
        dataset=train_data_dirt,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        resize_size=(224, 224)
    )
    
    # 迭代加载批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        # 统计当前batch的正样本数（label=1的总和）
        pos_in_batch = labels.sum().item()
        # 负样本数 = batch_size - 正样本数
        neg_in_batch = len(labels) - pos_in_batch
        
        # 打印详细信息
        print(f"Batch {batch_idx + 1}:")
        print(f"  样本总数: {len(labels)} (batch_size={data_loader.batch_size})")
        print(f"  正样本数: {pos_in_batch} (占比: {pos_in_batch/len(labels):.0%})")
        print(f"  负样本数: {neg_in_batch} (占比: {neg_in_batch/len(labels):.0%})")
        print(f"  图像形状: {images.shape}\n")
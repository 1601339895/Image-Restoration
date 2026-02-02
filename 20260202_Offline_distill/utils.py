import os
import pickle

import cv2
import nori2 as nori
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from meghair.utils.imgproc import imdecode as imdecode_1
from balls.imgproc import imdecode
from tensorboardX import SummaryWriter as Writer
from loguru import logger

import traceback


def load_config(file):
    with open(file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_config(file, obj):
    with open(file, "w+") as f:
        f.write(yaml.dump(obj))


def get_num_classes(cls_config):
    num_cls = load_config(cls_config)
    return sum(list(num_cls.values()))


def gather_tensor(tensor, trainer):
    tensor_all = [
        torch.ones_like(tensor).to(trainer.local_rank)
        for _ in range(trainer.world_size)
    ]
    dist.all_gather(tensor_all, tensor)
    tensor_all[trainer.rank] = tensor
    tensor_all = torch.cat(tensor_all)
    return tensor_all


class ModelResize(torch.nn.Module):
    def __init__(self, backbone, size=[112, 112]):
        super().__init__()
        self.backbone = backbone
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bicubic", align_corners=False)
        x = self.backbone(x)
        return x


class TensorBoardLogger:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.train_tb = None

    def create(self, log_dir):
        if self.rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.train_tb = Writer(log_dir)

    def write(self, name, data, step):
        if self.rank == 0:
            self.train_tb.add_scalar(name, data, step)


def oss2local(path):
    return path.replace("s3://", "/data/")


nr = nori.Fetcher()

# 全局统计变量
_load_stats = {
    "total_attempts": 0,
    "success_count": 0,
    "old_format_count": 0,
    "new_format_count": 0,
    "error_count": 0
}

def nori2feat(nr_key):
    feat = pickle.loads(nr.get(nr_key))
    return feat


def nori2img(nr_key):
    global _load_stats
    _load_stats["total_attempts"] += 1
    
    blob = nr.get(nr_key, retry=4)
    try:
        # 检查是否是 Pickle 格式
        if isinstance(blob, (bytes, bytearray)) and len(blob) > 0 and blob[0] == 0x80:
            # 老格式：Pickle + imdecode_1
            obj = pickle.loads(blob)
            img = imdecode_1(obj["img"])
            _load_stats["old_format_count"] += 1
            
            # 定期打印格式统计
            if _load_stats["total_attempts"] % 1000 == 0:
                logger.info(f"Image format stats: OLD={_load_stats['old_format_count']}, NEW={_load_stats['new_format_count']}")
        else:
            # 新格式：直接用 imdecode
            img = imdecode(blob, cv2.IMREAD_COLOR)
            _load_stats["new_format_count"] += 1
            
        _load_stats["success_count"] += 1
        
        # 图像质量检查
        if img is not None:
            h, w = img.shape[:2]
            if h < 50 or w < 50:  # 检查是否为异常小图
                if _load_stats["total_attempts"] % 100 == 0:
                    logger.warning(f"Small image detected: {w}x{h} for nr_key={nr_key}")
        
        return img

    except Exception as e:
        _load_stats["error_count"] += 1
        
        # 错误统计和详细日志
        if _load_stats["error_count"] % 10 == 1:  # 每10个错误打印一次
            logger.error(f"[ERROR {_load_stats['error_count']}] nr_key={nr_key}, error={str(e)[:100]}")
            success_rate = _load_stats["success_count"] / _load_stats["total_attempts"] * 100
            logger.info(f"Current success rate: {success_rate:.2f}% ({_load_stats['success_count']}/{_load_stats['total_attempts']})")

        # 同时把失败的 id 写入日志文件
        fail_log_path = '/data/projects/jg-face-unlock-ppl/base/fea/vivo_shufflent_s3_light1/txt/feature_train_fail.txt'
        try:
            with open(fail_log_path, 'a') as f:
                f.write(nr_key + "**" + str(e)[:50] + '\n')
        except:
            pass  # 避免日志写入失败影响主流程

        # 返回 None 表示失败
        return None


def imgpreprocess(img, out_size, aug):
    # 预处理监控
    if img is None:
        return None
        
    original_shape = img.shape
    
    img = aug(image=img)["image"]
    img = cv2.resize(img, tuple(out_size))
    img = img.transpose(2, 0, 1).astype(np.uint8)
    
    # 偶尔打印预处理信息
    if np.random.random() < 0.001:  # 0.1%概率打印
        logger.info(f"Preprocess: {original_shape} -> {tuple(out_size)} -> {img.shape}")
    
    return img


def get_load_stats():
    """获取图像加载统计信息"""
    return _load_stats.copy()


def reset_load_stats():
    """重置图像加载统计"""
    global _load_stats
    _load_stats = {
        "total_attempts": 0,
        "success_count": 0,
        "old_format_count": 0,
        "new_format_count": 0,
        "error_count": 0
    }

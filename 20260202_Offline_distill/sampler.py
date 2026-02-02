import argparse
import os
import pdb
import time
import sys
import yaml
import numpy as np
from dpflow import InputPipe, OutputPipe, control
from loguru import logger
import sampler_method as sampler
from frtrain.misc import utils

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="sampler")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
        help="train config",
    )

    args = parser.parse_args()

    # 加载主配置文件（YAML）
    config = utils.load_config(args.config)
    # 构建 dataset_collection 配置文件的完整路径（该文件列出所有可用数据集的存储路径）
    data_list_config = os.path.join(
        os.path.dirname(__file__), config["datasets"]["dataset_collection"]
    )
    try:
        data_path = utils.load_config(data_list_config)
        logger.info(f"Loaded dataset paths: {list(data_path.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset collection config: {e}")
        data_path = {}  
    
    # 初始化输出管道（用于将采样结果发送给训练进程）
    pipename = config["sampler_pipe_name"]
    out_pipe = OutputPipe(pipename, buffer_size=64)
    logger.info(f"out pipe name: {pipename}")

    # 获取要处理的数据集列表（来自 config["datasets"]["datasets"]）
    datasets = config["datasets"]["datasets"]
    start_label = 0   # 全局起始类别标签（用于不同数据集 label 不重叠）
    total_persons = 0   # 统计总人数
    total_images = 0     # 统计总图像数

    all_sample_datasets = []
    for set_id, dataset in enumerate(datasets):
        try:
            info_file = os.path.join(os.path.dirname(__file__), dataset["path"])
            data_info = utils.load_config(info_file)
        except Exception as e:
            logger.error(f"Failed to load dataset config from {info_file}: {e}")
            continue

        sample_datasets = []   # 存储该组内所有有效子数据集的元信息
        logger.info(f"Processing dataset group: {dataset['name']}")
        
        for idx, d in enumerate(data_info.get("datasets", [])):
            logger.info(f"[{idx} / {len(data_info.get('datasets', []))}]: {d.get('name', 'unknown')}")
            try:

                dataset_name = d["name"]
                # 从 data_path 中获取该数据集的实际存储路径列表
                paths = data_path.get(dataset_name, [])
                logger.info(f"Loading from paths: {paths}")
                
                # 调用 sampler.load_dataset 加载元数据（如 info, nr_keys 等）
                meta = sampler.load_dataset(dataset_name, paths)
                if meta is None:
                    logger.warning(f"Dataset {dataset_name} loaded as None, skipping.")
                    continue
                    
                # 数据格式检查
                info_shape = meta["info"].shape if hasattr(meta["info"], 'shape') else 'unknown'
                logger.info(f"Dataset {dataset_name}: info_shape={info_shape}, #persons={len(meta['info'])}, #images={len(meta['nr_keys'])}")
                
                # 检查是新数据还是旧数据格式
                if hasattr(meta["info"], 'shape') and len(meta["info"].shape) == 3:
                    logger.info(f"Detected NEW data format for {dataset_name}")
                    # 打印前几个person的info
                    logger.info(f"Sample person info: {meta['info'][:2]}")
                else:
                    logger.info(f"Detected OLD data format for {dataset_name}")
                
                # 将元数据附加到配置中，并分配全局 label 范围
                d["meta"] = meta
                d["sample_func"] = d.get("sample_method", "sampler_uniform_image")  # 默认采样策略
                d["start_label"] = start_label
                d["num_class"] = len(meta["info"]) # 该数据集的人数
                d["mask"] = (start_label, start_label + d["num_class"])  # label 范围 [start, end)
                start_label += d["num_class"] # 更新全局起始 label
                total_persons += d["num_class"]
                total_images += len(meta["nr_keys"])
                
                logger.info(f"{dataset_name}: sample_func={d['sample_func']}, labels=[{d['start_label']}-{start_label})")
                sample_datasets.append(d)
                
            except KeyError as e:
                logger.error(f"Missing key in dataset {d.get('name', 'unknown')}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading dataset {d.get('name', 'unknown')}: {e}")
                continue
        
         # 将处理好的子数据集列表存回原配置
        datasets[set_id]["sample_datasets"] = sample_datasets  
        all_sample_datasets.extend(sample_datasets)
        logger.info(f"Dataset group {dataset['name']}: loaded {len(sample_datasets)} sub-datasets")

    # Summary
    logger.info(f"TOTAL SUMMARY: {total_persons} persons, {total_images} images across all datasets")

    # Save num_class.yaml
    # 保存每个数据集的类别数到 num_class.yaml（供后续使用，如损失函数）
    num_class = {}
    for d in all_sample_datasets:
        name = d.get("name")
        if name and "num_class" in d:
            num_class[name] = d["num_class"]
    if num_class:
        save_dir = os.path.dirname(args.config)
        num_class_file = os.path.join(save_dir, "num_class.yaml")
        utils.save_config(num_class_file, num_class)
        logger.info(f"Saved num_class.yaml: {num_class}")
    else:
        logger.warning("No valid datasets loaded. Skipping num_class.yaml save.")

    sampler_batch_size = config.get("sampler_batch_size", 32)
    logger.info(f"Starting sampling loop with batch_size={sampler_batch_size}")
    
    count = 0
    # 使用 dpflow.control 管理管道生命周期（自动 cleanup）
    with control(io=[out_pipe]):
        while True:
            data = {}
            ts = time.time()
            batch_info = {}
            
            for dataset in datasets:
                try:
                    batch_data = sampler.batch_sample(dataset, sampler_batch_size)
                    data[dataset["name"]] = batch_data
                    
                    # 采样统计
                    batch_info[dataset["name"]] = {
                        "nr_keys_count": len(batch_data["nr_keys"]),
                        "labels_range": f"[{batch_data['labels'].min()}-{batch_data['labels'].max()}]" if len(batch_data['labels']) > 0 else "empty",
                        "unique_persons": len(np.unique(batch_data["labels"])) if len(batch_data['labels']) > 0 else 0
                    }
                    
                except Exception as e:
                    logger.error(f"Error in batch sampling for {dataset['name']}: {e}")
                    continue
                    
            te = time.time()
            
            # 详细的batch统计信息
            if count % 10 == 0:  # 每10个batch详细打印一次
                logger.info(f"[Sampler] [{count}] time: {te - ts:.4f}s | Batch details:")
                for dataset_name, info in batch_info.items():
                    logger.info(f"  {dataset_name}: {info['nr_keys_count']} images, {info['unique_persons']} persons, labels {info['labels_range']}")
            else:
                logger.info(f"[Sampler] [{count}] time: {te - ts:.4f}s | Total batches: {sum(info['nr_keys_count'] for info in batch_info.values())}")
            
            out_pipe.put_pyobj(data)
            count += 1

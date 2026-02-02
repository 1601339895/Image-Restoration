# validate_data.py
import pickle
import json
from brainpp.oss import OSSPath
import numpy as np

def validate_dataset(dataset_path):
    """验证单个数据集"""
    print(f"Validating dataset: {dataset_path}")
    
    # 加载info文件
    info_path = f"{dataset_path}/info"
    info_bytes = OSSPath(info_path).read_bytes()
    info = pickle.loads(info_bytes)
    
    # 加载nori_ids文件
    nori_ids_path = f"{dataset_path}/new_align5p.nori_id"
    nori_ids_bytes = OSSPath(nori_ids_path).read_bytes()
    nori_ids = pickle.loads(nori_ids_bytes)
    
    # 加载统计信息
    stats_path = f"{dataset_path}/stats.json"
    stats_bytes = OSSPath(stats_path).read_bytes()
    stats = json.loads(stats_bytes.decode('utf-8'))
    
    print(f"  Dataset: {stats['dataset_name']}")
    print(f"  Persons: {stats['num_persons']}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Info shape: {info.shape}")
    print(f"  Nori IDs length: {len(nori_ids)}")
    
    # 验证数据一致性
    assert len(nori_ids) == stats['total_images'], "Nori IDs count mismatch"
    assert len(info) == stats['num_persons'], "Info persons count mismatch"
    
    # 检查info结构
    total_images_from_info = 0
    for i, person_info in enumerate(info):
        if len(person_info) == 2:  # base-query结构
            (diku_start, diku_end), (jiesuo_start, jiesuo_end) = person_info
            person_images = (diku_end - diku_start) + (jiesuo_end - jiesuo_start)
            total_images_from_info += person_images
            
            if i < 3:  # 显示前3个person的信息
                print(f"  Person {i}: diku({diku_start}-{diku_end}), jiesuo({jiesuo_start}-{jiesuo_end})")
        else:
            print(f"  Warning: Person {i} has unexpected info structure: {person_info}")
    
    print(f"  Total images from info: {total_images_from_info}")
    assert total_images_from_info == stats['total_images'], "Image count mismatch between info and stats"
    
    print("  ✓ Validation passed!")
    return True

def main():
    base_path = "s3://jiigan-faceid/txy/rec/processed_datasets/processed_data"
    
    try:
        oss_base = OSSPath(base_path)
        datasets = [item.name for item in oss_base.list() if item.is_dir()]
        
        print(f"Found {len(datasets)} datasets to validate")
        
        for dataset_name in datasets:
            dataset_path = f"{base_path}/{dataset_name}"
            try:
                validate_dataset(dataset_path)
            except Exception as e:
                print(f"  ❌ Validation failed: {e}")
        
    except Exception as e:
        print(f"Error accessing base path: {e}")

if __name__ == "__main__":
    main()

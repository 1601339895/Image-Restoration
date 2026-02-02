import os
import pickle
from balls.imgproc import imencode
import numpy as np
import cv2
from tqdm import tqdm
import nori2 as nori

import refile
from collections import defaultdict
import json
import yaml
from frtrain.data.nori_compress import CompactNoriIDSet

class FaceRecognitionDatasetProcessor:
    def __init__(self, base_paths, output_base_path):
        """
        人脸识别数据集处理器
        Args:
            base_paths: list of str, 基础路径列表  
            output_base_path: str, 输出路径
        """
        self.base_paths = base_paths
        self.output_base_path = output_base_path
        
    def get_all_datasets(self):
        """获取所有数据集名称"""
        all_datasets = set()
        for base_path in self.base_paths:
            try:
                print(f"Scanning base path: {base_path}")
                # 使用refile来列出目录
                items = refile.smart_listdir(base_path)
                for item in items:
                    item_path = refile.smart_path_join(base_path, item)
                    if refile.smart_isdir(item_path):
                        all_datasets.add(item)
                        print(f"  Found dataset: {item}")
            except Exception as e:
                print(f"Error accessing {base_path}: {e}")
        return sorted(list(all_datasets))
    
    def collect_person_images(self, dataset_path):
        """
        收集一个数据集中所有person的图片
        Returns:
            dict: {person_id: {'diku': [paths], 'jiesuo': [paths]}}
        """
        persons_data = defaultdict(lambda: {'diku': [], 'jiesuo': []})
        
        # 处理diku目录 (base images)
        diku_path = refile.smart_path_join(dataset_path, "diku")
        try:
            if refile.smart_exists(diku_path) and refile.smart_isdir(diku_path):
                print(f"Processing diku directory: {diku_path}")
                person_dirs = refile.smart_listdir(diku_path)
                for person_id in person_dirs:
                    person_dir_path = refile.smart_path_join(diku_path, person_id)
                    if refile.smart_isdir(person_dir_path):
                        try:
                            img_files = refile.smart_listdir(person_dir_path)
                            img_count = 0
                            for img_file in img_files:
                                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    img_path = refile.smart_path_join(person_dir_path, img_file)
                                    persons_data[person_id]['diku'].append(img_path)
                                    img_count += 1
                            if img_count > 0:
                                print(f"  Found person {person_id}: {img_count} diku images")
                        except Exception as e:
                            print(f"  Error processing person {person_id} in diku: {e}")
            else:
                print(f"Diku directory does not exist or not a directory: {diku_path}")
        except Exception as e:
            print(f"Error processing diku in {dataset_path}: {e}")
        
        # 处理jiesuo目录 (query images)
        jiesuo_path = refile.smart_path_join(dataset_path, "jiesuo")
        try:
            if refile.smart_exists(jiesuo_path) and refile.smart_isdir(jiesuo_path):
                print(f"Processing jiesuo directory: {jiesuo_path}")
                person_dirs = refile.smart_listdir(jiesuo_path)
                for person_id in person_dirs:
                    person_dir_path = refile.smart_path_join(jiesuo_path, person_id)
                    if refile.smart_isdir(person_dir_path):
                        try:
                            img_files = refile.smart_listdir(person_dir_path)
                            img_count = 0
                            for img_file in img_files:
                                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    img_path = refile.smart_path_join(person_dir_path, img_file)
                                    persons_data[person_id]['jiesuo'].append(img_path)
                                    img_count += 1
                            if img_count > 0:
                                print(f"  Found person {person_id}: {img_count} jiesuo images")
                        except Exception as e:
                            print(f"  Error processing person {person_id} in jiesuo: {e}")
            else:
                print(f"Jiesuo directory does not exist or not a directory: {jiesuo_path}")
        except Exception as e:
            print(f"Error processing jiesuo in {dataset_path}: {e}")
        
        # 过滤并排序 - 必须同时有diku和jiesuo图片
        filtered_persons = {}
        for person_id, data in persons_data.items():
            if len(data['diku']) > 0 and len(data['jiesuo']) > 0:
                # 排序确保顺序一致
                data['diku'].sort()
                data['jiesuo'].sort()
                filtered_persons[person_id] = data
            else:
                print(f"Warning: Person {person_id} missing diku({len(data['diku'])}) or jiesuo({len(data['jiesuo'])}) images, skipping")
        
        return filtered_persons
    
    def upload_images_to_nori(self, persons_data, dataset_name):
        """
        将图片上传到nori并构造数据结构
        """
        nori_path = refile.smart_path_join(self.output_base_path, "nori_data", f"{dataset_name}.nori")
        print(f"Uploading to: {nori_path}")
        
        # 创建输出目录
        nori_dir = refile.smart_path_join(self.output_base_path, "nori_data")
        if not refile.smart_exists(nori_dir):
            refile.smart_makedirs(nori_dir)
        
        # 修正：先用list收集nori_id，最后转换为CompactNoriIDSet
        all_nori_ids = []
        info_list = []
        labels_list = []
        person_mapping = {}
        
        current_global_idx = 0
        
        # 按person_id排序确保一致性
        sorted_persons = sorted(persons_data.items())
        print(f"Processing {len(sorted_persons)} persons for {dataset_name}")
        
        with nori.remotewriteopen(nori_path) as nr:
            for person_idx, (person_id, data) in enumerate(tqdm(sorted_persons, desc=f"Processing {dataset_name}")):
                person_mapping[person_id] = person_idx
                
                # 上传diku图片 (base)
                diku_start = current_global_idx
                for img_path in data['diku']:
                    try:
                        nori_id = self.upload_single_image(nr, img_path, person_idx, person_id, 'diku')
                        if nori_id:
                            all_nori_ids.append(nori_id)  # 添加到普通list
                            labels_list.append(person_idx)
                            current_global_idx += 1
                    except Exception as e:
                        print(f"Error uploading diku image {img_path}: {e}")
                diku_end = current_global_idx
                
                # 上传jiesuo图片 (query)  
                jiesuo_start = current_global_idx
                for img_path in data['jiesuo']:
                    try:
                        nori_id = self.upload_single_image(nr, img_path, person_idx, person_id, 'jiesuo')
                        if nori_id:
                            all_nori_ids.append(nori_id)  # 添加到普通list
                            labels_list.append(person_idx)
                            current_global_idx += 1
                    except Exception as e:
                        print(f"Error uploading jiesuo image {img_path}: {e}")
                jiesuo_end = current_global_idx
                
                # 构造info格式 - 2D数组，形状为(2, 2)
                person_info = [
                    [diku_start, diku_end],      # base范围
                    [jiesuo_start, jiesuo_end]   # query范围  
                ]
                info_list.append(person_info)
                
                print(f"Person {person_id} (idx={person_idx}): diku[{diku_start}:{diku_end}], jiesuo[{jiesuo_start}:{jiesuo_end}]")
        
        # 修正：最后创建CompactNoriIDSet并extend所有nori_id
        nr_keys = CompactNoriIDSet()
        nr_keys.extend(all_nori_ids)
        
        # 转换为numpy数组
        info_array = np.array(info_list)
        
        # 验证数据一致性
        total_imgs = sum((end-start) for person_info in info_list for start, end in person_info)
        if total_imgs != len(nr_keys):
            print(f"Warning: Inconsistency between info ({total_imgs} images) and nori_ids ({len(nr_keys)} images)")
        
        print(f"Dataset {dataset_name} upload complete:")
        print(f"  - Total persons: {len(sorted_persons)}")  
        print(f"  - Total images: {len(nr_keys)}")
        print(f"  - Info shape: {info_array.shape}")
        
        return {
            'info': info_array,
            'nr_keys': nr_keys,
            'labels': np.array(labels_list),
            'person_mapping': person_mapping,
            'num_persons': len(sorted_persons),
            'total_images': len(nr_keys)
        }
    
    def upload_single_image(self, nr, img_path, person_idx, person_id, img_type):
        """上传单张图片到nori"""
        try:
            # 使用refile读取图片
            with refile.smart_open(img_path, 'rb') as f:
                img_data = f.read()
            
            # 解码验证
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Cannot decode image {img_path}")
                return None
            
            # 编码为.np4格式
            _, encoded = imencode('.np4', img)
            encoded_bytes = np.array(encoded).tobytes()
            
            # 构造extra信息
            extra = {
                "label": person_idx,  # person索引
                "img_types": img_type,  # 'diku' or 'jiesuo' 
                "person_id": person_id,  # 原始person_id
                "original_path": img_path,
            }
            
            # 上传到nori
            nori_id = nr.put(data=encoded_bytes, extra=extra)
            return nori_id
            
        except Exception as e:
            print(f"Error uploading {img_path}: {e}")
            return None
    
    def save_dataset_files(self, dataset_data, dataset_name):
        """保存数据集文件"""
        output_dir = refile.smart_path_join(self.output_base_path, "processed_data", dataset_name)
        
        # 创建输出目录
        if not refile.smart_exists(output_dir):
            refile.smart_makedirs(output_dir)
        
        # 保存info文件
        info_path = refile.smart_path_join(output_dir, "info")
        with refile.smart_open(info_path, 'wb') as f:
            pickle.dump(dataset_data['info'], f)
        print(f"Saved info with shape {dataset_data['info'].shape} to {info_path}")
        
        # 保存nori_ids文件
        nori_ids_path = refile.smart_path_join(output_dir, "new_align5p.nori_id")
        with refile.smart_open(nori_ids_path, 'wb') as f:
            pickle.dump(dataset_data['nr_keys'], f)
        print(f"Saved {len(dataset_data['nr_keys'])} nori_ids to {nori_ids_path}")
        
        # 保存统计信息
        stats = {
            'dataset_name': dataset_name,
            'num_persons': dataset_data['num_persons'],
            'total_images': dataset_data['total_images'],
            'person_mapping': dataset_data['person_mapping'],
            'labels_shape': list(dataset_data['labels'].shape),
            'info_shape': list(dataset_data['info'].shape)
        }
        
        stats_path = refile.smart_path_join(output_dir, "stats.json")
        with refile.smart_open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset {dataset_name} saved to {output_dir}")
        print(f"  - Persons: {dataset_data['num_persons']}")
        print(f"  - Images: {dataset_data['total_images']}")
        print(f"  - Info shape: {dataset_data['info'].shape}")
        
        return output_dir
    
    def generate_config_files(self, processed_datasets):
        """生成配置文件"""
        config_dir = refile.smart_path_join(self.output_base_path, "configs")
        if not refile.smart_exists(config_dir):
            refile.smart_makedirs(config_dir)
        
        # 生成data_list.yaml
        data_list = {}
        for dataset_name, output_path in processed_datasets.items():
            # 使用相对路径
            relative_path = f"../processed_data/{dataset_name}/"
            data_list[dataset_name] = [relative_path]
        
        data_list_path = refile.smart_path_join(config_dir, "data_list.yaml")
        data_list_yaml = yaml.dump(data_list, default_flow_style=False, allow_unicode=True)
        with refile.smart_open(data_list_path, 'w') as f:
            f.write(data_list_yaml)
        
        # 生成cls.yaml
        cls_config = {
            'datasets': []
        }
        
        for dataset_name in processed_datasets.keys():
            cls_config['datasets'].append({
                'name': dataset_name,
                'weight': 1.0,
                'sample_method': 'sampler_uniform_person_bq'
            })
        
        cls_config_path = refile.smart_path_join(config_dir, "cls.yaml")
        cls_yaml = yaml.dump(cls_config, default_flow_style=False, allow_unicode=True)
        with refile.smart_open(cls_config_path, 'w') as f:
            f.write(cls_yaml)
        
        print(f"\nConfig files generated:")
        print(f"  - Data list: {data_list_path}")
        print(f"  - CLS config: {cls_config_path}")
        
        return data_list_path, cls_config_path
    
    def verify_dataset(self, output_path, dataset_name):
        """验证生成的数据集"""
        print(f"Verifying dataset {dataset_name}...")
        
        try:
            # 验证info文件
            info_path = refile.smart_path_join(output_path, "info")
            with refile.smart_open(info_path, 'rb') as f:
                info = pickle.load(f)
            print(f"✓ Info loaded: shape={info.shape}, dtype={info.dtype}")
            
            # 检查info格式
            if len(info.shape) == 3 and info.shape[1:] == (2, 2):
                print(f"✓ Info format correct for sampler_uniform_person_bq")
                sample_person = info[0]
                print(f"  Sample: base[{sample_person[0][0]}:{sample_person[0][1]}], query[{sample_person[1][0]}:{sample_person[1][1]}]")
            else:
                print(f"✗ Info format may be incorrect: expected (N, 2, 2), got {info.shape}")
            
            # 验证nori_ids
            nori_ids_path = refile.smart_path_join(output_path, "new_align5p.nori_id")
            with refile.smart_open(nori_ids_path, 'rb') as f:
                nr_keys = pickle.load(f)
            print(f"✓ Nori IDs loaded: length={len(nr_keys)}, type={type(nr_keys)}")
            
            # 验证数据一致性
            total_images_from_info = sum((end-start) for person_info in info for start, end in person_info)
            if total_images_from_info == len(nr_keys):
                print(f"✓ Data consistency check passed: {total_images_from_info} images")
            else:
                print(f"✗ Data consistency check failed: info={total_images_from_info}, nori_ids={len(nr_keys)}")
                
        except Exception as e:
            print(f"✗ Verification failed: {e}")
    
    def process_all_datasets(self):
        """处理所有数据集"""
        print("Discovering datasets...")
        all_datasets = self.get_all_datasets()
        print(f"Found datasets: {all_datasets}")
        
        if not all_datasets:
            print("No datasets found!")
            return
        
        processed_datasets = {}
        
        for dataset_name in all_datasets:
            print(f"{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # 收集所有路径下该数据集的数据
            all_persons_data = {}
            
            for base_path in self.base_paths:
                dataset_path = refile.smart_path_join(base_path, dataset_name)
                print(f"Checking: {dataset_path}")
                
                if refile.smart_exists(dataset_path) and refile.smart_isdir(dataset_path):
                    persons_data = self.collect_person_images(dataset_path)
                    print(f"Found {len(persons_data)} persons in {dataset_path}")
                    
                    # 合并数据
                    for person_id, data in persons_data.items():
                        if person_id in all_persons_data:
                            all_persons_data[person_id]['diku'].extend(data['diku'])
                            all_persons_data[person_id]['jiesuo'].extend(data['jiesuo'])
                            all_persons_data[person_id]['diku'] = sorted(list(set(all_persons_data[person_id]['diku'])))
                            all_persons_data[person_id]['jiesuo'] = sorted(list(set(all_persons_data[person_id]['jiesuo'])))
                        else:
                            all_persons_data[person_id] = data
                else:
                    print(f"Dataset path {dataset_path} does not exist or not a directory")
            
            if not all_persons_data:
                print(f"No valid data found for dataset {dataset_name}, skipping...")
                continue
            
            print(f"Total valid persons in {dataset_name}: {len(all_persons_data)}")
            
            # 上传并构造数据
            dataset_data = self.upload_images_to_nori(all_persons_data, dataset_name)
            
            if dataset_data['total_images'] == 0:
                print(f"No images uploaded for {dataset_name}, skipping...")
                continue
            
            # 保存数据集文件
            output_path = self.save_dataset_files(dataset_data, dataset_name)
            processed_datasets[dataset_name] = output_path
            
            # 验证数据集
            self.verify_dataset(output_path, dataset_name)
        
        # 生成配置文件
        if processed_datasets:
            data_list_path, cls_config_path = self.generate_config_files(processed_datasets)
            
            print(f"{'='*60}")
            print("PROCESSING COMPLETE!")
            print(f"{'='*60}")
            
        else:
            print("No datasets were processed successfully!")


def main():
    """主函数"""
    base_paths = [
        # "s3://jiigan-faceid/txy/rec/data_train/",
        # "s3://jiigan-faceid/txy/rec/data/"
        's3://jiigan-faceid/txy/rec/richang_new/',
    ]
    
    output_base_path = "s3://jiigan-faceid/txy/rec/vivo_data/"
    
    print("人脸识别数据集处理器")
    print(f"输入路径: {base_paths}")
    print(f"输出路径: {output_base_path}")
    print("-" * 60)
    
    processor = FaceRecognitionDatasetProcessor(base_paths, output_base_path)
    processor.process_all_datasets()


if __name__ == "__main__":
    main()

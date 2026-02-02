# -*- coding: utf-8 -*-
import json
import os
import random
import cv2
import numpy as np
import refile
from tqdm import tqdm
import pickle as pkl
import nori2 as nori
from io import BytesIO
from meghair.train.base import DatasetMinibatch
from neupeak.dataset.meta import EpochDataset, GeneratorDataset
from neupeak.dataset.server import create_servable_dataset

from common import config
import utils_
import aug_light as augmentor 
from balls.imgproc import i01c_to_ic01

# ===================== 可视化相关配置 =====================
VIS_SAVE_DIR = "/data/jg-face-unlock-fea-impove/lmk/celian_lmk/vis/vis_data_new"
VIS_MAX_NUM = 5  
vis_count = 0    

os.makedirs(VIS_SAVE_DIR, exist_ok=True)

def denormalize_gray(img_norm, mean=0.449, std=0.226):
    img_norm = img_norm.squeeze()
    img_vis = (img_norm * std + mean) * 255.0
    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
    return img_vis

def visualize_landmarks(img, landmarks, save_path):
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img_vis.shape[:2]
    landmarks = landmarks.reshape(-1, 2)
    for idx, (x, y) in enumerate(landmarks):
        x_int = int(round(x))
        y_int = int(round(y))
        if 0 <= x_int < w and 0 <= y_int < h:
            cv2.circle(img_vis, (x_int, y_int), 2, (255, 0, 0), -1)
            cv2.putText(img_vis, str(idx), (x_int+2, y_int+2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(save_path, img_vis)
    print(f"可视化图像已保存：{save_path}")

def get(dataset_name):
    global vis_count
    dense_nids_dict = {}
    dataset_weight_dict = {}

    for dataset_block_name, dataset_block in config.train_dataset_dict.items():
        print(f'加载数据集块: {dataset_block_name}')
        dataset_block, dataset_block_w = dataset_block
        for dataset, w in dataset_block:
            print(f'加载子数据集: {dataset}')
            dataset_weight_dict[dataset] = dataset_block_w * w
            json_path = os.path.join(config.train_dataset_dir, dataset) + '_img2label_nori_head.json'
            with refile.smart_open(json_path, 'r') as f:
                data_info_dict = json.load(f)
            for img_nid, label_dict in tqdm(data_info_dict.items(), desc=f'解析{dataset}'):
                if 'ld297_nid' not in label_dict: continue
                atom = [img_nid, label_dict['ld297_nid'], None, None]
                dense_nids_dict.setdefault(dataset, []).append(atom)

    for dataset, items in dense_nids_dict.items():
        if len(items) == 0: raise RuntimeError(f'[ERROR] 数据集{dataset}无ld297样本！')

    dataset_list = list(dataset_weight_dict.keys())
    weight_list = np.array([dataset_weight_dict[d] for d in dataset_list], dtype=np.float32)
    weight_list = list(weight_list / weight_list.sum())

    batch_size = config.train_batch_size

    def generator():
        global vis_count

        def sample_one_batch():
            batch_items = []
            sample_datasets = random.choices(dataset_list, k=batch_size, weights=weight_list)
            for dataset in sample_datasets:
                batch_items.append(random.choice(dense_nids_dict[dataset]))
            return batch_items

        def batch_train_data(batch_items):
            global vis_count
            imgs, ld81s = [], [] 

            for img_id, meta_id, _, _ in batch_items:
                # 判断是否需要可视化
                need_vis = vis_count < VIS_MAX_NUM
                if need_vis:
                    print(f"处理可视化样本 {vis_count+1}/{VIS_MAX_NUM}")

                # 1. 读取图像
                img = utils_.nori2img(img_id)
                if img is None or img.mean() < 10: continue

                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=2)

                # 2. 读取关键点标注
                try:
                    info = pkl.load(BytesIO(nf.get(meta_id)))
                    sample_key = next(iter(info))
                    boxes = info[sample_key]
                except Exception as e:
                    print(f"读取标注失败: {e}，跳过该样本")
                    continue

                gt_box_lds = []
                for box in boxes:
                    if "face_bbox_mot" not in box or 'ld_81' not in box: continue
                    bbox, ld_data = box['face_bbox_mot'], box['ld_81']
                    if bbox[2] * bbox[3] < 400: continue
                    gt_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                    gt_box_lds.append((gt_box, ld_data))

                if len(gt_box_lds) == 0: continue
                gt_box, ld = gt_box_lds[rng.randint(0, len(gt_box_lds))]

                resize_h, resize_w = config.img_shape
                
                # ========================================================
                # 3. 数据增强 (核心修正)
                # ========================================================
                
                # A. 几何变换管道 (透视 + 仿射/侧脸优化 + 眼部遮挡)
                # 注意：这里不再需要手动调用 rot_clockwise_90，因为 affine_crop 已经支持 +/- 60度旋转
                try:
                    img, ld, _ = augmentor.augment_img_pipe(
                        img, gt_box, ld, rng, img_size=(resize_w, resize_h)
                    )
                    # augment_img_pipe 返回的是 (crop_img, new_ld, M)
                    # 其中的 box_area 我们不需要，所以 augment_img_pipe 没返回，如果需要可以在 aug 里改
                    # 但在这里我们的逻辑里暂时用不到 box_area 做判定，或者我们可以假设侧脸增强后不需要再模糊
                    
                except Exception as e:
                    print(f"Augmentation error: {e}")
                    continue
                
                # B. 像素级增强 (颜色/噪声/模糊) - 放在 crop 之后做更高效
                
                # 颜色抖动
                img = augmentor.color_jitter(
                    img, rng, prob=0.5,
                    brightness=0.5, contrast=0.5, saturation=0.5
                )
                
                # 转灰度图 (概率1.0，强制转灰)
                img = augmentor.bgr_to_gray(img, rng, prob=1.0)
                
                # 模糊与噪声 (针对大图/清晰图做退化模拟)
                # 由于已经crop了，这里简单根据概率做增强即可
                img = augmentor.gaussian_blur(img, rng, prob=0.3)
                img = augmentor.gaussian_noise(img, rng, max_std=0.1, prob=0.3)
                
                # YUV窄范围转换
                img = augmentor.yuv_narrow_range(img, rng, prob=0.3)

                # ========================================================
                # 4. 图像预处理（归一化）
                # ========================================================
                # 转单通道 (H, W, 1)
                img_out = img.mean(axis=2, keepdims=True)
                img_out = np.clip(img_out, 0, 255)
                img_out = i01c_to_ic01(img_out, allowed_nr_chl=(3, 1)) # (1, H, W)
                img_out = augmentor.normalize_gray(img_out.astype('float32'))

                # 5. 关键点归一化 (0~1)
                ld_np = np.array(ld, dtype='float32')
                ld_out = (ld_np / np.array([resize_w, resize_h])).flatten()

                # 6. 可视化
                if need_vis:
                    try:
                        img_vis = denormalize_gray(img_out)
                        if len(ld_out) != 162:
                            print(f"关键点维度异常：{len(ld_out)}")
                            vis_count += 1
                            continue
                        ld_reshaped = ld_out.reshape(-1, 2)
                        ld_vis = (ld_reshaped * np.array([resize_w, resize_h])).flatten()
                        save_path = os.path.join(VIS_SAVE_DIR, f"vis_{vis_count+1}.jpg")
                        visualize_landmarks(img_vis, ld_vis, save_path)
                        vis_count += 1
                    except Exception as e:
                        print(f"可视化失败: {e}")
                        vis_count += 1
                        continue

                imgs.append(img_out)
                ld81s.append(ld_out)

            if len(imgs) == 0: return None, None
            return np.array(imgs), np.array(ld81s)

        while True:
            batch_items = sample_one_batch()
            imgs, ld81s = batch_train_data(batch_items)
            if imgs is None: continue
            
            yield DatasetMinibatch(
                image=imgs,
                ld81=ld81s, # 字段名与 train.py 对齐
                check_minibatch_size=False,
            )

    service_name = config.make_service_name(dataset_name)
    dataset = GeneratorDataset(generator)
    dataset = EpochDataset(dataset, config.train_batch_num)
    dataset = create_servable_dataset(
        dataset, service_name, config.train_batch_num, serve_type='combiner'
    )
    return dataset

nf = nori.Fetcher()
rng = np.random.RandomState()


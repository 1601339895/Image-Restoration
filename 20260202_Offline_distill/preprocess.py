import argparse
import os
import pdb
import sys
import time
import hashlib
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import albumentations as A
from dpflow import InputPipe, OutputPipe, control
from loguru import logger

import utils as utils
from frtrain.misc.FastFeat import FastFeat
from frtrain.misc.teacher_pca import TeaFeatPCA
from helmet_aug_big import HelmetOcclusionAug_big

logger.remove()
logger.add(sys.stderr)

# ===== 全局变量 =====
lightweight_id_map = {}
enable_id_mapping = os.environ.get("ENABLE_ID_MAPPING", "1") == "1"
viz_save_count = 0
MAX_VIZ_SAVE = 5

# ===== 工具函数 =====
def build_lightweight_id_map(config):
    id_map = {}
    logger.info("Building lightweight ID map...")
    try:
        for dataset in config["datasets"]["datasets"]:
            if "helmet" in dataset["name"].lower():
                continue
            meta = dataset.get("meta")
            if meta is None or len(meta.get("info", [])) == 0:
                continue
            for idx, info in enumerate(meta["info"]):
                person_id = f"{dataset['name']}_{idx}"
                clean_idx = info[0] if len(info.shape) == 1 else (info[0,0] if info[0,0] < info[0,1] else info[0,1])
                if clean_idx < len(meta["nr_keys"]):
                    id_map[person_id] = meta["nr_keys"][clean_idx]
        logger.info(f"ID map built: {len(id_map)} entries")
        return id_map
    except Exception as e:
        logger.error(f"Build map failed: {e}")
        return {}

def get_teacher_optimal_input(nr_key, dataset_name, raw_img):
    """教师始终看干净图"""
    if "helmet" not in dataset_name.lower() or not enable_id_mapping:
        return raw_img
    try:
        parts = nr_key.split('/')
        pid = f"{dataset_name}_{parts[-2]}" if len(parts) >= 2 else f"{dataset_name}_{hashlib.md5(nr_key.encode()).hexdigest()[:8]}"
        if pid in lightweight_id_map:
            clean_img = utils.nori2img(lightweight_id_map[pid])
            if clean_img is not None and clean_img.shape == raw_img.shape:
                return clean_img
    except Exception as e:
        logger.debug(f"ID mapping fallback for {nr_key}: {e}")
    return raw_img

def build_student_aug(dataset_name,p=0.4):
    """头盔增强"""
    transforms = [HelmetOcclusionAug_big(p=p)]
    return A.Compose(transforms)

def imgpreprocess(img, out_size, aug=None):
    """统一 resize + 通道变换"""
    if isinstance(out_size, (list, tuple)):
        out_size = tuple(int(x) for x in out_size)
    elif isinstance(out_size, int):
        out_size = (out_size, out_size)
    else:
        logger.error(f"Invalid out_size type: {type(out_size)}, value: {out_size}")
        out_size = (128, 128)
    if img is None:
        return np.zeros((3, out_size[0], out_size[1]), dtype=np.uint8), "empty"
    try:
        mode_name = "none"
        if aug is not None:
            img = aug(image=img)["image"]
            mode_name = "aug"
        if img.shape[:2] != out_size:
            dsize = (out_size[1], out_size[0])
            img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.transpose(2,0,1)
        return np.clip(img, 0, 255).astype(np.uint8), mode_name
    except Exception as e:
        logger.error(f"Img preprocess failed: {e}")
        return np.zeros((3, out_size[0], out_size[1]), dtype=np.uint8), "error"

def visualize_augmentation(img, dataset_name, count, prefix="student"):
    global viz_save_count
    if viz_save_count >= MAX_VIZ_SAVE:
        return
    try:
        os.makedirs("aug_viz", exist_ok=True)
        img = img.transpose(1,2,0) if img.shape[0]==3 else img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2]==3 else img
        filename = f"aug_viz/{prefix}_{dataset_name}_{count}_{viz_save_count}.jpg"
        cv2.imwrite(filename, img)
        viz_save_count += 1
        logger.info(f"Saved visualization ({viz_save_count}/{MAX_VIZ_SAVE}): {filename}")
    except Exception as e:
        logger.error(f"Visualization save failed: {e}")


def save_batch_images(images, dataset_name, batch_id, prefix=""):
    """
    保存一个 batch 的所有图像
    images: np.ndarray of shape (B, C, H, W), uint8
    """
    if images.size == 0:
        return
    os.makedirs("Chinese_hard_batch_dump", exist_ok=True)
    B = images.shape[0]
    for i in range(B):
        img = images[i]  # (C, H, W)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # (H, W, C)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[0] == 1:
            img = img[0]  # (H, W)
        else:
            logger.warning(f"Unexpected channel dim: {img.shape}")
            continue

        filename = f"Chinese_hard_batch_dump/{prefix}_{dataset_name}_batch{batch_id:05d}_idx{i:03d}.jpg"
        try:
            cv2.imwrite(filename, img)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
    logger.info(f"Saved {B} images for {prefix} of {dataset_name} (batch {batch_id})")


# ===== 主流程 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", default=os.path.join(os.path.dirname(__file__), "config/config.yaml"))
    args = parser.parse_args()
    config = utils.load_config(args.config)

    if enable_id_mapping:
        lightweight_id_map = build_lightweight_id_map(config)

    input_pipe = InputPipe(config["sampler_pipe_name"], buffer_size=16)
    output_pipe = OutputPipe(config["provider_pipe_name"], buffer_size=64)
    tea_aug = A.Compose([])

    # ===== 加载教师 =====
    ff = {}
    for tea in config["teachers"]:
        try:
            ff[tea["name"]] = FastFeat(
                tea["model_path"], device_id=0, torch_model="pt" in tea["model_path"]
            )
            logger.info(f"Loaded teacher: {tea['name']}")

        except Exception as e:
            logger.error(f"Failed to load teacher {tea['name']}: {e}")
            raise e
        

    teafeatpca = {}
    for tea in config["teachers"]:
        if "pca_param" in tea:
            try: 
                teafeatpca[tea["name"]] = TeaFeatPCA(tea["pca_param"], tea["pca_dim"])
                logger.info(f"Loaded PCA for {tea['name']}")
            except Exception as e:
                logger.warning(f"Failed to load PCA for {tea['name']}: {e}")
    if "teachers_pca" in config:
        try: 
            teafeatpca["ensemble"] = TeaFeatPCA(config["teachers_pca"]["pca_param"], config["teachers_pca"]["pca_dim"])
            logger.info("Loaded ensemble PCA")
        except Exception as e: 
            logger.warning(f"Failed to load ensemble PCA: {e}")


    count = 0
    
    with control(io=[input_pipe, output_pipe]), ThreadPool(4) as pool:
        while True:
            count += 1
            try:
                data = input_pipe.get()
                logger.info(f"Preprocessing batch {count}, datasets: {list(data.keys())}")
               
                for dataset_name in data:
                    sub_dataset_names = data[dataset_name]["dataset_names"]  # 每个采样的数据集名称
                    sub_dataset_names = list(set(sub_dataset_names))   # 数据集名称去重，否则就是[batch_size大小的数据集名称列表]，比如：sub_dataset_names=['helmet_new_20251204_align5p', 'Foreigners_new', 'Chinese_hard']
                    
                    nr_keys = data[dataset_name]["nr_keys"]
                    raw_imgs = list(pool.map(utils.nori2img, nr_keys))
                    logger.info(f"Loaded {len(raw_imgs)} images for {dataset_name} (success: {sum(1 for img in raw_imgs if img is not None)})")

                    # 教师输入 (干净)
                    tea_inputs = [get_teacher_optimal_input(k, dataset_name, img if img is not None else np.zeros((128,128,3))) 
                                  for k,img in zip(nr_keys, raw_imgs)]
                    
                    # 判断增强概率
                    p = 0.0
                    if any(name.startswith("Chinese") for name in sub_dataset_names):
                        p = 0.1
                    elif any(name.startswith("Foreigners") for name in sub_dataset_names):
                        p = 0.5
                    # 其他情况： 不增强

                    # 构建增强器（仅当 p > 0）
                    stu_aug = build_student_aug(dataset_name, p=p) if p > 0 else None

                    images_clean, images_helmet = [], []
                    target_size = tuple(config["input_size"])
                    for img in raw_imgs:
                        img = img if img is not None else np.zeros((128, 128, 3))
                        c, _ = imgpreprocess(img, target_size, aug=None)
                        h, _ = imgpreprocess(img, target_size, aug=stu_aug)  # stu_aug 为 None 时无增强
                        images_clean.append(c)
                        images_helmet.append(h)
                    
                    data[dataset_name]["images_clean:cls"] = np.array(images_clean)
                    data[dataset_name]["images_helmet:cls"] = np.array(images_helmet)

                    # 可视化
                    if count % 100 == 0 and len(images_helmet)>0:
                        visualize_augmentation(images_helmet[0], dataset_name, count, prefix="helmet")
                        visualize_augmentation(images_clean[0], dataset_name, count, prefix="clean")


                    # 教师特征
                    tea_feats = []
                    for tea in config["teachers"]:
                        tea_name = tea["name"]
                        if tea_name not in ff:
                            logger.warning(f"Teacher {tea_name} not loaded, skip")
                            continue

                        tea_sz = tuple(tea["input_size"])

                        # 教师看干净图
                        tea_imgs_processed = []
                        for img in tea_inputs:
                            img_proc, _ = imgpreprocess(img, tea_sz, tea_aug)
                            tea_imgs_processed.append(img_proc)
                        tea_imgs_processed = np.array(tea_imgs_processed)
                        
                        # 特征提取 + NaN处理
                        feat = ff[tea_name].batchimg2feat(tea_imgs_processed, bs=128)
                        nan_count = np.isnan(feat).sum()
                        if nan_count > 0:
                            logger.warning(f"Teacher {tea_name} has {nan_count} NaN features")
                        feat = np.nan_to_num(feat)
                        
                        # PCA处理
                        if tea_name in teafeatpca:
                            try:
                                feat = teafeatpca[tea_name].make_renorm(feat)
                            except Exception as e:
                                logger.error(f"PCA failed for {tea_name}: {e}")
                        
                        data[dataset_name][f"feat_{tea_name}"] = feat
                        tea_feats.append(feat)
                    
                    if tea_feats:
                        final_feat = np.concatenate(tea_feats, axis=1)
                        if "ensemble" in teafeatpca:
                            try: 
                                final_feat = teafeatpca["ensemble"].make_renorm(final_feat)
                            except Exception as e:
                                logger.error(f"Ensemble PCA failed: {e}")

                        data[dataset_name]["feat_tea:cls"] = np.nan_to_num(final_feat)
                        logger.info(f"Final teacher feature shape for {dataset_name}: {data[dataset_name]['feat_tea'].shape}")
                    else:
                        logger.warning(f"No valid teacher features for {dataset_name}")


                output_pipe.put_pyobj(data)
                if count % 10 == 0:
                    logger.info(f"Preprocess batch {count} done")
            except Exception as e:
                logger.error(f"Preprocess batch {count} failed: {e}", exc_info=True)

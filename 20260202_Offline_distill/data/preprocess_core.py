# preprocess_core.py
import os
import cv2
import numpy as np
import hashlib
from loguru import logger
import albumentations as A

import nori2 as nori
from frtrain.misc.FastFeat import FastFeat
from frtrain.misc.teacher_pca import TeaFeatPCA
from helmet_aug_big import HelmetOcclusionAug_big
import utils


# ---------------- 基础工具 ---------------- #

_nf = nori.Fetcher()


def nori2img(nori_id):
    try:
        data = _nf.get(nori_id)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imgpreprocess(img, out_size, aug=None):
    if img is None:
        return np.zeros((3, out_size[0], out_size[1]), dtype=np.uint8)

    if aug is not None:
        img = aug(image=img)["image"]

    if img.shape[:2] != out_size:
        img = cv2.resize(img, (out_size[1], out_size[0]))

    img = img.transpose(2, 0, 1)
    return img.astype(np.uint8)


def build_student_aug(p):
    return A.Compose([HelmetOcclusionAug_big(p=p)])


# ---------------- Teacher 管理 ---------------- #

class TeacherManager:
    def __init__(self, config):
        self.ff = {}
        self.pca = {}

        for tea in config["teachers"]:
            self.ff[tea["name"]] = FastFeat(
                tea["model_path"], device_id=0, torch_model="pt" in tea["model_path"]
            )

            if "pca_param" in tea:
                self.pca[tea["name"]] = TeaFeatPCA(
                    tea["pca_param"], tea["pca_dim"]
                )

        if "teachers_pca" in config:
            self.pca["ensemble"] = TeaFeatPCA(
                config["teachers_pca"]["pca_param"],
                config["teachers_pca"]["pca_dim"],
            )

    def extract(self, imgs, config):
        feats = []
        for tea in config["teachers"]:
            name = tea["name"]
            ff = self.ff[name]

            sz = tuple(tea["input_size"])
            tea_imgs = np.stack([imgpreprocess(i, sz) for i in imgs])
            feat = ff.batchimg2feat(tea_imgs, bs=128)
            feat = np.nan_to_num(feat)

            if name in self.pca:
                feat = self.pca[name].make_renorm(feat)

            feats.append(feat)

        out = np.concatenate(feats, axis=1)
        if "ensemble" in self.pca:
            out = self.pca["ensemble"].make_renorm(out)

        return np.nan_to_num(out)


# ---------------- 主 preprocess ---------------- #

def preprocess_batch(batch, config, teacher_mgr):
    output = {}

    for dataset_name, data in batch.items():
        nr_keys = data["nr_keys"]
        labels = data["labels"]
        dataset_names = data["dataset_names"]

        raw_imgs = [nori2img(k) for k in nr_keys]

        # 增强概率策略（完全一致）
        p = 0.0
        if any(n.startswith("Chinese") for n in dataset_names):
            p = 0.1
        elif any(n.startswith("Foreigners") for n in dataset_names):
            p = 0.5

        stu_aug = build_student_aug(p) if p > 0 else None
        out_size = tuple(config["input_size"])

        images_clean = []
        images_helmet = []

        for img in raw_imgs:
            img = img if img is not None else np.zeros((128, 128, 3), np.uint8)
            images_clean.append(imgpreprocess(img, out_size))
            images_helmet.append(imgpreprocess(img, out_size, stu_aug))

        feat_tea = teacher_mgr.extract(raw_imgs, config)

        output[dataset_name] = {
            "images_clean:cls": np.stack(images_clean),
            "images_helmet:cls": np.stack(images_helmet),
            "labels": labels,
            "feat_tea:cls": feat_tea,
            "masks": data["masks"],
        }

    return output

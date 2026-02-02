import json
import os
import random
from collections import OrderedDict
import copy

import cv2
import numpy as np
from brainpp.oss import OSSPath
from IPython import embed
from meghair.train.base import DatasetMinibatch
from neupeak.dataset.meta import EpochDataset, GeneratorDataset
from neupeak.dataset.server import create_servable_dataset
import sys
import pickle as pkl
import json
import refile

import utils
from common import config

debug = False



def get(dataset_name):
    false_data_count = 0
    sift_keys = [('level', lambda x: x == '0'), ('level', lambda x: x != '0')]
    batch_ratio = [config.train_batch_size//2, config.train_batch_size//2]
    train_data_dirt = [dict() for sift_key in sift_keys]

    for train_meat in config.train_file_0428_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name:
                false_data_count+=1
                continue
            for id, sift_key in enumerate(sift_keys):
                key, sift = sift_key
                if sift(info_dict[key]):
                    train_data_dirt[id][dataid] = info_dict
    print(f'false_data_count:{false_data_count}')
    print(f'use_data_count:')
    for train_data_d in train_data_dirt:
        print('--------------', len(list(train_data_d.keys())))

    for train_meat in config.train_file_glass_0527_dirt:
        with refile.smart_open(train_meat, 'r') as f:
            info_result = json.load(f) 
        for dataid, info_dict in info_result.items():
            img_name = info_dict['img_path'].split('/')[-1]
            if img_name in config.false_datas_name_0527:
                false_data_count+=1
                continue
            for id, sift_key in enumerate(sift_keys):
                key, sift = sift_key
                if sift(info_dict[key]):
                    train_data_dirt[id][dataid] = info_dict
    print(f'false_data_count add0527:{false_data_count}')
    print(f'use_data_count add0527:')
    for train_data_d in train_data_dirt:
        print('--------------', len(list(train_data_d.keys())))

    def generator():
        def random_extract_train_data(train_datas, batch_ratios, train_data_item):
            for train_data, batch_ratio in zip(train_datas, batch_ratios):
                train_data_item.extend(random.sample(list(train_data.values()), batch_ratio))

        def batch_train_data(train_data_item, lock_batch, img_out_batch):
            for info_dict in train_data_item:
                img_raw = utils.load_nori(info_dict['nori_id'])
                img_raw = cv2.resize(img_raw, config.img_shape)
                img_label = not info_dict['level']=='0'
                lock_batch.append(img_label)
                img_out_batch.append(img_raw)

        rng = utils.get_rng()

        # 分片均匀的从四批数据中提取数据
        while True:
            train_data_item = []
            random_extract_train_data(train_data_dirt, batch_ratio, train_data_item)
            print('------batch infos num', len(train_data_item))

            lock_batch, img_out_batch = [], []
            batch_train_data(train_data_item, lock_batch, img_out_batch)
            print('------batch imgs num:', len(img_out_batch))

            print('------single batch split shape', img_out_batch[-1].shape)
            img_batch_np = np.array(img_out_batch).transpose(0,3,1,2)
            print('------final batch shape:', img_batch_np.shape)
            yield DatasetMinibatch(
                image=img_batch_np,  # face:112*112
                label=np.float32(lock_batch),
                check_minibatch_size=False,
            )

    # 初始化dataset
    service_name = config.make_service_name('train')
    dataset = GeneratorDataset(generator)
    dataset = EpochDataset(dataset, config.train_batch_num)
    dataset = create_servable_dataset(
        dataset, service_name,
        config.train_batch_num,
        serve_type='combiner'
    )
    return dataset

import math
import os

import numpy as np
import refile


class Config:
    def __init__(self):
        # ==========sift_log==========
        self.sift_log = 's3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/250511_siftlog.txt'
        logs = refile.smart_open(self.sift_log, 'r').readlines()
        self.false_datas_name = [a.strip().split(',')[0].split('\\')[-1] for a in logs if int(a.strip().split(',')[-1])!=1]
        
        self.sift_logs_0527_dir = 's3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/250527_siftlogs/'
        self.false_datas_name_0527 = []
        for file in refile.smart_listdir(self.sift_logs_0527_dir):
            logs = refile.smart_open(os.path.join(self.sift_logs_0527_dir, file), 'r').readlines()
            self.false_datas_name_0527.extend([a.strip().split(',')[0].split('\\')[-1] for a in logs if int(a.strip().split(',')[-1])!=1])
        print(f'sift false_datas: {len(self.false_datas_name)}')
        print(f'sift false_datas 0527: {len(self.false_datas_name_0527)}')

        # ==========train dataset==========
        self.train_file_0428_dirt = [
            # 5691
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_train_0422.json",
            # 2871
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_train_0423.json",
            # 6572
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_train_0424.json",
            # 2140
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_train_0425A.json",
            # 4445
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_train_0428.json",
        ]
        self.train_file_0428_occ = [
            # 5026
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_train_0422.json",
            # 6589
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_train_0423.json",
            # 7473
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_train_0424.json",
            # 4293
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_train_0425A.json",
            # 5299
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_train_0428.json",
        ]

        self.train_file_glass_0527_dirt = [
            # 3823
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0528.json",
            # 3804
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0529.json",
            # 5205
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0530.json",
            # 1727
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0603.json",
            # 1050
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0606.json",
            # 2609
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0607.json",
            # 1900
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_train_0608.json",
        ]
        self.train_file_glass_0527_occ = [
            # 4373
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0528.json",
            # 5733
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0529.json",
            # 4810
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0530.json",
            # 2824
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0603.json",
            # 1197
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0606.json",
            # 3416
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0607.json",
            # 2532
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_occ_train_0608.json",
        ]

        self.nori_file_0428_dirt = [
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_dirt_train_0422.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_dirt_train_0423.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_dirt_train_0424.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_dirt_train_0425A.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_dirt_train_0428.nori",
        ]
        self.nori_file_0428_occ = [
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_occ_train_0422.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_occ_train_0423.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_occ_train_0424.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_occ_train_0425A.nori",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/nori/250421_occ_train_0428.nori",
        ]
        self.nori_file_glass_0527 = "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/nori/"

config = Config()

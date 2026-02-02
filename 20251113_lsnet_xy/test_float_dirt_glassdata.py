import collections
import json
import os
import sys
import re

import cv2
import glob
import numpy as np
from tqdm import tqdm
from natsort import ns, natsorted
import json

import torch
import torch.nn.functional as F

from test_utils import setup_logger, fallback_nori_fetcher
from loguru import logger

from data_nori.common import config
import refile
from thop import profile

class LoadAndDumpModel():
    """Functions for loading or dumping a model"""

    def __init__(self):
        pass

    def get_test_func(self, model, model_path, dump_flag=False):
        print("Loading model from {}".format(model_path))
        print(f"Creating model: {args.model}")
        checkpoint = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
        model.eval()
        model.to("cuda:0")

        return model



class LoadTestData():
    """Functions for loading test dataset"""

    def __init__(self):
        self.dirt_val_path = [
            # True:1585, False:269
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_test_0423.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_dirt_test_0425A.json",
        ]
        self.occ_val_path = [
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_test_0423.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250421_traindata_unzip/anno/250421_occ_test_0425A.json",
        ]
        self.dirt_val_glass_path = [
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0528.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0529.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0530.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0603.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0606.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0607.json",
            "s3://jiigan-dirt-occ/dataset/train_format_data/250527_traindata_unzip/anno/250527_dirt_test_0608.json",
        ]
        self.dirt_QA_path = [
            "s3://jiigan-dirt-occ/dataset/test/250612_QA_data_format/anno/250612_dirt_QAtest.json",
        ]
        self.fallback_nori_fetcher = fallback_nori_fetcher

    def unpack_img(self, nori_id):
        r = pkl.loads(self.fallback_nori_fetcher.get(nori_id))
        img = cv2.imdecode(np.frombuffer(r.pop('img'), np.uint8), 0)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 16, cv2.BORDER_CONSTANT, value=0)  # (320, 256，3)
        return img
    
    def load_nori(self, nori_id):
        str_b = self.fallback_nori_fetcher.get(nori_id)
        img_raw = cv2.imdecode(np.frombuffer(str_b, np.uint8), cv2.IMREAD_UNCHANGED)
        return img_raw


    def get_s3_data(self, s3_path, false_datas_name=[], subdir=None):
        img_batch_t = []
        gt_batch = []
        name_batch = []

        false_data_count = 0
        s3_data_dict = dict()
        for train_meat in s3_path:
            f = refile.smart_open(train_meat, 'r')
            info_result = json.load(f) 
            with refile.smart_open(train_meat, 'r') as f:
                info_result = json.load(f) 
            
            for dataid, info_dict in info_result.items():
                img_name = info_dict['img_path'].split('/')[-1]
                if img_name in false_datas_name:
                    false_data_count+=1
                    continue
                s3_data_dict[dataid] = info_dict
        print(f'skip false data: {false_data_count}')

        inverse_flag = False
        if subdir:
            if '反' in subdir:
                subdir = subdir.replace('反', '')
                inverse_flag = True
        count = 0
        for data_id, info_dict in s3_data_dict.items():
        # for data_id, info_dict in tqdm(s3_data_dict.items()):
            # count+=1
            # if count>100:
            #     break
            img_name = info_dict['img_path'].split('/')[-1]
            if subdir:
                if inverse_flag:
                    if bool(re.compile(subdir).search(img_name)):
                        # if subdir not in ori_infos:
                        continue
                else:
                    if not bool(re.compile(subdir).search(img_name)):
                        # if subdir not in ori_infos:
                        continue
            ori_img = self.load_nori(data_id)[...,:3]
            ori_img = cv2.resize(ori_img, (224,224))
            label = not info_dict["level"]=="0"
            # print(subdir)
            # print(img_name)

            gt_batch.append(label)
            name_batch.append(img_name)
            img_batch_t.append(ori_img)
        img_batch_t = np.array(img_batch_t).transpose(0,3,1,2)
        gt_batch_t = np.array(gt_batch)
        return img_batch_t, gt_batch_t, name_batch

    def get_local_data(self, QA_folder, label_def=None, data_flag=None, subdir=None):
        img_batch_t = []
        gt_batch = []
        name_batch = []
        box_xy_batch = []
        box_wh_batch = []
        img_shape = (112, 112)
        label_dict = {"no": 0, "watch": 1}
        rotate_angel = {'up': 1, 'left': 0.25, 'down': 0.5, 'right': 0.75}
        box_json = json.load(open(os.path.join(QA_folder, 'NIO_box0112_infos.json'), 'r'))
        ori_path = QA_folder.replace('face_data', 'png_data') if '_LMR' in QA_folder else QA_folder.replace('QVGA', 'raw_data')
        count = 0
        for root, dirs_name, files_name in os.walk(ori_path):
            for file_name in files_name:
                if ".png" in file_name or '.jpg' in file_name:
                    img_path = os.path.join(root, file_name)
                    sub_path = img_path.replace(ori_path, '')
                    try:
                        box = box_json[sub_path]
                    except:
                        # print(file_name)
                        continue
                    direction = self.rotate_angel_ori[int(box["label"][0])]

                    if subdir:
                        if not bool(re.compile(subdir).search(root)):
                            continue
                    # if "no_gaze" not in root:
                    #     continue
                    if label_def:
                        label = label_def
                    else:
                        label = "watch" if "watch" in root else "no"

                    ori_img = cv2.imread(img_path, 0)
                    ori_img = cv2.rotate(ori_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_out_112, xc, yc, wc, hc = self.crop_test_face(ori_img, box)
                    img_out_112 = img_out_112.astype(int)
                    img_out_112 -= 128

                    gt_batch.append(label_dict[label])
                    name_batch.append(img_path)
                    img_batch_t.append(img_out_112)
                    box_xy_batch.append(np.append(np.array([xc, yc]), [direction] * 14))
                    box_wh_batch.append(np.append(np.array([wc, hc]), [direction] * 14))
        img_batch_t = np.array(img_batch_t).reshape(-1, 1, img_shape[1], img_shape[0])
        gt_batch_t = np.array(gt_batch)
        box_xy_batch_t = np.array(box_xy_batch)
        box_wh_batch_t = np.array(box_wh_batch)
        return img_batch_t, gt_batch_t, box_xy_batch_t, box_wh_batch_t, name_batch

    def get_one_dataset(self, dataset_flag):
        if "val_dirt" in dataset_flag:
            print("================= {} =================".format(dataset_flag))
            subdir = dataset_flag.split(':')[-1] if len(dataset_flag.split(':')) > 1 else None
            img_batch, gt_batch, name_batch = self.get_s3_data(self.dirt_val_path, config.false_datas_name, subdir)
        elif 'val_occ' in dataset_flag:
            print("================= {} =================".format(dataset_flag))
            subdir = dataset_flag.split(':')[-1] if len(dataset_flag.split(':')) > 1 else None
            img_batch, gt_batch, name_batch = self.get_s3_data(self.occ_val_path, subdir)
        elif 'glass_dirt' in dataset_flag:
            print("================= {} =================".format(dataset_flag))
            subdir = dataset_flag.split(':')[-1] if len(dataset_flag.split(':')) > 1 else None
            img_batch, gt_batch, name_batch = self.get_s3_data(self.dirt_val_glass_path, config.false_datas_name_0527, subdir)
        elif 'QA_dirt' in dataset_flag:
            print("================= {} =================".format(dataset_flag))
            subdir = dataset_flag.split(':')[-1] if len(dataset_flag.split(':')) > 1 else None
            img_batch, gt_batch, name_batch = self.get_s3_data(self.dirt_QA_path, [], subdir)
        print("load imgs:{}, {}".format(img_batch.shape, gt_batch.shape))
        print('True:{}, False:{}'.format(gt_batch.sum(), gt_batch.shape[0] - gt_batch.sum()))
        return img_batch, gt_batch, name_batch


class CalResult():
    """Functions for calculating indicators"""

    def __init__(self, model_paths, model, threshold_list, dataset_flags, dump_flag=False, more_infos_flag=False):
        self.model_paths = model_paths
        self.model = model
        self.model_tool = LoadAndDumpModel()
        self.data_tool = LoadTestData()
        self.threshold_list = threshold_list
        self.dataset_flags = dataset_flags
        self.dump_flag = dump_flag
        self.more_infos_flag = more_infos_flag

    def cal_result(self, score, gt_batch, threshold_list):
        total_dict = {}
        th_Recall = {}
        th_Accuracy = {}
        th_Precision = {}
        th_mean_Recall = {}
        th_F_Recall = {}
        # th_Fpr = {}
        th_TP = {}
        th_TN = {}
        th_FP = {}
        th_FN = {}
        th_FN_badcase, th_FP_badcase = {}, {}
        th_FN_badcase_score, th_FP_badcase_score = {}, {}
        th_start = threshold_list[0]
        th_end = threshold_list[1]
        th_step = threshold_list[2]
        for th in range(th_start, th_end, th_step):
            th /= 100
            th_T_P, th_T_N, th_F_P, th_F_N = 0, 0, 0, 0
            F_P_img_list, F_N_img_list = [], []
            F_P_img_score_list, F_N_img_score_list = [], []
            gt = (gt_batch == 1)
            rec = (score > th)
            # rec = np.squeeze(rec, 1)
            T_P = (gt & rec)
            F_N = (gt & (1 - rec))
            T_N = ((1 - gt) & (1 - rec))
            F_P = ((1 - gt) & (rec))
            T_P_val = sum(T_P)
            F_N_val = sum(F_N)
            T_N_val = sum(T_N)
            F_P_val = sum(F_P)
            th_T_P += T_P_val
            th_T_N += T_N_val
            th_F_P += F_P_val
            th_F_N += F_N_val
            for i in range(len(F_P)):
                if F_P[i]:
                    F_P_img_list.append(i)
                    F_P_img_score_list.append(score[i])
            for i in range(len(F_N)):
                if F_N[i]:
                    F_N_img_list.append(i)
                    F_N_img_score_list.append(score[i])
            Recall = float(T_P_val / (T_P_val + F_N_val + 1e-5))
            Precision = float(T_P_val / (T_P_val + F_P_val + 1e-5))
            Accuracy = float(T_P_val + T_N_val) / (T_P_val + T_N_val + F_P_val + F_N_val + 1e-5)
            F_Recall = float(T_N_val / (F_P_val + T_N_val + 1e-5))
            mean_Recall = float((Recall + F_Recall) / 2)
            # Fpr = float(F_P_val) / (T_N_val + F_P_val + 1e-5)
            th_TP[str(th)] = th_T_P
            th_TN[str(th)] = th_T_N
            th_FP[str(th)] = th_F_P
            th_FN[str(th)] = th_F_N
            th_Recall[str(th)] = Recall
            th_Accuracy[str(th)] = Accuracy
            th_Precision[str(th)] = Precision
            th_mean_Recall[str(th)] = mean_Recall
            th_F_Recall[str(th)] = F_Recall
            # th_Fpr[str(th)] = Fpr
            th_FP_badcase[str(th)] = F_P_img_list
            th_FN_badcase[str(th)] = F_N_img_list
            th_FP_badcase_score[str(th)] = F_P_img_score_list
            th_FN_badcase_score[str(th)] = F_N_img_score_list
        total_dict["TP"] = th_TP
        total_dict["TN"] = th_TN
        total_dict["FP"] = th_FP
        total_dict["FN"] = th_FN
        total_dict["Recall"] = th_Recall
        total_dict["Accuracy"] = th_Accuracy
        total_dict["Precision"] = th_Precision
        total_dict["mean_Recall"] = th_mean_Recall
        total_dict["F_Recall"] = th_F_Recall
        # total_dict["th_Fpr"] = th_Fpr
        # total_dict["th_FP_badcase"] = th_FP_badcase
        # total_dict["th_FN_badcase"] = th_FN_badcase
        # total_dict["th_FP_badcase_score"] = th_FP_badcase_score
        # total_dict["th_FN_badcase_score"] = th_FN_badcase_score
        return total_dict

    def get_scores(self, img_batch, model):
        batch_size = 64

        if img_batch.shape[0] <= batch_size:
            with torch.no_grad():
                img_batch = torch.tensor(img_batch).float().to("cuda:0") / 255.0
                score = F.sigmoid(model(img_batch_t))[:, 1]
                score = score.cpu().numpy()
        else:
            score = []
            n_batch = img_batch.shape[0] // batch_size
            tail_batch = img_batch.shape[0] % batch_size
            with torch.no_grad():
                # for batch in tqdm(range(n_batch)):
                for batch in range(n_batch):
                    img_batch_t = img_batch[batch_size * batch: batch_size * (batch + 1)]
                    img_batch_t = torch.tensor(img_batch_t).float().to("cuda:0") / 255.0
                    score_t = F.sigmoid(model(img_batch_t))[:, 1]
                    score.append(score_t.cpu().numpy())
                img_batch_t = img_batch[-tail_batch:]
                img_batch_t = torch.tensor(img_batch_t).float().to("cuda:0") / 255.0
                score_t = F.sigmoid(model(img_batch_t))[:, 1]
                score.append(score_t.cpu().numpy())
                score = np.concatenate(score, axis=0)
        return score

    def test_datasets(self):
        # load all data
        data_dict = {}
        for dataset_flag in self.dataset_flags:
            img_batch, gt_batch, name_batch = self.data_tool.get_one_dataset(dataset_flag)
            data_dict[dataset_flag] = (img_batch, gt_batch, name_batch)

        # exit()
        # run and save result
        result_dict = {}
        th_start = self.threshold_list[0]
        th_end = self.threshold_list[1]
        th_step = self.threshold_list[2]
        for model_path in self.model_paths:
            # print(model_path)
            model = self.model_tool.get_test_func(self.model, model_path, self.dump_flag)
            for dataset_flag in self.dataset_flags:
                img_batch, gt_batch, name_batch = data_dict[dataset_flag]

                # print("==================get {} scores=================".format(dataset_flag))
                scores = self.get_scores(img_batch, model)
                # print("==================calculate result=================")
                result_dict[dataset_flag] = self.cal_result(scores, gt_batch, self.threshold_list)

            print_result = '\n'
            for th in range(th_start, th_end, th_step):
                threshold = th / 100
                result = "th: " + str("%.2f" % (threshold))
                for dataset_flag in self.dataset_flags:
                    result += " " + dataset_flag
                    if self.more_infos_flag:
                        result += " TP:" + str(result_dict[dataset_flag]["TP"][str(threshold)])
                        result += " TN:" + str(result_dict[dataset_flag]["TN"][str(threshold)])
                        result += " FP:" + str(result_dict[dataset_flag]["FP"][str(threshold)])
                        result += " FN:" + str(result_dict[dataset_flag]["FN"][str(threshold)])
                    # result += " Rec：" + str("%.3f" % (result_dict[dataset_flag]["Recall"][str(threshold)] * 100))
                    # result += " Acc：" + str("%.3f" % (result_dict[dataset_flag]["Accuracy"][str(threshold)] * 100))
                    # result += " Pre：" + str("%.3f" % (result_dict[dataset_flag]["Precision"][str(threshold)] * 100))
                    result += " T_R:" + str("%.3f" % (result_dict[dataset_flag]["Recall"][str(threshold)] * 100))
                    result += " M_R:" + str("%.3f" % (result_dict[dataset_flag]["mean_Recall"][str(threshold)] * 100))
                    result += " F_R:" + str("%.3f" % (result_dict[dataset_flag]["F_Recall"][str(threshold)] * 100))
                print_result += result + '\n'
            print(print_result)
    def get_QA_datapath(self, dataset_flag):
        if "honor_qa" in dataset_flag:
            return self.data_tool.honor_data_path
        if "badcase" in dataset_flag:
            return self.data_tool.honor_badcase_path
        if "zicai" in dataset_flag:
            return self.data_tool.honor_zicai_path


    def get_badcase(self, model_path, th, write=True):
        # load all data
        for dataset_flag in self.dataset_flags:
            img_batch, gt_batch, box_xy_batch, box_wh_batch, name_batch = self.data_tool.get_one_dataset(
                dataset_flag)
            model = self.model_tool.get_test_func(self.model, model_path, self.dump_flag)
            scores = self.get_scores(img_batch, box_xy_batch, box_wh_batch, model)

            data_name = self.get_QA_datapath(dataset_flag).split('/')[-1]
            print("threshhold:", th)
            rec = (scores > th).squeeze()
            gt = (gt_batch == 1)
            T_map = rec == gt
            for face_img, img_ori_path, score, TF in zip(img_batch, name_batch, scores, T_map):
                if TF:
                    continue
                img_ori = cv2.imread(img_ori_path, 0)
                save_ori_path = img_ori_path.replace(data_name, data_name+'_badcase').replace('.png', '_{}.png'.format(score))
                save_face_path = save_ori_path.replace('raw_data', 'QVGA')
                if write:
                    if not os.path.exists(os.path.dirname(save_ori_path)):
                        os.makedirs(os.path.dirname(save_ori_path))
                    if not os.path.exists(os.path.dirname(save_face_path)):
                        os.makedirs(os.path.dirname(save_face_path))
                    cv2.imwrite(save_ori_path, img_ori)
                    cv2.imwrite(save_face_path, face_img.squeeze()+128)
                    print(save_ori_path)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--exp', default="251110_lsnet_t", type=str)
    args = parser.parse_args()

    args.model = 'lsnet_t'
    # setup_logger('./', filename=f"eval_results_dirt/{args.exp}.txt", mode="o")
    model_dirpath = f'/data/workspace/git_research/lsnet/checkpoints/lsnet_t/*'
    model_paths = glob.glob(os.path.join(model_dirpath, "checkpoint_*"))
    model_paths = natsorted(model_paths, alg=ns.PATH)
    model_paths = model_paths
    # model_paths = [
    #     "/data/workspace/eye_chengdu/honor_4dir_train_log/20240307.1124base.tiny.addzicai0111.badcase0118.TF_balance.morebiyan.addkehu.augno3.blackmore.pretrain/models/epoch_3.pth",
    # ]
    print('epoch_num:{}'.format(len(model_paths)))

    from timm.models import create_model
    from model import build
    model = create_model('lsnet_t',num_classes=2,).to("cuda:0")


    # 测试阈值范围
    threshold_list = [30, 80, 1]

    # 测试标签范围
    # 80cm_no-glasses_outdoor_normallight_frontlight_hand_down_A-border_watch
    dataset_flags = []


    # dataset_flags += ["val_dirt"]
    # dataset_flags += ["glass_dirt:"]
    # dataset_flags += ["glass_dirt:.*_室外_.*_夜晚_.*"]
    # dataset_flags += ["glass_dirt:.*_室外_.*_夜晚_.*反"]
    dataset_flags += ["QA_dirt"]
    # dataset_flags += ["QA_dirt:.*_室外_.*_夜晚_.*"]
    # dataset_flags += ["QA_dirt:.*_室外_.*_夜晚_.*反"]


    # dataset_flags += ["val_dirt:.*_室内_.*"]
    # dataset_flags += ["val_dirt:.*_室外_.*"]

    # dataset_flags += ["val_occ:.*_0_.*"]
    # dataset_flags += ["val_occ:.*_[1234ABC]_.*"]


    dump_flag = True
    more_infos_flag = False

    cal_tool = CalResult(model_paths, model, threshold_list, dataset_flags, dump_flag, more_infos_flag)
    cal_tool.test_datasets()
    # cal_tool.get_badcase(model_paths[-1], 0.51)


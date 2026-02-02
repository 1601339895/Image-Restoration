import argparse
import os
import time
import numpy as np
import refile
import torch
import torch.distributed as dist
from dpflow import InputPipe, control
from frtrain.loss.basic_loss import l2_loss
from frtrain.misc import checkpoint, utils
from frtrain.train.lr import LRWorker
from frtrain.train.trainer import Trainer
import torch.nn.functional as F

from resnet_sigma_inf import InfNet

# ===================== 数据验证 =====================
def validate_training_data(data, minibatch, trainer):
    """
    验证训练数据完整性，适配原逻辑和头盔增强输出
    - 必要键：
        原始场景: 'images:cls', 'feat_tea:cls'
        头盔增强: 'images_helmet:cls', 'images_clean:cls', 'feat_tea:cls'
    """
    # 支持的键组合
    if 'images:cls' in data and 'feat_tea:cls' in data:
        required_keys = ['images:cls', 'feat_tea:cls']
    elif 'images_helmet:cls' in data and 'images_clean:cls' in data and 'feat_tea:cls' in data:
        required_keys = ['images_helmet:cls', 'images_clean:cls', 'feat_tea:cls']
    else:
        # 缺少必要键
        missing_keys = list(data.keys())  # 方便日志输出
        trainer.log(f"Missing required keys for batch {minibatch}. Available keys: {missing_keys}")
        return False, missing_keys

    # 检查每个键是否存在以及类型是否正确
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
            continue
        value = data[key]
        if not isinstance(value, np.ndarray):
            missing_keys.append(f"{key} type error: {type(value)}")
            continue
        # 图像检查
        if 'images' in key and len(value.shape) != 4:
            missing_keys.append(f"{key} shape error: {value.shape}, expected 4D")
        # 特征检查
        if 'feat' in key and len(value.shape) != 2:
            missing_keys.append(f"{key} shape error: {value.shape}, expected 2D")

    if missing_keys:
        trainer.log(f"Data validation failed (batch {minibatch}): {missing_keys}")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                trainer.log(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, '__len__'):
                trainer.log(f"  {key}: len={len(value)}, type={type(value)}")
            else:
                trainer.log(f"  {key}: type={type(value)}")
        return False, missing_keys

    # 验证通过
    shapes = {k: data[k].shape for k in required_keys}
    trainer.log(f"Data validation passed (batch {minibatch}): {shapes}")
    return True, []


# ===================== 虚拟数据 =====================
def create_dummy_data(batch_size, img_size, feat_size, device):
    dummy_img = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
    dummy_feat = torch.randn(batch_size, feat_size).to(device)
    return dummy_img, dummy_img, dummy_feat  # helmet, clean, feat_tea

# ===================== 主训练 =====================
def main(args, config):
    trainer = Trainer(args.local_rank, config)
    trainer.init()
    trainer.config_logger()

    train_tb = utils.TensorBoardLogger()
    log_dir = utils.oss2local(refile.smart_path_join(trainer.output_dir, "train_log"))
    train_tb.create(log_dir)

    trainer.log("Init student model...")
    model = InfNet(num_features=config.get("feat_dim", 256))
    model = model.to(trainer.local_rank)

    optimizer_cls = getattr(torch.optim, config["optimizer"]["type"])
    model_opt = optimizer_cls(
        params=[{"params": model.parameters()}],
        lr=config["lr_info"]["lr"],
        weight_decay=config["weight_decay"],
        **config["optimizer"]["params"]
    )

    trainer.log("load checkpoint...")
    checkpoint.load_checkpoint(model, model_opt, trainer)

    # 加载预训练模型
    if config.get("pretrained_path"):
        state = torch.load(config["pretrained_path"], map_location="cpu")
        model.load_state_dict(state, strict=False)
        trainer.log(f"Loaded pretrained: {config['pretrained_path']}")

    trainer.log("broadcast parameters...")
    dist.broadcast(model.parameters(), 0)


    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        device_ids=[trainer.local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True
    )
    model.train()

    pipe_name = "{}.{}.{}".format(
        config["provider_pipe_name"], os.getcwd().split("/")[-1], trainer.rank
    )
    in_pipe = InputPipe(pipe_name)
    in_pipe._meta = {"group_id": in_pipe.name}

    fp16 = config.get("fp16", True)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    data_error_count = 0
    total_batches = 0
    use_dummy_data = config.get("use_dummy_data", False)

    with control(io=[in_pipe]):
        lrworker = LRWorker(config["max_epoch"], config["lr_info"]["lr"], config["lr_info"]["lr_func"])
        trainer.log("Starting training loop...")

        if use_dummy_data:
            trainer.log("Running in DUMMY DATA mode for testing")

        for epoch in range(trainer.start_epoch, config["max_epoch"] + 1):
            lr = lrworker.get_lr(epoch)
            lrworker.set_lr(epoch, model_opt)

            for minibatch in range(config["num_batch_per_epoch"]):
                ts = time.time()
                try:
                    data = in_pipe.get()
                    total_batches += 1

                    if use_dummy_data:
                        img_helmet, img_clean, feat_tea = create_dummy_data(
                            batch_size=32,
                            img_size=config["input_size"],
                            feat_size=config["feat_dim"],
                            device=trainer.local_rank
                        )
                        trainer.log(f"Using dummy data: img_helmet {img_helmet.shape}, img_clean{img_clean.shape}, feat {feat_tea.shape}")
 
                    else:
                        data_valid, errors = validate_training_data(data, minibatch, trainer)
                        
                        if not data_valid:
                            data_error_count += 1
                            error_rate = data_error_count / total_batches * 100
                            
                            trainer.log(f" Data validation failed (error #{data_error_count}, rate: {error_rate:.1f}%)")
                            trainer.log(f"Errors: {errors}")
                            
                            if error_rate > 50:
                                trainer.log("Too many data errors, consider switching to dummy mode")
                            
                            continue

                        img_helmet = torch.from_numpy(data['images_helmet:cls'].copy().astype(np.float32)).to(trainer.local_rank)
                        img_clean  = torch.from_numpy(data['images_clean:cls'].copy().astype(np.float32)).to(trainer.local_rank)
                        feat_tea   = torch.from_numpy(data['feat_tea:cls'].copy()).to(trainer.local_rank)

                    td = time.time()

                    with torch.cuda.amp.autocast(enabled=fp16):
                        f_helmet = model(img_helmet)
                        f_clean  = model(img_clean)

                        # 学生特征统计
                        if minibatch % 50 == 0:
                            trainer.log(f"Student f_helmet output: {f_helmet.shape}, mean={f_helmet.mean():.4f}, std={f_helmet.std():.4f}")
                            trainer.log(f"Student f_clean output: {f_clean.shape}, mean={f_clean.mean():.4f}, std={f_clean.std():.4f}")

                        # 归一化
                        f_helmet_norm = F.normalize(f_helmet, dim=1)
                        f_clean_norm  = F.normalize(f_clean, dim=1)
                        tea_norm      = F.normalize(feat_tea, dim=1)

                        loss_all = 0
                        loss_info = {}

                        # 余弦蒸馏
                        cos_loss = 1 - torch.mean(torch.sum(f_helmet_norm * tea_norm, dim=1))
                        cos_loss *= config["l2_loss"].get("cos_weight", 1.0)

                        # L2 蒸馏
                        l2_feat_loss = F.mse_loss(f_helmet_norm, tea_norm) * config["l2_loss"].get("l2_weight", 1.0)

                        loss_all += cos_loss + l2_feat_loss
                        loss_info["distill_cos_loss"] = cos_loss
                        loss_info["distill_l2_loss"]  = l2_feat_loss

                        # 自我一致性损失（helmet vs clean）
                        sc_weight = config.get("self_consistency_weight", 0.5)
                        self_consistency_loss = F.mse_loss(f_helmet_norm, f_clean_norm) * sc_weight
                        loss_all += self_consistency_loss
                        loss_info["self_consistency_loss"] = self_consistency_loss

                        # 学生特征统计与对齐度量
                        if minibatch % 50 == 0:
                            # 使用原来的 f_norm 代表学生输出
                            f_norm = f_helmet_norm
                            cosine_sim = torch.mean(torch.sum(f_norm * tea_norm, dim=1))
                            l2_distance = torch.mean(torch.norm(f_norm - tea_norm, dim=1))
                            trainer.log(f"Feature alignment: cosine_sim={cosine_sim:.4f}, l2_dist={l2_distance:.4f}")
                            # 新增自我一致性统计
                            sc_cosine = torch.mean(torch.sum(f_helmet_norm * f_clean_norm, dim=1))
                            sc_l2     = torch.mean(torch.norm(f_helmet_norm - f_clean_norm, dim=1))
                            trainer.log(f"Self-consistency alignment: cos_sim={sc_cosine:.4f}, l2_dist={sc_l2:.4f}")

                        loss_info["sum_loss"] = loss_all


                    # ====== 反向 ======
                    model_opt.zero_grad()
                    scaler.scale(loss_all).backward()
                    scaler.unscale_(model_opt)
                    
                    if config["clip_grad"]["clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"]["max_norm"])
                    scaler.step(model_opt)
                    scaler.update()

                    te = time.time()
                    log_msg = f"Epoch[{epoch}/{config['max_epoch']}] Batch[{minibatch}/{config['num_batch_per_epoch']}] " \
                              f"Time_all:{te-ts:.3f} Data:{td-ts:.3f} | " + \
                              " | ".join([f"{k}:{v.item():.4f}" for k, v in loss_info.items()])
                    trainer.log(log_msg)

                except Exception as e:
                    data_error_count += 1
                    trainer.log(f"Training error in batch {minibatch}: {e}")
                    continue

            # Epoch summary
            if total_batches > 0:
                success_rate = (total_batches - data_error_count) / total_batches * 100
                trainer.log(f"Epoch {epoch} done | lr:{lr:.6f} | data_success:{success_rate:.1f}%")
            else:
                trainer.log(f"Epoch {epoch} done | lr:{lr:.6f} | NO DATA")

            # Save checkpoint
            if epoch % config["checkpoint_interval"] == 0 and trainer.rank == 0:
                checkpoint.save_checkpoint(model, model_opt, epoch, trainer, dump_flag=False)
                trainer.log(f"Checkpoint saved at epoch {epoch}")

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("-c", "--config", default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"))
    args = parser.parse_args()
    config = utils.load_config(args.config)
    if args.dummy:
        config["use_dummy_data"] = True
    main(args, config)

import argparse
import os
import time
import numpy as np
import refile
import torch
import torch.distributed as dist
import torch.nn.functional as F

from frtrain.loss.basic_loss import l2_loss
from frtrain.misc import checkpoint, utils
from frtrain.train.lr import LRWorker
from frtrain.train.trainer import Trainer

from resnet_sigma_inf import InfNet

# NEW: dataloader
from data.build_dataloader import build_train_dataloader


def validate_training_data(data, minibatch, trainer):
    """验证训练数据完整性"""
    required_keys = ['images:cls', 'feat_tea:cls']
    missing_keys = []

    for key in required_keys:
        if key not in data:
            missing_keys.append(key)

    if missing_keys:
        trainer.log(f"Missing required keys: {missing_keys}")
        trainer.log(f"Available keys: {list(data.keys())}")

        for key, value in data.items():
            if hasattr(value, 'shape'):
                trainer.log(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, '__len__'):
                trainer.log(f"  {key}: len={len(value)}, type={type(value)}")
            else:
                trainer.log(f"  {key}: {type(value)}")

        return False, missing_keys

    try:
        img_data = data['images:cls']
        feat_data = data['feat_tea:cls']

        if not isinstance(img_data, np.ndarray):
            return False, ['images:cls type error']
        if not isinstance(feat_data, np.ndarray):
            return False, ['feat_tea:cls type error']
        if len(img_data.shape) != 4:
            return False, ['images:cls shape error']
        if len(feat_data.shape) != 2:
            return False, ['feat_tea:cls shape error']

        return True, []

    except Exception as e:
        return False, [str(e)]


def create_dummy_data(batch_size, img_size, feat_size, device):
    img = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
    feat = torch.randn(batch_size, feat_size).to(device)
    return img, feat


def main(args, config):
    trainer = Trainer(args.local_rank, config)
    trainer.init()
    trainer.config_logger()

    train_tb = utils.TensorBoardLogger()
    log_dir = utils.oss2local(refile.smart_path_join(trainer.output_dir, "train_log"))
    train_tb.create(log_dir)

    trainer.log("init model")
    model = InfNet(num_features=256)

    optimizer = getattr(torch.optim, config["optimizer"]["type"])
    model_opt = optimizer(
        params=model.parameters(),
        lr=trainer.lr,
        weight_decay=trainer.weight_decay,
        **config["optimizer"]["params"],
    )

    model = model.to(trainer.local_rank)

    trainer.log("load checkpoint...")
    checkpoint.load_checkpoint(model, model_opt, trainer)

    if config.get("pretrained_path"):
        state = torch.load(config["pretrained_path"], map_location="cpu")
        model.load_state_dict(state, strict=False)
        trainer.log(f"loaded pretrained: {config['pretrained_path']}")


    if trainer.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[trainer.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        
    model.train()

    # -----------------------------
    # Build DataLoader (替代 dpflow)
    # -----------------------------
    trainer.log("building dataloader...")
    train_loader = build_train_dataloader(config)
    data_iter = iter(train_loader)
    trainer.log("dataloader ready")

    fp16 = config["fp16"]
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    data_error_count = 0
    total_batches = 0
    use_dummy_data = config.get("use_dummy_data", False)

    lrworker = LRWorker(trainer.max_epoch, trainer.lr, config["lr_info"]["lr_func"])
    trainer.log("Starting training loop...")

    for epoch in range(trainer.start_epoch, trainer.max_epoch + 1):
        lr = lrworker.get_lr(epoch)
        lrworker.set_lr(epoch, model_opt)

        for minibatch in range(trainer.num_batch_per_epoch):
            ts = time.time()

            try:
                data = next(data_iter)
                total_batches += 1

                # DataLoader 输出是 dict[dataset_name -> batch]
                # 取第一个 dataset（与原 dpflow 行为一致）
                if isinstance(data, dict) and isinstance(next(iter(data.values())), dict):
                    data = next(iter(data.values()))

                if use_dummy_data:
                    img, feat_tea = create_dummy_data(
                        batch_size=config["batch_size"] // trainer.get_world_size(),
                        img_size=config["input_size"],
                        feat_size=256,
                        device=trainer.local_rank,
                    )
                else:
                    valid, errors = validate_training_data(data, minibatch, trainer)
                    if not valid:
                        data_error_count += 1
                        continue

                    img = torch.from_numpy(
                        data['images:cls'].astype(np.float32)
                    ).to(trainer.local_rank)

                    feat_tea = torch.from_numpy(
                        data['feat_tea:cls']
                    ).to(trainer.local_rank)

                td = time.time()

                with torch.cuda.amp.autocast(enabled=fp16):
                    f = model(img)

                    loss_all = 0
                    loss_info = {}

                    if config.get("l2_loss"):
                        f_norm = F.normalize(f, dim=1)
                        tea_norm = F.normalize(feat_tea, dim=1)

                        cos_loss = 1 - torch.mean(torch.sum(f_norm * tea_norm, dim=1))
                        cos_loss *= config["l2_loss"].get("cos_weight", 1.0)

                        l2_feat_loss = F.mse_loss(
                            f_norm, tea_norm
                        ) * config["l2_loss"].get("l2_weight", 1.0)

                        loss_all = cos_loss + l2_feat_loss
                        loss_info["distill_cos_loss"] = cos_loss
                        loss_info["distill_l2_loss"] = l2_feat_loss

                    loss_info["sum_loss"] = loss_all

                model_opt.zero_grad()
                scaler.scale(loss_all).backward()
                scaler.unscale_(model_opt)

                if config["clip_grad"]["clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["clip_grad"]["max_norm"]
                    )

                scaler.step(model_opt)
                scaler.update()

                te = time.time()

                log_print = (
                    f"epoch: [{epoch}/{trainer.max_epoch}] | "
                    f"minibatch: [{minibatch}/{trainer.num_batch_per_epoch}] | "
                    f"time_all: {te - ts:.3f} | time_data: {td - ts:.3f} | "
                )
                log_print += " | ".join(
                    [f"{k}: {v:.4f}" for k, v in loss_info.items()]
                )
                trainer.log(log_print)

                if trainer.rank == 0:
                    step = epoch * trainer.num_batch_per_epoch + minibatch + 1
                    train_tb.write("learning_rate", lr, step)
                    for k, v in loss_info.items():
                        train_tb.write(k, v, step)

            except Exception as e:
                trainer.log(f"Training error in batch {minibatch}: {e}")
                continue

        if epoch % config["checkpoint_interval"] == 0 and trainer.rank == 0:
            checkpoint.save_checkpoint(
                model, model_opt, epoch, trainer, dump_flag=False
            )
            trainer.log(f"Checkpoint saved at epoch {epoch}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser("InfNet-distill")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
    )

    args = parser.parse_args()
    config = utils.load_config(args.config)

    if args.dummy:
        config["use_dummy_data"] = True

    main(args, config)

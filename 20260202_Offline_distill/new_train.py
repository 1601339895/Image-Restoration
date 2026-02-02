import argparse
import os
import time
import numpy as np
import refile
import torch
import torch.distributed as dist
from dpflow import InputPipe, control
from frtrain.misc import checkpoint, utils
from frtrain.train.lr import LRWorker
from frtrain.train.trainer import Trainer
import torch.nn.functional as F

from freeze_strategy import FreezeStrategy, GradientMonitor
from resnet_sigma_inf import InfNet



def validate_training_data(data, minibatch, trainer):
    """验证训练数据完整性"""
    required_keys = ['images:cls', 'feat_tea:cls']
    missing_keys = []
    
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        trainer.log(f"❌ Missing required keys: {missing_keys}")
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
            trainer.log(f"images:cls is not numpy array: {type(img_data)}")
            return False, ['images:cls type error']
            
        if not isinstance(feat_data, np.ndarray):
            trainer.log(f"feat_tea:cls is not numpy array: {type(feat_data)}")
            return False, ['feat_tea:cls type error']
            
        if len(img_data.shape) != 4:
            trainer.log(f"images:cls wrong shape: {img_data.shape}, expected 4D")
            return False, ['images:cls shape error']
            
        if len(feat_data.shape) != 2:
            trainer.log(f"feat_tea:cls wrong shape: {feat_data.shape}, expected 2D")
            return False, ['feat_tea:cls shape error']
            
        trainer.log(f"✅ Data validation passed: img {img_data.shape}, feat {feat_data.shape}")
        return True, []
        
    except Exception as e:
        trainer.log(f"❌ Data validation error: {e}")
        import traceback
        trainer.log(traceback.format_exc())
        return False, [str(e)]


def create_dummy_data(batch_size, img_size, feat_size, device):
    """创建虚拟数据用于测试"""
    dummy_img = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
    dummy_feat = torch.randn(batch_size, feat_size).to(device)
    return dummy_img, dummy_feat


def main(args, config):
    trainer = Trainer(args.local_rank, config)
    trainer.init()
    trainer.config_logger()

    train_tb = utils.TensorBoardLogger()
    log_dir = utils.oss2local(refile.smart_path_join(trainer.output_dir, "train_log"))
    train_tb.create(log_dir)

    # ==================== 初始化模型 ====================
    trainer.log("="*80)
    trainer.log("🚀 Initializing Feature Module with Freeze Strategy")
    trainer.log("="*80)
    
    model = InfNet(num_features=256)

    optimizer = getattr(torch.optim, config["optimizer"]["type"])
    model_opt = optimizer(
        params=[{"params": model.parameters()}],
        lr=trainer.lr,
        weight_decay=trainer.weight_decay,
        **config["optimizer"]["params"],
    )
    model = model.to(trainer.local_rank)
    
    # ==================== 加载预训练权重 ====================
    trainer.log("📥 Loading checkpoint...")
    checkpoint.load_checkpoint(model, model_opt, trainer)

    if config.get("pretrained_path"):
        state = torch.load(config["pretrained_path"], map_location="cpu")
        model.load_state_dict(state, strict=False)
        trainer.log(f"✅ Loaded pretrained: {config['pretrained_path']}")
    
    # ==================== 应用冻结策略（关键步骤） ====================
    freeze_config = config.get("freeze_config", {})
    
    if freeze_config.get("enable", True):
        freeze_strategy = FreezeStrategy(model, freeze_config, trainer.log)
        freeze_strategy.apply_freeze()
        
        # 验证预训练权重是否正确加载
        freeze_strategy.validate_pretrained_loading()
        
        # 创建梯度监控器
        grad_monitor = GradientMonitor(model, trainer.log)
        
        # 获取冻结信息用于TensorBoard
        freeze_info = freeze_strategy.get_freeze_info()
    else:
        freeze_strategy = None
        grad_monitor = None
        freeze_info = {}
        trainer.log("⚠️  Freeze strategy disabled")
    
    # ==================== DDP 初始化 ====================
    trainer.log("📡 Broadcasting parameters...")
    for p in model.parameters():
        dist.broadcast(p, 0)

    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        broadcast_buffers=False,
        device_ids=[trainer.local_rank],
        find_unused_parameters=True
    )
    model.train()

    # ==================== 配置数据管道 ====================
    pipe_name = "{}.{}.{}".format(
        config["provider_pipe_name"], os.getcwd().split("/")[-1], trainer.rank
    )
    in_pipe = InputPipe(pipe_name)
    in_pipe._meta = {"group_id": pipe_name}

    fp16 = config["fp16"]
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    
    # 数据统计
    data_error_count = 0
    total_batches = 0
    use_dummy_data = config.get("use_dummy_data", False)
    
    with control(io=[in_pipe]):
        lrworker = LRWorker(trainer.max_epoch, trainer.lr, config["lr_info"]["lr_func"])
        trainer.log("🎯 Starting training loop...")
        trainer.log(f"   Freeze enabled: {freeze_config.get('enable', True)}")
        trainer.log(f"   Freeze strategy: {freeze_config.get('strategy', 'unknown')}")
        
        if use_dummy_data:
            trainer.log("⚠️  Running in DUMMY DATA mode for testing")
        
        # ==================== 训练循环 ====================
        for epoch in range(trainer.start_epoch, trainer.max_epoch + 1):
            lr = lrworker.get_lr(epoch)
            lrworker.set_lr(epoch, model_opt)

            epoch_loss_dict = {}
            epoch_batch_count = 0

            for minibatch in range(trainer.num_batch_per_epoch):
                ts = time.time()
                
                try:
                    if use_dummy_data:
                        # 测试模式：使用虚拟数据
                        img, feat_tea = create_dummy_data(
                            batch_size=config["batch_size"] // trainer.world_size, 
                            img_size=config["input_size"], 
                            feat_size=256, 
                            device=trainer.local_rank
                        )
                        if minibatch % 100 == 0:
                            trainer.log(f"🧪 Using dummy data: img {img.shape}, feat {feat_tea.shape}")
                    else:
                        # 正常模式：获取真实数据
                        data = in_pipe.get()
                        total_batches += 1
                        epoch_batch_count += 1
                        
                        # 数据验证
                        data_valid, errors = validate_training_data(data, minibatch, trainer)
                        
                        if not data_valid:
                            data_error_count += 1
                            error_rate = data_error_count / total_batches * 100 if total_batches > 0 else 0
                            
                            if minibatch % 50 == 0:
                                trainer.log(f"❌ Data validation failed (error #{data_error_count}, rate: {error_rate:.1f}%)")
                            
                            if error_rate > 50:
                                trainer.log("⚠️  Too many data errors, consider checking data pipeline")
                            
                            continue
                        
                        # 数据转换
                        img = torch.from_numpy(data['images:cls'].copy().astype(np.float32)).to(trainer.local_rank)
                        feat_tea = torch.from_numpy(data['feat_tea:cls'].copy()).to(trainer.local_rank)
                    
                    td = time.time()

                    # ==================== 前向传播 ====================
                    with torch.cuda.amp.autocast(enabled=fp16):
                        f = model(img)
                        
                        # NaN/Inf 检测
                        if torch.isnan(f).any() or torch.isinf(f).any():
                            trainer.log("⚠️  NaN/Inf detected in model output! Skipping batch")
                            continue
                        
                        loss_all = 0
                        loss_info = {}
                        
                        if config.get("l2_loss"):
                            f_norm = F.normalize(f, dim=1)
                            tea_norm = F.normalize(feat_tea, dim=1)

                            # 余弦蒸馏
                            cos_loss = 1 - torch.mean(torch.sum(f_norm * tea_norm, dim=1))
                            cos_loss = cos_loss * config["l2_loss"].get("cos_weight", 1.0)

                            # L2 蒸馏
                            l2_feat_loss = F.mse_loss(f_norm, tea_norm) * config["l2_loss"].get("l2_weight", 1.0)

                            distill_loss = cos_loss + l2_feat_loss
                            loss_all += distill_loss

                            loss_info["distill_cos_loss"] = cos_loss.detach()
                            loss_info["distill_l2_loss"] = l2_feat_loss.detach()
                            
                            if minibatch % 50 == 0:
                                cosine_sim = torch.mean(torch.sum(f_norm * tea_norm, dim=1))
                                l2_distance = torch.mean(torch.norm(f_norm - tea_norm, dim=1))
                                loss_info["cosine_similarity"] = cosine_sim.detach()
                                loss_info["l2_distance"] = l2_distance.detach()
                                
                        loss_info["sum_loss"] = loss_all.detach()

                    # ==================== 反向传播 ====================
                    model_opt.zero_grad()
                    scaler.scale(loss_all).backward()
                    
                    # 梯度监控（仅当启用冻结时）
                    step = epoch * trainer.num_batch_per_epoch + minibatch + 1
                    if grad_monitor is not None:
                        grad_monitor.monitor_gradients(step, train_tb)
                    
                    scaler.unscale_(model_opt)
                    
                    # 梯度裁剪 - 仅应用于可训练参数
                    if config["clip_grad"]["clip"]:
                        trainable_params = [p for p in model.parameters() if p.requires_grad]
                        if trainable_params:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params, config["clip_grad"]["max_norm"]
                            )
                    scaler.step(model_opt)
                    scaler.update()

                    te = time.time()
                    
                    # ==================== 日志记录 ====================
                    log_print = (f"epoch: [{epoch}/{trainer.max_epoch}] | "
                                 f"minibatch: [{minibatch}/{trainer.num_batch_per_epoch}] | "
                                 f"lr: {lr:.6f} | "
                                 f"time_all: {te - ts:.3f}s | time_data: {td - ts:.3f}s |")
                    log_print += " | ".join([f"{k}: {v.item() if hasattr(v, 'item') else v:.4f}" 
                                            for k, v in loss_info.items()])
                    
                    # 数据流效率监控
                    data_time_ratio = (td - ts) / (te - ts) * 100 if (te - ts) > 0 else 0
                    if data_time_ratio > 50:
                        log_print += f" | ⚠️DATA_SLOW: {data_time_ratio:.1f}%"
                    
                    # 数据质量监控
                    if not use_dummy_data and total_batches > 0:
                        data_success_rate = (total_batches - data_error_count) / total_batches * 100
                        if data_success_rate < 90:
                            log_print += f" | ⚠️DATA_SUCCESS: {data_success_rate:.1f}%"
                    
                    if minibatch % 20 == 0:
                        trainer.log(log_print)

                    # ==================== TensorBoard 记录 ====================
                    if trainer.rank == 0:
                        train_tb.write("learning_rate", lr, step)
                        for k, v in loss_info.items():
                            v_item = v.item() if hasattr(v, 'item') else v
                            train_tb.write(k, v_item, step)
                        
                        # 额外监控指标
                        train_tb.write("data_time_ratio", data_time_ratio, step)
                        if not use_dummy_data and total_batches > 0:
                            data_success_rate = (total_batches - data_error_count) / total_batches * 100
                            train_tb.write("data_success_rate", data_success_rate, step)
                    
                    # 累计loss用于epoch统计
                    for k, v in loss_info.items():
                        v_item = v.item() if hasattr(v, 'item') else v
                        if k not in epoch_loss_dict:
                            epoch_loss_dict[k] = []
                        epoch_loss_dict[k].append(v_item)
                
                except KeyError as e:
                    data_error_count += 1
                    trainer.log(f"❌ KeyError in batch {minibatch}: {e}")
                    if 'data' in locals():
                        trainer.log(f"Available keys: {list(data.keys())}")
                    continue
                    
                except Exception as e:
                    trainer.log(f"❌ Training error in batch {minibatch}: {str(e)}")
                    import traceback
                    trainer.log(traceback.format_exc())
                    continue

            # ==================== Epoch 总结 ====================
            if total_batches > 0:
                epoch_success_rate = (total_batches - data_error_count) / total_batches * 100
            else:
                epoch_success_rate = 0
            
            epoch_summary = f"\n{'='*80}\n"
            epoch_summary += f"📊 Epoch {epoch} Summary:\n"
            epoch_summary += f"   Learning Rate: {lr:.8f}\n"
            epoch_summary += f"   Data Success Rate: {epoch_success_rate:.1f}%\n"
            epoch_summary += f"   Batches Processed: {epoch_batch_count}/{trainer.num_batch_per_epoch}\n"
            
            # 平均loss
            if epoch_loss_dict:
                for k, v_list in epoch_loss_dict.items():
                    if v_list:
                        avg_v = np.mean(v_list)
                        epoch_summary += f"   Avg {k}: {avg_v:.6f}\n"
            
            # 冻结策略信息
            if freeze_strategy is not None and (epoch % 10 == 0 or epoch == trainer.max_epoch):
                freeze_info = freeze_strategy.get_freeze_info()
                epoch_summary += f"\n   🔒 Freeze Strategy Info:\n"
                epoch_summary += f"      Strategy: {freeze_config.get('strategy', 'unknown')}\n"
                epoch_summary += f"      Frozen Params: {freeze_info.get('frozen_ratio', 0):.1%} ({freeze_info.get('frozen_count', 0):,} params)\n"
                epoch_summary += f"      Trainable Params: {freeze_info.get('trainable_ratio', 0):.1%} ({freeze_info.get('trainable_count', 0):,} params)\n"
                
                # 显示关键层的冻结状态
                if 'layer_status' in freeze_info:
                    epoch_summary += f"      Layer Status:\n"
                    for layer_name, status in freeze_info['layer_status'].items():
                        epoch_summary += f"         {layer_name}: {'FROZEN' if status['frozen'] else 'TRAINABLE'} ({status['param_count']:,} params)\n"
            
            epoch_summary += f"{'='*80}\n"
            trainer.log(epoch_summary)
            
            # ==================== 保存Checkpoint ====================
            if epoch % config["checkpoint_interval"] == 0:
                if trainer.rank == 0:
                    checkpoint.save_checkpoint(model, model_opt, epoch, trainer, dump_flag=False)
                    trainer.log(f"✅ Checkpoint saved at epoch {epoch}")

    dist.destroy_process_group()
    trainer.log("🎉 Training completed successfully!")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="InfNet-distill with Freeze Strategy")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--dummy", action="store_true", help="Use dummy data for testing")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
    )
    args = parser.parse_args()
    config = utils.load_config(args.config)
    
    # 命令行覆盖
    if args.dummy:
        config["use_dummy_data"] = True
    
    main(args, config)
import argparse
import os
import pathlib
from typing import Tuple

# Helpers
def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def base_parser():
    parser = argparse.ArgumentParser()

    # Basic training settings
    parser.add_argument('--model', type=str, required=True, choices=["model_s", "model"],
                        help='Model to use: model_s (MoCE_IR_S/Light) or model (MoCE_IR/Heavy).')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs (auto-adjust by train_scene).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--de_type', nargs='+', default=[], help='Degradation types for training/testing (auto-set by train_scene).')
    parser.add_argument('--trainset', type=str, default="standard", 
                        choices=["standard", "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"],
                        help='Training dataset type (auto-adjust by train_scene):\n'
                             'CDD11_all: all 11 degradations | CDD11_single: 4 single degradations |\n'
                             'CDD11_double: 5 double degradations | CDD11_triple: 2 triple degradations.')
    parser.add_argument('--patch_size', type=int, default=128, help='Input patch size (CDD-11 uses 1080×720, patch=256 recommended).')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    parser.add_argument('--accum_grad', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint (path suffix in ckpt_dir).')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='Fine-tune from checkpoint (path suffix in ckpt_dir).')
    parser.add_argument('--checkpoint_id', type=str, default="", help='Custom checkpoint ID for saving.')
    parser.add_argument('--benchmarks', nargs='+', default=[], help='Which benchmarks to test on (e.g., cdd11_single cdd11_double).')
    parser.add_argument('--save_results', action="store_true", help="Save restored outputs during testing.")

    # Paths
    parser.add_argument('--data_file_dir', type=str, default=os.path.join(pathlib.Path.home(), "datasets"), 
                        help='Root path to all datasets (CDD-11 should be under ${data_file_dir}/cdd11).')
    parser.add_argument('--output_path', type=str, default="output/", help='Output save path for test results.')
    parser.add_argument('--wblogger', action="store_true", help='Log training to Weights & Biases.')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help='Root directory for saving checkpoints.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for training (>=1).')

    # ========== CDD-11 场景配置 ==========
    parser.add_argument('--train_scene', type=str, required=True, choices=['three', 'five', 'cdd11'],
                        help='Training scene (match paper):\n'
                             'three: 3 degradations (dehaze/derain/denoise) |\n'
                             'five: 5 degradations (dehaze/derain/denoise/deblur/synllie) |\n'
                             'cdd11: CDD-11 composite degradations (11 types).')
    # ========== ALL-in-One 任务编码配置 ==========
    parser.add_argument('--task_emb_dim', type=int, default=16, 
                        help='Dimension of task embedding (fixed for all scenes, 16 for CDD-11 11 tasks).')

    return parser

def model_ir_s(parser):
    """MoCE_IR_S (Light model) parameters (match CDD-11 paper)"""
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8])
    parser.add_argument('--heads', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--gate_type', type=str, default="elementwise", 
                        help='GateType: None/headwise/elementwise')
    parser.add_argument('--ffn_scales', type=int, default=2, 
                        help='FFN分块数量，控制GatedFFNWithSplitConv的scales参数')
    parser.add_argument('--LoRa_ffn_ratio', type=float, default=0.5, 
                        help='LoRA FFN层压缩比例 (CDD-11 training: 0.5 recommended)')
    parser.add_argument('--LoRa_attn_ratio', type=float, default=0.8, 
                        help='LoRA Attention层压缩比例 (CDD-11 training: 0.8 recommended)')
    return parser

def model_ir(parser):
    """MoCE_IR (Heavy model) parameters (match CDD-11 paper)"""
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[6, 8, 8, 10])  # 增强Heavy模型适配CDD-11
    parser.add_argument('--heads', nargs='+', type=int, default=[2, 4, 6, 8])       # 适配11类任务的注意力头数
    parser.add_argument('--num_refinement_blocks', type=int, default=6)             # 更多细化块适配复合退化
    parser.add_argument('--gate_type', type=str, default="elementwise",
                        help='GateType: None/headwise/elementwise')
    parser.add_argument('--ffn_scales', type=int, default=2, 
                        help='FFN分块数量，控制GatedFFNWithSplitConv的scales参数')
    parser.add_argument('--LoRa_ffn_ratio', type=float, default=0.5, 
                        help='LoRA FFN层压缩比例 (CDD-11 training: 0.5 recommended)')
    parser.add_argument('--LoRa_attn_ratio', type=float, default=0.8, 
                        help='LoRA Attention层压缩比例 (CDD-11 training: 0.8 recommended)')
    return parser

def train_options():
    # Step1: 解析基础参数（用于选择模型配置）
    base_parser_instance = base_parser()
    base_args, _ = base_parser_instance.parse_known_args()
    
    # Step2: 根据模型类型加载对应参数
    if base_args.model == "model_s":
        parser = model_ir_s(base_parser_instance)
    elif base_args.model == "model":
        parser = model_ir(base_parser_instance)
    else:
        raise NotImplementedError(f"Model '{base_args.model}' not supported (only model_s/model).")

    # Step3: 解析所有参数
    options = parser.parse_args()

    # Step4: 根据train_scene自动配置核心参数（适配CDD-11 11类退化）
    ## 场景1: Three degradations (去雾+去雨+去噪)
    if options.train_scene == 'three':
        options.de_type = ['dehaze', 'derain', 'denoise_15', 'denoise_25', 'denoise_50']
        options.epochs = 120
        options.trainset = "standard"
        options.task_num = 5  # 3类核心任务（去噪含3个sigma）
    ## 场景2: Five degradations (去雾+去雨+去噪+去模糊+低光增强)
    elif options.train_scene == 'five':
        options.de_type = ['dehaze', 'derain', 'denoise_15', 'denoise_25', 'denoise_50', 'deblur', 'synllie']
        options.epochs = 120
        options.trainset = "standard"
        options.task_num = 7  # 5类核心任务
    ## 场景3: CDD-11 (11类复合退化，按single/double/triple细分)
    elif options.train_scene == 'cdd11':
        # 适配CDD-11的11类退化（按trainset细分）
        if options.trainset == "CDD11_all":
            options.de_type = [
                'low', 'haze', 'rain', 'snow', 
                'low+haze', 'low+rain', 'low+snow', 'haze+rain', 'haze+snow',
                'low+haze+rain', 'low+haze+snow'
            ]
            options.task_num = 11  # 全部11类退化
        elif options.trainset == "CDD11_single":
            options.de_type = ['low', 'haze', 'rain', 'snow']
            options.task_num = 4   # 4类单一退化
        elif options.trainset == "CDD11_double":
            options.de_type = ['low+haze', 'low+rain', 'low+snow', 'haze+rain', 'haze+snow']
            options.task_num = 5   # 5类双重退化
        elif options.trainset == "CDD11_triple":
            options.de_type = ['low+haze+rain', 'low+haze+snow']
            options.task_num = 2   # 2类三重退化
        options.epochs = 200  # CDD-11训练200 epoch（匹配论文）
        # CDD-11推荐更大patch size（适配1080×720分辨率）
        if options.patch_size < 256:
            options.patch_size = 256
            print(f"Warning: CDD-11 uses 1080×720 images, patch_size auto-adjusted to {options.patch_size}")

    # Step5: 梯度累积适配（保持原有逻辑）
    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad
        if options.batch_size < 1:
            raise ValueError(f"Batch size {options.batch_size} too small after accumulation (accum_grad={options.accum_grad}).")

    # Step6: 生成唯一时间戳（含CDD-11细分类型）
    from datetime import datetime
    options.time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + \
                         f"_{options.train_scene}_{options.trainset}_{options.model}"

    return options
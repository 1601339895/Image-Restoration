#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模块级测试脚本 - 用于排查CUDA错误"""

import torch
import torch.nn as nn
import sys
import os

# 设置CUDA同步模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 80)
print("FrequencyAwareBlock 调试测试")
print("=" * 80)

# 环境检查
print("\n【环境信息】")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
else:
    print("警告：CUDA不可用，将使用CPU测试")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from FPN_Restormer_CA_CNN_Encoder import (
        FrequencyAwareBlock,
        Light_FFT_DSConv_Block,
        Restormer_FFT_DSConv_Fusion
    )
    print("\n✅ 成功导入模块")
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试1：原始Block（基线对比）
# ============================================================================
print("\n" + "=" * 80)
print("【测试1】Light_FFT_DSConv_Block（原始编码器）")
print("=" * 80)

try:
    block_original = Light_FFT_DSConv_Block(dim=96, bias=False, dilation_rate=1).to(device)
    x = torch.randn(2, 96, 64, 64).to(device)

    with torch.no_grad():
        out = block_original(x)

    print(f"✅ 成功！")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in block_original.parameters()) / 1e3:.1f}K")

except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n提示：如果原始Block也失败，可能是环境问题而非代码问题")

# ============================================================================
# 测试2：FrequencyAwareBlock逐步测试
# ============================================================================
print("\n" + "=" * 80)
print("【测试2】FrequencyAwareBlock（改进编码器）- 逐步测试")
print("=" * 80)

try:
    block_freq = FrequencyAwareBlock(dim=96, bias=False, dilation_rate=1).to(device)
    x = torch.randn(2, 96, 64, 64).to(device)

    print("\n步骤1: 通道拆分...")
    spatial_x, freq_x = x.chunk(2, dim=1)
    print(f"  ✅ spatial_x: {spatial_x.shape}")
    print(f"  ✅ freq_x: {freq_x.shape}")

    print("\n步骤2: FFT变换...")
    fft = torch.fft.fft2(freq_x, dim=(-2, -1))
    print(f"  ✅ fft: {fft.shape} (复数类型)")

    print("\n步骤3: 幅度和相位提取...")
    fft_mag = torch.abs(fft)
    fft_phase = torch.angle(fft)
    print(f"  ✅ fft_mag: {fft_mag.shape}")
    print(f"  ✅ fft_phase: {fft_phase.shape}")

    # 检查数值稳定性
    if torch.isnan(fft_mag).any():
        print("  ⚠️ 警告：fft_mag包含NaN")
    if torch.isnan(fft_phase).any():
        print("  ⚠️ 警告：fft_phase包含NaN")
    if torch.isinf(fft_phase).any():
        print("  ⚠️ 警告：fft_phase包含Inf")

    print("\n步骤4: 拼接幅度和相位...")
    fft_mag_phase = torch.cat([fft_mag, fft_phase], dim=1)
    print(f"  ✅ fft_mag_phase: {fft_mag_phase.shape}")
    print(f"     预期: (B=2, C=96, H=64, W=64)")

    print("\n步骤5: 完整前向传播...")
    with torch.no_grad():
        out = block_freq(x)

    print(f"  ✅ 成功！")
    print(f"     输入: {x.shape}")
    print(f"     输出: {out.shape}")
    print(f"     参数量: {sum(p.numel() for p in block_freq.parameters()) / 1e3:.1f}K")

    # 验证输出
    assert out.shape == x.shape, "输出形状应与输入相同"
    print("\n  ✅ 形状验证通过")

except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n调试建议：")
    print("1. 检查上述哪一步失败")
    print("2. 使用 CUDA_LAUNCH_BLOCKING=1 运行以获取准确堆栈")
    print("3. 查看 CUDA错误排查指南.md 获取详细解决方案")

# ============================================================================
# 测试3：不同维度和膨胀率测试
# ============================================================================
print("\n" + "=" * 80)
print("【测试3】不同维度和膨胀率组合")
print("=" * 80)

test_configs = [
    {'dim': 48, 'dilation': 1, 'name': 'level1'},
    {'dim': 96, 'dilation': 2, 'name': 'level2'},
    {'dim': 192, 'dilation': 4, 'name': 'level3'},
    {'dim': 384, 'dilation': 8, 'name': 'latent'},
]

for config in test_configs:
    try:
        dim = config['dim']
        dilation = config['dilation']
        name = config['name']

        block = FrequencyAwareBlock(dim=dim, bias=False, dilation_rate=dilation).to(device)
        x = torch.randn(1, dim, 32, 32).to(device)

        with torch.no_grad():
            out = block(x)

        params = sum(p.numel() for p in block.parameters()) / 1e6
        print(f"✅ {name:8s} (dim={dim:3d}, dilation={dilation}): "
              f"输入{tuple(x.shape)} -> 输出{tuple(out.shape)}, "
              f"参数={params:.2f}M")

    except Exception as e:
        print(f"❌ {name} (dim={dim}, dilation={dilation}) 失败: {e}")

# ============================================================================
# 测试4：完整模型测试
# ============================================================================
print("\n" + "=" * 80)
print("【测试4】完整模型前向传播")
print("=" * 80)

test_model_configs = [
    {'use_freq': False, 'task_aware': False, 'name': '原始模型'},
    {'use_freq': True, 'task_aware': False, 'name': 'FrequencyAware'},
    {'use_freq': True, 'task_aware': True, 'name': '完整版（推荐）'},
]

for config in test_model_configs:
    print(f"\n测试配置: {config['name']}")
    print("-" * 40)
    try:
        model = Restormer_FFT_DSConv_Fusion(
            inp_channels=3,
            out_channels=3,
            dim=48,  # 使用较小dim以节省显存
            num_blocks=[2, 3, 3, 4],  # 减少block数量
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
            fusion_type="PAFPN",
            gate_type="elementwise",
            use_frequency_aware=config['use_freq'],
            task_aware_fusion=config['task_aware']
        ).to(device)

        inp = torch.randn(1, 3, 128, 128).to(device)  # 使用较小图像尺寸

        with torch.no_grad():
            out = model(inp)

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 成功！")
        print(f"   输入: {tuple(inp.shape)}")
        print(f"   输出: {tuple(out.shape)}")
        print(f"   参数量: {params:.2f}M")

        if device == 'cuda':
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"   显存占用: {mem_allocated:.1f}MB")

    except Exception as e:
        print(f"❌ 失败: {e}")
        if device == 'cuda':
            print(f"   显存状态: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ============================================================================
# 显存压力测试（可选）
# ============================================================================
if device == 'cuda':
    print("\n" + "=" * 80)
    print("【测试5】显存压力测试（可选）")
    print("=" * 80)
    print("提示：如果前面测试都通过，但实际训练时OOM，可能需要：")
    print("  1. 减小batch_size")
    print("  2. 减小图像尺寸")
    print("  3. 减小dim参数")
    print("  4. 使用梯度检查点（gradient checkpointing）")
    print("  5. 使用混合精度训练（AMP）")

    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU总显存: {total_mem:.1f}GB")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("【测试总结】")
print("=" * 80)
print("\n如果所有测试通过：")
print("  ✅ 代码逻辑正确")
print("  ✅ 维度匹配正确")
print("  ✅ 可以开始训练")
print("\n如果部分测试失败：")
print("  1. 查看上述具体错误信息")
print("  2. 参考 CUDA错误排查指南.md")
print("  3. 使用 CUDA_LAUNCH_BLOCKING=1 重新运行")
print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
